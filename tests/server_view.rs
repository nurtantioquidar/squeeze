//! The proxy's `--context-edits cold` against an emulated API (see
//! `common::server`): a session logged the way Claude Code logs one is rebuilt
//! as `squeeze replay --cache-sim` rebuilds it, paced like Fermat's runs on a
//! fake clock, and sent through the real proxy to an API that applies the
//! edits, caches the edited prompt and checks thinking. Free and offline.

mod common;

use axum::http::{HeaderMap, HeaderValue};
use common::server::{self, Api, PLACEHOLDER, Served};
use common::{FakeClock, Recorded};
use serde_json::{Value, json};
use squeeze::cachesim::{self, Rebuilt, Regime, SimConfig};
use squeeze::ctxmgmt::CtxConfig;
use squeeze::proxy::{Proxy, ProxyConfig, Totals};
use squeeze::store::Store;
use squeeze::transform::TransformConfig;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

const BETAS: &str = "claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27,\
                     per-turn-control-2026-07-01,thinking-token-count-2026-05-13,effort-2025-11-24";

/// Writes a session the way Claude Code 2.1.318 logs one: the request shape,
/// entries for prompts, responses and tool results, a blob for its effort
/// message, and each request as an `api-request` delta against the one
/// before, with its parameters and cache markers. Usage counts tokens as the
/// emulated API does, so the simulator's fit recovers the emulator's counts.
struct Transcript {
    lines: Vec<Value>,
    /// What the next request sends, in the form Claude Code logs.
    messages: Vec<Value>,
    sent: usize,
    t: u64,
    requests: usize,
    /// Tokens of the system prompt and tools.
    head: f64,
}

impl Transcript {
    fn new() -> Self {
        let system: Vec<Value> = ["You are a Claude agent.", "CWD: /repo\nDate: 2026-11-04"]
            .map(|t| json!({ "type": "text", "text": t }))
            .into();
        let tools: Vec<Value> = ["Bash", "Read", "Grep", "Edit", "TodoWrite"]
            .map(
                |n| json!({ "name": n, "description": format!("The {n} tool."), "input_schema": { "type": "object" } }),
            )
            .into();
        let head = system.iter().chain(&tools).map(server::tokens).sum();
        let shape =
            json!({ "type": "api-request-shape", "shapeHash": "s1", "shape": { "system": system, "tools": tools } });
        Self { lines: vec![shape], messages: Vec::new(), sent: 0, t: 0, requests: 0, head }
    }

    fn prompt(&mut self, text: String) {
        self.lines.push(json!({ "type": "user", "uuid": format!("u{}", self.lines.len()),
                                "message": { "role": "user", "content": text } }));
        self.messages.push(json!({ "role": "user", "content": text }));
        if self.requests == 0 {
            let effort = json!({ "role": "system", "content": [], "output_config": { "effort": "medium" } });
            self.lines.push(
                json!({ "type": "api-request-blob", "hash": cachesim::message_hash(&effort), "message": effort }),
            );
            self.messages.push(effort);
        }
    }

    /// The results of the latest response's tool calls, and any reminder after them.
    fn results(&mut self, content: Vec<Value>) {
        self.lines.push(json!({ "type": "user", "uuid": format!("u{}", self.lines.len()),
            "sourceToolAssistantUUID": format!("a{}", self.requests - 1), "message": { "role": "user", "content": content } }));
        self.messages.push(json!({ "role": "user", "content": content }));
    }

    /// A request `after` seconds after the previous one, answered with `reply`.
    fn request(&mut self, after: u64, reply: Vec<Value>) {
        self.t += after;
        let id = format!("r{}", self.requests);
        // On the last block, which Claude Code's effort message (no content) doesn't have.
        let last = self.messages.iter().rposition(|m| !server::blocks(m).is_empty()).unwrap();
        let block = server::blocks(&self.messages[last]).len() - 1;
        let marker = json!({ "type": "ephemeral" });
        let markers: Vec<Value> =
            [json!(["system", 0]), json!(["system", 1]), json!(["messages", last, "content", block])]
                .into_iter()
                .map(|path| json!({ "path": path, "value": marker }))
                .collect();
        let total = self.head
            + self.messages.iter().map(|m| server::blocks(m).iter().map(server::tokens).sum::<f64>()).sum::<f64>();
        let thinking: f64 = reply.iter().filter(|b| b["type"] == "thinking").map(server::tokens).sum();
        let (h, m, s) = (self.t / 3600, self.t % 3600 / 60, self.t % 60);
        self.lines.push(json!({
            "type": "api-request", "id": id, "timestamp": format!("2026-11-03T{h:02}:{m:02}:{s:02}.000Z"),
            "shapeHash": "s1",
            "params": { "model": "claude-opus-5-5", "betas": BETAS.split(',').collect::<Vec<_>>(), "max_tokens": 128000,
                        "thinking": { "type": "adaptive", "display": "omitted" },
                        "context_management": { "edits": [{ "type": "clear_thinking_20251015", "keep": "all" }] },
                        "output_config": { "effort": "medium" } },
            "base": self.requests.checked_sub(1).map(|i| format!("r{i}")), "keep": self.sent,
            "tail": self.messages[self.sent..].iter().map(cachesim::message_hash).collect::<Vec<_>>(),
            "cacheControl": markers,
        }));
        self.sent = self.messages.len();
        self.lines.push(json!({ "type": "assistant", "uuid": format!("a{}", self.requests), "requestRef": id,
            "message": { "id": format!("m{}", self.requests), "role": "assistant", "content": reply,
                "usage": { "input_tokens": 0, "cache_read_input_tokens": total, "cache_creation_input_tokens": 0,
                           "output_tokens": 100, "output_tokens_details": { "thinking_tokens": thinking } } } }));
        self.messages.push(json!({ "role": "assistant", "content": reply }));
        self.requests += 1;
    }

    fn write(&self, dir: &Path) -> PathBuf {
        let path = dir.join("session.jsonl");
        std::fs::write(&path, self.lines.iter().map(|l| format!("{l}\n")).collect::<String>()).unwrap();
        path
    }
}

fn thinking(k: usize) -> Value {
    json!({ "type": "thinking", "thinking": "", "signature": format!("sig-{k}-{}", "s".repeat(40 + k % 7 * 13)) })
}

fn text(t: String) -> Value {
    json!({ "type": "text", "text": t })
}

/// A file view or search output, unique to `k` and `j`, of `lines` lines.
fn output(k: usize, j: usize, lines: usize) -> String {
    (0..lines).map(|n| format!("{n:>6}\tconst panel_{k}_{j}_{n} = render({n}, \"{k}\");\n")).collect()
}

/// Claude Code's answer to a Read of a file it already showed, unchanged since.
const UNCHANGED: &str = "File unchanged since last read. The content from the earlier Read tool_result in this \
                         conversation is still current \u{2014} refer to that instead of re-reading.";

/// The `k`th tool loop: a response with thinking, sometimes text, and its
/// calls (Read, a pair of Reads, Grep, a short Bash, an Edit, or the excluded
/// TodoWrite), and the results, sometimes with a reminder. Loop 30 reads loop
/// 2's first file again, which Claude Code answers with its stand-in.
fn tool_loop(k: usize) -> (Vec<Value>, Vec<Value>) {
    let file =
        |j: usize| (json!({ "file_path": format!("/repo/src/panel_{k}_{j}.ts") }), output(k, j, 20 + k * 13 % 9 * 10));
    let calls = match k % 7 {
        _ if k == 30 => vec![("Read", json!({ "file_path": "/repo/src/panel_2_0.ts" }), UNCHANGED.to_string())],
        _ if k % 11 == 6 => {
            let (path, edit) =
                (format!("/repo/src/panel_{k}_0.ts"), |n: usize| format!("render({n}, \"{k}\");\n").repeat(6));
            let input = json!({ "file_path": path, "old_string": edit(k), "new_string": edit(k + 1) });
            vec![("Edit", input, format!("The file {path} has been updated."))]
        }
        0 => vec![(
            "TodoWrite",
            json!({ "todos": [{ "content": format!("step {k}"), "status": "in_progress" }] }),
            "Todos updated.".into(),
        )],
        1 | 4 => vec![("Grep", json!({ "pattern": format!("panel_{k}"), "path": "src" }), output(k, 9, 5 + k % 4 * 4))],
        2 => vec![("Read", file(0).0, file(0).1), ("Read", file(1).0, file(1).1)],
        3 => {
            vec![("Bash", json!({ "command": format!("git log --oneline -n {}", 2 + k % 5) }), output(k, 8, 2 + k % 5))]
        }
        _ => vec![("Read", file(0).0, file(0).1)],
    };
    let mut reply = vec![thinking(k)];
    if k.is_multiple_of(3) {
        reply.push(text(format!("Looking at step {k}. {}", "n".repeat(20 + k % 5 * 37))));
    }
    let mut results = Vec::new();
    for (j, (name, input, out)) in calls.into_iter().enumerate() {
        let id = format!("toolu_{k:03}_{j}");
        reply.push(json!({ "type": "tool_use", "id": id, "name": name, "input": input }));
        results.push(json!({ "tool_use_id": id, "type": "tool_result", "content": out, "is_error": false }));
    }
    if k % 4 == 1 {
        results.push(text(format!(
            "<system-reminder>{}</system-reminder>",
            "Keep the todo list current. ".repeat(1 + k % 3)
        )));
    }
    (reply, results)
}

/// Three tickets of four human turns each; every turn is 5 to 10 tool loops
/// and a closing reply, 5 to 27 seconds apart.
fn session() -> Transcript {
    let mut tr = Transcript::new();
    let mut k = 0;
    for ticket in 0..3 {
        for turn in 0..4 {
            tr.prompt(if turn == 0 {
                format!("Ticket {ticket}: {}", "Fix the panel rendering. ".repeat(100 + 7 * ticket))
            } else {
                format!("Follow-up {ticket}.{turn}: {}", "Also check the tests. ".repeat(3 + turn))
            });
            for _ in 0..5 + (3 * ticket + 2 * turn) % 6 {
                k += 1;
                let (reply, results) = tool_loop(k);
                tr.request(5 + (k * 7 % 23) as u64, reply);
                tr.results(results);
            }
            tr.request(5 + (k * 11 % 23) as u64, vec![thinking(1000 + k), text(format!("Done with {ticket}.{turn}."))]);
        }
    }
    tr
}

/// What came of sending a session's requests through a proxy to the emulated API.
struct Replay {
    served: Vec<Served>,
    totals: Totals,
    /// Each request's `ctx` stats, with `--context-edits`.
    ctx: Vec<Value>,
}

impl Replay {
    /// Tokens of the messages read from and written to the cache; the
    /// simulator bills the system prompt and tools as always read.
    fn message_tokens(&self) -> (f64, f64) {
        let missed = |r: &Served| if r.read > 0.0 { 0.0 } else { r.head };
        let read = self.served.iter().map(|r| r.read - (r.head - missed(r))).sum();
        let write = self.served.iter().map(|r| r.write + r.input - missed(r)).sum();
        (read, write)
    }
}

/// Send each request at its time on a fake clock, the way Claude Code does,
/// through a proxy with `ctx` (or without context edits) to the emulated API.
async fn replay(requests: &[Rebuilt], ctx: Option<CtxConfig>, store: &Path) -> Replay {
    let clock = Arc::new(FakeClock::default());
    let api = Api::new(clock.clone());
    let answer = api.clone();
    let ok = json!({ "type": "text", "text": "ok" });
    let mock = common::start(Arc::new(move |r: &Recorded| answer.respond(r, vec![ok.clone()], "end_turn"))).await;
    let transform = TransformConfig { ctx_window: ctx.map(|c| c.window), ..Default::default() };
    let cfg = ProxyConfig { upstream: mock.url(), transform: Some(transform), store: Store::open(store).unwrap(), ctx };
    let proxy = Proxy::with_clock(cfg, clock.clone()).unwrap();
    let addr = proxy.spawn("127.0.0.1:0").await.unwrap();
    let client = reqwest::Client::new();
    for (i, r) in requests.iter().enumerate() {
        clock.set(r.at.round() as u64);
        client
            .post(format!("http://{addr}/v1/messages?beta=true"))
            .header("x-api-key", "sk-ant-api03-replay-not-real")
            .header("anthropic-beta", &r.betas)
            .header("x-claude-code-session-id", "replay")
            .header("content-type", "application/json")
            .body(serde_json::to_vec(&r.body).unwrap())
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        // The next request needs this one settled, as it would be by the time Claude Code sends it.
        while proxy.totals().requests <= i as u64 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
    let records = Store::open(store).unwrap().records().unwrap();
    Replay {
        served: api.served(),
        totals: proxy.totals(),
        ctx: records.into_iter().filter_map(|r| r.get("ctx").cloned()).collect(),
    }
}

fn assert_close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-6 * want.abs().max(1.0), "{what}: emulated {got}, simulated {want}");
}

#[tokio::test]
async fn replayed_requests_cache_as_the_simulator_bills_them() {
    replay_matches_the_simulator(false).await;
}

#[tokio::test]
async fn replayed_requests_cache_as_the_simulator_bills_them_with_inputs_cleared() {
    replay_matches_the_simulator(true).await;
}

async fn replay_matches_the_simulator(clear_inputs: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = session().write(dir.path());
    let ctx = CtxConfig { clear_inputs, ..CtxConfig::default() };
    let placeholder = PLACEHOLDER.len() as f64 / 4.0;
    assert_eq!(
        server::tokens(&json!({ "type": "tool_use", "input": { "cleared": server::INPUT_PLACEHOLDER } })),
        placeholder
    );
    let cfg = SimConfig { regime: Regime::FiveMinFermat, read_price: Some(0.1), reread: 0.0, placeholder, ctx };
    let sim = cachesim::simulate_files(std::slice::from_ref(&path), &cfg).unwrap();
    let s = &sim.sessions[0];
    let requests = cachesim::rebuild(&path, Regime::FiveMinFermat).unwrap().expect("requests logged");
    assert_eq!((requests.len(), s.unresolved), (s.requests, 0));
    assert!(s.fit_error < 1e-9, "the fit recovers the emulator's token counts: {}", s.fit_error);

    let unchanged = replay(&requests, None, &dir.path().join("unchanged")).await;
    let edited = replay(&requests, Some(ctx), &dir.path().join("edited")).await;
    assert_eq!((s.refilled, edited.totals.refilled), (1, 1), "the stand-in is refilled, by both");
    for (run, bill, what) in [(&unchanged, &s.unchanged, "unchanged"), (&edited, &s.context_edits, "edited")] {
        let refused: Vec<&String> = run.served.iter().filter_map(|r| r.error.as_ref()).collect();
        assert!(refused.is_empty(), "{what}: the API refused {refused:?}");
        assert_eq!(run.totals.rewritten(), 0, "nothing the simulator leaves out");
        let cold = run.served.iter().filter(|r| r.read == 0.0).count();
        assert_eq!(cold, s.cold + 1, "{what}: the first request and each Fermat pause find the cache cold");
        let (read, write) = run.message_tokens();
        let head = s.requests as f64 * run.served[0].head;
        assert_close(read, bill.cache_read - head, &format!("{what} cache reads"));
        assert_close(write, bill.cache_write, &format!("{what} cache writes"));
    }

    // Unchanged, each request's prompt extends the one before; edited, all
    // but the advances do, and those come at cold moments.
    assert!(unchanged.served.windows(2).all(|w| w[1].extends(&w[0])));
    assert_eq!(edited.ctx.len(), requests.len());
    for (i, w) in edited.served.windows(2).enumerate() {
        let line = &edited.ctx[i + 1];
        if line["advance"] == true {
            assert_eq!((line["cold"].clone(), w[1].read), (json!(true), 0.0), "request {}", i + 1);
        } else {
            assert!(w[1].extends(&w[0]), "request {} is no advance, yet the server's view changed", i + 1);
        }
    }
    eprintln!(
        "{} requests, {} cold, {} advances; message tokens read/written: unchanged {:.0}/{:.0}, edited {:.0}/{:.0}; simulated ratio {:.3}",
        s.requests,
        s.cold,
        s.advances,
        unchanged.message_tokens().0,
        unchanged.message_tokens().1,
        edited.message_tokens().0,
        edited.message_tokens().1,
        s.ratio
    );
    for (i, (r, l)) in edited.served.iter().zip(&edited.ctx).enumerate().filter(|(_, (_, l))| l["cold"] == true) {
        eprintln!(
            "  cold request {i}: advance {} C {} C_th {} K {} read {} write {}",
            l["advance"], l["C"], l["C_th"], l["K"], r.read, r.write
        );
    }
    assert!(s.advances >= 2, "{} advances", s.advances);
    assert_eq!(edited.totals.ctx_advances, s.advances as u64);
    assert!(edited.ctx.iter().all(|l| l["breakers"] == json!([])) && s.breakers.is_empty());
    assert!(edited.message_tokens().0 < 0.8 * unchanged.message_tokens().0);
    // The server reported clearing exactly what the proxy pinned.
    for (r, line) in edited.served.iter().zip(&edited.ctx) {
        assert_eq!(r.cleared("cleared_tool_uses"), line["C"].as_u64().unwrap());
        assert_eq!(r.cleared("cleared_thinking_turns"), line["C_th"].as_u64().unwrap());
    }
}

fn recorded(path: &str, betas: &str, body: &Value) -> Recorded {
    let mut headers = HeaderMap::new();
    headers.insert("anthropic-beta", HeaderValue::from_str(betas).unwrap());
    Recorded { method: "POST".into(), path: path.into(), headers, body: serde_json::to_vec(body).unwrap() }
}

/// A short Claude Code conversation on Opus 5.5: a prompt, three tool loops
/// with thinking, and markers on the system prompt and the last block.
fn conversation() -> Value {
    let mut messages = vec![json!({ "role": "user", "content": "look" })];
    for k in 1..=3 {
        let (reply, results) = tool_loop(5 * k);
        messages.push(json!({ "role": "assistant", "content": reply }));
        messages.push(json!({ "role": "user", "content": results }));
    }
    messages.last_mut().unwrap()["content"][0]["cache_control"] = json!({ "type": "ephemeral" });
    json!({
        "model": "claude-opus-5-5",
        "thinking": { "type": "adaptive" },
        "tools": [{ "name": "Read" }, { "name": "Bash" }],
        "system": [{ "type": "text", "text": "You are a Claude agent.", "cache_control": { "type": "ephemeral" } }],
        "context_management": { "edits": [{ "type": "clear_thinking_20251015", "keep": "all" }] },
        "messages": messages
    })
}

/// Send `body` to `api` and return the error it answered with, if any.
fn refusal(api: &Api, betas: &str, body: &Value) -> Option<String> {
    api.respond(&recorded("/v1/messages", betas, body), vec![], "end_turn");
    api.served().last().unwrap().error.clone()
}

#[test]
fn the_emulated_api_refuses_what_the_api_refuses() {
    let api = Api::new(Arc::new(FakeClock::default()));
    let base = conversation();
    assert_eq!(refusal(&api, BETAS, &base), None);
    let pinned = json!([
        { "type": "clear_thinking_20251015", "keep": { "type": "thinking_turns", "value": 1 } },
        { "type": "clear_tool_uses_20250919", "trigger": { "type": "tool_uses", "value": 1 }, "keep": { "type": "tool_uses", "value": 1 } }
    ]);
    // Server-side clearing, moved markers, a reordered tool list, string
    // content and dropping the oldest thinking don't count as edits: the
    // thinking blocks still verify.
    let mut fine = base.clone();
    fine["context_management"]["edits"] = pinned.clone();
    fine["tools"] = json!([{ "name": "Bash" }, { "name": "Read" }]);
    fine["messages"][0]["content"] =
        json!([{ "type": "text", "text": "look", "cache_control": { "type": "ephemeral" } }]);
    assert_eq!(refusal(&api, BETAS, &fine), None);
    let served = api.served().pop().unwrap();
    // Two thinking turns as the API counts them: the prompt, and the reminder
    // next to the first results. The current one is kept.
    assert_eq!((served.cleared("cleared_tool_uses"), served.cleared("cleared_thinking_turns")), (2, 1));
    let drop_thinking = |b: &mut Value, at: usize| b["messages"][at]["content"].as_array_mut().unwrap().remove(0);
    let mut front = base.clone();
    drop_thinking(&mut front, 1);
    assert_eq!(refusal(&api, BETAS, &front), None);

    let edit = |f: &dyn Fn(&mut Value)| {
        let mut b = base.clone();
        f(&mut b);
        b
    };
    let cases: Vec<(&str, Value, &str)> = vec![
        ("claude-code-20250219", base.clone(), "requires the context-management-2025-06-27 beta"),
        (BETAS, edit(&|b| b["context_management"]["edits"] = json!([pinned[1], pinned[0]])), "must be the first edit"),
        (
            BETAS,
            edit(&|b| b["context_management"]["edits"][0]["keep"] = json!({ "type": "thinking_turns", "value": 0 })),
            "keep",
        ),
        (
            BETAS,
            edit(&|b| b["context_management"]["edits"][0]["type"] = json!("clear_everything")),
            "unknown edit type",
        ),
        (BETAS, edit(&|b| b["thinking"] = json!({ "type": "disabled" })), "requires thinking"),
        (
            BETAS,
            edit(&|b| {
                for m in b["messages"].as_array_mut().unwrap().iter_mut().skip(1) {
                    m["content"][0]["cache_control"] = json!({ "type": "ephemeral" });
                }
            }),
            "A maximum of 4 blocks",
        ),
        (BETAS, edit(&|b| b["messages"][6]["content"][0]["cache_control"]["ttl"] = json!("1h")), "must not come after"),
        // Edited history: thinking sent back after its prompt, the tools or an earlier result changed.
        (BETAS, edit(&|b| b["messages"][0]["content"] = json!("look again")), "bound to a different conversation"),
        (BETAS, edit(&|b| b["tools"][0]["description"] = json!("changed")), "bound to a different conversation"),
        (BETAS, edit(&|b| b["messages"][2]["content"][0]["content"] = json!("trimmed")), "messages.3.content.0"),
        (BETAS, edit(&|b| _ = drop_thinking(b, 3)), "messages.5.content.0"),
    ];
    for (betas, body, why) in cases {
        let error = refusal(&api, betas, &body).unwrap_or_else(|| panic!("accepted, but the API refuses: {why}"));
        assert!(error.contains(why), "{error} (expected: {why})");
    }
}

#[test]
fn the_emulated_cache_reads_within_its_lookback_and_lifetime() {
    let clock = Arc::new(FakeClock::default());
    let api = Api::new(clock.clone());
    let send = |at: u64, body: &Value| {
        clock.set(at);
        api.respond(&recorded("/v1/messages", BETAS, body), vec![], "end_turn");
        api.served().pop().unwrap()
    };
    let total = |s: &Served| s.read + s.write + s.input;
    // Append to `body` and mark the new last block.
    let grow = |body: &Value, blocks: Vec<Value>| {
        let mut b = body.clone();
        let messages = b["messages"].as_array_mut().unwrap();
        for m in messages.iter_mut() {
            for block in m["content"].as_array_mut().into_iter().flatten() {
                block.as_object_mut().unwrap().remove("cache_control");
            }
        }
        let mut blocks = blocks;
        blocks.last_mut().unwrap()["cache_control"] = json!({ "type": "ephemeral" });
        messages.push(json!({ "role": "assistant", "content": [{ "type": "text", "text": "next" }] }));
        messages.push(json!({ "role": "user", "content": blocks }));
        b
    };
    let base = conversation();
    let first = send(1000, &base);
    assert_eq!((first.read, first.write, first.input), (0.0, total(&first), 0.0));

    // Warm: everything before the appended blocks is read.
    let two = grow(&base, vec![text("a".repeat(400))]);
    let warm = send(1100, &two);
    assert_eq!((warm.read, warm.write), (total(&first), total(&warm) - total(&first)));

    // An entry lives 5 minutes from the start of the request that last read it.
    assert_eq!(send(1399, &two).read, total(&warm));
    assert_eq!(send(1650, &two).read, total(&warm), "refreshed at 1399");
    assert_eq!(send(1951, &two).read, 0.0, "expired at 1950");

    let three = grow(&two, vec![text("c".repeat(40))]);
    let resent = send(2000, &three);
    assert_eq!(resent.read, total(&warm));

    // More than 20 positions appended: the previous entry is out of reach,
    // except as a run of tool results, which is one position.
    let texts: Vec<Value> = (0..25).map(|i| text(format!("note {i}"))).collect();
    let far = send(2001, &grow(&three, texts));
    assert_eq!(far.read, resent.head, "only the system prompt's own breakpoint hits");
    let results: Vec<Value> = (0..25)
        .map(|i| json!({ "type": "tool_result", "tool_use_id": format!("t{i}"), "content": format!("out {i}") }))
        .collect();
    let run = send(2002, &grow(&three, results));
    assert_eq!(run.read, total(&resent));

    // A 1-hour breakpoint outlives a 330 s pause.
    let mut hour = grow(&three, vec![text("d".repeat(40))]);
    hour["system"][0]["cache_control"]["ttl"] = json!("1h");
    let last = hour["messages"].as_array().unwrap().len() - 1;
    hour["messages"][last]["content"][0]["cache_control"]["ttl"] = json!("1h");
    let wrote = send(3000, &hour);
    assert!(wrote.write > 0.0);
    assert_eq!(send(3330, &hour).read, total(&wrote));
}
