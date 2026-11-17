//! End-to-end: the real `claude` CLI -> squeeze proxy -> scripted mock API.
//!
//! Free and offline: Claude Code runs with `--bare` (never reads OAuth or the
//! keychain), a dummy API key, an isolated CLAUDE_CONFIG_DIR, and a base URL
//! that points at squeeze, whose upstream is a local mock. Nothing reaches
//! Anthropic and no model is involved; the mock plays a fixed script of tool
//! calls, and Claude Code really executes them (Bash, squeeze's MCP tools).
//!
//! Needs `claude` on PATH (or CLAUDE_BIN). Run with:
//!     cargo test --test e2e_claude -- --ignored --nocapture

mod common;

use common::server::{self, Api, Served};
use common::{FakeClock, Recorded, Reply, sse_text, sse_tool_use, tool_result_text};
use regex::Regex;
use serde_json::{Value, json};
use squeeze::ctxmgmt::{CtxConfig, KeepWindow};
use squeeze::proxy::{Proxy, ProxyConfig, Totals};
use squeeze::store::Store;
use squeeze::transform::{ExpandHint, TransformConfig};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

const FAKE_KEY: &str = "sk-ant-api03-mock-key-not-real";
const LOG_CMD: &str =
    r#"for i in $(seq 1 400); do echo "   Compiling dep$i v0.$i.0"; done; echo "error: mock link failure""#;

fn is_main_thread(body: &Value) -> bool {
    body["tools"].as_array().is_some_and(|t| t.iter().any(|x| x["name"] == "Bash"))
}

fn assistant_turns(body: &Value) -> usize {
    body["messages"].as_array().map_or(0, |m| m.iter().filter(|x| x["role"] == "assistant").count())
}

/// The fixed "model": Bash, the same Bash again, read_many, expand, done.
fn script(r: &Recorded) -> Reply {
    match r.path.as_str() {
        "/v1/messages" => {}
        "/v1/messages/count_tokens" => return Reply::Json(200, json!({ "input_tokens": 1000 })),
        "/v1/models" => {
            return Reply::Json(
                200,
                json!({ "data": [{ "type": "model", "id": "claude-mock", "display_name": "Mock" }], "has_more": false }),
            );
        }
        _ => {
            return Reply::Json(
                404,
                json!({ "type": "error", "error": { "type": "not_found_error", "message": "mock" } }),
            );
        }
    }
    let body = r.json();
    let system = body["system"].to_string();
    if system.contains("simulate the developer") {
        // The simulated user pushes back exactly once per test process, then approves.
        static PUSHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let verdict = if PUSHED.swap(true, std::sync::atomic::Ordering::SeqCst) {
            json!({ "approve": true, "message": "looks good" })
        } else {
            json!({ "approve": false, "message": "Please confirm you are done." })
        };
        return Reply::Sse(sse_text(98, &verdict.to_string()));
    }
    if system.contains("grade a coding agent") {
        return Reply::Sse(sse_text(97, &json!({ "score": 0.9, "reason": "fine" }).to_string()));
    }
    if !is_main_thread(&body) {
        return Reply::Sse(sse_text(99, "ok"));
    }
    let n = assistant_turns(&body);
    match n {
        0 => Reply::Sse(sse_tool_use(
            n,
            "toolu_mock_bash1",
            "Bash",
            json!({ "command": LOG_CMD, "description": "Build" }),
        )),
        1 => Reply::Sse(sse_tool_use(
            n,
            "toolu_mock_bash2",
            "Bash",
            json!({ "command": LOG_CMD, "description": "Build" }),
        )),
        2 => Reply::Sse(sse_tool_use(
            n,
            "toolu_mock_read",
            "mcp__squeeze__read_many",
            json!({ "files": [{ "path": "hello.txt" }] }),
        )),
        3 => {
            let first = tool_result_text(&body, "toolu_mock_bash1").unwrap_or_default();
            let id = Regex::new(r#"id "([0-9a-f]{24})""#).unwrap().captures(&first).map(|c| c[1].to_string());
            match id {
                Some(id) => Reply::Sse(sse_tool_use(
                    n,
                    "toolu_mock_expand",
                    "mcp__squeeze__expand",
                    json!({ "id": id, "offset": 199, "limit": 3 }),
                )),
                None => Reply::Sse(sse_text(n, "NO_EXPAND_ID")),
            }
        }
        _ => Reply::Sse(sse_text(n, "done")),
    }
}

#[tokio::test]
#[ignore = "needs the claude CLI; run with --ignored"]
async fn real_claude_code_through_squeeze() {
    let claude = std::env::var("CLAUDE_BIN").unwrap_or_else(|_| "claude".into());
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("hello.txt"), "hello from squeeze\n").unwrap();
    let home = tmp.path().join("squeeze-home");
    let store = Store::open(&home).unwrap();

    let mock = common::start(Arc::new(script)).await;
    let proxy = Proxy::new(ProxyConfig {
        upstream: mock.url(),
        transform: Some(TransformConfig { expand_hint: ExpandHint::Mcp, ..Default::default() }),
        store,
        ctx: None,
    })
    .unwrap();
    let addr = proxy.spawn("127.0.0.1:0").await.unwrap();

    let mcp = json!({ "mcpServers": { "squeeze": {
        "type": "stdio",
        "command": env!("CARGO_BIN_EXE_squeeze"),
        "args": ["mcp", "--root", work],
        "env": { "SQUEEZE_HOME": home }
    }}});
    let mut cmd = tokio::process::Command::new(&claude);
    cmd.current_dir(&work)
        .args(["-p", "--bare", "--output-format", "json", "--model", "claude-mock"])
        .arg("--mcp-config")
        .arg(mcp.to_string())
        .args(["--allowedTools", "Bash", "mcp__squeeze__read_many", "mcp__squeeze__expand"])
        .env("ANTHROPIC_BASE_URL", format!("http://{addr}"))
        .env("ANTHROPIC_API_KEY", FAKE_KEY)
        .env("CLAUDE_CONFIG_DIR", tmp.path().join("claude-config"))
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .env("DISABLE_AUTOUPDATER", "1")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .env_remove("CLAUDE_CODE_OAUTH_TOKEN")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = cmd.spawn().expect("claude CLI not found (set CLAUDE_BIN)");
    {
        use tokio::io::AsyncWriteExt;
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(b"Run the build, check it, then read hello.txt.").await.unwrap();
    }
    let out = tokio::time::timeout(Duration::from_secs(180), child.wait_with_output())
        .await
        .expect("claude timed out")
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    let main: Vec<Value> = mock
        .requests()
        .iter()
        .filter(|r| r.path == "/v1/messages")
        .inspect(|r| {
            assert_eq!(r.headers["x-api-key"], FAKE_KEY, "API key must be forwarded unchanged");
            let beta = r.headers.get("anthropic-beta").map(|v| v.to_str().unwrap().to_string()).unwrap_or_default();
            assert!(!beta.contains("oauth-"), "no subscription traffic may pass");
        })
        .map(Recorded::json)
        .filter(is_main_thread)
        .collect();
    eprintln!("main-thread requests seen by the mock: {}", main.len());
    for (i, b) in main.iter().enumerate() {
        eprintln!("  request {i}: {} assistant turns, {} bytes", assistant_turns(b), b.to_string().len());
    }
    assert!(out.status.success(), "claude failed\nstdout: {stdout}\nstderr: {stderr}");
    let result: Value = serde_json::from_str(stdout.trim()).expect("claude JSON output");
    assert_eq!(result["result"], "done", "unexpected result: {result}");

    let at = |turns: usize| {
        main.iter().find(|b| assistant_turns(b) == turns).unwrap_or_else(|| panic!("no request with {turns} turns"))
    };

    // 1. The first build log reached the API compressed, with the error kept and an expand id.
    let first = tool_result_text(at(1), "toolu_mock_bash1").expect("first Bash result");
    eprintln!("\n--- first Bash result as the API saw it ({} bytes) ---\n{first}\n---", first.len());
    assert!(first.contains("error: mock link failure"));
    assert!(first.contains("[squeeze: output compressed from"));
    assert!(!first.contains("dep200 v0.200.0"));

    // 2. The identical second run was replaced by a reference.
    let second = tool_result_text(at(2), "toolu_mock_bash2").expect("second Bash result");
    eprintln!("--- second Bash result ---\n{second}\n---");
    assert!(second.contains("identical to the output of the earlier Bash call"), "{second}");

    // 3. squeeze's MCP read_many ran inside Claude Code.
    let read = tool_result_text(at(3), "toolu_mock_read").expect("read_many result");
    assert!(read.contains("1\thello from squeeze"), "{read}");

    // 4. expand returned the original lines the compression dropped.
    let expanded = tool_result_text(at(4), "toolu_mock_expand").expect("expand result");
    eprintln!("--- expand result ---\n{expanded}\n---");
    assert!(expanded.contains("Compiling dep200 v0.200.0"), "{expanded}");

    // 5. Prefix stability on real Claude Code traffic: the rewritten first
    //    result is byte-identical in every later request.
    for turns in 2..=4 {
        assert_eq!(tool_result_text(at(turns), "toolu_mock_bash1").unwrap(), first, "rewrite changed at turn {turns}");
    }

    let t = proxy.totals();
    eprintln!("\nproxy totals: {}", serde_json::to_string(&t).unwrap());
    assert_eq!(t.refused, 0);
    assert!(t.compressed >= 1 && t.deduped >= 1);
}

/// Files the `--context-edits` sessions read, one per tool call.
const CM_READS: usize = 12;
/// The first run reads this many; the second, resumed with a new prompt, the rest.
const CM_FIRST_RUN: usize = 6;
/// The fake clock's step per request, and the pauses past a 5-minute cache's lifetime.
const CM_STEP: u64 = 10;
const CM_PAUSE: u64 = 400;

fn cm_file(i: usize) -> String {
    (0..200).map(|l| format!("{i}.{l}\tlet value_{i}_{l} = compute({l});\n")).collect()
}

/// What reached the emulated API in a `--context-edits cold` session, and what the proxy recorded.
struct CmSession {
    /// Main-thread messages requests: their `anthropic-beta` header and body.
    requests: Vec<(String, Value)>,
    /// The emulated API's view of each.
    served: Vec<Served>,
    /// Each one's `ctx` stats line.
    ctx: Vec<Value>,
    totals: Totals,
}

/// `cache_control` markers in a request body, as (where, TTL).
fn cache_markers(body: &Value) -> Vec<(String, Option<String>)> {
    let mut out = Vec::new();
    let mut visit = |at: String, b: &Value| {
        if let Some(cc) = b.get("cache_control") {
            out.push((at, cc["ttl"].as_str().map(str::to_string)));
        }
    };
    for (i, s) in body["system"].as_array().into_iter().flatten().enumerate() {
        visit(format!("system.{i}"), s);
    }
    for (i, m) in body["messages"].as_array().into_iter().flatten().enumerate() {
        for (j, b) in m["content"].as_array().into_iter().flatten().enumerate() {
            visit(format!("messages.{i}.content.{j}"), b);
        }
    }
    out
}

/// Two `claude -p` runs on one conversation through `--context-edits cold`
/// to the emulated API (free): the first reads 6 files one at a time, the
/// second (`--resume`, a new prompt) the other 6. The fake clock moves 10 s
/// per request, so a 5-minute cache stays warm, except for two 400 s pauses:
/// before the second prompt, and after its fourth read. With `one_hour`,
/// Claude Code runs with `ENABLE_PROMPT_CACHING_1H`.
async fn context_edits_session(one_hour: bool) -> CmSession {
    let claude = std::env::var("CLAUDE_BIN").unwrap_or_else(|_| "claude".into());
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    for i in 0..CM_READS {
        std::fs::write(work.join(format!("f{i}.txt")), cm_file(i)).unwrap();
    }
    let home = tmp.path().join("squeeze-home");
    let clock = Arc::new(FakeClock::default());
    clock.set(1_000_000);
    let api = Api::new(clock.clone());
    let (answer, ticks) = (api.clone(), clock.clone());
    let mock = common::start(Arc::new(move |r: &Recorded| {
        let body = r.json();
        match r.path.as_str() {
            "/v1/messages/count_tokens" => return answer.respond(r, vec![], ""),
            "/v1/messages" if is_main_thread(&body) => {}
            _ => return script(r),
        }
        let messages = body["messages"].as_array().unwrap();
        let uses =
            messages.iter().flat_map(|m| server::blocks(m).into_owned()).filter(|b| b["type"] == "tool_use").count();
        let prompted = messages.last().is_some_and(|m| server::blocks(m).iter().all(|b| b["type"] != "tool_result"));
        let read = uses < CM_FIRST_RUN || (uses < CM_READS && (uses > CM_FIRST_RUN || prompted));
        let reply = if read {
            let input = json!({ "command": format!("cat f{uses}.txt"), "description": "Read a file" });
            answer.respond(
                r,
                vec![json!({ "type": "tool_use", "id": format!("toolu_cm_{uses}"), "name": "Bash", "input": input })],
                "tool_use",
            )
        } else {
            answer.respond(r, vec![json!({ "type": "text", "text": "done" })], "end_turn")
        };
        let pause = read && uses == CM_FIRST_RUN + 3;
        ticks.0.fetch_add(if pause { CM_PAUSE } else { CM_STEP }, Ordering::SeqCst);
        reply
    }))
    .await;
    let ctx = CtxConfig { window: KeepWindow { min: 4, max: 4 }, ..Default::default() };
    let transform = TransformConfig { ctx_window: Some(ctx.window), ..Default::default() };
    let cfg = ProxyConfig {
        upstream: mock.url(),
        transform: Some(transform),
        store: Store::open(&home).unwrap(),
        ctx: Some(ctx),
    };
    let proxy = Proxy::with_clock(cfg, clock.clone()).unwrap();
    let addr = proxy.spawn("127.0.0.1:0").await.unwrap();

    let mut session: Option<String> = None;
    for prompt in ["Read the files one at a time.", "Now read the rest, one at a time."] {
        let mut cmd = tokio::process::Command::new(&claude);
        cmd.current_dir(&work)
            .args(["-p", prompt, "--bare", "--output-format", "json"])
            .args(["--model", "claude-opus-5-5", "--allowedTools", "Bash"])
            .env("ANTHROPIC_BASE_URL", format!("http://{addr}"))
            .env("ANTHROPIC_API_KEY", FAKE_KEY)
            .env("CLAUDE_CONFIG_DIR", tmp.path().join("claude-config"))
            .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
            .env("DISABLE_AUTOUPDATER", "1")
            .env_remove("ANTHROPIC_AUTH_TOKEN")
            .env_remove("CLAUDE_CODE_OAUTH_TOKEN")
            .env_remove("ENABLE_PROMPT_CACHING_1H")
            .stdin(Stdio::null());
        if one_hour {
            cmd.env("ENABLE_PROMPT_CACHING_1H", "1");
        }
        if let Some(id) = &session {
            cmd.arg("--resume").arg(id);
            clock.0.fetch_add(CM_PAUSE, Ordering::SeqCst);
        }
        let out =
            tokio::time::timeout(Duration::from_secs(240), cmd.output()).await.expect("claude timed out").unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "claude failed\nstdout: {stdout}\nstderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let result: Value = serde_json::from_str(stdout.trim()).expect("claude JSON output");
        assert_eq!(result["result"], "done", "Claude Code accepts applied_edits and keeps going: {result}");
        session = result["session_id"].as_str().map(str::to_string);
    }

    let requests: Vec<(String, Value)> = mock
        .requests()
        .iter()
        .filter(|r| r.path == "/v1/messages")
        .map(|r| (r.headers.get("anthropic-beta").map_or("", |v| v.to_str().unwrap()).to_string(), r.json()))
        .filter(|(_, b)| is_main_thread(b))
        .collect();
    let served: Vec<Served> = api.served().into_iter().filter(|s| s.path == "/v1/messages").collect();
    let ctx: Vec<Value> =
        Store::open(&home).unwrap().records().unwrap().into_iter().filter_map(|r| r.get("ctx").cloned()).collect();
    let (beta, first) = &requests[0];
    eprintln!("Claude Code on claude-opus-5-5 (ENABLE_PROMPT_CACHING_1H {}):", if one_hour { "on" } else { "off" });
    eprintln!("  anthropic-beta: {beta}");
    eprintln!("  context_management: {}", first["context_management"]);
    eprintln!("  thinking: {}, output_config: {}", first["thinking"], first["output_config"]);
    eprintln!("  cache markers: {:?}", cache_markers(first));
    eprintln!("  proxy totals: {}", serde_json::to_string(&proxy.totals()).unwrap());
    for (s, l) in served.iter().zip(&ctx) {
        eprintln!(
            "  request at +{:>4} s: read {:>7.0} write {:>6.0} | cold {} advance {} C {} C_th {} applied {}{}",
            s.at - 1_000_000,
            s.read,
            s.write,
            l["cold"],
            l["advance"],
            l["C"],
            l["C_th"],
            serde_json::to_string(&s.applied).unwrap(),
            s.error.as_ref().map_or(String::new(), |e| format!(" REFUSED: {e}"))
        );
    }
    CmSession { requests, served, ctx, totals: proxy.totals() }
}

impl CmSession {
    /// What Claude Code sends on every main-thread request, whatever the cache TTL.
    fn assert_claude_codes_shape(&self, ttl: Option<&str>) {
        assert_eq!(self.requests.len(), CM_READS + 2, "a request per read and per reply");
        assert_eq!((self.served.len(), self.ctx.len()), (self.requests.len(), self.requests.len()));
        for (beta, b) in &self.requests {
            let betas: Vec<&str> = beta.split(',').collect();
            assert!(betas.contains(&"context-management-2025-06-27"), "{beta}");
            assert!(!beta.contains("message-threads"), "{beta}");
            assert_eq!(betas.contains(&"extended-cache-ttl-2025-04-11"), ttl == Some("1h"), "{beta}");
            assert_eq!(b["thinking"]["type"], "adaptive");
            let edits = b["context_management"]["edits"].as_array().expect("Claude Code sends context_management");
            assert_eq!(edits[0]["type"], "clear_thinking_20251015", "Claude Code's own edit, first");
            let markers = cache_markers(b);
            assert!(markers.len() <= 4 && markers.iter().all(|(_, t)| t.as_deref() == ttl), "{markers:?}");
            // The last block is marked: Claude Code's effort message after the first prompt has none.
            let messages = b["messages"].as_array().unwrap();
            let last = messages.iter().rposition(|m| !server::blocks(m).is_empty()).unwrap();
            let block = server::blocks(&messages[last]).len() - 1;
            assert!(markers.iter().any(|(at, _)| *at == format!("messages.{last}.content.{block}")), "{markers:?}");
            // Message bytes are Claude Code's own: every result as the command printed it.
            for i in 0..CM_READS {
                if let Some(text) = tool_result_text(b, &format!("toolu_cm_{i}")) {
                    assert_eq!(text.trim_end(), cm_file(i).trim_end(), "result {i}");
                }
            }
        }
        let refused: Vec<&String> = self.served.iter().filter_map(|s| s.error.as_ref()).collect();
        assert!(refused.is_empty(), "the API refused {refused:?}");
        let last = &self.requests.last().unwrap().1;
        let emitted = last["messages"].as_array().unwrap().iter().flat_map(|m| server::blocks(m).into_owned());
        let bound = emitted.filter(|b| b["signature"].as_str().is_some_and(|s| s.starts_with("emu-sig-"))).count();
        assert_eq!(bound, CM_READS + 1, "every thinking block came back and verified");
        assert!(self.ctx.iter().all(|l| l["breakers"] == json!([])), "{:?}", self.ctx);
    }
}

/// Real Claude Code through `--context-edits cold` against the emulated API.
/// Until the first cold moment the proxy leaves Claude Code's own
/// `context_management` alone; from it on, a numeric thinking `keep` and a
/// `clear_tool_uses` edit after it pin what the server clears. The API
/// accepts every request (thinking blocks verify across the resume),
/// reports what it cleared in `message_delta`, and between the two cold
/// moments its view only grows, so each warm request reads the whole
/// previous prompt from cache.
#[tokio::test]
#[ignore = "needs the claude CLI; run with --ignored"]
async fn real_claude_code_with_context_edits() {
    let s = context_edits_session(false).await;
    s.assert_claude_codes_shape(None);
    let cold: Vec<usize> = (0..s.ctx.len()).filter(|&i| s.ctx[i]["cold"] == true).collect();
    assert_eq!(cold, [CM_FIRST_RUN + 1, CM_FIRST_RUN + 5], "the second prompt, and after its fourth read");
    let own = json!([{ "type": "clear_thinking_20251015", "keep": "all" }]);
    for (i, ((_, b), (served, line))) in s.requests.iter().zip(s.served.iter().zip(&s.ctx)).enumerate() {
        let edits = &b["context_management"]["edits"];
        if i < cold[0] {
            assert_eq!(edits, &own, "{i}: Claude Code's own, until the cache is cold");
            assert!(served.applied.is_empty());
            continue;
        }
        // Pinned: the oldest C uses and C_th thinking turns, as the server reported.
        let (c, c_th) = (line["C"].as_u64().unwrap(), line["C_th"].as_u64().unwrap());
        assert_eq!(line["advance"], cold.contains(&i), "{i}: an advance at each cold moment, and only there");
        let uses = b["messages"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|m| server::blocks(m).into_owned())
            .filter(|x| x["type"] == "tool_use");
        let n = uses.count() as u64;
        let thinking_turns =
            b["messages"].as_array().unwrap().iter().filter(|m| m["role"] == "assistant").count() as u64;
        let mut ours = json!({ "type": "clear_tool_uses_20250919", "trigger": { "type": "tool_uses", "value": n - 1 },
                               "keep": { "type": "tool_uses", "value": n - c } });
        // Every offered tool but the clearable ones (none, with `--bare`'s Bash, Edit and Read).
        let mut others: Vec<&str> = b["tools"].as_array().unwrap().iter().filter_map(|t| t["name"].as_str()).collect();
        others.retain(|t| !squeeze::ctxmgmt::is_clearable(t));
        others.sort();
        if !others.is_empty() {
            ours["exclude_tools"] = json!(others);
        }
        let thinking = json!({ "type": "clear_thinking_20251015", "keep": { "type": "thinking_turns", "value": thinking_turns - c_th } });
        assert_eq!(edits, &json!([thinking, ours]), "{i}");
        assert_eq!((served.cleared("cleared_tool_uses"), served.cleared("cleared_thinking_turns")), (c, c_th));
    }
    assert_eq!(s.totals.ctx_advances, 2);
    assert_eq!((s.ctx[cold[0]]["C"].clone(), s.ctx[cold[1]]["C"].clone()), (json!(2), json!(6)));
    for (i, w) in s.served.windows(2).enumerate().map(|(i, w)| (i + 1, w)) {
        if cold.contains(&i) {
            assert_eq!(w[1].read, 0.0, "{i}: expired");
        } else {
            assert!(w[1].extends(&w[0]), "{i}: the server's view grows append-only between advances");
            assert_eq!(w[1].read, w[0].read + w[0].write + w[0].input, "{i}: the whole previous prompt is read");
        }
    }
}

/// With `ENABLE_PROMPT_CACHING_1H`, Claude Code marks every breakpoint
/// 1-hour, so the same 400 s pauses leave the cache warm: the proxy never
/// edits, and every request reads the whole previous prompt.
#[tokio::test]
#[ignore = "needs the claude CLI; run with --ignored"]
async fn one_hour_cache_markers_never_go_cold_on_short_pauses() {
    let s = context_edits_session(true).await;
    s.assert_claude_codes_shape(Some("1h"));
    let own = json!([{ "type": "clear_thinking_20251015", "keep": "all" }]);
    assert!(s.requests.iter().all(|(_, b)| b["context_management"]["edits"] == own));
    assert!(s.ctx.iter().all(|l| l["cold"] == false && l["advance"] == false));
    for w in s.served.windows(2) {
        assert!(w[1].extends(&w[0]));
        assert_eq!(w[1].read, w[0].read + w[0].write + w[0].input);
    }
}

/// `squeeze run --proxy`: credential preflight, proxy, MCP wiring and
/// the `--settings` pre-approval of squeeze's read-only tools (note that
/// `--allowedTools` below lists only Bash).
#[tokio::test]
#[ignore = "needs the claude CLI; run with --ignored"]
async fn squeeze_run_cli_wires_claude_code() {
    let claude = std::env::var("CLAUDE_BIN").unwrap_or_else(|_| "claude".into());
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("hello.txt"), "hello from squeeze\n").unwrap();
    let mock = common::start(Arc::new(script)).await;

    let out = tokio::process::Command::new(env!("CARGO_BIN_EXE_squeeze"))
        .current_dir(&work)
        .args(["run", "--proxy", "--quiet", "--upstream", &mock.url(), "--claude-bin", &claude, "--"])
        // Not --bare: it drops the ToolSearch tool, and with it tool search.
        .args(["-p", "Run the build, check it, then read hello.txt.", "--output-format", "json"])
        .args(["--model", "claude-mock", "--allowedTools", "Bash"])
        .env("ANTHROPIC_API_KEY", FAKE_KEY)
        .env("SQUEEZE_HOME", tmp.path().join("squeeze-home"))
        .env("CLAUDE_CONFIG_DIR", tmp.path().join("claude-config"))
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .env("DISABLE_AUTOUPDATER", "1")
        // On by default against Anthropic's API; a custom base URL (the proxy) turns it off.
        .env("ENABLE_TOOL_SEARCH", "true")
        .env_remove("ANTHROPIC_BASE_URL")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .env_remove("CLAUDE_CODE_OAUTH_TOKEN")
        .stdin(Stdio::null())
        .output();
    let out = tokio::time::timeout(Duration::from_secs(180), out).await.expect("timed out").unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "squeeze run failed\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let result: Value = serde_json::from_str(stdout.trim()).expect("claude JSON output");
    assert_eq!(result["result"], "done", "{result}");
    assert_eq!(result["permission_denials"], json!([]), "squeeze tools should be pre-approved");

    let main: Vec<Value> = mock
        .requests()
        .iter()
        .filter(|r| r.path == "/v1/messages")
        .map(Recorded::json)
        .filter(is_main_thread)
        .collect();
    let at = |turns: usize| main.iter().find(|b| assistant_turns(b) == turns).unwrap();
    // Claude Code defers MCP tools behind tool search by default (the model sees
    // only their names). squeeze's tools opt out, so the model sees them in full.
    let tools = at(0)["tools"].as_array().expect("tools");
    for name in ["read_many", "outline", "search", "expand"] {
        let t = tools.iter().find(|t| t["name"] == format!("mcp__squeeze__{name}"));
        let t = t.unwrap_or_else(|| {
            panic!("mcp__squeeze__{name} not sent in full: {:?}", tools.iter().map(|t| &t["name"]).collect::<Vec<_>>())
        });
        assert!(t["defer_loading"] != true && t["input_schema"].is_object(), "deferred: {t}");
    }
    // The server's instructions reach the model, early in the cached prefix
    // (Claude Code 2.1.x sends them as a system-role entry among the first messages).
    assert!(
        at(0)["messages"]
            .as_array()
            .unwrap()
            .iter()
            .take(3)
            .any(|m| m.to_string().contains("instead of reading one per turn")),
        "squeeze's MCP instructions in the prompt"
    );
    assert!(tool_result_text(at(1), "toolu_mock_bash1").unwrap().contains("[squeeze: output compressed"));
    assert!(tool_result_text(at(3), "toolu_mock_read").unwrap().contains("1\thello from squeeze"));
    assert!(tool_result_text(at(4), "toolu_mock_expand").unwrap().contains("Compiling dep200 v0.200.0"));
}

/// `squeeze run --proxy` needs API-key credentials (the proxy sees API traffic);
/// hook mode does not, because squeeze never handles credentials there.
#[test]
fn proxy_mode_requires_api_credentials_hook_mode_does_not() {
    let tmp = tempfile::tempdir().unwrap();
    let run = |args: &[&str], oauth: Option<&str>| {
        let mut c = std::process::Command::new(env!("CARGO_BIN_EXE_squeeze"));
        c.args(args)
            .env("SQUEEZE_HOME", tmp.path())
            .env_remove("ANTHROPIC_API_KEY")
            .env_remove("ANTHROPIC_AUTH_TOKEN")
            .env_remove("CLAUDE_CODE_USE_BEDROCK")
            .env_remove("CLAUDE_CODE_USE_VERTEX")
            .env_remove("CLAUDE_CODE_USE_FOUNDRY");
        match oauth {
            Some(t) => c.env("CLAUDE_CODE_OAUTH_TOKEN", t),
            None => c.env_remove("CLAUDE_CODE_OAUTH_TOKEN"),
        };
        let out = c.output().unwrap();
        (out.status.success(), String::from_utf8_lossy(&out.stderr).to_string())
    };
    let (ok, err) = run(&["run", "--proxy", "--claude-bin", "/nonexistent/claude", "--", "-p", "hi"], None);
    assert!(!ok && err.contains("API-key or gateway credentials"), "{err}");
    let (ok, err) = run(&["run", "--proxy", "--claude-bin", "/nonexistent/claude"], Some("sk-ant-oat01-x"));
    assert!(!ok && err.contains("subscription OAuth token"), "{err}");
    // Hook mode gets as far as launching claude (which doesn't exist here).
    let (ok, err) = run(&["run", "--claude-bin", "/nonexistent/claude", "--", "-p", "hi"], Some("sk-ant-oat01-x"));
    assert!(!ok && err.contains("could not start"), "{err}");
}

/// `squeeze bench` end to end against the mock (free): every arm, a
/// two-ticket resumed session, simulated-user pushback, judge scoring, resume.
#[tokio::test]
#[ignore = "needs the claude CLI; run with --ignored"]
async fn bench_runs_all_arms_with_usersim_judge_and_resume() {
    let claude = std::env::var("CLAUDE_BIN").unwrap_or_else(|_| "claude".into());
    let tmp = tempfile::tempdir().unwrap();
    let repo = tmp.path().join("fixture");
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(repo.join("hello.txt"), "hello from squeeze\n").unwrap();
    let tasks = tmp.path().join("tasks.toml");
    std::fs::write(
        &tasks,
        r#"[defaults]
model = "claude-mock"
usersim_model = "claude-mock"
judge_model = "claude-mock"
max_pushbacks = 1
allowed_tools = ["Bash"]

[[bundle]]
name = "mock-bundle"
repo = "fixture"

[[bundle.ticket]]
id = "t1"
prompt = "Run the build, check it, then read hello.txt."
test = "test -f hello.txt"
rubric = "anything"

[[bundle.ticket]]
id = "t2"
prompt = "Say done."
test = "true"
rubric = "anything"
"#,
    )
    .unwrap();
    let mock = common::start(Arc::new(script)).await;
    let results = tmp.path().join("results.jsonl");
    let bench = |extra: &[&str]| {
        let mut c = tokio::process::Command::new(env!("CARGO_BIN_EXE_squeeze"));
        c.arg("bench")
            .arg(&tasks)
            .args(["--auth", "api-key", "--repeats", "1", "--yes", "--claude-bin", &claude, "--out"])
            .arg(&results)
            .args([
                "--arms",
                "vanilla,squeeze,squeeze-full,squeeze-scout,squeeze-compact,squeeze-max,squeeze-hook,squeeze-mcp,squeeze-proxy,squeeze-proxy-cm",
            ])
            .args(extra)
            .env("ANTHROPIC_API_KEY", FAKE_KEY)
            .env("ANTHROPIC_BASE_URL", mock.url())
            .env("SQUEEZE_HOME", tmp.path().join("squeeze-home"))
            .env("CLAUDE_CONFIG_DIR", tmp.path().join("claude-config"))
            .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
            .env("DISABLE_AUTOUPDATER", "1")
            .env_remove("ANTHROPIC_AUTH_TOKEN")
            .env_remove("CLAUDE_CODE_OAUTH_TOKEN");
        c.output()
    };
    let out = tokio::time::timeout(Duration::from_secs(600), bench(&[])).await.expect("timed out").unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    eprintln!("{stdout}");
    assert!(out.status.success(), "bench failed: {}", String::from_utf8_lossy(&out.stderr));

    let rows: Vec<Value> =
        std::fs::read_to_string(&results).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(rows.len(), 10, "one session per arm");
    let mut pushbacks = 0;
    for r in &rows {
        assert!(r["error"].is_null(), "{r}");
        assert_eq!(r["tickets"].as_array().unwrap().len(), 2, "{r}");
        for t in r["tickets"].as_array().unwrap() {
            assert_eq!(t["tests_passed"], true, "{r}");
            assert_eq!(t["rubric"], 0.9, "{r}");
            pushbacks += t["pushbacks"].as_u64().unwrap();
        }
        assert!(r["agent"]["invocations"].as_u64().unwrap() >= 2);
        assert!(r["helpers"]["invocations"].as_u64().unwrap() >= 3);
    }
    assert_eq!(pushbacks, 1, "the simulated user pushed back once");

    let reqs: Vec<Value> = mock.requests().iter().filter(|r| r.path == "/v1/messages").map(Recorded::json).collect();
    // Resumed sessions: ticket 2 arrives with ticket 1's history.
    assert!(reqs.iter().any(|b| is_main_thread(b) && assistant_turns(b) >= 4 && b.to_string().contains("Say done.")));
    // Simulated user and judge run without tools.
    for b in reqs.iter().filter(|b| b["system"].to_string().contains("simulate the developer")) {
        assert!(b["tools"].as_array().is_none_or(|t| t.is_empty()), "usersim must have no tools");
    }
    // Hook and proxy arms compressed the build log; vanilla did not.
    let compressed = reqs
        .iter()
        .filter_map(|b| tool_result_text(b, "toolu_mock_bash1"))
        .filter(|t| t.contains("[squeeze: output compressed"))
        .count();
    assert!(compressed >= 5, "expected compression in every hook and proxy arm");
    // squeeze-full: the built-in file tools are gone and edit_many is offered.
    let full: Vec<&Value> = reqs
        .iter()
        .filter(|b| is_main_thread(b) && b["system"].to_string().contains("turned off in this session"))
        .collect();
    assert!(!full.is_empty(), "squeeze-full sessions carry the guidance");
    // squeeze-scout: Explore is redefined with squeeze's tools, and the agent is told to use it.
    assert!(
        full.iter().any(|b| b["system"].to_string().contains("delegate it to the Explore agent")
            && b["messages"].to_string().contains("Read-only code scout")),
        "squeeze-scout offers the scout"
    );
    for b in full {
        let names: Vec<&str> = b["tools"].as_array().unwrap().iter().filter_map(|t| t["name"].as_str()).collect();
        assert!(!names.contains(&"Read") && !names.contains(&"Edit"), "{names:?}");
        assert!(names.contains(&"mcp__squeeze__edit_many"), "{names:?}");
    }
    assert!(stdout.contains("cost vs vanilla"));

    // Rerunning with the same results file resumes: nothing left to do.
    let again = tokio::time::timeout(Duration::from_secs(60), bench(&[])).await.expect("timed out").unwrap();
    assert!(again.status.success());
    assert!(String::from_utf8_lossy(&again.stderr).contains("0 session(s) to run"));
}

/// Default `squeeze run` (hook mode): no proxy at all. Claude Code talks to the
/// API itself (here the mock, as a test double); squeeze compresses Bash output
/// through the documented PostToolUse hook and serves MCP tools. This is the
/// mode that works with any login, including a Claude subscription.
#[tokio::test]
#[ignore = "needs the claude CLI; run with --ignored"]
async fn squeeze_run_hook_mode_compresses_without_a_proxy() {
    let claude = std::env::var("CLAUDE_BIN").unwrap_or_else(|_| "claude".into());
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("hello.txt"), "hello from squeeze\n").unwrap();
    let hook_log = tmp.path().join("hook-payloads.jsonl");
    let mock = common::start(Arc::new(script)).await;

    let out = tokio::process::Command::new(env!("CARGO_BIN_EXE_squeeze"))
        .current_dir(&work)
        .args(["run", "--quiet", "--claude-bin", &claude, "--"])
        .args(["-p", "Run the build, check it, then read hello.txt.", "--output-format", "json"])
        .args(["--model", "claude-mock", "--allowedTools", "Bash", "--strict-mcp-config"])
        .env("ANTHROPIC_BASE_URL", mock.url())
        .env("ANTHROPIC_API_KEY", FAKE_KEY)
        .env("SQUEEZE_HOME", tmp.path().join("squeeze-home"))
        .env("SQUEEZE_HOOK_LOG", &hook_log)
        .env("CLAUDE_CONFIG_DIR", tmp.path().join("claude-config"))
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .env("DISABLE_AUTOUPDATER", "1")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .env_remove("CLAUDE_CODE_OAUTH_TOKEN")
        .stdin(Stdio::null())
        .output();
    let out = tokio::time::timeout(Duration::from_secs(180), out).await.expect("timed out").unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "squeeze run failed\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // What Claude Code actually hands the hook (undocumented shape, so record it).
    let payloads: Vec<Value> = std::fs::read_to_string(&hook_log)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    for p in payloads.iter().filter(|p| p["tool_name"] == "Bash").take(1) {
        let keys: Vec<&String> = p.as_object().unwrap().keys().collect();
        let resp_keys: Vec<&String> = p["tool_response"].as_object().map(|o| o.keys().collect()).unwrap_or_default();
        eprintln!("hook payload keys: {keys:?}\nBash tool_response keys: {resp_keys:?}");
    }
    assert!(payloads.iter().any(|p| p["tool_name"] == "Bash"), "the hook never ran");

    let result: Value = serde_json::from_str(stdout.trim()).expect("claude JSON output");
    assert_eq!(result["result"], "done", "{result}");
    assert_eq!(result["permission_denials"], json!([]));

    let main: Vec<Value> = mock
        .requests()
        .iter()
        .filter(|r| r.path == "/v1/messages")
        .map(Recorded::json)
        .filter(is_main_thread)
        .collect();
    let at = |turns: usize| main.iter().find(|b| assistant_turns(b) == turns).unwrap();
    let first = tool_result_text(at(1), "toolu_mock_bash1").unwrap();
    eprintln!("--- first Bash result as the API saw it ({} bytes) ---\n{first}\n---", first.len());
    assert!(first.contains("[squeeze: output compressed"), "hook replacement not applied: {first}");
    assert!(first.contains("error: mock link failure"));
    // Claude Code stored the replaced output: identical bytes in every later request.
    for turns in 2..=4 {
        assert_eq!(tool_result_text(at(turns), "toolu_mock_bash1").unwrap(), first, "changed at turn {turns}");
    }
    assert!(tool_result_text(at(3), "toolu_mock_read").unwrap().contains("1\thello from squeeze"));
    assert!(tool_result_text(at(4), "toolu_mock_expand").unwrap().contains("Compiling dep200 v0.200.0"));
}

/// Experiment: which hook events fire for a failing Bash command, and with what payload.
#[tokio::test]
#[ignore = "experiment; needs the claude CLI"]
async fn hook_events_for_failing_bash() {
    let claude = std::env::var("CLAUDE_BIN").unwrap_or_else(|_| "claude".into());
    let tmp = tempfile::tempdir().unwrap();
    let log = tmp.path().join("events.jsonl");
    let fail_cmd = r#"for i in $(seq 1 300); do echo "case_$i ... ok"; done; echo "FAIL: x"; exit 1"#;
    let mock = common::start(Arc::new(move |r: &Recorded| {
        if r.path != "/v1/messages" {
            return Reply::Json(404, json!({}));
        }
        let b = r.json();
        if !is_main_thread(&b) {
            return Reply::Sse(sse_text(99, "ok"));
        }
        match assistant_turns(&b) {
            0 => Reply::Sse(sse_tool_use(0, "toolu_fail", "Bash", json!({ "command": fail_cmd }))),
            _ => Reply::Sse(sse_text(1, "done")),
        }
    }))
    .await;
    let logger = format!("cat >> {}; echo >> {}", log.display(), log.display());
    // HOOK_REPLY (optional): JSON the failure hook prints, to test output replacement.
    let failure_cmd = match std::env::var("HOOK_REPLY") {
        Ok(_) => format!("cat >> {} ; echo >> {}; printf '%s' \"$HOOK_REPLY\"", log.display(), log.display()),
        Err(_) => logger.clone(),
    };
    let settings = json!({ "hooks": {
        "PostToolUse": [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": logger }] }],
        "PostToolUseFailure": [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": failure_cmd }] }]
    }});
    let out = tokio::process::Command::new(&claude)
        .current_dir(tmp.path())
        .args(["-p", "go", "--output-format", "json", "--model", "claude-mock", "--settings", &settings.to_string()])
        .args(["--allowedTools", "Bash"])
        .env("ANTHROPIC_BASE_URL", mock.url())
        .env("ANTHROPIC_API_KEY", FAKE_KEY)
        .env("CLAUDE_CONFIG_DIR", tmp.path().join("cfg"))
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .env_remove("CLAUDE_CODE_OAUTH_TOKEN")
        .output()
        .await
        .unwrap();
    eprintln!("claude exit: {:?} stderr: {}", out.status.code(), String::from_utf8_lossy(&out.stderr));
    for line in std::fs::read_to_string(&log).unwrap_or_default().lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).unwrap();
        let resp_keys: Vec<String> =
            v["tool_response"].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
        let other: Vec<&String> = v.as_object().unwrap().keys().collect();
        eprintln!(
            "EVENT {} | payload keys {:?} | tool_response keys {:?} | error field: {}",
            v["hook_event_name"],
            other,
            resp_keys,
            crate_trunc(&v["error"].to_string())
        );
    }
    let api_saw =
        mock.requests().iter().filter_map(|r| tool_result_text(&r.json(), "toolu_fail")).next().unwrap_or_default();
    eprintln!(
        "API saw {} bytes; is_error flag: {}; text starts: {:?}",
        api_saw.len(),
        mock.requests().iter().any(|r| r.json().to_string().contains("\"is_error\":true")),
        crate_trunc(&api_saw)
    );
}

fn crate_trunc(s: &str) -> String {
    s.chars().take(160).collect()
}

/// The fixed "model" for --replace-file-tools: read with read_many, then edit one
/// file and create another in a single edit_many call, then finish.
fn full_tools_script(r: &Recorded) -> Reply {
    if r.path != "/v1/messages" {
        return script(r);
    }
    let body = r.json();
    if !is_main_thread(&body) {
        return Reply::Sse(sse_text(99, "ok"));
    }
    let n = assistant_turns(&body);
    match n {
        0 => {
            Reply::Sse(sse_tool_use(n, "toolu_full_read", "mcp__squeeze__read_many", json!({ "files": ["hello.txt"] })))
        }
        1 => Reply::Sse(sse_tool_use(
            n,
            "toolu_full_edit",
            "mcp__squeeze__edit_many",
            json!({ "edits": [
                { "path": "hello.txt", "old_string": "hello", "new_string": "goodbye" },
                { "path": "src/new.txt", "new_string": "created by edit_many\n" }
            ] }),
        )),
        _ => Reply::Sse(sse_text(n, "done")),
    }
}

/// `squeeze run --replace-file-tools`: Claude Code's file tools are gone, squeeze's
/// read and edit tools are sent in full with guidance, and edits land on disk.
#[tokio::test]
#[ignore = "needs the claude CLI; run with --ignored"]
async fn squeeze_run_replace_file_tools() {
    let claude = std::env::var("CLAUDE_BIN").unwrap_or_else(|_| "claude".into());
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("hello.txt"), "hello from squeeze\n").unwrap();
    let mock = common::start(Arc::new(full_tools_script)).await;

    let out = tokio::process::Command::new(env!("CARGO_BIN_EXE_squeeze"))
        .current_dir(&work)
        .args(["run", "--quiet", "--replace-file-tools", "--trim-tools", "--claude-bin", &claude, "--"])
        .args(["-p", "Change hello to goodbye and add src/new.txt.", "--output-format", "json"])
        // edit_many is approved without asking exactly when Edit would be: here, in acceptEdits.
        .args(["--permission-mode", "acceptEdits", "--append-system-prompt", "USER-PROMPT-MARKER"])
        .args(["--model", "claude-mock", "--allowedTools", "Bash", "--strict-mcp-config"])
        .env("ANTHROPIC_BASE_URL", mock.url())
        .env("ANTHROPIC_API_KEY", FAKE_KEY)
        .env("ENABLE_TOOL_SEARCH", "true")
        .env("SQUEEZE_HOME", tmp.path().join("squeeze-home"))
        .env("CLAUDE_CONFIG_DIR", tmp.path().join("claude-config"))
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .env("DISABLE_AUTOUPDATER", "1")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .env_remove("CLAUDE_CODE_OAUTH_TOKEN")
        .stdin(Stdio::null())
        .output();
    let out = tokio::time::timeout(Duration::from_secs(180), out).await.expect("timed out").unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "squeeze run failed\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let result: Value = serde_json::from_str(stdout.trim()).expect("claude JSON output");
    assert_eq!(result["result"], "done", "{result}");
    assert_eq!(result["permission_denials"], json!([]), "{result}");

    let main: Vec<Value> = mock
        .requests()
        .iter()
        .filter(|r| r.path == "/v1/messages")
        .map(Recorded::json)
        .filter(is_main_thread)
        .collect();
    let first = main.iter().find(|b| assistant_turns(b) == 0).unwrap();
    let names: Vec<&str> = first["tools"].as_array().unwrap().iter().filter_map(|t| t["name"].as_str()).collect();
    for builtin in ["Read", "Edit", "Write", "Grep", "Glob", "NotebookEdit", "Workflow", "ScheduleWakeup", "ListAgents"]
    {
        assert!(!names.contains(&builtin), "{builtin} should be turned off: {names:?}");
    }
    for tool in ["mcp__squeeze__read_many", "mcp__squeeze__edit_many", "mcp__squeeze__search"] {
        assert!(names.contains(&tool), "{tool} not sent in full: {names:?}");
    }
    let system = first["system"].to_string();
    assert!(system.contains("turned off in this session"), "guidance in the system prompt");
    assert!(system.contains("USER-PROMPT-MARKER"), "the user's own --append-system-prompt is kept");

    let last = main.iter().max_by_key(|b| assistant_turns(b)).unwrap();
    let edit = tool_result_text(last, "toolu_full_edit").unwrap();
    assert!(edit.contains("hello.txt: 1 replacement(s) at line(s) 1"), "{edit}");
    assert!(edit.contains("src/new.txt: created (1 lines)"), "{edit}");
    assert_eq!(std::fs::read_to_string(work.join("hello.txt")).unwrap(), "goodbye from squeeze\n");
    assert_eq!(std::fs::read_to_string(work.join("src/new.txt")).unwrap(), "created by edit_many\n");
}

/// In the default permission mode, edit_many needs approval like Edit: `-p`
/// can't ask, so the edit is denied and nothing changes on disk.
#[tokio::test]
#[ignore = "needs the claude CLI; run with --ignored"]
async fn edit_many_asks_like_edit_in_the_default_mode() {
    let claude = std::env::var("CLAUDE_BIN").unwrap_or_else(|_| "claude".into());
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("hello.txt"), "hello from squeeze\n").unwrap();
    let mock = common::start(Arc::new(full_tools_script)).await;
    let out = tokio::process::Command::new(env!("CARGO_BIN_EXE_squeeze"))
        .current_dir(&work)
        .args(["run", "--quiet", "--replace-file-tools", "--claude-bin", &claude, "--"])
        .args(["-p", "Change hello to goodbye.", "--output-format", "json"])
        .args(["--model", "claude-mock", "--allowedTools", "Bash", "--strict-mcp-config"])
        .env("ANTHROPIC_BASE_URL", mock.url())
        .env("ANTHROPIC_API_KEY", FAKE_KEY)
        .env("SQUEEZE_HOME", tmp.path().join("squeeze-home"))
        .env("CLAUDE_CONFIG_DIR", tmp.path().join("claude-config"))
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .env("DISABLE_AUTOUPDATER", "1")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .env_remove("CLAUDE_CODE_OAUTH_TOKEN")
        .stdin(Stdio::null())
        .output();
    let out = tokio::time::timeout(Duration::from_secs(180), out).await.expect("timed out").unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let result: Value = serde_json::from_str(stdout.trim()).expect("claude JSON output");
    let denied = result["permission_denials"].to_string();
    assert!(denied.contains("mcp__squeeze__edit_many"), "{result}");
    assert_eq!(std::fs::read_to_string(work.join("hello.txt")).unwrap(), "hello from squeeze\n");
}
