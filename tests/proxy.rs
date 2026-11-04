//! Proxy integration tests against a scripted upstream (no network, no API key).

mod common;

use common::{FakeClock, Reply, sse_text, tool_result_text};
use serde_json::{Value, json};
use squeeze::ctxmgmt::{CtxConfig, KeepWindow};
use squeeze::proxy::{Proxy, ProxyConfig};
use squeeze::store::Store;
use squeeze::transform::TransformConfig;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

fn big_log() -> String {
    let mut s: String = (0..500).map(|i| format!("   Compiling dep{i} v0.{i}.0\n")).collect();
    s.push_str("error: linker failed\n");
    s
}

fn messages_body() -> Value {
    json!({
        "model": "mock-model",
        "max_tokens": 100,
        "stream": true,
        "system": [{ "type": "text", "text": "sys", "cache_control": { "type": "ephemeral", "ttl": "1h" } }],
        "tools": [{ "name": "Bash", "input_schema": { "type": "object" } }],
        "messages": [
            { "role": "user", "content": "build" },
            { "role": "assistant", "content": [{ "type": "tool_use", "id": "t1", "name": "Bash", "input": { "command": "cargo build" } }] },
            { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "t1", "content": big_log() }] }
        ],
        "some_future_field": { "kept": true }
    })
}

async fn setup(transform: bool) -> (common::Mock, Proxy, String, tempfile::TempDir) {
    let mock = common::start(Arc::new(|r: &common::Recorded| match r.path.as_str() {
        "/v1/messages" => Reply::Sse(sse_text(0, "hello")),
        "/v1/messages/count_tokens" => Reply::Json(200, json!({ "input_tokens": 42 })),
        "/v1/bad" => Reply::Json(400, json!({ "type": "error", "error": { "type": "invalid_request_error", "message": "exact upstream wording" } })),
        _ => Reply::Json(200, json!({ "data": [], "path": r.path })),
    }))
    .await;
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let proxy = Proxy::new(ProxyConfig {
        upstream: mock.url(),
        transform: transform.then(TransformConfig::default),
        store,
        ctx: None,
    })
    .unwrap();
    let addr = proxy.spawn("127.0.0.1:0").await.unwrap();
    (mock, proxy, format!("http://{addr}"), dir)
}

fn client() -> reqwest::Client {
    reqwest::Client::new()
}

#[tokio::test]
async fn compresses_tool_output_and_streams_response_unchanged() {
    let (mock, proxy, url, _dir) = setup(true).await;
    let resp = client()
        .post(format!("{url}/v1/messages?beta=true"))
        .header("x-api-key", "sk-ant-api03-test")
        .header("anthropic-version", "2023-06-01")
        .header("anthropic-beta", "context-management-2025-06-27")
        .header("x-claude-code-session-id", "sess-1")
        .json(&messages_body())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
    assert_eq!(resp.text().await.unwrap(), sse_text(0, "hello"), "response must pass through byte-for-byte");

    let reqs = mock.requests();
    assert_eq!(reqs.len(), 1);
    let up = &reqs[0];
    assert_eq!(up.path, "/v1/messages");
    // Headers forwarded unchanged.
    assert_eq!(up.headers["x-api-key"], "sk-ant-api03-test");
    assert_eq!(up.headers["anthropic-version"], "2023-06-01");
    assert_eq!(up.headers["anthropic-beta"], "context-management-2025-06-27");
    assert_eq!(up.headers["x-claude-code-session-id"], "sess-1");
    let body = up.json();
    let text = tool_result_text(&body, "t1").unwrap();
    assert!(text.len() < big_log().len() / 5, "tool output should be compressed");
    assert!(text.contains("error: linker failed"));
    assert!(text.contains("squeeze expand"));
    // Everything else untouched.
    let original = messages_body();
    for key in ["system", "tools", "some_future_field", "model", "max_tokens", "stream"] {
        assert_eq!(body[key], original[key], "{key} must be forwarded unchanged");
    }

    // Wait for the stream finalizer, then check accounting.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let t = proxy.totals();
    assert_eq!(t.requests, 1);
    assert_eq!(t.compressed, 1);
    assert_eq!(t.usage.output_tokens, 20);
    assert!(t.saved_input_eq > 0.0);
}

#[tokio::test]
async fn refuses_subscription_oauth_without_forwarding() {
    let (mock, proxy, url, _dir) = setup(true).await;
    let resp = client()
        .post(format!("{url}/v1/messages"))
        .header("authorization", "Bearer sk-ant-oat01-secret")
        .header("anthropic-beta", "oauth-2025-04-20,interleaved-thinking-2025-05-14")
        .json(&messages_body())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
    let err: Value = resp.json().await.unwrap();
    assert_eq!(err["type"], "error");
    assert!(err["error"]["message"].as_str().unwrap().contains("subscription"));
    assert!(mock.requests().is_empty(), "nothing may reach the upstream");
    assert_eq!(proxy.totals().refused, 1);
}

#[tokio::test]
async fn passthrough_mode_forwards_bytes_exactly() {
    let (mock, _proxy, url, _dir) = setup(false).await;
    let raw = serde_json::to_vec(&messages_body()).unwrap();
    client()
        .post(format!("{url}/v1/messages"))
        .header("x-api-key", "k")
        .header("content-type", "application/json")
        .body(raw.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(mock.requests()[0].body, raw);
}

#[tokio::test]
async fn error_bodies_and_other_paths_pass_through() {
    let (mock, _proxy, url, _dir) = setup(true).await;
    let resp = client().post(format!("{url}/v1/bad")).header("x-api-key", "k").body("{}").send().await.unwrap();
    assert_eq!(resp.status(), 400);
    let err: Value = resp.json().await.unwrap();
    assert_eq!(err["error"]["message"], "exact upstream wording");

    let resp = client().get(format!("{url}/v1/models?limit=5")).header("x-api-key", "k").send().await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(mock.requests().last().unwrap().path, "/v1/models");
}

#[tokio::test]
async fn count_tokens_sees_the_same_rewrite_and_bad_json_is_forwarded() {
    let (mock, _proxy, url, _dir) = setup(true).await;
    let resp = client()
        .post(format!("{url}/v1/messages/count_tokens"))
        .header("x-api-key", "k")
        .json(&messages_body())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.json::<Value>().await.unwrap()["input_tokens"], 42);
    let text = tool_result_text(&mock.requests()[0].json(), "t1").unwrap();
    assert!(text.contains("[squeeze:"));

    client().post(format!("{url}/v1/messages")).header("x-api-key", "k").body("not json").send().await.unwrap();
    assert_eq!(mock.requests()[1].body, b"not json");
}

#[tokio::test]
async fn rewrite_is_stable_across_turns() {
    // Turn 2 resends turn 1's history; its rewritten bytes must not change.
    let (mock, _proxy, url, _dir) = setup(true).await;
    let turn1 = messages_body();
    let mut turn2 = turn1.clone();
    let msgs = turn2["messages"].as_array_mut().unwrap();
    msgs.push(json!({ "role": "assistant", "content": [{ "type": "tool_use", "id": "t2", "name": "Bash", "input": { "command": "ls" } }] }));
    msgs.push(
        json!({ "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "t2", "content": "a\nb" }] }),
    );
    for body in [&turn1, &turn2] {
        client().post(format!("{url}/v1/messages")).header("x-api-key", "k").json(body).send().await.unwrap();
    }
    let reqs = mock.requests();
    let m1 = reqs[0].json()["messages"].clone();
    let m2 = reqs[1].json()["messages"].clone();
    assert_eq!(m1.as_array().unwrap()[..], m2.as_array().unwrap()[..3]);
}

const T0: u64 = 1_000_000;

/// A proxy with cold-cache pruning on `store_dir`; each upstream request takes
/// `request_secs` of fake time.
async fn prune_proxy(clock: &Arc<FakeClock>, store_dir: &Path, request_secs: u64) -> (common::Mock, String) {
    let c = clock.clone();
    let mock = common::start(Arc::new(move |_: &common::Recorded| {
        c.0.fetch_add(request_secs, Ordering::SeqCst);
        Reply::Sse(sse_text(0, "ok"))
    }))
    .await;
    let mut cfg = TransformConfig::default();
    cfg.prune.enabled = true;
    let store = Store::open(store_dir).unwrap();
    let proxy =
        Proxy::with_clock(ProxyConfig { upstream: mock.url(), transform: Some(cfg), store, ctx: None }, clock.clone())
            .unwrap();
    let addr = proxy.spawn("127.0.0.1:0").await.unwrap();
    (mock, format!("http://{addr}"))
}

/// Claude Code's shape: a Read of a file, then an Edit of it, with the cache
/// marker (`ttl`, if any) on the last block. At a cold moment the stale Read
/// is pruned, which is how these tests see that the proxy judged it cold.
fn edit_conversation(ttl: Option<&str>) -> Value {
    let mut marker = json!({ "type": "ephemeral" });
    if let Some(t) = ttl {
        marker["ttl"] = json!(t);
    }
    let file: String = (1..=40).map(|i| format!("{i}\tlet v{i} = {i};\n")).collect();
    json!({
        "model": "mock-model",
        "max_tokens": 100,
        "stream": true,
        "system": [{ "type": "text", "text": "sys", "cache_control": { "type": "ephemeral" } }],
        "messages": [
            { "role": "user", "content": "fix a.rs" },
            { "role": "assistant", "content": [{ "type": "tool_use", "id": "toolu_r1", "name": "Read", "input": { "file_path": "/w/a.rs" } }] },
            { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "toolu_r1", "content": file }] },
            { "role": "assistant", "content": [{ "type": "tool_use", "id": "toolu_e1", "name": "Edit", "input": { "file_path": "/w/a.rs", "old_string": "v1", "new_string": "w1" } }] },
            { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "toolu_e1", "content": "ok", "cache_control": marker }] }
        ]
    })
}

/// Send at fake time `at` and say whether the stale Read went out pruned.
async fn send_at(
    url: &str,
    mock: &common::Mock,
    clock: &FakeClock,
    at: u64,
    body: &Value,
    headers: &[(&str, &str)],
) -> bool {
    clock.set(at);
    let mut req = client().post(format!("{url}/v1/messages")).header("x-api-key", "sk-ant-api03-test").json(body);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    req.send().await.unwrap().text().await.unwrap();
    // Let the stream finalizer record the response's end.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let sent = mock.requests().last().unwrap().json();
    tool_result_text(&sent, "toolu_r1").unwrap().contains("pruned because the file was edited")
}

const S1: &[(&str, &str)] = &[("x-claude-code-session-id", "s1")];

#[tokio::test]
async fn cold_after_the_message_ttl_plus_five_seconds() {
    let (clock, dir) = (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap());
    let (mock, url) = prune_proxy(&clock, dir.path(), 0).await;
    let body = edit_conversation(None);
    assert!(!send_at(&url, &mock, &clock, T0, &body, S1).await, "a thread's first request is never cold");
    assert!(!send_at(&url, &mock, &clock, T0 + 304, &body, S1).await, "TTL + 4 s: still warm");
    // A count_tokens preview isn't a request start: it mustn't reset the clock.
    clock.set(T0 + 500);
    client().post(format!("{url}/v1/messages/count_tokens")).header("x-api-key", "k").json(&body).send().await.unwrap();
    assert!(send_at(&url, &mock, &clock, T0 + 304 + 306, &body, S1).await, "TTL + 6 s: cold");
}

#[tokio::test]
async fn a_330_second_gap_after_a_5_second_request_is_cold() {
    let (clock, dir) = (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap());
    let (mock, url) = prune_proxy(&clock, dir.path(), 5).await;
    let body = edit_conversation(None);
    send_at(&url, &mock, &clock, T0, &body, S1).await;
    assert!(send_at(&url, &mock, &clock, T0 + 5 + 330, &body, S1).await);

    // With Claude Code's 1-hour cache on the history, the same gap is warm.
    let (clock, dir) = (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap());
    let (mock, url) = prune_proxy(&clock, dir.path(), 5).await;
    let body = edit_conversation(Some("1h"));
    send_at(&url, &mock, &clock, T0, &body, S1).await;
    assert!(!send_at(&url, &mock, &clock, T0 + 5 + 330, &body, S1).await);
}

#[tokio::test]
async fn the_ttl_runs_from_the_request_start_not_its_end() {
    // A 200 s request, then 110 s idle: the cache entry written at its start is gone.
    let (clock, dir) = (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap());
    let (mock, url) = prune_proxy(&clock, dir.path(), 200).await;
    let body = edit_conversation(None);
    send_at(&url, &mock, &clock, T0, &body, S1).await;
    assert!(send_at(&url, &mock, &clock, T0 + 200 + 110, &body, S1).await);
}

#[tokio::test]
async fn thread_state_survives_a_restart() {
    let (clock, dir) = (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap());
    let body = edit_conversation(None);
    let (mock, url) = prune_proxy(&clock, dir.path(), 0).await;
    send_at(&url, &mock, &clock, T0, &body, S1).await;
    assert!(send_at(&url, &mock, &clock, T0 + 400, &body, S1).await);
    let before = mock.requests().last().unwrap().json();

    // A new proxy on the same store, 10 s later (warm): the next turn goes out
    // with the same rewritten history.
    let (mock, url) = prune_proxy(&clock, dir.path(), 0).await;
    let mut next = body.clone();
    let msgs = next["messages"].as_array_mut().unwrap();
    msgs[4]["content"][0].as_object_mut().unwrap().remove("cache_control");
    msgs.push(json!({ "role": "assistant", "content": [{ "type": "text", "text": "done" }] }));
    msgs.push(json!({ "role": "user", "content": [{ "type": "text", "text": "thanks", "cache_control": { "type": "ephemeral" } }] }));
    assert!(send_at(&url, &mock, &clock, T0 + 410, &next, S1).await);
    let after = mock.requests().last().unwrap().json();
    assert_eq!(after["messages"].as_array().unwrap()[..4], before["messages"].as_array().unwrap()[..4]);
}

#[tokio::test]
async fn a_thread_is_found_by_its_first_tool_call() {
    let (clock, dir) = (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap());
    let (mock, url) = prune_proxy(&clock, dir.path(), 0).await;
    let body = edit_conversation(None);
    send_at(&url, &mock, &clock, T0, &body, S1).await;

    // Resumed in a new process: another session id, another model, and the
    // first message in the string form Claude Code sends once it's no longer last.
    let mut resumed = body.clone();
    resumed["model"] = json!("other-model");
    resumed["messages"][0] = json!({ "role": "user", "content": [{ "type": "text", "text": "fix a.rs" }] });
    assert!(send_at(&url, &mock, &clock, T0 + 400, &resumed, &[("x-claude-code-session-id", "s2")]).await);

    // The same history under another agent id starts from that thread's state.
    let agent = [("x-claude-code-session-id", "s2"), ("x-claude-code-agent-id", "agent-x")];
    assert!(send_at(&url, &mock, &clock, T0 + 401, &body, &agent).await);
}

// --- Server-side context editing (`--context-edits cold`) ---

/// How the mock API answers in the context-editing tests.
#[derive(Clone, Copy)]
struct Api {
    /// Prompt tokens each reply says it read from cache (of 50,100).
    cache_read: u64,
    /// Report one cleared tool use fewer than asked.
    misreport: bool,
    /// Refuse requests that carry a clear_tool_uses edit with a 400.
    reject_edits: bool,
    /// Answer everything with a 529.
    overloaded: bool,
    /// Cut every stream short after `message_start`.
    cut: bool,
    /// Drop every connection unanswered.
    unreachable: bool,
}

impl Default for Api {
    fn default() -> Self {
        Self {
            cache_read: 50_000,
            misreport: false,
            reject_edits: false,
            overloaded: false,
            cut: false,
            unreachable: false,
        }
    }
}

/// A proxy with `--context-edits cold` and a keep window of 4 tool uses, in
/// front of a mock that applies the edits it receives and reports them.
async fn cm_proxy(clock: &Arc<FakeClock>, store_dir: &Path, api: &Arc<Mutex<Api>>) -> (common::Mock, String, Proxy) {
    let api = api.clone();
    cm_proxy_with(clock, store_dir, move |r: &common::Recorded| {
        let api = *api.lock().unwrap();
        if r.path == "/v1/messages/count_tokens" {
            return Reply::Json(200, json!({ "input_tokens": 42 }));
        }
        let body = r.json();
        if api.unreachable {
            return Reply::Hangup;
        }
        if api.overloaded {
            return Reply::Json(
                529,
                json!({ "type": "error", "error": { "type": "overloaded_error", "message": "busy" } }),
            );
        }
        if api.reject_edits && body["context_management"].to_string().contains("clear_tool_uses") {
            let err = json!({ "type": "error", "error": { "type": "invalid_request_error", "message": "bad edit" } });
            return Reply::Json(400, err);
        }
        let mut cm = common::emulate_context_edits(&body);
        if api.misreport {
            for e in cm["applied_edits"].as_array_mut().unwrap() {
                if let Some(n) = e["cleared_tool_uses"].as_u64() {
                    e["cleared_tool_uses"] = json!(n - 1);
                }
            }
        }
        let usage = json!({ "input_tokens": 100, "cache_creation_input_tokens": 50_000 - api.cache_read,
                            "cache_read_input_tokens": api.cache_read, "output_tokens": 1 });
        if api.cut {
            return Reply::Sse(common::sse_cut_short(0, usage));
        }
        Reply::Sse(common::sse_text_with(0, "ok", usage, cm))
    })
    .await
}

async fn cm_proxy_with(
    clock: &Arc<FakeClock>,
    store_dir: &Path,
    script: impl Fn(&common::Recorded) -> Reply + Send + Sync + 'static,
) -> (common::Mock, String, Proxy) {
    let mock = common::start(Arc::new(script)).await;
    let ctx = CtxConfig { window: KeepWindow { min: 4, max: 4 }, ..Default::default() };
    let transform = TransformConfig { ctx_window: Some(ctx.window), ..Default::default() };
    let cfg = ProxyConfig {
        upstream: mock.url(),
        transform: Some(transform),
        store: Store::open(store_dir).unwrap(),
        ctx: Some(ctx),
    };
    let proxy = Proxy::with_clock(cfg, clock.clone()).unwrap();
    let addr = proxy.spawn("127.0.0.1:0").await.unwrap();
    (mock, format!("http://{addr}"), proxy)
}

fn tool_loop(id: &str, name: &str, input: Value, result: String) -> [Value; 2] {
    [
        json!({ "role": "assistant", "content": [
            { "type": "thinking", "thinking": "", "signature": format!("sig-{id}") },
            { "type": "tool_use", "id": id, "name": name, "input": input }
        ] }),
        json!({ "role": "user", "content": [{ "type": "tool_result", "tool_use_id": id, "content": result }] }),
    ]
}

/// The ~7 KB file `cat_loop(id)` shows.
fn cat_file(id: &str) -> String {
    (0..250).map(|l| format!("{l}\tlet {id}_{l} = {l};\n")).collect()
}

fn cat_loop(id: &str) -> [Value; 2] {
    tool_loop(id, "Bash", json!({ "command": format!("cat src/{id}.rs") }), cat_file(id))
}

/// Claude Code's shape on Opus 5.5: a prompt, `first` tool loops (thinking and
/// a `cat` of a distinct ~5 KB file) with a TodoWrite call after the first,
/// and a reply; then a second prompt and `second` more loops.
fn cm_conversation(first: usize, second: usize) -> Value {
    let mut m = vec![json!({ "role": "user", "content": "look around" })];
    for i in 0..first {
        m.extend(cat_loop(&format!("toolu_a{i:02}")));
        if i == 0 {
            m.extend(tool_loop("toolu_todo", "TodoWrite", json!({ "todos": [] }), "ok".into()));
        }
    }
    m.push(json!({ "role": "assistant", "content": [
        { "type": "thinking", "thinking": "", "signature": "sig-reply" }, { "type": "text", "text": "found it" }
    ] }));
    m.push(json!({ "role": "user", "content": "now fix it" }));
    m.extend((0..second).flat_map(|i| cat_loop(&format!("toolu_b{i:02}"))));
    json!({
        "model": "claude-opus-5-5",
        "max_tokens": 100,
        "stream": true,
        "thinking": { "type": "adaptive" },
        "tools": [{ "name": "Bash", "input_schema": { "type": "object" } }, { "name": "TodoWrite", "input_schema": { "type": "object" } }],
        "context_management": { "edits": [{ "type": "clear_thinking_20251015", "keep": "all" }] },
        "messages": m
    })
}

const CM_BETAS: &str = "claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27";

/// Send `body` at fake time `at` the way Claude Code does, and return what reached the API.
async fn cm_send(
    url: &str,
    mock: &common::Mock,
    clock: &FakeClock,
    at: u64,
    path: &str,
    body: &Value,
) -> common::Recorded {
    clock.set(at);
    client()
        .post(format!("{url}{path}?beta=true"))
        .header("x-api-key", "sk-ant-api03-test")
        .header("anthropic-beta", CM_BETAS)
        .header("content-type", "application/json")
        .body(serde_json::to_vec(body).unwrap())
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    // Let the stream finalizer settle the response.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    mock.requests().last().unwrap().clone()
}

const MESSAGES: &str = "/v1/messages";

/// The edits a forwarded request carried.
fn edits(r: &common::Recorded) -> Value {
    r.json()["context_management"]["edits"].clone()
}

/// The edits pinned by an advance at `cm_conversation(6, 4)`, as sent with
/// `cm_conversation(6, second)`: 6 tool uses and 8 thinking turns cleared.
fn pinned(second: usize) -> Value {
    let (n, n_th) = (6 + second, 8 + second);
    json!([
        { "type": "clear_thinking_20251015", "keep": { "type": "thinking_turns", "value": n_th - 8 } },
        { "type": "clear_tool_uses_20250919", "trigger": { "type": "tool_uses", "value": n - 1 },
          "keep": { "type": "tool_uses", "value": n - 6 }, "exclude_tools": ["TodoWrite"] }
    ])
}

/// Two warm requests, then a cold one that advances; returns the time of the advance.
async fn advance(url: &str, mock: &common::Mock, clock: &FakeClock) -> u64 {
    cm_send(url, mock, clock, T0, MESSAGES, &cm_conversation(6, 3)).await;
    let up = cm_send(url, mock, clock, T0 + 306, MESSAGES, &cm_conversation(6, 4)).await;
    assert_eq!(edits(&up), pinned(4));
    T0 + 306
}

fn ctx_lines(dir: &Path) -> Vec<Value> {
    Store::open(dir).unwrap().records().unwrap().into_iter().filter_map(|r| r.get("ctx").cloned()).collect()
}

#[tokio::test]
async fn context_edits_off_forwards_context_management_unchanged() {
    let (mock, _proxy, url, _dir) = setup(true).await;
    let raw = serde_json::to_vec(&cm_conversation(6, 3)).unwrap();
    client()
        .post(format!("{url}/v1/messages"))
        .header("x-api-key", "k")
        .header("anthropic-beta", CM_BETAS)
        .body(raw.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(mock.requests()[0].body, raw);
}

#[tokio::test]
async fn cm_advances_at_a_cold_moment_and_keeps_the_cleared_set_pinned() {
    let (clock, dir, api) =
        (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap(), Arc::new(Mutex::new(Api::default())));
    let (mock, url, proxy) = cm_proxy(&clock, dir.path(), &api).await;

    // Warm requests go out exactly as sent until the first advance.
    for (at, second) in [(T0, 2), (T0 + 100, 3)] {
        let body = cm_conversation(6, second);
        let up = cm_send(&url, &mock, &clock, at, MESSAGES, &body).await;
        assert_eq!(up.body, serde_json::to_vec(&body).unwrap(), "{second}: byte-identical");
    }

    // Cold: all but the newest 4 of 10 clearable uses, and the thinking from
    // before the latest prompt (8 of 12 turns). Messages untouched.
    let body = cm_conversation(6, 4);
    let up = cm_send(&url, &mock, &clock, T0 + 100 + 306, MESSAGES, &body).await.json();
    assert_eq!(up["messages"], body["messages"], "message bytes are never edited");
    assert_eq!(up["context_management"]["edits"], pinned(4));
    let mut rest = up.clone();
    rest.as_object_mut().unwrap().remove("context_management");
    let mut sent = body.clone();
    sent.as_object_mut().unwrap().remove("context_management");
    assert_eq!(rest, sent, "every other field as sent");

    // Warm turns after it keep the same 6 uses and 8 thinking turns cleared.
    for (i, second) in [5, 6, 9].into_iter().enumerate() {
        let body = cm_conversation(6, second);
        let up = cm_send(&url, &mock, &clock, T0 + 410 + i as u64, MESSAGES, &body).await;
        assert_eq!(edits(&up), pinned(second), "{second}");
        assert_eq!(up.json()["messages"], body["messages"]);
    }
    let lines = ctx_lines(dir.path());
    assert_eq!(lines.len(), 6);
    assert_eq!((lines[1]["advance"].clone(), lines[1]["C"].clone()), (json!(false), json!(0)));
    let adv = &lines[2];
    assert_eq!(
        (adv["advance"].clone(), adv["C"].clone(), adv["C_th"].clone(), adv["K"].clone()),
        (json!(true), json!(6), json!(8), json!(4))
    );
    assert_eq!(adv["applied"]["cleared_tool_uses"], 6);
    let cleared = adv["cleared_input_tokens"].as_u64().unwrap();
    assert!(cleared > 6 * 1000, "six ~5 KB results and eight thinking blocks: {cleared}");
    assert!(lines.iter().all(|l| l["breakers"] == json!([])), "{lines:?}");
    assert_eq!(lines[3]["cold"], false);
    let t = proxy.totals();
    assert_eq!((t.ctx_advances, t.cleared_input_tokens), (1, 4 * cleared), "the same set cleared each time");
}

#[tokio::test]
async fn count_tokens_previews_the_edits_but_never_advances() {
    let (clock, dir, api) =
        (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap(), Arc::new(Mutex::new(Api::default())));
    let (mock, url, _proxy) = cm_proxy(&clock, dir.path(), &api).await;
    cm_send(&url, &mock, &clock, T0, MESSAGES, &cm_conversation(6, 3)).await;
    // At a cold moment, a count_tokens call goes out as sent and records nothing.
    let body = cm_conversation(6, 4);
    let up = cm_send(&url, &mock, &clock, T0 + 400, "/v1/messages/count_tokens", &body).await;
    assert_eq!(up.body, serde_json::to_vec(&body).unwrap());
    // So the next messages request is still cold, and advances.
    let up = cm_send(&url, &mock, &clock, T0 + 401, MESSAGES, &body).await;
    assert_eq!(edits(&up), pinned(4));
    // Later previews carry the pinned edits.
    let up = cm_send(&url, &mock, &clock, T0 + 1000, "/v1/messages/count_tokens", &cm_conversation(6, 9)).await;
    assert_eq!(edits(&up), pinned(9));
    let up = cm_send(&url, &mock, &clock, T0 + 1001, MESSAGES, &cm_conversation(6, 9)).await;
    assert_ne!(edits(&up), pinned(9), "the preview didn't stamp the clock either");
}

#[tokio::test]
async fn a_mismatch_freezes_the_thread() {
    let (clock, dir, api) =
        (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap(), Arc::new(Mutex::new(Api::default())));
    let (mock, url, _proxy) = cm_proxy(&clock, dir.path(), &api).await;
    api.lock().unwrap().misreport = true;
    let at = advance(&url, &mock, &clock).await;
    assert_eq!(ctx_lines(dir.path()).last().unwrap()["breakers"], json!(["ctx_mismatch"]));
    api.lock().unwrap().misreport = false;
    // Much later and much longer: still the same cleared set.
    let up = cm_send(&url, &mock, &clock, at + 1000, MESSAGES, &cm_conversation(6, 20)).await;
    assert_eq!(edits(&up), pinned(20));
}

#[tokio::test]
async fn cache_regressed_twice_goes_back_to_claude_codes_setting() {
    let (clock, dir, api) =
        (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap(), Arc::new(Mutex::new(Api::default())));
    let (mock, url, _proxy) = cm_proxy(&clock, dir.path(), &api).await;
    let at = advance(&url, &mock, &clock).await;
    api.lock().unwrap().cache_read = 1000;
    for second in [5, 6] {
        let up = cm_send(&url, &mock, &clock, at + second as u64, MESSAGES, &cm_conversation(6, second)).await;
        assert_eq!(edits(&up), pinned(second), "pinned while the cache is warm");
        assert_eq!(ctx_lines(dir.path()).last().unwrap()["breakers"], json!(["cache_regressed"]));
    }
    api.lock().unwrap().cache_read = 50_000;
    let body = cm_conversation(6, 7);
    let up = cm_send(&url, &mock, &clock, at + 1000, MESSAGES, &body).await;
    assert_eq!(up.body, serde_json::to_vec(&body).unwrap(), "back to Claude Code's own setting at a cold moment");
}

#[tokio::test]
async fn a_rejected_edit_goes_back_to_claude_codes_setting_at_once() {
    let (clock, dir, api) =
        (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap(), Arc::new(Mutex::new(Api::default())));
    let (mock, url, _proxy) = cm_proxy(&clock, dir.path(), &api).await;
    api.lock().unwrap().reject_edits = true;
    cm_send(&url, &mock, &clock, T0, MESSAGES, &cm_conversation(6, 3)).await;
    clock.set(T0 + 306);
    let resp = client()
        .post(format!("{url}/v1/messages"))
        .header("x-api-key", "sk-ant-api03-test")
        .header("anthropic-beta", CM_BETAS)
        .json(&cm_conversation(6, 4))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    assert_eq!(resp.json::<Value>().await.unwrap()["error"]["message"], "bad edit", "the error passes through");
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(ctx_lines(dir.path()).last().unwrap()["breakers"], json!(["rejected"]));
    let body = cm_conversation(6, 4);
    let up = cm_send(&url, &mock, &clock, T0 + 310, MESSAGES, &body).await;
    assert_eq!(up.body, serde_json::to_vec(&body).unwrap(), "the retry goes out as sent");
}

#[tokio::test]
async fn cm_state_survives_a_restart() {
    let (clock, dir, api) =
        (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap(), Arc::new(Mutex::new(Api::default())));
    let (mock, url, _proxy) = cm_proxy(&clock, dir.path(), &api).await;
    let at = advance(&url, &mock, &clock).await;
    let before = cm_send(&url, &mock, &clock, at + 5, MESSAGES, &cm_conversation(6, 5)).await;
    // A new proxy on the same store, still warm: the same request goes out the same.
    let (mock, url, _proxy) = cm_proxy(&clock, dir.path(), &api).await;
    let after = cm_send(&url, &mock, &clock, at + 10, MESSAGES, &cm_conversation(6, 5)).await;
    assert_eq!(after.body, before.body);
    assert!(ctx_lines(dir.path()).iter().all(|l| l["breakers"] == json!([])));
}

#[tokio::test]
async fn a_failed_request_leaves_the_cache_clock_alone() {
    let (clock, dir, api) =
        (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap(), Arc::new(Mutex::new(Api::default())));
    let (mock, url, _proxy) = cm_proxy(&clock, dir.path(), &api).await;
    cm_send(&url, &mock, &clock, T0, MESSAGES, &cm_conversation(6, 3)).await;
    api.lock().unwrap().overloaded = true;
    cm_send(&url, &mock, &clock, T0 + 200, MESSAGES, &cm_conversation(6, 4)).await;
    api.lock().unwrap().overloaded = false;
    // 110 s after the failed request, but 306 s after the last one that cached anything.
    let up = cm_send(&url, &mock, &clock, T0 + 306, MESSAGES, &cm_conversation(6, 4)).await;
    assert_eq!(edits(&up), pinned(4), "cold");
}

#[tokio::test]
async fn an_advance_is_on_disk_before_the_request_goes_out() {
    let (clock, dir) = (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap());
    // The mock looks at the saved thread state as each request reaches it.
    let threads = dir.path().join("threads");
    let on_disk = Arc::new(Mutex::new(Vec::new()));
    let seen = on_disk.clone();
    let (mock, url, _proxy) = cm_proxy_with(&clock, dir.path(), move |r: &common::Recorded| {
        let cleared = std::fs::read_dir(&threads)
            .unwrap()
            .flatten()
            .filter_map(|e| serde_json::from_slice::<Value>(&std::fs::read(e.path()).ok()?).ok())
            .filter_map(|t| t["ctx"]["cleared"].as_u64())
            .max();
        seen.lock().unwrap().push(cleared);
        let usage = json!({ "input_tokens": 100, "cache_read_input_tokens": 50_000, "output_tokens": 1 });
        Reply::Sse(common::sse_text_with(0, "ok", usage, common::emulate_context_edits(&r.json())))
    })
    .await;
    advance(&url, &mock, &clock).await;
    assert_eq!(*on_disk.lock().unwrap(), [None, Some(6)]);
}

/// The `ctx` line and the whole stats line of the latest request.
fn last_record(dir: &Path) -> Value {
    Store::open(dir).unwrap().records().unwrap().pop().unwrap()
}

/// `cm_conversation(6, second)` with one more loop: `cat` of the first
/// file again, with the same output.
fn cat_again(second: usize) -> Value {
    let mut body = cm_conversation(6, second);
    let again = tool_loop("toolu_again", "Bash", json!({ "command": "cat src/toolu_a00.rs" }), cat_file("toolu_a00"));
    body["messages"].as_array_mut().unwrap().extend(again);
    body
}

#[tokio::test]
async fn an_output_repeated_after_its_copy_was_cleared_goes_in_full() {
    let (clock, dir, api) =
        (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap(), Arc::new(Mutex::new(Api::default())));
    let (mock, url, _proxy) = cm_proxy(&clock, dir.path(), &api).await;
    let at = advance(&url, &mock, &clock).await;
    // The server cleared the first `cat`; a dedup stub would point at its placeholder.
    let up = cm_send(&url, &mock, &clock, at + 5, MESSAGES, &cat_again(4)).await.json();
    assert_eq!(up["messages"], cat_again(4)["messages"], "sent as Claude Code sent it");
    let view = common::server::cleared(&up).unwrap().messages;
    let seen = json!({ "messages": view });
    assert_eq!(tool_result_text(&seen, "toolu_a00").unwrap(), common::server::PLACEHOLDER);
    assert_eq!(tool_result_text(&seen, "toolu_again").unwrap(), cat_file("toolu_a00"), "the agent sees the output");
}

#[tokio::test]
async fn a_response_cut_short_trips_no_breaker() {
    let (clock, dir, api) =
        (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap(), Arc::new(Mutex::new(Api::default())));
    let (mock, url, _proxy) = cm_proxy(&clock, dir.path(), &api).await;
    let at = advance(&url, &mock, &clock).await;
    // Esc, or an overloaded error mid-stream: no applied_edits ever arrive.
    api.lock().unwrap().cut = true;
    cm_send(&url, &mock, &clock, at + 5, MESSAGES, &cm_conversation(6, 5)).await;
    assert_eq!(ctx_lines(dir.path()).last().unwrap()["breakers"], json!([]));
    api.lock().unwrap().cut = false;
    // So the thread still advances at the next cold moment: 26 uses, all but 4 cleared.
    let up = cm_send(&url, &mock, &clock, at + 1000, MESSAGES, &cm_conversation(6, 20)).await;
    assert_eq!(edits(&up)[1]["keep"]["value"], 4);
}

#[tokio::test]
async fn a_mismatching_thread_that_misses_the_cache_goes_back_to_claude_codes_setting() {
    let (clock, dir, api) =
        (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap(), Arc::new(Mutex::new(Api::default())));
    let (mock, url, _proxy) = cm_proxy(&clock, dir.path(), &api).await;
    api.lock().unwrap().misreport = true;
    let at = advance(&url, &mock, &clock).await;
    api.lock().unwrap().cache_read = 1000;
    for second in [5, 6] {
        cm_send(&url, &mock, &clock, at + second as u64, MESSAGES, &cm_conversation(6, second)).await;
        assert_eq!(ctx_lines(dir.path()).last().unwrap()["breakers"], json!(["ctx_mismatch", "cache_regressed"]));
    }
    *api.lock().unwrap() = Api::default();
    let body = cm_conversation(6, 7);
    let up = cm_send(&url, &mock, &clock, at + 1000, MESSAGES, &body).await;
    assert_eq!(up.body, serde_json::to_vec(&body).unwrap(), "back to Claude Code's own setting at a cold moment");
}

#[tokio::test]
async fn an_unreachable_api_leaves_the_cache_clock_alone() {
    let (clock, dir, api) =
        (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap(), Arc::new(Mutex::new(Api::default())));
    let (mock, url, _proxy) = cm_proxy(&clock, dir.path(), &api).await;
    let at = advance(&url, &mock, &clock).await;
    cm_send(&url, &mock, &clock, at + 4, MESSAGES, &cm_conversation(6, 5)).await;
    // The laptop wakes 400 s later and the first try can't reach the API.
    api.lock().unwrap().unreachable = true;
    clock.set(at + 404);
    let resp = client()
        .post(format!("{url}/v1/messages"))
        .header("x-api-key", "sk-ant-api03-test")
        .header("anthropic-beta", CM_BETAS)
        .json(&cm_conversation(6, 5))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 502);
    assert_eq!(last_record(dir.path())["status"], 502);
    // Claude Code retries 4 s later. The cache expired long ago, so it reads nothing.
    *api.lock().unwrap() = Api { cache_read: 0, ..Api::default() };
    cm_send(&url, &mock, &clock, at + 408, MESSAGES, &cm_conversation(6, 5)).await;
    let line = ctx_lines(dir.path()).pop().unwrap();
    assert_eq!((line["cold"].clone(), line["breakers"].clone()), (json!(true), json!([])), "{line}");
}

#[tokio::test]
async fn a_model_switch_trips_no_breaker() {
    let (clock, dir, api) =
        (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap(), Arc::new(Mutex::new(Api::default())));
    let (mock, url, _proxy) = cm_proxy(&clock, dir.path(), &api).await;
    let at = advance(&url, &mock, &clock).await;
    // opusplan: Sonnet takes over, and its cache has nothing of this thread yet.
    api.lock().unwrap().cache_read = 0;
    let mut sonnet = cm_conversation(6, 5);
    sonnet["model"] = json!("claude-sonnet-5-5");
    let up = cm_send(&url, &mock, &clock, at + 5, MESSAGES, &sonnet).await;
    assert_eq!(edits(&up), pinned(5), "the same thread");
    assert_eq!(ctx_lines(dir.path()).last().unwrap()["breakers"], json!([]));
}

#[tokio::test]
async fn a_pause_just_past_the_ttl_is_neither_warm_nor_cold() {
    let (clock, dir, api) =
        (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap(), Arc::new(Mutex::new(Api::default())));
    let (mock, url, _proxy) = cm_proxy(&clock, dir.path(), &api).await;
    let at = advance(&url, &mock, &clock).await;
    cm_send(&url, &mock, &clock, at + 4, MESSAGES, &cm_conversation(6, 5)).await;
    // 303 s later the cache may have expired: a miss is no regression, and no advance either.
    api.lock().unwrap().cache_read = 0;
    let up = cm_send(&url, &mock, &clock, at + 4 + 303, MESSAGES, &cm_conversation(6, 6)).await;
    assert_eq!(edits(&up), pinned(6));
    let line = ctx_lines(dir.path()).pop().unwrap();
    assert_eq!((line["cold"].clone(), line["breakers"].clone()), (json!(false), json!([])), "{line}");
    // Well within the TTL, the same miss is one.
    cm_send(&url, &mock, &clock, at + 4 + 303 + 290, MESSAGES, &cm_conversation(6, 7)).await;
    assert_eq!(ctx_lines(dir.path()).last().unwrap()["breakers"], json!(["cache_regressed"]));
}

#[tokio::test]
async fn a_refill_is_counted_as_a_cost() {
    let (clock, dir, api) =
        (Arc::new(FakeClock::default()), tempfile::tempdir().unwrap(), Arc::new(Mutex::new(Api::default())));
    let (mock, url, proxy) = cm_proxy(&clock, dir.path(), &api).await;
    // A Read, five other calls, then Claude Code's stand-in for an unchanged file.
    let read = |id: &str| {
        json!({ "role": "assistant", "content": [
        { "type": "tool_use", "id": id, "name": "Read", "input": { "file_path": "/w/big.rs" } }] })
    };
    let result = |id: &str, text: String| {
        json!({ "role": "user", "content": [
        { "type": "tool_result", "tool_use_id": id, "content": text }] })
    };
    let stand_in = "File unchanged since last read. Refer to that earlier tool_result instead.";
    let mut m =
        vec![json!({ "role": "user", "content": "look" }), read("toolu_r1"), result("toolu_r1", cat_file("big"))];
    m.extend((0..5).flat_map(|i| cat_loop(&format!("toolu_a{i:02}"))));
    m.extend([read("toolu_r2"), result("toolu_r2", stand_in.into())]);
    let mut body = cm_conversation(0, 0);
    body["messages"] = json!(m);

    let up = cm_send(&url, &mock, &clock, T0, MESSAGES, &body).await.json();
    assert_eq!(tool_result_text(&up, "toolu_r2").unwrap(), cat_file("big"), "refilled");
    let added = (cat_file("big").len() - stand_in.len()) as f64 / 4.0;
    let rec = last_record(dir.path());
    assert_eq!(rec["ctx"]["new_refilled"], 1);
    assert!((rec["saved_input_eq"].as_f64().unwrap() + 1.25 * added).abs() < 1e-6, "written once: {rec}");
    cm_send(&url, &mock, &clock, T0 + 10, MESSAGES, &body).await;
    let rec = last_record(dir.path());
    assert!((rec["saved_input_eq"].as_f64().unwrap() + 0.1 * added).abs() < 1e-6, "then read: {rec}");
    assert!(proxy.totals().saved_input_eq < 0.0);
}
