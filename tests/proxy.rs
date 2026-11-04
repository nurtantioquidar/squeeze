//! Proxy integration tests against a scripted upstream (no network, no API key).

mod common;

use common::{Reply, sse_text, tool_result_text};
use serde_json::{Value, json};
use squeeze::proxy::{Proxy, ProxyConfig};
use squeeze::store::Store;
use squeeze::transform::TransformConfig;
use std::sync::Arc;

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
    let proxy =
        Proxy::new(ProxyConfig { upstream: mock.url(), transform: transform.then(TransformConfig::default), store })
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
