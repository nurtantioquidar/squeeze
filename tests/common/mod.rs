//! A scripted stand-in for the Anthropic Messages API, for tests. It records
//! every request it receives and replies with canned SSE streams or JSON.
#![allow(dead_code)]

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::Response;
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct Recorded {
    pub method: String,
    pub path: String,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

impl Recorded {
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
}

pub enum Reply {
    Sse(String),
    Json(u16, Value),
}

pub type Script = Arc<dyn Fn(&Recorded) -> Reply + Send + Sync>;

#[derive(Clone)]
struct MockState {
    log: Arc<Mutex<Vec<Recorded>>>,
    script: Script,
}

pub struct Mock {
    pub addr: SocketAddr,
    pub log: Arc<Mutex<Vec<Recorded>>>,
}

impl Mock {
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.log.lock().unwrap().clone()
    }
}

async fn handle(State(st): State<MockState>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let body = to_bytes(body, usize::MAX).await.unwrap().to_vec();
    let rec = Recorded {
        method: parts.method.to_string(),
        path: parts.uri.path().to_string(),
        headers: parts.headers.clone(),
        body,
    };
    st.log.lock().unwrap().push(rec.clone());
    match (st.script)(&rec) {
        Reply::Sse(s) => Response::builder()
            .status(200)
            .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
            .body(Body::from(s))
            .unwrap(),
        Reply::Json(code, v) => Response::builder()
            .status(StatusCode::from_u16(code).unwrap())
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(v.to_string()))
            .unwrap(),
    }
}

pub async fn start(script: Script) -> Mock {
    start_on("127.0.0.1:0", script).await
}

/// Bind on a specific address, e.g. `0.0.0.0:0` so containers can reach the mock.
pub async fn start_on(addr: &str, script: Script) -> Mock {
    let log = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new().fallback(handle).with_state(MockState { log: log.clone(), script });
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Mock { addr, log }
}

fn event(name: &str, data: Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

fn message_start(n: usize) -> String {
    event(
        "message_start",
        json!({ "type": "message_start", "message": {
            "id": format!("msg_mock_{n}"), "type": "message", "role": "assistant", "model": "mock-model",
            "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": { "input_tokens": 100, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 1 }
        }}),
    )
}

fn message_end(stop: &str) -> String {
    event(
        "message_delta",
        json!({ "type": "message_delta", "delta": { "stop_reason": stop, "stop_sequence": null }, "usage": { "output_tokens": 20 } }),
    ) + &event("message_stop", json!({ "type": "message_stop" }))
}

pub fn sse_text(n: usize, text: &str) -> String {
    message_start(n)
        + &event(
            "content_block_start",
            json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        )
        + &event(
            "content_block_delta",
            json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } }),
        )
        + &event("content_block_stop", json!({ "type": "content_block_stop", "index": 0 }))
        + &message_end("end_turn")
}

pub fn sse_tool_use(n: usize, id: &str, name: &str, input: Value) -> String {
    message_start(n)
        + &event(
            "content_block_start",
            json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "tool_use", "id": id, "name": name, "input": {} } }),
        )
        + &event(
            "content_block_delta",
            json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": input.to_string() } }),
        )
        + &event("content_block_stop", json!({ "type": "content_block_stop", "index": 0 }))
        + &message_end("tool_use")
}

/// Text of the tool_result for `tool_use_id`, if present in a request body.
pub fn tool_result_text(body: &Value, tool_use_id: &str) -> Option<String> {
    for m in body["messages"].as_array()? {
        for b in m["content"].as_array().into_iter().flatten() {
            if b["type"] == "tool_result" && b["tool_use_id"] == tool_use_id {
                return Some(match &b["content"] {
                    Value::String(s) => s.clone(),
                    Value::Array(items) => {
                        items.iter().filter_map(|i| i["text"].as_str()).collect::<Vec<_>>().join("\n")
                    }
                    _ => String::new(),
                });
            }
        }
    }
    None
}
