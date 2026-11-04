//! The local HTTP proxy Claude Code reaches through `ANTHROPIC_BASE_URL`.
//!
//! Follows Claude Code's LLM gateway protocol
//! (<https://code.claude.com/docs/en/llm-gateway-protocol>):
//! - `anthropic-*` headers and all body fields are forwarded unchanged; only
//!   `tool_result` text inside `messages` is rewritten (see [`crate::transform`]);
//! - responses, including error bodies, stream through unmodified and unbuffered;
//! - every other path is a transparent pass-through.
//!
//! Requests carrying Claude subscription (OAuth) credentials are refused, not
//! forwarded (see [`crate::auth`]).

use crate::auth;
use crate::sse::{Usage, UsageTap};
use crate::store::{Store, ThreadState};
use crate::transform::{self, ChangeKind, TransformConfig, TransformStats};
use crate::util::{approx_tokens, now_secs, short_hash};
use axum::Router;
use axum::body::{Body, Bytes, to_bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, header};
use axum::response::Response;
use futures_util::Stream;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::net::TcpListener;

pub const DEFAULT_UPSTREAM: &str = "https://api.anthropic.com";
const MAX_BODY: usize = 512 * 1024 * 1024;
/// Extra idle time beyond the cache TTL before the cache is treated as cold.
const COLD_MARGIN_SECS: u64 = 60;

#[derive(Clone)]
pub struct ProxyConfig {
    pub upstream: String,
    /// `None` forwards every request byte-for-byte (usage is still logged).
    pub transform: Option<TransformConfig>,
    pub store: Store,
}

#[derive(Clone)]
pub struct Proxy {
    inner: Arc<Inner>,
}

struct Inner {
    client: reqwest::Client,
    upstream: String,
    transform: Option<TransformConfig>,
    store: Store,
    threads: Mutex<HashMap<String, ThreadState>>,
    /// tool_use ids whose rewrite has already been billed once (as a cache write).
    counted: Mutex<HashSet<String>>,
    totals: Mutex<Totals>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Totals {
    pub requests: u64,
    pub refused: u64,
    pub compressed: u64,
    pub deduped: u64,
    pub pruned: u64,
    pub bytes_saved: u64,
    /// Estimated input-side savings, in base-input-token equivalents.
    pub saved_input_eq: f64,
    /// Actual input-side cost reported by the API, in the same units.
    pub actual_input_eq: f64,
    pub usage: Usage,
}

impl Totals {
    pub fn rewritten(&self) -> u64 {
        self.compressed + self.deduped + self.pruned
    }

    /// Estimated share of input-side cost avoided.
    pub fn savings_pct(&self) -> Option<f64> {
        let denom = self.saved_input_eq + self.actual_input_eq;
        (denom > 0.0).then(|| 100.0 * self.saved_input_eq / denom)
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Proxy {
    pub fn new(cfg: ProxyConfig) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .tcp_keepalive(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self {
            inner: Arc::new(Inner {
                client,
                upstream: cfg.upstream.trim_end_matches('/').to_string(),
                transform: cfg.transform,
                store: cfg.store,
                threads: Mutex::default(),
                counted: Mutex::default(),
                totals: Mutex::default(),
            }),
        })
    }

    pub fn router(&self) -> Router {
        Router::new().fallback(handle).with_state(self.inner.clone())
    }

    /// Bind and serve in the background; returns the bound address.
    pub async fn spawn(&self, addr: &str) -> anyhow::Result<SocketAddr> {
        let listener = TcpListener::bind(addr).await?;
        let local = listener.local_addr()?;
        let app = self.router();
        tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                tracing::error!("proxy stopped: {e}");
            }
        });
        Ok(local)
    }

    pub fn totals(&self) -> Totals {
        lock(&self.inner.totals).clone()
    }
}

async fn handle(State(inner): State<Arc<Inner>>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    if let Err(refusal) = auth::check_request(&parts.headers) {
        lock(&inner.totals).refused += 1;
        tracing::warn!("refused a request carrying Claude subscription (OAuth) credentials");
        return error_response(StatusCode::FORBIDDEN, "permission_error", &auth::refusal_message(refusal));
    }
    let bytes = match to_bytes(body, MAX_BODY).await {
        Ok(b) => b,
        Err(e) => {
            return error_response(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request_too_large",
                &format!("squeeze: could not read the request body: {e}"),
            );
        }
    };
    let path = parts.uri.path();
    let path_and_query = parts.uri.path_and_query().map_or(path, |p| p.as_str()).to_string();
    let is_messages = parts.method == Method::POST && path == "/v1/messages";
    let is_count = parts.method == Method::POST && path == "/v1/messages/count_tokens";

    let mut out = bytes.clone();
    let mut pending = None;
    if (is_messages || is_count) && !parts.headers.contains_key(header::CONTENT_ENCODING) {
        let (rewritten, p) = prepare(&inner, &parts.headers, &bytes, is_messages);
        if let Some(b) = rewritten {
            out = b;
        }
        pending = p;
    }

    let url = format!("{}{}", inner.upstream, path_and_query);
    let upstream = inner
        .client
        .request(parts.method.clone(), &url)
        .headers(forward_headers(&parts.headers))
        .body(out)
        .send()
        .await;
    let resp = match upstream {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("upstream request failed: {e}");
            return error_response(
                StatusCode::BAD_GATEWAY,
                "api_error",
                &format!("squeeze: upstream request failed: {e}"),
            );
        }
    };

    let status = resp.status();
    let is_sse = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"));
    let mut builder = Response::builder().status(status);
    for (k, v) in resp.headers() {
        if !is_hop_by_hop(k) {
            builder = builder.header(k, v);
        }
    }
    let stream = resp.bytes_stream();
    let body = match pending {
        Some(p) => Body::from_stream(TapStream {
            inner: Box::pin(stream),
            tap: Some(UsageTap::new(is_sse)),
            done: Some(Box::new(move |usage| p.finish(usage, status.as_u16()))),
        }),
        None => Body::from_stream(stream),
    };
    builder.body(body).unwrap_or_else(|e| {
        error_response(StatusCode::INTERNAL_SERVER_ERROR, "api_error", &format!("squeeze: bad upstream response: {e}"))
    })
}

struct Pending {
    inner: Arc<Inner>,
    session: Option<String>,
    model: Option<String>,
    stats: TransformStats,
    ttl_1h: bool,
}

/// Parse and rewrite a messages request. Fails soft: any problem means the
/// original bytes are forwarded.
fn prepare(
    inner: &Arc<Inner>,
    headers: &HeaderMap,
    bytes: &Bytes,
    is_messages: bool,
) -> (Option<Bytes>, Option<Pending>) {
    let Ok(mut body) = serde_json::from_slice::<Value>(bytes) else {
        return (None, None);
    };
    let session = header_str(headers, "x-claude-code-session-id");
    let agent = header_str(headers, "x-claude-code-agent-id");
    let model = body.get("model").and_then(Value::as_str).map(str::to_string);
    let ttl = transform::max_cache_ttl_secs(&body);

    let mut stats = TransformStats::default();
    let mut rewritten = None;
    if let Some(cfg) = &inner.transform {
        let boundary = thread_boundary(
            inner,
            cfg,
            &body,
            session.as_deref(),
            agent.as_deref(),
            model.as_deref(),
            ttl,
            is_messages,
        );
        let mut sink = inner.store.clone();
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            transform::transform(&mut body, cfg, &mut sink, boundary.as_deref())
        }));
        match result {
            Ok(s) if s.changed() => match serde_json::to_vec(&body) {
                Ok(v) => {
                    rewritten = Some(Bytes::from(v));
                    stats = s;
                }
                Err(e) => tracing::warn!("could not re-serialize the request, forwarding it unchanged: {e}"),
            },
            Ok(s) => stats = s,
            Err(_) => tracing::error!("transform panicked; forwarding the request unchanged"),
        }
    }
    let pending = is_messages.then(|| Pending { inner: inner.clone(), session, model, stats, ttl_1h: ttl >= 3600 });
    (rewritten, pending)
}

/// Track idle time per conversation thread and move the prune boundary only
/// when the prompt cache must already have expired, so pruning never costs a
/// cache rewrite that wasn't going to happen anyway.
#[allow(clippy::too_many_arguments)]
fn thread_boundary(
    inner: &Inner,
    cfg: &TransformConfig,
    body: &Value,
    session: Option<&str>,
    agent: Option<&str>,
    model: Option<&str>,
    ttl: u64,
    is_messages: bool,
) -> Option<String> {
    if !cfg.prune.enabled {
        return None;
    }
    let session = session?;
    let first = body.get("messages").and_then(|m| m.get(0)).map(Value::to_string).unwrap_or_default();
    let key = short_hash(&[session, agent.unwrap_or("main"), model.unwrap_or(""), &first]);
    let mut threads = lock(&inner.threads);
    let state = threads.entry(key.clone()).or_insert_with(|| inner.store.load_thread(&key).unwrap_or_default());
    if is_messages {
        let now = now_secs();
        if let Some(last) = state.last_seen
            && now.saturating_sub(last) > ttl + COLD_MARGIN_SECS
            && let Some(id) = transform::last_tool_result_id(body)
        {
            tracing::info!("prompt cache is cold for thread {key}; moving prune boundary to {id}");
            state.boundary = Some(id);
        }
        state.last_seen = Some(now);
        inner.store.save_thread(&key, state);
    }
    state.boundary.clone()
}

impl Pending {
    fn finish(self, usage: Option<Usage>, status: u16) {
        let ok = (200..300).contains(&status);
        let write_mult = if self.ttl_1h { 2.0 } else { 1.25 };
        let (mut saved_eq, mut new_bytes) = (0.0, 0u64);
        let mut new_counts = (0u64, 0u64, 0u64);
        if ok {
            let mut counted = lock(&self.inner.counted);
            for c in &self.stats.changes {
                let tokens = approx_tokens(c.saved_bytes);
                // First appearance: the rewritten block would have been a cache
                // write. Afterwards it would have been a cache read each turn.
                if counted.insert(c.tool_use_id.clone()) {
                    saved_eq += tokens * write_mult;
                    new_bytes += c.saved_bytes as u64;
                    match c.kind {
                        ChangeKind::Compressed => new_counts.0 += 1,
                        ChangeKind::Deduped => new_counts.1 += 1,
                        ChangeKind::Pruned => new_counts.2 += 1,
                    }
                } else {
                    saved_eq += tokens * 0.1;
                }
            }
        }
        let actual_eq = usage.as_ref().map_or(0.0, Usage::input_equivalent);
        {
            let mut t = lock(&self.inner.totals);
            t.requests += 1;
            if ok {
                t.compressed += new_counts.0;
                t.deduped += new_counts.1;
                t.pruned += new_counts.2;
                t.bytes_saved += new_bytes;
                t.saved_input_eq += saved_eq;
                t.actual_input_eq += actual_eq;
                if let Some(u) = &usage {
                    t.usage.add(u);
                }
            }
        }
        let rec = json!({
            "ts": now_secs(),
            "session": self.session,
            "model": usage.as_ref().and_then(|u| u.model.clone()).or(self.model),
            "status": status,
            "tool_results": self.stats.tool_results,
            "rewritten": self.stats.changes.len(),
            "new_compressed": new_counts.0,
            "new_deduped": new_counts.1,
            "new_pruned": new_counts.2,
            "new_bytes_saved": new_bytes,
            "saved_input_eq": saved_eq,
            "actual_input_eq": actual_eq,
            "usage": usage,
        });
        if let Err(e) = self.inner.store.append_record(&rec) {
            tracing::warn!("could not write stats: {e}");
        }
    }
}

/// Passes chunks through untouched while feeding them to a [`UsageTap`].
struct TapStream {
    inner: Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>,
    tap: Option<UsageTap>,
    done: Option<Box<dyn FnOnce(Option<Usage>) + Send>>,
}

impl TapStream {
    fn complete(&mut self) {
        if let Some(done) = self.done.take() {
            done(self.tap.take().and_then(UsageTap::finish));
        }
    }
}

impl Stream for TapStream {
    type Item = reqwest::Result<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                if let Some(tap) = this.tap.as_mut() {
                    tap.feed(&chunk);
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(None) => {
                this.complete();
                Poll::Ready(None)
            }
            other => other,
        }
    }
}

impl Drop for TapStream {
    fn drop(&mut self) {
        self.complete();
    }
}

fn header_str(h: &HeaderMap, name: &str) -> Option<String> {
    h.get(name).and_then(|v| v.to_str().ok()).map(str::to_string)
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-connection"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

fn forward_headers(src: &HeaderMap) -> HeaderMap {
    let mut h = HeaderMap::with_capacity(src.len());
    for (k, v) in src {
        if is_hop_by_hop(k) || k == header::HOST || k == header::CONTENT_LENGTH || k == header::ACCEPT_ENCODING {
            continue;
        }
        h.append(k.clone(), v.clone());
    }
    // Uncompressed responses let usage be read from the stream as it passes.
    h.insert(header::ACCEPT_ENCODING, HeaderValue::from_static("identity"));
    h
}

fn error_response(status: StatusCode, kind: &str, message: &str) -> Response {
    let body = json!({ "type": "error", "error": { "type": kind, "message": message } }).to_string();
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("static response is valid")
}
