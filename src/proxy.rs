//! The local HTTP proxy Claude Code reaches through `ANTHROPIC_BASE_URL`.
//!
//! Follows Claude Code's LLM gateway protocol
//! (<https://code.claude.com/docs/en/llm-gateway-protocol>):
//! - `anthropic-*` headers and all body fields are forwarded unchanged; only
//!   `tool_result` text inside `messages` is rewritten (see [`crate::transform`]),
//!   and with `--context-edits`, edits are merged into `context_management`
//!   (see [`crate::ctxmgmt`]);
//! - responses, including error bodies, stream through unmodified and unbuffered;
//! - every other path is a transparent pass-through.
//!
//! Requests carrying Claude subscription (OAuth) credentials are refused, not
//! forwarded (see [`crate::auth`]).

use crate::auth;
use crate::compact::{self, CompactConfig, Credentials};
use crate::ctxmgmt::{self, CtxConfig, Expected};
use crate::sse::{Usage, UsageTap};
use crate::store::{CtxState, Store, ThreadState};
use crate::transform::{self, ChangeKind, TransformConfig, TransformStats};
use crate::util::{approx_tokens, now_secs, read_price, short_hash};
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
/// How far a thread's time since its previous request must be from the cache
/// TTL to be sure of the cache: this much beyond it is cold, this much within
/// it is warm. Room for clock and queueing skew, small enough that a pause
/// just past the TTL counts as cold.
pub(crate) const COLD_MARGIN_SECS: u64 = 5;

/// Wall-clock time in unix seconds; tests substitute their own.
pub trait Clock: Send + Sync {
    fn now_secs(&self) -> u64;
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now_secs(&self) -> u64 {
        now_secs()
    }
}

#[derive(Clone)]
pub struct ProxyConfig {
    pub upstream: String,
    /// `None` forwards every request byte-for-byte (usage is still logged).
    pub transform: Option<TransformConfig>,
    pub store: Store,
    /// Server-side context editing (`--context-edits cold`); `None` leaves
    /// `context_management` as the client sent it.
    pub ctx: Option<CtxConfig>,
    /// Compaction at pauses (`--compact-pauses`, API keys only).
    pub compact: Option<CompactConfig>,
}

#[derive(Clone)]
pub struct Proxy {
    inner: Arc<Inner>,
}

struct Inner {
    client: reqwest::Client,
    upstream: String,
    transform: Option<TransformConfig>,
    ctx: Option<CtxConfig>,
    compact: Option<CompactConfig>,
    store: Store,
    clock: Arc<dyn Clock>,
    threads: Mutex<HashMap<String, ThreadState>>,
    /// Messages requests seen per thread, so a scheduled compaction knows
    /// whether another request came after the one that scheduled it.
    generations: Mutex<HashMap<String, u64>>,
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
    /// Unchanged-file Read results given their content back (`--context-edits`).
    pub refilled: u64,
    /// Cold moments at which server-side context editing cleared more.
    pub ctx_advances: u64,
    /// Input tokens server-side context editing cleared, summed over requests.
    pub cleared_input_tokens: u64,
    /// Summarizing requests the proxy made itself (`--compact-pauses`), and
    /// the summaries they produced.
    pub compaction_requests: u64,
    pub compactions: u64,
    /// Keep-warm requests the proxy made itself (`--keep-warm`).
    pub keep_warm_requests: u64,
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
        Self::with_clock(cfg, Arc::new(SystemClock))
    }

    pub fn with_clock(cfg: ProxyConfig, clock: Arc<dyn Clock>) -> anyhow::Result<Self> {
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
                ctx: cfg.ctx,
                compact: cfg.compact,
                store: cfg.store,
                clock,
                threads: Mutex::default(),
                generations: Mutex::default(),
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
    // The body asks for compaction's thinking-binding control, which needs its beta.
    let mut binding = false;
    if (is_messages || is_count) && !parts.headers.contains_key(header::CONTENT_ENCODING) {
        let (rewritten, p, bind) = prepare(&inner, &parts.headers, &bytes, is_messages);
        if let Some(b) = rewritten {
            out = b;
        }
        pending = p;
        binding = bind;
    }

    if is_messages && let Some(dir) = std::env::var_os("SQUEEZE_DUMP_REQUESTS") {
        // Debugging aid only: the request bodies as forwarded (conversation
        // content; never headers, so never credentials).
        let n = lock(&inner.totals).requests;
        let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{}-{n}.json", now_secs())), &out);
    }
    if let Some(job) = pending.as_mut().and_then(|p| p.compact.as_mut()) {
        job.forwarded = Some(out.clone());
        job.credentials = Credentials::from_headers(&parts.headers);
        job.bound = binding;
    }
    let mut headers = forward_headers(&parts.headers);
    if inner.compact.is_some_and(|c| c.warm) {
        without_claude_code_beta(&mut headers);
    }
    let url = format!("{}{}", inner.upstream, path_and_query);
    let upstream = inner
        .client
        .request(parts.method.clone(), &url)
        .headers(with_binding_beta(headers, binding))
        .body(out)
        .send()
        .await;
    let resp = match upstream {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("upstream request failed: {e}");
            // Undo the start this request stamped, as for any failed request.
            if let Some(p) = pending {
                p.finish(None, StatusCode::BAD_GATEWAY.as_u16());
            }
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
    /// The conversation thread, when the proxy tracks threads.
    thread: Option<String>,
    /// The start this request stamped on its thread, and the one it replaced.
    stamp: Option<(u64, Option<u64>)>,
    ctx: Option<CtxRequest>,
    /// What a compaction at the next pause needs (`--compact-pauses`).
    compact: Option<CompactJob>,
}

/// A request that may end a turn, and what summarizing its conversation needs.
struct CompactJob {
    /// The thread's generation this request made.
    generation: u64,
    /// The client's messages and their hash.
    covered: usize,
    prefix_hash: String,
    /// The body as forwarded.
    forwarded: Option<Bytes>,
    credentials: Option<Credentials>,
    /// It went out with [`compact::THINKING_BINDING_BETA`] added.
    bound: bool,
}

/// What context editing did to a request.
struct CtxRequest {
    /// What the server should report clearing, when the proxy added edits.
    expected: Option<Expected>,
    advance: bool,
    cold: bool,
    /// Surely warm and the cleared set unmoved: the response should read the
    /// previous prompt from cache (see `ctxmgmt::settle`).
    steady: bool,
    idle_secs: Option<u64>,
    /// The keep window an advance would use now.
    k: usize,
}

/// Parse and rewrite a messages request. Fails soft: any problem means the
/// original bytes are forwarded.
fn prepare(
    inner: &Arc<Inner>,
    headers: &HeaderMap,
    bytes: &Bytes,
    is_messages: bool,
) -> (Option<Bytes>, Option<Pending>, bool) {
    let Ok(mut body) = serde_json::from_slice::<Value>(bytes) else {
        return (None, None, false);
    };
    let session = header_str(headers, "x-claude-code-session-id");
    let model = body.get("model").and_then(Value::as_str).map(str::to_string);
    let ttl_1h = transform::max_cache_ttl_secs(&body) >= 3600;

    let mut stats = TransformStats::default();
    let mut rewritten = None;
    let (mut thread, mut stamp, mut ctx, mut job, mut binding) = (None, None, None, None, false);
    if inner.transform.is_some() || inner.ctx.is_some() || inner.compact.is_some() {
        let tracked =
            inner.ctx.is_some() || inner.compact.is_some() || inner.transform.as_ref().is_some_and(|c| c.prune.enabled);
        let mut sink = inner.store.clone();
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let tick = tracked.then(|| thread_tick(inner, &body, headers, is_messages)).flatten();
            // The summary goes in first: everything after works on what's sent.
            let (compacted, job, bound) = match (&inner.compact, &tick) {
                (Some(_), Some(t)) => compact_step(inner, &mut body, t, is_messages),
                _ => (false, None, false),
            };
            let boundary = tick.as_ref().and_then(|t| t.boundary.clone());
            let s = inner
                .transform
                .as_ref()
                .map(|cfg| transform::transform(&mut body, cfg, &mut sink, boundary.as_deref()))
                .unwrap_or_default();
            let c = match (&inner.ctx, &tick) {
                (Some(cfg), Some(t)) => Some(ctx_step(inner, cfg, &mut body, headers, t, is_messages)),
                _ => None,
            };
            (tick, s, c, compacted || bound, job, bound)
        }));
        match result {
            Ok((tick, s, c, compacted, j, bound)) => {
                job = j;
                binding = bound;
                if let Some(t) = tick {
                    if t.cold {
                        tracing::info!(
                            "prompt cache is cold for thread {} ({} s idle){}",
                            t.key,
                            t.idle_secs.map_or("?".into(), |s| s.to_string()),
                            t.boundary.as_ref().map_or(String::new(), |b| format!("; prune boundary at {b}"))
                        );
                    }
                    stamp = t.started.map(|start| (start, t.prev_start));
                    thread = is_messages.then_some(t.key);
                }
                let edited = c.as_ref().is_some_and(|c| c.expected.is_some()) || compacted;
                if s.changed() || edited {
                    match serde_json::to_vec(&body) {
                        Ok(v) => {
                            rewritten = Some(Bytes::from(v));
                            stats = s;
                            ctx = c;
                        }
                        Err(e) => {
                            tracing::warn!("could not re-serialize the request, forwarding it unchanged: {e}");
                            ctx = c.map(|c| CtxRequest { expected: None, ..c });
                            (job, binding) = (None, false);
                        }
                    }
                } else {
                    stats = s;
                    ctx = c;
                }
            }
            Err(_) => tracing::error!("transform panicked; forwarding the request unchanged"),
        }
    }
    let pending = is_messages.then(|| Pending {
        inner: inner.clone(),
        session,
        model,
        stats,
        ttl_1h,
        thread,
        stamp,
        ctx,
        compact: job,
    });
    (rewritten, pending, binding)
}

/// Compaction for one request: put the thread's summary in, and note what
/// summarizing this request's conversation would need, should it end a turn.
fn compact_step(inner: &Inner, body: &mut Value, tick: &Tick, is_messages: bool) -> (bool, Option<CompactJob>, bool) {
    let job = match (is_messages, tick.generation, body.get("messages").and_then(Value::as_array)) {
        (true, Some(generation), Some(m)) => Some(CompactJob {
            generation,
            covered: m.len(),
            prefix_hash: compact::messages_hash(m),
            forwarded: None,
            credentials: None,
            bound: false,
        }),
        _ => None,
    };
    let mut threads = lock(&inner.threads);
    let Some(state) = threads.get_mut(&tick.key) else { return (false, job, false) };
    let before = state.compact.clone();
    let changed = compact::apply(body, &mut state.compact, is_messages);
    let bound = compact::guard(body, &state.compact);
    if state.compact != before {
        inner.store.save_thread(&tick.key, state);
    }
    (changed, job, bound)
}

/// Context editing for one request: census its history, advance the thread's
/// cleared set at a cold moment (saved before the request goes out), and
/// merge the pinned edits into the body. A count_tokens request previews the
/// same edits but never advances.
fn ctx_step(
    inner: &Inner,
    cfg: &CtxConfig,
    body: &mut Value,
    headers: &HeaderMap,
    tick: &Tick,
    is_messages: bool,
) -> CtxRequest {
    let census = ctxmgmt::census(body, &header_str(headers, "anthropic-beta").unwrap_or_default(), cfg.clear_inputs);
    let (state, advance) = {
        let mut threads = lock(&inner.threads);
        if is_messages && let Some(thread) = threads.get_mut(&tick.key) {
            let mut ctx = thread.ctx.clone().unwrap_or_default();
            let advance = ctxmgmt::step(&mut ctx, &census, cfg, tick.cold);
            let ctx = (ctx != CtxState::default()).then_some(ctx);
            if ctx != thread.ctx {
                thread.ctx = ctx;
                inner.store.save_thread(&tick.key, thread);
            }
            (thread.ctx.clone().unwrap_or_default(), advance)
        } else {
            let saved = match threads.get(&tick.key) {
                Some(t) => t.ctx.clone(),
                None => inner.store.load_thread(&tick.key).and_then(|t| t.ctx),
            };
            (saved.unwrap_or_default(), None)
        }
    };
    if let Some(a) = &advance {
        tracing::info!(
            "context editing for thread {}: clearing {} tool uses and {} thinking turns (~{:.0} tokens more)",
            tick.key,
            a.cleared,
            a.cleared_th,
            a.new_tokens
        );
    }
    CtxRequest {
        expected: ctxmgmt::apply(body, &census, &state),
        steady: tick.warm && advance.is_none(),
        advance: advance.is_some(),
        cold: tick.cold,
        idle_secs: tick.idle_secs,
        k: cfg.window.k(census.uses_since_prev_human),
    }
}

/// A request's place in its conversation thread.
struct Tick {
    key: String,
    /// The prompt cache must have expired since the thread's previous request.
    cold: bool,
    /// The prompt cache must still be live. Within the margin either side of
    /// the TTL a request is neither: the cache may or may not have expired.
    warm: bool,
    /// Seconds since the thread's previous response finished.
    idle_secs: Option<u64>,
    /// Prune boundary, moved to the newest tool result at a cold moment.
    boundary: Option<String>,
    /// The start this request stamped (messages requests only), and the one before.
    started: Option<u64>,
    prev_start: Option<u64>,
    /// The thread's generation this request made (messages requests only).
    generation: Option<u64>,
}

/// Stamp a messages request's start in its thread and tell whether the prompt
/// cache has expired since the thread's previous request: the moment stale
/// context can change without costing a cache rewrite that wasn't going to
/// happen anyway. The cache's TTL runs from a request's start, so a long
/// request followed by a short pause can already be cold. count_tokens only
/// previews a request: same view, nothing recorded.
fn thread_tick(inner: &Inner, body: &Value, headers: &HeaderMap, is_messages: bool) -> Option<Tick> {
    let anchor = transform::first_tool_use_id(body);
    let key = thread_key(body, headers, anchor.as_deref())?;
    let mut threads = lock(&inner.threads);
    if !is_messages {
        let boundary = match threads.get(&key) {
            Some(state) => state.boundary.clone(),
            None => inner.store.load_thread(&key).and_then(|s| s.boundary),
        };
        return Some(Tick {
            key,
            cold: false,
            warm: false,
            idle_secs: None,
            boundary,
            started: None,
            prev_start: None,
            generation: None,
        });
    }
    let state = threads.entry(key.clone()).or_insert_with(|| open_thread(&inner.store, &key, anchor.as_deref()));
    let now = inner.clock.now_secs();
    let ttl = transform::message_cache_ttl_secs(body);
    let since_start = state.last_start.map(|start| now.saturating_sub(start));
    let cold = since_start.is_some_and(|t| t > ttl + COLD_MARGIN_SECS);
    let warm = since_start.is_some_and(|t| t + COLD_MARGIN_SECS <= ttl);
    let idle_secs = state.last_stop.map(|stop| now.saturating_sub(stop));
    if cold
        && inner.transform.as_ref().is_some_and(|c| c.prune.enabled)
        && let Some(id) = transform::last_tool_result_id(body)
    {
        state.boundary = Some(id);
    }
    let prev_start = state.last_start.replace(now);
    inner.store.save_thread(&key, state);
    let boundary = state.boundary.clone();
    drop(threads);
    let generation = {
        let mut generations = lock(&inner.generations);
        // The conversation's first tool call renamed its thread (see
        // `thread_key`): what the request before it left waiting (a keep-warm)
        // waits under the old name, and ends here as it would on that thread.
        if anchor.is_some()
            && !generations.contains_key(&key)
            && let Some(before) = thread_key(body, headers, None)
            && let Some(g) = generations.get_mut(&before)
        {
            *g += 1;
        }
        let g = generations.entry(key.clone()).or_default();
        *g += 1;
        *g
    };
    Some(Tick { key, cold, warm, idle_secs, boundary, started: Some(now), prev_start, generation: Some(generation) })
}

/// Which thread a request belongs to. Once the conversation has a tool call,
/// its first tool_use id (with the agent) names the thread: unlike the session
/// id, the model or the first message, it survives `--resume`, a model switch
/// and Claude Code moving its cache marker off the first message. Until then,
/// the session and its first message (without cache markers) do.
fn thread_key(body: &Value, headers: &HeaderMap, anchor: Option<&str>) -> Option<String> {
    let agent = header_str(headers, "x-claude-code-agent-id");
    let agent = agent.as_deref().unwrap_or("main");
    if let Some(anchor) = anchor {
        return Some(short_hash(&[agent, anchor]));
    }
    let session = header_str(headers, "x-claude-code-session-id")?;
    let model = body.get("model").and_then(Value::as_str).unwrap_or("");
    let first = body.get("messages").and_then(|m| m.get(0)).map(unmarked).unwrap_or_default();
    Some(short_hash(&[&session, agent, model, &first]))
}

/// A message as text without its cache markers, which move along the
/// conversation as it grows.
fn unmarked(message: &Value) -> String {
    let mut m = message.clone();
    if let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut) {
        for block in blocks.iter_mut().filter_map(Value::as_object_mut) {
            block.remove("cache_control");
        }
    }
    m.to_string()
}

/// A thread's saved state. A thread that's new under its key but whose first
/// tool call another thread already made (a history continued under another
/// agent id, such as a subagent given its parent's context) starts from that
/// thread's state, which describes the prefix the two share.
fn open_thread(store: &Store, key: &str, anchor: Option<&str>) -> ThreadState {
    if let Some(state) = store.load_thread(key) {
        return state;
    }
    let Some(anchor) = anchor else { return ThreadState::default() };
    match store.thread_by_anchor(anchor) {
        Some(owner) if owner != key => store.load_thread(&owner).unwrap_or_default(),
        Some(_) => ThreadState::default(),
        None => {
            store.link_anchor(anchor, key);
            ThreadState::default()
        }
    }
}

impl Pending {
    fn finish(self, usage: Option<Usage>, status: u16) {
        let ok = (200..300).contains(&status);
        let mut breakers = Vec::new();
        if let Some(key) = &self.thread {
            let mut threads = lock(&self.inner.threads);
            if let Some(state) = threads.get_mut(key) {
                state.last_stop = Some(self.inner.clock.now_secs());
                // A failed request cached nothing: the TTL still runs from the start before it.
                if !ok
                    && let Some((started, prev)) = self.stamp
                    && state.last_start == Some(started)
                {
                    state.last_start = prev;
                }
                if let Some(c) = &self.ctx
                    && let Some(expected) = c.expected
                {
                    let ctx = state.ctx.get_or_insert_default();
                    breakers = ctxmgmt::settle(ctx, expected, c.steady, self.model.as_deref(), usage.as_ref(), status);
                    for b in &breakers {
                        tracing::warn!("context editing for thread {key}: breaker {b} tripped");
                    }
                }
                self.inner.store.save_thread(key, state);
            }
        }
        let model = usage.as_ref().and_then(|u| u.model.clone()).or(self.model);
        let read = read_price(model.as_deref().unwrap_or(""));
        let write_mult = if self.ttl_1h { 2.0 } else { 1.25 };
        let (mut saved_eq, mut new_bytes) = (0.0, 0u64);
        let mut new_counts = (0u64, 0u64, 0u64, 0u64);
        // A prefix rebuilt anyway (the cache was cold, or the cleared set moved)
        // is all cache writes.
        let rebuilt = self.ctx.as_ref().is_some_and(|c| c.cold || c.advance);
        if ok {
            let mut counted = lock(&self.inner.counted);
            for c in &self.stats.changes {
                // A refill sends more than the client did: a cost, not a saving.
                let tokens = approx_tokens(c.saved_bytes) - approx_tokens(c.added_bytes);
                // The rewritten block would have been a cache write the first
                // time and when rebuilt, and a cache read on every other turn.
                let first = counted.insert(c.tool_use_id.clone());
                saved_eq += tokens * if first || rebuilt { write_mult } else { read };
                if first {
                    new_bytes += c.saved_bytes as u64;
                    match c.kind {
                        ChangeKind::Compressed => new_counts.0 += 1,
                        ChangeKind::Deduped => new_counts.1 += 1,
                        ChangeKind::Pruned => new_counts.2 += 1,
                        ChangeKind::Refilled => new_counts.3 += 1,
                    }
                }
            }
        }
        // What the server cleared would have been a cache write when the
        // prefix was rebuilt anyway, and a cache read otherwise. A cleared
        // refill nets out: its cost above already counts what the server
        // reports clearing beyond the client's stand-in.
        let applied = usage.as_ref().and_then(|u| u.applied_edits.clone());
        let cleared_tokens = match (&self.ctx, &applied) {
            (Some(c), Some(a)) if ok && c.expected.is_some() => {
                saved_eq += a.cleared_input_tokens as f64 * if rebuilt { write_mult } else { read };
                a.cleared_input_tokens
            }
            _ => 0,
        };
        let actual_eq = usage.as_ref().map_or(0.0, |u| u.input_equivalent(read));
        {
            let mut t = lock(&self.inner.totals);
            t.requests += 1;
            if ok {
                t.compressed += new_counts.0;
                t.deduped += new_counts.1;
                t.pruned += new_counts.2;
                t.refilled += new_counts.3;
                t.ctx_advances += u64::from(self.ctx.as_ref().is_some_and(|c| c.advance));
                t.cleared_input_tokens += cleared_tokens;
                t.bytes_saved += new_bytes;
                t.saved_input_eq += saved_eq;
                t.actual_input_eq += actual_eq;
                if let Some(u) = &usage {
                    t.usage.add(u);
                }
            }
        }
        let mut rec = json!({
            "ts": now_secs(),
            "session": self.session,
            "model": model,
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
        if let Some(c) = &self.ctx {
            rec["ctx"] = json!({
                "advance": c.advance,
                "C": c.expected.map_or(0, |e| e.tool_uses),
                "C_th": c.expected.and_then(|e| e.thinking_turns).unwrap_or(0),
                "K": c.k,
                "expected": c.expected,
                "applied": applied,
                "cleared_input_tokens": cleared_tokens,
                "cold": c.cold,
                "idle_secs": c.idle_secs,
                "breakers": breakers,
                "new_refilled": new_counts.3,
            });
        }
        if let Err(e) = self.inner.store.append_record(&rec) {
            tracing::warn!("could not write stats: {e}");
        }
        // A turn that ended: summarize its conversation if the thread then
        // pauses. A tool call to run: with --keep-warm, keep the conversation
        // cached should it run past the TTL.
        let stop = usage.as_ref().and_then(|u| u.stop_reason.as_deref());
        if let (Some(cfg), Some(job), Some(key), Some((started, _))) =
            (self.inner.compact, self.compact, self.thread, self.stamp)
            && ok
            && job.credentials.is_some()
            && job.forwarded.is_some()
        {
            let inner = self.inner.clone();
            match stop {
                Some("end_turn") => {
                    tokio::spawn(compact_later(inner, cfg, key, self.session, model, started, job));
                }
                Some("tool_use") if cfg.keep_warm => {
                    tokio::spawn(keep_warm(inner, cfg, key, self.session, model, started, job));
                }
                _ => {}
            }
        }
    }
}

/// While the tool call a response asked for runs (or the pause after a turn
/// too short to summarize lasts), send the request it answered again, for one
/// output token ([`compact::keep_warm_request`]), shortly before that
/// request's cache would expire: a cache read that keeps the conversation
/// cached for Claude Code's next request, which would otherwise write it all
/// again. Stops at the thread's next request, after a failed request, or after
/// [`compact::KEEP_WARM_MAX_PINGS`], dropping the credentials.
async fn keep_warm(
    inner: Arc<Inner>,
    cfg: CompactConfig,
    key: String,
    session: Option<String>,
    model: Option<String>,
    started: u64,
    job: CompactJob,
) {
    let CompactJob { generation, forwarded, credentials, bound, .. } = job;
    let (Some(forwarded), Some(credentials)) = (forwarded, credentials) else { return };
    let Ok(body) = serde_json::from_slice::<Value>(&forwarded) else { return };
    let Some(payload) = compact::keep_warm_request(&body).and_then(|r| serde_json::to_vec(&r).ok()) else { return };
    let every = cfg
        .idle_secs
        .unwrap_or_else(|| transform::message_cache_ttl_secs(&body).saturating_sub(compact::IDLE_LEAD_SECS))
        .max(1);
    // Exactly the forwarded request's betas, as for a warm summary.
    let extra: &[&str] = if bound { &[compact::THINKING_BINDING_BETA] } else { &[] };
    let current = |inner: &Inner| lock(&inner.generations).get(&key).copied() == Some(generation);
    let mut last = started;
    for ping in 1..=compact::KEEP_WARM_MAX_PINGS {
        let wait = (last + every).saturating_sub(inner.clock.now_secs());
        tokio::time::sleep(Duration::from_secs(wait)).await;
        if !current(&inner) {
            return;
        }
        let at = inner.clock.now_secs();
        let sent = credentials
            .apply(inner.client.post(format!("{}/v1/messages", inner.upstream)), extra)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::USER_AGENT, concat!("squeeze/", env!("CARGO_PKG_VERSION")))
            .timeout(Duration::from_secs(120))
            .body(payload.clone())
            .send()
            .await;
        let (status, bytes) = match sent {
            Ok(r) => (r.status().as_u16(), r.bytes().await.unwrap_or_default()),
            Err(e) => {
                tracing::warn!("keep-warm for thread {key}: the request failed: {e}");
                return;
            }
        };
        let mut tap = UsageTap::new(false);
        tap.feed(&bytes);
        let usage = tap.finish();
        let ok = (200..300).contains(&status);
        let error = (!ok).then(|| String::from_utf8_lossy(&bytes).chars().take(2_000).collect::<String>());
        {
            let mut t = lock(&inner.totals);
            t.keep_warm_requests += 1;
            if let Some(u) = &usage {
                t.usage.add(u);
            }
        }
        // The cache's TTL now runs from this request (unless the thread's
        // next request started meanwhile, later still).
        if ok && let Some(state) = lock(&inner.threads).get_mut(&key) {
            state.last_start = state.last_start.max(Some(at));
            inner.store.save_thread(&key, state);
        }
        let rec = json!({
            "ts": now_secs(),
            "session": session,
            "model": usage.as_ref().and_then(|u| u.model.clone()).or_else(|| model.clone()),
            "status": status,
            "keep_warm": { "thread": key, "ping": ping, "error": error },
            "usage": usage,
        });
        if let Err(e) = inner.store.append_record(&rec) {
            tracing::warn!("could not write stats: {e}");
        }
        match &error {
            Some(e) => {
                tracing::warn!("keep-warm for thread {key}: HTTP {status}: {e}");
                return;
            }
            None => tracing::info!("keep-warm for thread {key}: conversation still cached (request {ping})"),
        }
        last = at;
    }
}

/// Wait for the pause, then have the turn's conversation summarized and keep
/// the summary for the thread's next request. Gives up, dropping the
/// credentials, when another request on the thread comes first.
async fn compact_later(
    inner: Arc<Inner>,
    cfg: CompactConfig,
    key: String,
    session: Option<String>,
    model: Option<String>,
    started: u64,
    job: CompactJob,
) {
    let Some(body) = job.forwarded.as_deref().and_then(|f| serde_json::from_slice::<Value>(f).ok()) else { return };
    let conversation = body.get("messages").map_or(0, |m| m.to_string().len());
    if approx_tokens(conversation) < cfg.min_tokens as f64 {
        // Too short for a summary to pay for itself. A keep-warm request costs
        // a 25th of rewriting it after the pause, so keep it cached instead.
        if cfg.keep_warm {
            keep_warm(inner, cfg, key, session, model, started, job).await;
        }
        return;
    }
    let CompactJob { generation, covered, prefix_hash, credentials, bound, .. } = job;
    let Some(credentials) = credentials else { return };
    let idle = cfg
        .idle_secs
        .unwrap_or_else(|| transform::message_cache_ttl_secs(&body).saturating_sub(compact::IDLE_LEAD_SECS));
    let wait = (started + idle).saturating_sub(inner.clock.now_secs());
    tokio::time::sleep(Duration::from_secs(wait)).await;
    let current = |inner: &Inner| lock(&inner.generations).get(&key).copied() == Some(generation);
    if !current(&inner) {
        return;
    }
    // Warm: the forwarded request plus the instruction, with exactly its betas,
    // so it reads the conversation from the cache. Otherwise a cheaper-to-read
    // skeleton that needs the binding beta.
    let (request, extra): (_, &[&str]) = if cfg.warm {
        (compact::warm_summary_request(&body), if bound { &[compact::THINKING_BINDING_BETA] } else { &[] })
    } else {
        (compact::summary_request(&body), &[compact::THINKING_BINDING_BETA])
    };
    let Some(request) = request else { return };
    let Ok(payload) = serde_json::to_vec(&request) else { return };
    let sent = credentials
        .apply(inner.client.post(format!("{}/v1/messages", inner.upstream)), extra)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::USER_AGENT, concat!("squeeze/", env!("CARGO_PKG_VERSION")))
        .timeout(Duration::from_secs(600))
        .body(payload)
        .send()
        .await;
    drop(credentials);
    let (status, bytes) = match sent {
        Ok(r) => (r.status().as_u16(), r.bytes().await.unwrap_or_default()),
        Err(e) => {
            tracing::warn!("compaction for thread {key}: the summarizing request failed: {e}");
            return;
        }
    };
    let mut tap = UsageTap::new(false);
    tap.feed(&bytes);
    let usage = tap.finish();
    let summary = serde_json::from_slice::<Value>(&bytes).ok().as_ref().and_then(compact::summary_text);
    let stored = summary.as_ref().is_some_and(|text| {
        if !current(&inner) {
            return false;
        }
        let mut threads = lock(&inner.threads);
        let Some(state) = threads.get_mut(&key) else { return false };
        state.compact.pending = Some(compact::Summary { covered, prefix_hash, text: text.clone() });
        state.compact.compactions += 1;
        inner.store.save_thread(&key, state);
        true
    });
    // Kept so a failure can be diagnosed without paying for it again.
    let error = (!(200..300).contains(&status))
        .then(|| String::from_utf8_lossy(&bytes).chars().take(2_000).collect::<String>());
    match &error {
        Some(e) => tracing::warn!("compaction for thread {key}: the summarizing request got HTTP {status}: {e}"),
        None => tracing::info!(
            "compaction for thread {key}: {}",
            if stored { "summary ready for the next request" } else { "no summary kept" }
        ),
    }
    {
        let mut t = lock(&inner.totals);
        t.compaction_requests += 1;
        t.compactions += u64::from(stored);
        if let Some(u) = &usage {
            t.usage.add(u);
        }
    }
    let rec = json!({
        "ts": now_secs(),
        "session": session,
        "model": usage.as_ref().and_then(|u| u.model.clone()).or(model),
        "status": status,
        "compaction": { "thread": key, "covered": covered, "stored": stored,
                        "summary_chars": summary.as_ref().map_or(0, String::len),
                        "summary": summary, "error": error },
        "usage": usage,
    });
    if let Err(e) = inner.store.append_record(&rec) {
        tracing::warn!("could not write stats: {e}");
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

/// Add [`compact::THINKING_BINDING_BETA`] to `anthropic-beta` when `add`.
fn with_binding_beta(mut h: HeaderMap, add: bool) -> HeaderMap {
    if !add {
        return h;
    }
    let mut betas: Vec<String> = h
        .get_all("anthropic-beta")
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(|b| b.trim().to_string())
        .filter(|b| !b.is_empty())
        .collect();
    if !betas.iter().any(|b| b == compact::THINKING_BINDING_BETA) {
        betas.push(compact::THINKING_BINDING_BETA.into());
    }
    if let Ok(v) = HeaderValue::from_str(&betas.join(",")) {
        h.insert("anthropic-beta", v);
    }
    h
}

/// `--compact-warm`: take Claude Code's beta out of a forwarded request.
fn without_claude_code_beta(h: &mut HeaderMap) {
    let all: Vec<String> =
        h.get_all("anthropic-beta").iter().filter_map(|v| v.to_str().ok()).map(String::from).collect();
    if all.is_empty() {
        return;
    }
    h.remove("anthropic-beta");
    if let Some(v) = compact::without_claude_code_beta(&all.join(",")).and_then(|b| HeaderValue::from_str(&b).ok()) {
        h.insert("anthropic-beta", v);
    }
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
