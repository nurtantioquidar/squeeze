//! An emulated Anthropic Messages API, for what context editing and prompt
//! caching depend on. Free and offline: the caller scripts each reply's
//! content; the emulator does what the server would do around it.
//!
//! - Context editing: it clears what `context_management` asks for, under the
//!   counting convention squeeze assumes (uses of tools outside
//!   `exclude_tools`, one by one; a thinking turn is a run of assistant
//!   messages, counted once one has thinking), and reports `applied_edits`.
//!   A cleared result or input becomes a 10-token placeholder, unless it's no
//!   bigger than that, as `squeeze replay --cache-sim` bills it.
//! - Prompt cache: the edited prompt (tools, system, messages) gets an entry
//!   at each `cache_control` breakpoint. Each breakpoint looks back at most 20
//!   positions for a live entry, a run of `tool_use` or of `tool_result`
//!   blocks being one position, and an entry lives for its TTL from the start
//!   of the request that wrote or last read it, on a fake clock.
//! - Thinking: a block is bound to the system prompt, the set of tools and
//!   every message before it as the client sent it (earlier thinking,
//!   server-side edits and cache markers don't count), and to the thinking
//!   block before it. Sending it after any of that changed is refused; only a
//!   leading run of thinking blocks may be dropped.
//! - Rejections: a 400 where the API documents one (edits without the beta,
//!   `clear_thinking` not first, a thinking `keep` of 0, more than 4
//!   breakpoints, a 1-hour breakpoint after a 5-minute one, a rebound
//!   thinking block).
//! - Tokens: a quarter of each block's content bytes (text, tool input, result
//!   text, or the whole block otherwise), as `squeeze replay --cache-sim`
//!   counts them.

use super::{FakeClock, Recorded, Reply, sse_message};
use serde_json::{Value, json};
use squeeze::proxy::Clock;
use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

/// What the server leaves in place of a cleared tool result (40 bytes, 10 tokens).
pub const PLACEHOLDER: &str = "[cleared to save context: run it again.]";
/// The input of a cleared tool call, as `{"cleared": ...}`: the same 40 bytes.
pub const INPUT_PLACEHOLDER: &str = "to save context: rerun it.";

const BETA: &str = "context-management-2025-06-27";
const CLEAR_TOOL_USES: &str = "clear_tool_uses_20250919";
const CLEAR_THINKING: &str = "clear_thinking_20251015";
const LOOKBACK: usize = 20;
const MAX_BREAKPOINTS: usize = 4;

fn json_len(v: &Value) -> usize {
    serde_json::to_string(v).map_or(0, |s| s.len())
}

fn without_marker(v: &Value) -> Value {
    let mut v = v.clone();
    if let Some(o) = v.as_object_mut() {
        o.shift_remove("cache_control");
    }
    v
}

/// Tokens of a content block, tool or system block.
pub fn tokens(b: &Value) -> f64 {
    let bytes = match b.get("type").and_then(Value::as_str) {
        Some("text") => b["text"].as_str().map_or(0, str::len),
        Some("tool_use") => b.get("input").map_or(2, json_len),
        Some("tool_result") => match &b["content"] {
            Value::String(s) => s.len(),
            Value::Array(parts) => parts
                .iter()
                .map(|p| if p["type"] == "text" { p["text"].as_str().map_or(0, str::len) } else { json_len(p) })
                .sum(),
            _ => 0,
        },
        _ => json_len(&without_marker(b)),
    };
    bytes as f64 / 4.0
}

/// A message's content as blocks; string content is one text block.
pub fn blocks(m: &Value) -> Cow<'_, [Value]> {
    match &m["content"] {
        Value::String(s) => Cow::Owned(vec![json!({ "type": "text", "text": s })]),
        Value::Array(a) => Cow::Borrowed(a),
        _ => Cow::Borrowed(&[]),
    }
}

fn is_thinking(b: &Value) -> bool {
    matches!(b["type"].as_str(), Some("thinking" | "redacted_thinking"))
}

/// Hash what the server compares and caches: `v` without `cache_control`
/// markers or the `skip` keys at its top, keys in any order.
fn hash_without(v: &Value, skip: &[&str]) -> u64 {
    fn feed(v: &Value, skip: &[&str], h: &mut DefaultHasher) {
        match v {
            Value::Object(m) => {
                let mut keys: Vec<&String> =
                    m.keys().filter(|k| *k != "cache_control" && !skip.contains(&k.as_str())).collect();
                keys.sort();
                h.write_u8(1);
                for k in keys {
                    k.hash(h);
                    feed(&m[k], &[], h);
                }
            }
            Value::Array(a) => {
                h.write_u8(2);
                h.write_usize(a.len());
                a.iter().for_each(|x| feed(x, &[], h));
            }
            Value::String(s) => {
                h.write_u8(3);
                s.hash(h);
            }
            other => {
                h.write_u8(4);
                other.to_string().hash(h);
            }
        }
    }
    let mut h = DefaultHasher::new();
    feed(v, skip, &mut h);
    h.finish()
}

fn hash(v: &Value) -> u64 {
    hash_without(v, &[])
}

fn mix(a: u64, b: u64) -> u64 {
    let mut h = DefaultHasher::new();
    (a, b).hash(&mut h);
    h.finish()
}

/// Hash of a message without its content: its role and any other fields.
fn header(m: &Value) -> u64 {
    hash_without(m, &["content"])
}

/// Starts a thinking turn as the API counts them for `clear_thinking`: a user
/// message holding anything besides tool results (measured on the API,
/// 2026-11-17). The current turn counts.
pub fn starts_turn(m: &Value) -> bool {
    m["role"] == "user"
        && match &m["content"] {
            Value::Array(items) => items.iter().any(|b| b["type"] != "tool_result"),
            _ => true,
        }
}

/// Each assistant message's thinking turn, and how many turns there are.
fn thinking_turns(messages: &[Value]) -> (Vec<Option<u64>>, u64) {
    let mut n = 0;
    let turns = messages
        .iter()
        .map(|m| {
            n += u64::from(starts_turn(m));
            (m["role"] == "assistant").then_some(n)
        })
        .collect();
    (turns, n)
}

/// The API leaves a tool use out of `keep` and `trigger` when its result is too
/// short to be worth clearing: under 47 characters of text, unless it's empty
/// or holds something besides text, or `clear_tool_inputs` is on (measured on
/// the API, 2026-11-17).
pub fn counted_use(result: Option<&Value>, inputs: bool) -> bool {
    let chars: usize = match result.map(|r| &r["content"]) {
        Some(Value::String(s)) => s.chars().count(),
        Some(Value::Array(items)) if items.iter().all(|b| b["type"] == "text") => {
            items.iter().filter_map(|b| b["text"].as_str()).map(|t| t.chars().count()).sum()
        }
        Some(Value::Array(_)) => return true,
        _ => 0,
    };
    inputs || chars == 0 || chars >= 47
}

/// A request's messages once its context edits are applied, and what they cleared.
pub struct Cleared {
    pub messages: Vec<Value>,
    /// `context_management.applied_edits`, as the response reports them.
    pub applied: Vec<Value>,
}

fn allowed(e: &Value, keys: &[&str], at: &str) -> Result<(), String> {
    match e.as_object().into_iter().flatten().find(|(k, _)| !keys.contains(&k.as_str())) {
        Some((k, _)) => Err(format!("{at}.{k}: Extra inputs are not permitted")),
        None => Ok(()),
    }
}

/// A `{type, value}` count of one of `kinds`, or `default` when absent.
fn count(e: &Value, key: &str, kinds: &[&str], default: (&str, u64), at: &str) -> Result<(String, u64), String> {
    let Some(v) = e.get(key) else { return Ok((default.0.to_string(), default.1)) };
    match (v["type"].as_str(), v["value"].as_u64()) {
        (Some(kind), Some(n)) if kinds.contains(&kind) => Ok((kind.to_string(), n)),
        _ => Err(format!("{at}.{key}: expected {{type: {}, value: integer}}", kinds.join("|"))),
    }
}

/// Apply a request's `context_management` edits, or say why the API refuses them.
pub fn cleared(body: &Value) -> Result<Cleared, String> {
    let mut messages: Vec<Value> = body["messages"].as_array().cloned().unwrap_or_default();
    let mut applied = Vec::new();
    let Some(cm) = body.get("context_management") else { return Ok(Cleared { messages, applied }) };
    let edits = cm.get("edits").and_then(Value::as_array).ok_or("context_management.edits: Field required")?;
    let mut seen = HashSet::new();
    for (i, e) in edits.iter().enumerate() {
        let at = format!("context_management.edits.{i}");
        let kind = e["type"].as_str().unwrap_or_default();
        if !seen.insert(kind) {
            return Err(format!("{at}: {kind} is listed twice"));
        }
        match kind {
            CLEAR_THINKING => {
                allowed(e, &["type", "keep"], &at)?;
                if i != 0 {
                    return Err(format!("{at}: {CLEAR_THINKING} must be the first edit"));
                }
                if !matches!(body["thinking"]["type"].as_str(), Some("adaptive" | "enabled")) {
                    return Err(format!("{at}: {CLEAR_THINKING} requires thinking"));
                }
                let keep = match &e["keep"] {
                    Value::String(s) if s == "all" => continue,
                    Value::Null => 1,
                    k => match (k["type"].as_str(), k["value"].as_u64()) {
                        (Some("thinking_turns"), Some(n)) if n >= 1 => n,
                        _ => return Err(format!("{at}.keep: \"all\" or {{type: thinking_turns, value: 1 or more}}")),
                    },
                };
                let (turns, n) = thinking_turns(&messages);
                let cut = n.saturating_sub(keep);
                let mut freed = 0.0;
                for (m, turn) in messages.iter_mut().zip(&turns) {
                    if turn.is_some_and(|t| t <= cut)
                        && let Some(content) = m["content"].as_array_mut()
                    {
                        freed += content.iter().filter(|b| is_thinking(b)).map(tokens).sum::<f64>();
                        content.retain(|b| !is_thinking(b));
                    }
                }
                if cut > 0 {
                    applied.push(json!({ "type": CLEAR_THINKING, "cleared_thinking_turns": cut,
                                         "cleared_input_tokens": freed.round() as u64 }));
                }
            }
            CLEAR_TOOL_USES => {
                let keys = ["type", "trigger", "keep", "clear_at_least", "exclude_tools", "clear_tool_inputs"];
                allowed(e, &keys, &at)?;
                let (trigger_kind, trigger) =
                    count(e, "trigger", &["input_tokens", "tool_uses"], ("input_tokens", 100_000), &at)?;
                let (_, keep) = count(e, "keep", &["tool_uses"], ("tool_uses", 3), &at)?;
                let excluded: Vec<&str> = match e.get("exclude_tools") {
                    None => Vec::new(),
                    Some(v) => v
                        .as_array()
                        .and_then(|a| a.iter().map(Value::as_str).collect())
                        .ok_or(format!("{at}.exclude_tools: a list of tool names"))?,
                };
                let inputs = match e.get("clear_tool_inputs") {
                    None => false,
                    Some(v) => v.as_bool().ok_or(format!("{at}.clear_tool_inputs: a boolean"))?,
                };
                let results: HashMap<String, Value> = messages
                    .iter()
                    .flat_map(|m| blocks(m).into_owned())
                    .filter(|b| b["type"] == "tool_result")
                    .filter_map(|b| b["tool_use_id"].as_str().map(|id| (id.to_string(), b.clone())))
                    .collect();
                let uses: Vec<String> = messages
                    .iter()
                    .filter(|m| m["role"] == "assistant")
                    .flat_map(|m| blocks(m).into_owned())
                    .filter(|b| b["type"] == "tool_use" && !b["name"].as_str().is_some_and(|n| excluded.contains(&n)))
                    .filter_map(|b| b["id"].as_str().map(str::to_string))
                    .filter(|id| counted_use(results.get(id), inputs))
                    .collect();
                let size = match trigger_kind.as_str() {
                    "tool_uses" => uses.len() as f64,
                    _ => messages.iter().map(|m| blocks(m).iter().map(tokens).sum::<f64>()).sum(),
                };
                if size <= trigger as f64 || uses.len() as u64 <= keep {
                    continue;
                }
                let cut: HashSet<&str> = uses[..uses.len() - keep as usize].iter().map(String::as_str).collect();
                let placeholder = PLACEHOLDER.len() as f64 / 4.0;
                let mut freed = 0.0;
                for b in messages.iter_mut().filter_map(|m| m["content"].as_array_mut()).flatten() {
                    let before = tokens(b);
                    if before <= placeholder {
                        continue;
                    }
                    if b["type"] == "tool_result" && b["tool_use_id"].as_str().is_some_and(|id| cut.contains(id)) {
                        b["content"] = json!(PLACEHOLDER);
                    } else if inputs && b["type"] == "tool_use" && b["id"].as_str().is_some_and(|id| cut.contains(id)) {
                        b["input"] = json!({ "cleared": INPUT_PLACEHOLDER });
                    } else {
                        continue;
                    }
                    freed += before - tokens(b);
                }
                applied.push(json!({ "type": CLEAR_TOOL_USES, "cleared_tool_uses": cut.len(),
                                     "cleared_input_tokens": freed.round() as u64 }));
            }
            other => return Err(format!("{at}.type: unknown edit type {other:?}")),
        }
    }
    Ok(Cleared { messages, applied })
}

/// A step of the prompt as rendered: a tool, a system block, a message's
/// role (and other fields), or a content block.
struct Item {
    /// Hash of the prompt up to and including this item.
    prefix: u64,
    tokens: f64,
    /// Lookback position, for blocks.
    position: Option<usize>,
    /// Lifetime of a breakpoint on it, in seconds.
    ttl: Option<u64>,
}

/// Tokens of a stretch of the prompt (`0.0`, not `-0.0`, when empty).
fn tokens_of(items: &[Item]) -> f64 {
    items.iter().fold(0.0, |sum, i| sum + i.tokens)
}

fn ttl(b: &Value, at: &str) -> Result<Option<u64>, String> {
    let Some(cc) = b.get("cache_control") else { return Ok(None) };
    if cc["type"] != "ephemeral" {
        return Err(format!("{at}.cache_control.type: must be \"ephemeral\""));
    }
    match cc.get("ttl").and_then(Value::as_str) {
        None | Some("5m") => Ok(Some(300)),
        Some("1h") => Ok(Some(3600)),
        Some(other) => Err(format!("{at}.cache_control.ttl: \"5m\" or \"1h\", not {other:?}")),
    }
}

/// The prompt the server renders: tools, system, then `messages` (edited),
/// with the client's breakpoints; and how many items are tools and system.
fn render(body: &Value, messages: &[Value]) -> Result<(Vec<Item>, usize), String> {
    let mut items: Vec<Item> = Vec::new();
    let (mut prefix, mut position, mut run) = (0, 0, None);
    let mut push = |v: &Value, at: String, block: bool| -> Result<(), String> {
        prefix = mix(prefix, if block { hash(v) } else { header(v) });
        let kind = v["type"].as_str().filter(|k| matches!(*k, "tool_use" | "tool_result")).map(str::to_string);
        let pos = block.then(|| {
            if kind.is_none() || kind != run {
                position += 1;
            }
            position
        });
        run = if block { kind } else { None };
        let ttl = if block { ttl(v, &at)? } else { None };
        items.push(Item { prefix, tokens: if block { tokens(v) } else { 0.0 }, position: pos, ttl });
        Ok(())
    };
    let tools = body["tools"].as_array().cloned().unwrap_or_default();
    let system = blocks(&json!({ "content": body["system"] })).into_owned();
    for (i, t) in tools.iter().enumerate() {
        push(t, format!("tools.{i}"), true)?;
    }
    for (i, s) in system.iter().enumerate() {
        push(s, format!("system.{i}"), true)?;
    }
    for (i, m) in messages.iter().enumerate() {
        push(m, format!("messages.{i}"), false)?;
        for (j, b) in blocks(m).iter().enumerate() {
            push(b, format!("messages.{i}.content.{j}"), true)?;
        }
    }
    let breakpoints: Vec<&Item> = items.iter().filter(|i| i.ttl.is_some()).collect();
    if breakpoints.len() > MAX_BREAKPOINTS {
        return Err(format!(
            "A maximum of {MAX_BREAKPOINTS} blocks with cache_control may be provided. Found {}.",
            breakpoints.len()
        ));
    }
    if breakpoints.windows(2).any(|w| w[0].ttl < w[1].ttl) {
        return Err("a ttl='1h' cache_control block must not come after a ttl='5m' cache_control block".into());
    }
    Ok((items, tools.len() + system.len()))
}

/// What a thinking block is bound to: the conversation before it as sent,
/// without thinking blocks, and the thinking block before it.
#[derive(Clone, Debug, PartialEq)]
struct Binding {
    prefix: u64,
    previous: Option<String>,
}

/// Check every thinking block the client sends back against the
/// conversation it came from (binding blocks seen for the first time), and
/// return the binding of a block produced in reply.
fn check_thinking(bound: &mut HashMap<String, Binding>, body: &Value) -> Result<Binding, String> {
    let mut tools = body["tools"].as_array().cloned().unwrap_or_default();
    tools.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    let mut at = mix(hash(&body["system"]), hash(&Value::Array(tools)));
    let mut previous: Option<String> = None;
    for (i, m) in body["messages"].as_array().into_iter().flatten().enumerate() {
        at = mix(at, header(m));
        for (j, b) in blocks(m).iter().enumerate() {
            let Some(key) = b.get("signature").or(b.get("data")).and_then(Value::as_str).filter(|_| is_thinking(b))
            else {
                at = mix(at, hash(b));
                continue;
            };
            match bound.get(key) {
                // With no thinking before it, the blocks before it were dropped from the front.
                Some(was) if was.prefix != at || (previous.is_some() && was.previous != previous) => {
                    return Err(format!(
                        "messages.{i}.content.{j}: Invalid `signature` in `thinking` block. \
                         The block is bound to a different conversation."
                    ));
                }
                Some(_) => {}
                None => _ = bound.insert(key.to_string(), Binding { prefix: at, previous: previous.clone() }),
            }
            previous = Some(key.to_string());
        }
    }
    Ok(Binding { prefix: mix(at, header(&json!({ "role": "assistant" }))), previous })
}

/// What the emulated API made of one request.
#[derive(Clone, Debug, Default)]
pub struct Served {
    pub path: String,
    pub at: u64,
    /// Why it was refused with a 400.
    pub error: Option<String>,
    /// Hash of the edited prompt after each item, tools and system first.
    pub prefix: Vec<u64>,
    /// Tokens of the tools and system prompt.
    pub head: f64,
    /// Tokens read from the cache, written to it, and after the last breakpoint.
    pub read: f64,
    pub write: f64,
    pub input: f64,
    pub applied: Vec<Value>,
}

impl Served {
    /// This prompt, as the server saw it, starts with all of `earlier`'s.
    pub fn extends(&self, earlier: &Served) -> bool {
        match earlier.prefix.last() {
            None => true,
            Some(last) => self.prefix.get(earlier.prefix.len() - 1) == Some(last),
        }
    }

    /// `applied_edits` summed: `cleared_tool_uses`, `cleared_thinking_turns` or `cleared_input_tokens`.
    pub fn cleared(&self, field: &str) -> u64 {
        self.applied.iter().filter_map(|e| e[field].as_u64()).sum()
    }
}

#[derive(Default)]
struct State {
    /// Cache entries by prompt prefix: expiry and lifetime.
    cache: HashMap<u64, (u64, u64)>,
    /// Each thinking block's binding, by signature.
    bound: HashMap<String, Binding>,
    served: Vec<Served>,
}

pub struct Api {
    clock: Arc<FakeClock>,
    state: Mutex<State>,
}

impl Api {
    pub fn new(clock: Arc<FakeClock>) -> Arc<Self> {
        Arc::new(Self { clock, state: Mutex::default() })
    }

    /// Every request answered so far, in order.
    pub fn served(&self) -> Vec<Served> {
        self.state.lock().unwrap().served.clone()
    }

    /// Answer a messages or count_tokens request as the API would. A messages
    /// reply, from the model asked for, streams a thinking block (when
    /// thinking is on), then `content`, ending with `stop`, with the cache's
    /// usage and the edits applied.
    pub fn respond(&self, r: &Recorded, content: Vec<Value>, stop: &str) -> Reply {
        let body = r.json();
        let mut st = self.state.lock().unwrap();
        let mut served = Served { path: r.path.clone(), at: self.clock.now_secs(), ..Default::default() };
        let prepared = (|| {
            let betas = r.headers.get("anthropic-beta").and_then(|v| v.to_str().ok()).unwrap_or_default();
            if body.get("context_management").is_some() && !betas.split(',').any(|b| b.trim() == BETA) {
                return Err(format!("context_management: requires the {BETA} beta"));
            }
            let c = cleared(&body)?;
            let (items, head) = render(&body, &c.messages)?;
            let binding = check_thinking(&mut st.bound, &body)?;
            Ok((c.applied, items, head, binding))
        })();
        let (applied, items, head, binding) = match prepared {
            Ok(p) => p,
            Err(e) => {
                served.error = Some(e.clone());
                st.served.push(served);
                let err = json!({ "type": "error", "error": { "type": "invalid_request_error", "message": e } });
                return Reply::Json(400, err);
            }
        };
        served.prefix = items.iter().map(|i| i.prefix).collect();
        served.head = tokens_of(&items[..head]);
        served.applied = applied;
        if r.path.ends_with("/count_tokens") {
            let total = tokens_of(&items);
            st.served.push(served);
            return Reply::Json(200, json!({ "input_tokens": total.round() as u64 }));
        }

        // Each breakpoint reads the longest live entry within its lookback.
        let now = served.at;
        let mut hits = Vec::new();
        for (b, item) in items.iter().enumerate().filter(|(_, i)| i.ttl.is_some()) {
            let pos = item.position.unwrap_or_default();
            hits.extend(
                (0..=b)
                    .rev()
                    .take_while(|&j| items[j].position.is_none_or(|p| pos - p <= LOOKBACK))
                    .find(|&j| st.cache.get(&items[j].prefix).is_some_and(|&(expiry, _)| expiry >= now)),
            );
        }
        let read_to = hits.iter().max().map_or(0, |&j| j + 1);
        let (mut write_5m, mut write_1h, mut from) = (0.0, 0.0, read_to);
        for (b, item) in items.iter().enumerate().filter(|(b, i)| i.ttl.is_some() && *b >= read_to) {
            let tokens = tokens_of(&items[from..=b]);
            if item.ttl == Some(3600) {
                write_1h += tokens;
            } else {
                write_5m += tokens;
            }
            from = b + 1;
        }
        served.read = tokens_of(&items[..read_to]);
        served.write = write_5m + write_1h;
        served.input = tokens_of(&items[from..]);
        for j in hits {
            if let Some(e) = st.cache.get_mut(&items[j].prefix) {
                e.0 = e.0.max(now + e.1);
            }
        }
        for item in &items {
            if let Some(ttl) = item.ttl {
                let e = st.cache.entry(item.prefix).or_insert((0, ttl));
                e.1 = e.1.max(ttl);
                e.0 = e.0.max(now + ttl);
            }
        }

        let n = st.served.len();
        let mut reply = Vec::new();
        if matches!(body["thinking"]["type"].as_str(), Some("adaptive" | "enabled")) {
            let signature = format!("emu-sig-{n}");
            st.bound.insert(signature.clone(), binding);
            reply.push(json!({ "type": "thinking", "thinking": "", "signature": signature }));
        }
        reply.extend(content);
        let usage = json!({
            "input_tokens": served.input.round() as u64,
            "cache_creation_input_tokens": served.write.round() as u64,
            "cache_read_input_tokens": served.read.round() as u64,
            "cache_creation": { "ephemeral_5m_input_tokens": write_5m.round() as u64,
                                "ephemeral_1h_input_tokens": write_1h.round() as u64 },
            "output_tokens": 1
        });
        let cm = json!({ "applied_edits": served.applied });
        st.served.push(served);
        Reply::Sse(sse_message(n, body["model"].as_str().unwrap_or_default(), &reply, stop, usage, cm))
    }
}
