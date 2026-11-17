//! Server-side context editing pinned at cold-cache moments (`--context-edits cold`).
//!
//! Claude Code already asks the API to manage thinking blocks
//! (`context_management.edits: [{clear_thinking_20251015, keep: "all"}]`). The
//! proxy merges in a `clear_tool_uses_20250919` edit and a numeric thinking
//! `keep`, chosen so the server clears exactly the oldest `C` clearable tool
//! uses and `C_th` thinking turns. `keep = n - C` grows with the history, so the
//! cleared set stays put and the server's view of the conversation grows
//! append-only, which keeps it cached. `C` and `C_th` move forward only when the
//! prompt cache has already expired, when a new prefix costs nothing extra.
//!
//! Message bytes are never touched: the server does the clearing, and
//! server-side context management never invalidates thinking. A thread whose
//! state is lost goes out as the client sent it: one cache rebuild, never a
//! rejected request.
//!
//! Counting convention, as measured on the API (probe-cache-key-v2,
//! 2026-11-17): `keep` and `trigger` count the uses of tools outside
//! `exclude_tools`, each parallel use on its own, but only uses whose result is
//! worth clearing ([`counts_toward_keep`]); a thinking turn starts at each user
//! message that holds anything besides tool results ([`starts_turn`]), and the
//! current turn counts. The `context_management` values themselves are not part
//! of the prompt-cache key: requests that only change `keep` and `trigger`
//! read the whole previous prompt from cache. The `ctx_mismatch` breaker stops
//! advancing a thread whose server counts otherwise.

use crate::sse::{AppliedEdits, Usage};
use crate::store::CtxState;
use anyhow::bail;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeSet, HashMap};

pub const BETA: &str = "context-management-2025-06-27";
pub(crate) const CLEAR_TOOL_USES: &str = "clear_tool_uses_20250919";
pub(crate) const CLEAR_THINKING: &str = "clear_thinking_20251015";

/// Tools whose uses the server may clear: file views, searches, edits and shell
/// output (squeeze's own tools included), which the agent can get back by
/// running the call again. Every other tool goes into `exclude_tools`.
const CLEARABLE: &[&str] = &["Read", "Grep", "Glob", "Edit", "MultiEdit", "Write", "NotebookEdit", "Bash"];

/// An advance must clear at least this many more tool uses, and about this many
/// more tokens: a smaller one isn't worth a new cache prefix.
const MIN_NEW_USES: usize = 2;
const MIN_NEW_TOKENS: f64 = 5000.0;

/// Tokens per 4 bytes of tool result text and of tool input JSON, as measured
/// on Claude Code transcripts.
const RESULT_TOKENS_PER_4_BYTES: f64 = 1.78;
const INPUT_TOKENS_PER_4_BYTES: f64 = 1.68;

/// A warm request should read at least this share of the previous prompt from cache.
const MIN_CACHE_READ_SHARE: f64 = 0.9;

pub fn is_clearable(name: &str) -> bool {
    CLEARABLE.contains(&name) || name.starts_with("mcp__squeeze__")
}

/// `--context-edits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextEdits {
    /// Claude Code's own `context_management` goes out unchanged.
    Off,
    /// Clear old tool uses and thinking, advancing only at cold-cache moments.
    Cold,
}

impl std::str::FromStr for ContextEdits {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.trim() {
            "off" => Ok(Self::Off),
            "cold" => Ok(Self::Cold),
            "cold+warm" => bail!("cold+warm (advancing while the cache is warm) is reserved and not implemented yet"),
            other => bail!("unknown context-edits mode {other:?} (off, cold)"),
        }
    }
}

/// How many recent clearable tool uses an advance keeps: those since the
/// start of the previous human turn, so the agent's working set survives,
/// between `min` and `max`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeepWindow {
    pub min: usize,
    pub max: usize,
}

impl Default for KeepWindow {
    fn default() -> Self {
        Self { min: 8, max: 24 }
    }
}

impl KeepWindow {
    /// From `--keep-tool-uses` and `--keep-tool-uses-max`.
    pub fn new(min: usize, max: usize) -> anyhow::Result<Self> {
        if min == 0 || max < min {
            bail!("--keep-tool-uses must be at least 1 and at most --keep-tool-uses-max");
        }
        Ok(Self { min, max })
    }

    pub fn k(self, uses_since_prev_human: usize) -> usize {
        uses_since_prev_human.max(self.min).min(self.max)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CtxConfig {
    pub window: KeepWindow,
    /// Clear tool inputs along with results (`clear_tool_inputs`).
    pub clear_inputs: bool,
    /// Also clear thinking turns from before the latest human prompt.
    pub clear_thinking: bool,
}

impl Default for CtxConfig {
    fn default() -> Self {
        Self { window: KeepWindow::default(), clear_inputs: false, clear_thinking: true }
    }
}

/// A clearable tool use and estimates of what clearing it saves.
#[derive(Debug, Clone)]
pub struct ToolUse {
    pub id: String,
    pub result_tokens: f64,
    pub input_tokens: f64,
}

/// What a request's history holds, as far as context editing goes.
#[derive(Debug, Default)]
pub struct Census {
    /// Clearable tool uses, oldest first, parallel ones one by one. Error and
    /// image results count too: `exclude_tools` can only exclude whole tools,
    /// so the server clears those like any other.
    pub uses: Vec<ToolUse>,
    /// Thinking turns as the server counts them ([`starts_turn`]), the current one included.
    pub n_th: u64,
    /// Thinking turns before the current one.
    pub th_before_human: u64,
    /// Clearable uses since the start of the previous human turn.
    pub uses_since_prev_human: usize,
    /// Every tool the request offers or the history used, minus the clearable
    /// ones. Rebuilt each request, so tools loaded mid-session are excluded too.
    pub exclude_tools: Vec<String>,
    /// The `context-management-2025-06-27` beta is on.
    pub beta: bool,
    /// The body has a `context_management.edits` list to merge into.
    pub field: bool,
    /// The client asks for `clear_tool_uses` itself.
    pub client_clears: bool,
    /// The client's `clear_thinking` keeps every turn, so a number can be pinned.
    pub thinking_keep_all: bool,
}

fn role(m: &Value) -> Option<&str> {
    m.get("role").and_then(Value::as_str)
}

fn blocks(m: &Value) -> &[Value] {
    m.get("content").and_then(Value::as_array).map_or(&[][..], Vec::as_slice)
}

fn block_type(b: &Value) -> Option<&str> {
    b.get("type").and_then(Value::as_str)
}

/// A human prompt: a user message that isn't returning tool results.
fn is_human(m: &Value) -> bool {
    role(m) == Some("user") && !blocks(m).iter().any(|b| block_type(b) == Some("tool_result"))
}

/// Starts a thinking turn as the server counts them for `clear_thinking`: a
/// user message holding anything besides tool results (a prompt, or the text
/// Claude Code adds next to results).
pub fn starts_turn(m: &Value) -> bool {
    role(m) == Some("user")
        && match m.get("content") {
            Some(Value::Array(items)) => items.iter().any(|b| block_type(b) != Some("tool_result")),
            _ => true,
        }
}

/// Results shorter than this many characters aren't worth clearing, and the
/// server leaves their uses out of its `keep` and `trigger` count.
pub const MIN_COUNTED_RESULT_CHARS: usize = 47;

/// Whether the server counts a tool use toward `keep` and `trigger`, given its
/// `tool_result` block: when the result has [`MIN_COUNTED_RESULT_CHARS`] or more
/// characters of text, is empty (the server fills one in), or holds an image or
/// a document. With `clear_tool_inputs` every use counts.
pub fn counts_toward_keep(result: Option<&Value>, clear_inputs: bool) -> bool {
    if clear_inputs {
        return true;
    }
    let chars: usize = match result.and_then(|r| r.get("content")) {
        Some(Value::String(s)) => s.chars().count(),
        Some(Value::Array(items)) => {
            if items.iter().any(|b| block_type(b) != Some("text")) {
                return true;
            }
            items.iter().filter_map(|b| b.get("text").and_then(Value::as_str)).map(|t| t.chars().count()).sum()
        }
        _ => 0,
    };
    chars == 0 || chars >= MIN_COUNTED_RESULT_CHARS
}

fn clearable_uses(m: &Value) -> impl Iterator<Item = &Value> {
    blocks(m).iter().filter(|b| {
        block_type(b) == Some("tool_use") && b.get("name").and_then(Value::as_str).is_some_and(is_clearable)
    })
}

/// Where the turn before the current one started, in clearable uses made
/// before it, given the count at each human prompt so far.
fn prev_turn_start(humans: &[usize]) -> usize {
    humans.len().checked_sub(2).map_or(0, |i| humans[i])
}

/// A clearable tool use's place in the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsePos {
    /// Clearable uses before it.
    pub pos: usize,
    /// Clearable uses since the start of the previous human turn, itself included.
    pub since_prev_human: usize,
}

/// Every clearable tool use by id. Each entry depends only on the messages up
/// to the use, so it's the same on every request of a conversation.
pub fn use_positions(messages: &[Value]) -> HashMap<String, UsePos> {
    let (mut out, mut humans, mut n) = (HashMap::new(), Vec::new(), 0);
    for m in messages {
        if is_human(m) {
            humans.push(n);
        } else if role(m) == Some("assistant") {
            for b in clearable_uses(m) {
                n += 1;
                if let Some(id) = b.get("id").and_then(Value::as_str) {
                    out.insert(id.to_string(), UsePos { pos: n - 1, since_prev_human: n - prev_turn_start(&humans) });
                }
            }
        }
    }
    out
}

fn edits(body: &Value) -> Option<&Vec<Value>> {
    body.get("context_management")?.get("edits")?.as_array()
}

pub fn census(body: &Value, betas: &str, clear_inputs: bool) -> Census {
    let messages = body.get("messages").and_then(Value::as_array).map_or(&[][..], Vec::as_slice);
    let mut results: HashMap<&str, &Value> = HashMap::new();
    for b in messages.iter().filter(|m| role(m) == Some("user")).flat_map(blocks) {
        if block_type(b) == Some("tool_result")
            && let Some(id) = b.get("tool_use_id").and_then(Value::as_str)
        {
            results.insert(id, b);
        }
    }

    let mut c = Census::default();
    let mut names: BTreeSet<String> = body
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|t| t.get("name").and_then(Value::as_str).map(str::to_string))
        .collect();
    let mut humans = Vec::new();
    for m in messages {
        if role(m) != Some("assistant") {
            c.n_th += u64::from(starts_turn(m));
            if is_human(m) {
                humans.push(c.uses.len());
            }
            continue;
        }
        for b in blocks(m).iter().filter(|b| block_type(b) == Some("tool_use")) {
            let (Some(id), Some(name)) = (b.get("id").and_then(Value::as_str), b.get("name").and_then(Value::as_str))
            else {
                continue;
            };
            names.insert(name.to_string());
            let result = results.get(id).copied();
            if is_clearable(name) && counts_toward_keep(result, clear_inputs) {
                let input_bytes = b.get("input").map_or(0, |i| i.to_string().len());
                let result_bytes = result.map_or(0, crate::transform::text_len);
                c.uses.push(ToolUse {
                    id: id.to_string(),
                    result_tokens: RESULT_TOKENS_PER_4_BYTES * result_bytes as f64 / 4.0,
                    input_tokens: INPUT_TOKENS_PER_4_BYTES * input_bytes as f64 / 4.0,
                });
            }
        }
    }
    c.th_before_human = c.n_th.saturating_sub(1);
    c.uses_since_prev_human = c.uses.len() - prev_turn_start(&humans);
    c.exclude_tools = names.into_iter().filter(|n| !is_clearable(n)).collect();
    c.beta = betas.split(',').any(|b| b.trim() == BETA);
    c.field = edits(body).is_some();
    let list = edits(body).map_or(&[][..], Vec::as_slice);
    c.client_clears = list.iter().any(|e| block_type(e).is_some_and(|t| t.starts_with("clear_tool_uses")));
    c.thinking_keep_all = list
        .iter()
        .any(|e| block_type(e) == Some(CLEAR_THINKING) && e.get("keep").and_then(Value::as_str) == Some("all"));
    c
}

/// The state still describes this history: its `C`-th clearable use is the
/// one recorded, and each cut leaves something after it.
pub fn lines_up(c: &Census, s: &CtxState) -> bool {
    let at = s.cleared as usize;
    let tools = at == 0 || (at < c.uses.len() && s.cleared_upto_id.as_deref() == Some(c.uses[at - 1].id.as_str()));
    tools && (s.cleared_th == 0 || s.cleared_th < c.n_th)
}

/// Go back to sending the client's own `context_management`.
pub fn unpin(s: &mut CtxState) {
    (s.cleared, s.cleared_upto_id, s.cleared_th) = (0, None, 0);
}

/// A move of the cleared set at a cold moment.
#[derive(Debug, Clone, PartialEq)]
pub struct Advance {
    pub cleared: u64,
    pub cleared_upto_id: String,
    pub cleared_th: u64,
    pub clear_inputs: bool,
    /// Estimated tokens newly cleared.
    pub new_tokens: f64,
}

/// Whether to move the cleared set, and where to: only at a cold moment, to
/// all but the newest `K` clearable uses, and only when that clears at least
/// two more uses and about 5k more tokens. Thinking goes up to the latest
/// human prompt, always keeping one turn (the API refuses `keep` 0).
pub fn plan_advance(c: &Census, s: &CtxState, cfg: &CtxConfig, cold: bool) -> Option<Advance> {
    if !cold || s.frozen.is_some() || s.client_managed || c.client_clears || !c.beta || !c.field {
        return None;
    }
    let from = s.cleared as usize;
    let target = c.uses.len().saturating_sub(cfg.window.k(c.uses_since_prev_human));
    if target < from + MIN_NEW_USES {
        return None;
    }
    let clear_inputs = if from == 0 { cfg.clear_inputs } else { s.clear_inputs };
    let new_tokens: f64 = c
        .uses
        .get(from..target)?
        .iter()
        .map(|u| u.result_tokens + if clear_inputs { u.input_tokens } else { 0.0 })
        .sum();
    if new_tokens < MIN_NEW_TOKENS {
        return None;
    }
    let cleared_th = if cfg.clear_thinking && c.thinking_keep_all {
        s.cleared_th.max(c.th_before_human.min(c.n_th.saturating_sub(1)))
    } else {
        s.cleared_th
    };
    let cleared_upto_id = c.uses[target - 1].id.clone();
    Some(Advance { cleared: target as u64, cleared_upto_id, cleared_th, clear_inputs, new_tokens })
}

/// Bring a thread's state up to date with a messages request and, at a cold
/// moment, advance it. A history that no longer lines up (Claude Code
/// compacted or rewound it) starts over; after a second `cache_regressed` the
/// thread goes back to the client's own setting at the next cold moment.
pub fn step(s: &mut CtxState, c: &Census, cfg: &CtxConfig, cold: bool) -> Option<Advance> {
    if c.client_clears {
        s.client_managed = true;
    }
    if !lines_up(c, s) || (cold && s.regressions >= 2) {
        unpin(s);
    }
    let a = plan_advance(c, s, cfg, cold)?;
    (s.cleared, s.cleared_upto_id, s.cleared_th) = (a.cleared, Some(a.cleared_upto_id.clone()), a.cleared_th);
    s.clear_inputs = a.clear_inputs;
    s.advances += 1;
    Some(a)
}

/// What the proxy asked the server to clear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Expected {
    pub tool_uses: u64,
    /// Set when the proxy pinned the thinking `keep`.
    pub thinking_turns: Option<u64>,
}

impl Expected {
    pub fn matches(&self, applied: Option<&AppliedEdits>) -> bool {
        applied.is_some_and(|a| {
            a.cleared_tool_uses == self.tool_uses && self.thinking_turns.is_none_or(|t| a.cleared_thinking_turns == t)
        })
    }
}

/// Merge the pinned edits into the request: the client's `clear_thinking`
/// keeps `n_th - C_th` turns, and a `clear_tool_uses` edit is appended after it
/// keeping `n - C` uses. Returns what the server should report clearing, or
/// `None` with the body untouched: nothing cleared yet, no beta or field, the
/// client clearing tool uses itself, or a history that doesn't line up.
pub fn apply(body: &mut Value, c: &Census, s: &CtxState) -> Option<Expected> {
    if s.cleared == 0 || s.client_managed || c.client_clears || !c.beta || !c.field || !lines_up(c, s) {
        return None;
    }
    let list = body.get_mut("context_management")?.get_mut("edits")?.as_array_mut()?;
    let mut thinking_turns = None;
    if s.cleared_th > 0
        && c.thinking_keep_all
        && let Some(e) = list.iter_mut().find(|e| block_type(e) == Some(CLEAR_THINKING))
    {
        e["keep"] = json!({ "type": "thinking_turns", "value": c.n_th - s.cleared_th });
        thinking_turns = Some(s.cleared_th);
    }
    let n = c.uses.len() as u64;
    let mut edit = json!({
        "type": CLEAR_TOOL_USES,
        "trigger": { "type": "tool_uses", "value": n - 1 },
        "keep": { "type": "tool_uses", "value": n - s.cleared }
    });
    if !c.exclude_tools.is_empty() {
        edit["exclude_tools"] = json!(c.exclude_tools);
    }
    if s.clear_inputs {
        edit["clear_tool_inputs"] = json!(true);
    }
    list.push(edit);
    Some(Expected { tool_uses: s.cleared, thinking_turns })
}

/// Why a thread stopped advancing.
pub const CTX_MISMATCH: &str = "ctx_mismatch";
pub const CACHE_REGRESSED: &str = "cache_regressed";
pub const REJECTED: &str = "rejected";

fn trip(s: &mut CtxState, why: &'static str) -> &'static str {
    s.frozen.get_or_insert_with(|| why.to_string());
    why
}

/// Record what the response to an edited request says, and return the
/// breakers it trips:
/// - `rejected`: a 400. The thread goes back to the client's own setting at once.
/// - `ctx_mismatch`: the server cleared something other than asked. No more advances.
/// - `cache_regressed`: a `steady` request (surely warm, cleared set unmoved)
///   read under 90% of the previous prompt from cache, so the edits seem to
///   cost cache. No more advances; the second time, back to the client's own
///   setting at the next cold moment. Checked whatever the mismatch check
///   says: a server that counts otherwise may also be moving its cut, and
///   only this breaker gets the thread out of paying for that.
///
/// A response cut short (Esc, or an error event mid-stream) never reached the
/// end where the edits are reported, so, like a failed request, it says
/// nothing. Neither does the first response from another `model` about the
/// cache: the prompt cache is per model.
pub fn settle(
    s: &mut CtxState,
    expected: Expected,
    steady: bool,
    model: Option<&str>,
    usage: Option<&Usage>,
    status: u16,
) -> Vec<&'static str> {
    if status == 400 {
        unpin(s);
        return vec![trip(s, REJECTED)];
    }
    let Some(u) = usage.filter(|u| (200..300).contains(&status) && u.stop_reason.is_some()) else {
        return Vec::new();
    };
    let mut tripped = Vec::new();
    if !expected.matches(u.applied_edits.as_ref()) {
        tripped.push(trip(s, CTX_MISMATCH));
    }
    let same_model = s.model.is_none() || s.model.as_deref() == model;
    if steady && same_model && (u.cache_read_input_tokens as f64) < MIN_CACHE_READ_SHARE * s.last_prompt as f64 {
        s.regressions += 1;
        tripped.push(trip(s, CACHE_REGRESSED));
    }
    s.last_prompt = u.input_tokens + u.cache_creation_input_tokens + u.cache_read_input_tokens;
    s.model = model.map(str::to_string);
    tripped
}

#[cfg(test)]
mod tests {
    use super::*;

    const BETAS: &str = "claude-code-20250219,context-management-2025-06-27";

    fn human(text: &str) -> Value {
        json!({ "role": "user", "content": [{ "type": "text", "text": text }] })
    }

    /// A tool loop: thinking and a call of `name`, then its ~6 KB result.
    fn tool_loop(id: &str, name: &str) -> [Value; 2] {
        [
            json!({ "role": "assistant", "content": [
                { "type": "thinking", "thinking": "", "signature": format!("sig-{id}") },
                { "type": "tool_use", "id": id, "name": name, "input": { "command": format!("cat {id}.rs") } }
            ] }),
            json!({ "role": "user", "content": [{ "type": "tool_result", "tool_use_id": id, "content": "x".repeat(6000) }] }),
        ]
    }

    /// One human prompt, `first` tool loops and a reply, then a second prompt
    /// and `second` tool loops (ids `a0..`, `b0..`).
    fn convo(first: usize, second: usize) -> Vec<Value> {
        let mut m = vec![human("look around")];
        m.extend((0..first).flat_map(|i| tool_loop(&format!("a{i}"), "Bash")));
        m.push(json!({ "role": "assistant", "content": [
            { "type": "thinking", "thinking": "", "signature": "sig-reply" }, { "type": "text", "text": "found it" }
        ] }));
        m.push(human("now fix it"));
        m.extend((0..second).flat_map(|i| tool_loop(&format!("b{i}"), "Bash")));
        m
    }

    fn body(messages: Vec<Value>) -> Value {
        json!({
            "model": "claude-opus-5-5",
            "tools": [{ "name": "Bash" }, { "name": "TodoWrite" }],
            "context_management": { "edits": [{ "type": CLEAR_THINKING, "keep": "all" }] },
            "messages": messages
        })
    }

    fn cfg(k: usize) -> CtxConfig {
        CtxConfig { window: KeepWindow { min: k, max: k }, ..Default::default() }
    }

    /// What the API would clear for `body`, under the counting convention
    /// above: tool use ids, and thinking turns.
    fn server_clears(body: &Value) -> (Vec<String>, u64) {
        let edits = body["context_management"]["edits"].as_array().unwrap();
        let results: HashMap<&str, &Value> = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(blocks)
            .filter(|b| block_type(b) == Some("tool_result"))
            .map(|b| (b["tool_use_id"].as_str().unwrap(), b))
            .collect();
        let mut ids = Vec::new();
        let mut thinking = 0;
        for e in edits {
            match e["type"].as_str() {
                Some(CLEAR_TOOL_USES) => {
                    let excluded: Vec<&str> =
                        e["exclude_tools"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
                    let uses: Vec<String> = body["messages"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .flat_map(blocks)
                        .filter(|b| {
                            block_type(b) == Some("tool_use") && !excluded.contains(&b["name"].as_str().unwrap())
                        })
                        .map(|b| b["id"].as_str().unwrap())
                        .filter(|id| counts_toward_keep(results.get(id).copied(), e["clear_tool_inputs"] == true))
                        .map(str::to_string)
                        .collect();
                    let (keep, trigger) =
                        (e["keep"]["value"].as_u64().unwrap(), e["trigger"]["value"].as_u64().unwrap());
                    if uses.len() as u64 > trigger {
                        ids = uses[..uses.len() - keep as usize].to_vec();
                    }
                }
                Some(CLEAR_THINKING) => {
                    if let Some(keep) = e["keep"]["value"].as_u64() {
                        thinking = census(body, BETAS, false).n_th - keep;
                    }
                }
                _ => {}
            }
        }
        (ids, thinking)
    }

    /// Advance at a cold moment, then apply.
    fn advanced(b: &mut Value, k: usize) -> (CtxState, Option<Expected>) {
        let mut s = CtxState::default();
        let c = census(b, BETAS, false);
        assert!(step(&mut s, &c, &cfg(k), true).is_some(), "expected an advance");
        let e = apply(b, &c, &s);
        (s, e)
    }

    #[test]
    fn counts_parallel_uses_redacted_thinking_and_error_and_image_results() {
        let messages = vec![
            human("go"),
            json!({ "role": "assistant", "content": [{ "type": "redacted_thinking", "data": "abc" }] }),
            json!({ "role": "assistant", "content": [
                { "type": "thinking", "thinking": "", "signature": "s" },
                { "type": "tool_use", "id": "err", "name": "Bash", "input": {} },
                { "type": "tool_use", "id": "img", "name": "Read", "input": { "file_path": "/a.png" } },
                { "type": "tool_use", "id": "todo", "name": "TodoWrite", "input": {} }
            ] }),
            json!({ "role": "user", "content": [
                { "type": "tool_result", "tool_use_id": "err", "is_error": true, "content": "boom".repeat(100) },
                { "type": "tool_result", "tool_use_id": "img", "content": [{ "type": "image", "source": { "type": "base64", "data": "AAAA" } }] },
                { "type": "tool_result", "tool_use_id": "todo", "content": "ok" }
            ] }),
            json!({ "role": "assistant", "content": [{ "type": "tool_use", "id": "g", "name": "Grep", "input": {} }] }),
            json!({ "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "g", "content": "hit" }] }),
            json!({ "role": "assistant", "content": [{ "type": "text", "text": "done" }] }),
            human("thanks"),
        ];
        let c = census(&body(messages.clone()), BETAS, false);
        let ids: Vec<&str> = c.uses.iter().map(|u| u.id.as_str()).collect();
        assert_eq!(
            ids,
            ["err", "img"],
            "parallel uses one by one, error and image results included; 3-char results not"
        );
        assert!((c.uses[0].result_tokens - 1.78 * 400.0 / 4.0).abs() < 1e-9);
        assert_eq!(c.n_th, 2, "one turn per human prompt, the current one included");
        assert_eq!((c.th_before_human, c.uses_since_prev_human), (1, 2));
        let ids: Vec<String> = census(&body(messages), BETAS, true).uses.into_iter().map(|u| u.id).collect();
        assert_eq!(ids, ["err", "img", "g"], "with clear_tool_inputs every use counts");
        assert_eq!(c.exclude_tools, ["TodoWrite"]);
        assert!(c.beta && c.field && c.thinking_keep_all && !c.client_clears);
    }

    #[test]
    fn keep_stays_pinned_as_turns_are_appended() {
        let mut b = body(convo(6, 3));
        let (s, e) = advanced(&mut b, 4);
        assert_eq!(e, Some(Expected { tool_uses: 5, thinking_turns: Some(1) }));
        let pinned = server_clears(&b);
        assert_eq!(pinned.0, ["a0", "a1", "a2", "a3", "a4"]);
        assert_eq!(pinned.1, 1);
        for more in [1, 5, 20] {
            let mut later = body(convo(6, 3 + more));
            let mut st = s.clone();
            let c = census(&later, BETAS, false);
            assert_eq!(step(&mut st, &c, &cfg(4), false), None, "a warm request never advances");
            assert_eq!(apply(&mut later, &c, &st), e);
            assert_eq!(server_clears(&later), pinned, "{more} turns later");
            let edits = later["context_management"]["edits"].as_array().unwrap();
            assert_eq!(edits[1]["keep"]["value"], json!(9 + more - 5));
            assert_eq!(edits[0]["keep"]["value"], json!(1));
        }
        // New turns, from a prompt or from a reminder next to results, grow the thinking keep.
        let mut messages = convo(6, 3);
        messages.push(json!({ "role": "assistant", "content": [{ "type": "text", "text": "done" }] }));
        messages.push(human("one more thing"));
        messages.extend(tool_loop("c0", "Bash"));
        let reminder = json!({ "type": "text", "text": "<system-reminder>todos</system-reminder>" });
        messages.last_mut().unwrap()["content"].as_array_mut().unwrap().push(reminder);
        let mut later = body(messages);
        let c = census(&later, BETAS, false);
        assert_eq!(c.n_th, 4);
        assert_eq!(apply(&mut later, &c, &s), e);
        assert_eq!(later["context_management"]["edits"][0]["keep"]["value"], json!(3));
        assert_eq!(server_clears(&later), pinned);
    }

    #[test]
    fn clear_thinking_stays_first() {
        let mut b = body(convo(6, 3));
        advanced(&mut b, 4);
        let edits = b["context_management"]["edits"].as_array().unwrap();
        let types: Vec<&str> = edits.iter().filter_map(block_type).collect();
        assert_eq!(types, [CLEAR_THINKING, CLEAR_TOOL_USES]);
        assert_eq!(
            edits[1],
            json!({ "type": CLEAR_TOOL_USES, "trigger": { "type": "tool_uses", "value": 8 },
                    "keep": { "type": "tool_uses", "value": 4 }, "exclude_tools": ["TodoWrite"] })
        );
    }

    #[test]
    fn thinking_keep_is_at_least_one_at_a_fresh_turn_advance() {
        // The latest message is a new human prompt: every thinking turn is before it.
        let mut messages = convo(8, 0);
        messages.pop();
        messages.push(human("next ticket"));
        let mut b = body(messages);
        let c = census(&b, BETAS, false);
        assert_eq!((c.n_th, c.th_before_human), (2, 1));
        let (_, e) = advanced(&mut b, 4);
        assert_eq!(e.unwrap().thinking_turns, Some(1));
        assert_eq!(b["context_management"]["edits"][0]["keep"], json!({ "type": "thinking_turns", "value": 1 }));
    }

    #[test]
    fn the_body_is_untouched_before_the_first_advance_and_without_beta_or_field() {
        let original = body(convo(6, 3));
        let mut b = original.clone();
        let mut s = CtxState::default();
        let c = census(&b, BETAS, false);
        assert_eq!(step(&mut s, &c, &cfg(4), false), None);
        assert_eq!(apply(&mut b, &c, &s), None);
        assert_eq!(b, original, "warm, nothing cleared yet");

        let (pinned, _) = advanced(&mut original.clone(), 4);
        for (betas, field) in [("claude-code-20250219", true), (BETAS, false)] {
            let mut b = original.clone();
            if !field {
                b.as_object_mut().unwrap().remove("context_management");
            }
            let mut s = CtxState::default();
            let c = census(&b, betas, false);
            assert_eq!(step(&mut s, &c, &cfg(4), true), None, "no advance without the beta and field");
            assert_eq!(apply(&mut b, &c, &pinned), None, "a pinned thread passes through too");
            let mut expect = original.clone();
            if !field {
                expect.as_object_mut().unwrap().remove("context_management");
            }
            assert_eq!(b, expect);
        }
    }

    #[test]
    fn a_client_clear_tool_uses_edit_means_passthrough() {
        let mut b = body(convo(6, 3));
        let (mut s, _) = advanced(&mut b.clone(), 4);
        b["context_management"]["edits"]
            .as_array_mut()
            .unwrap()
            .push(json!({ "type": CLEAR_TOOL_USES, "keep": { "type": "tool_uses", "value": 3 } }));
        let original = b.clone();
        let c = census(&b, BETAS, false);
        assert_eq!(step(&mut s, &c, &cfg(4), true), None);
        assert_eq!(apply(&mut b, &c, &s), None);
        assert_eq!(b, original);
        // It stays the client's from then on, even when a request omits it.
        let mut next = body(convo(6, 4));
        let c = census(&next, BETAS, false);
        assert!(s.client_managed);
        assert_eq!(step(&mut s, &c, &cfg(4), true), None);
        assert_eq!(apply(&mut next, &c, &s), None);
    }

    #[test]
    fn a_moved_cleared_upto_id_resets_the_state() {
        let (mut s, _) = advanced(&mut body(convo(6, 3)), 4);
        assert_eq!(s.cleared_upto_id.as_deref(), Some("a4"));
        // Compacted: a summary, then new tool calls.
        let mut compacted = vec![human("summary of earlier work")];
        compacted.extend((0..9).flat_map(|i| tool_loop(&format!("c{i}"), "Bash")));
        let mut b = body(compacted);
        let original = b.clone();
        let c = census(&b, BETAS, false);
        assert_eq!(step(&mut s, &c, &cfg(4), false), None);
        assert_eq!((s.cleared, s.cleared_upto_id.as_deref(), s.cleared_th), (0, None, 0));
        assert_eq!(apply(&mut b, &c, &s), None);
        assert_eq!(b, original);
        // Rewound to before the cut: the same.
        let (mut s, _) = advanced(&mut body(convo(6, 3)), 4);
        assert!(!lines_up(&census(&body(convo(4, 0)), BETAS, false), &s));
        step(&mut s, &census(&body(convo(4, 0)), BETAS, false), &cfg(4), false);
        assert_eq!(s.cleared, 0);
    }

    #[test]
    fn count_tokens_and_warm_requests_never_advance() {
        let c = census(&body(convo(6, 3)), BETAS, false);
        assert_eq!(plan_advance(&c, &CtxState::default(), &cfg(4), false), None);
        assert!(plan_advance(&c, &CtxState::default(), &cfg(4), true).is_some());
    }

    #[test]
    fn exclude_tools_grows_when_a_new_tool_appears() {
        let mut messages = convo(6, 3);
        let before = census(&body(messages.clone()), BETAS, false).exclude_tools;
        messages.extend(tool_loop("m0", "mcp__linear__get_issue"));
        let after = census(&body(messages), BETAS, false);
        assert_eq!(before, ["TodoWrite"]);
        assert_eq!(after.exclude_tools, ["TodoWrite", "mcp__linear__get_issue"]);
        assert_eq!(after.uses.len(), 9, "an excluded tool's use isn't counted");
    }

    #[test]
    fn an_advance_needs_two_more_uses_and_5k_more_tokens() {
        let c = census(&body(convo(6, 3)), BETAS, false);
        let at = |cleared: u64| CtxState {
            cleared,
            cleared_upto_id: Some(format!("a{}", cleared - 1)),
            ..Default::default()
        };
        assert_eq!(plan_advance(&c, &at(4), &cfg(4), true), None, "one more use isn't enough");
        assert_eq!(plan_advance(&c, &at(3), &cfg(4), true).unwrap().cleared, 5);
        let mut small = body(convo(6, 3));
        for m in small["messages"].as_array_mut().unwrap() {
            for b in m["content"].as_array_mut().into_iter().flatten() {
                if b["type"] == "tool_result" {
                    b["content"] = json!("tiny");
                }
            }
        }
        assert_eq!(plan_advance(&census(&small, BETAS, false), &CtxState::default(), &cfg(4), true), None);
        // Clearing inputs too counts their tokens, and pins the choice.
        let inputs = CtxConfig { clear_inputs: true, ..cfg(4) };
        let a = plan_advance(&c, &CtxState::default(), &inputs, true).unwrap();
        assert!(
            a.clear_inputs && a.new_tokens > plan_advance(&c, &CtxState::default(), &cfg(4), true).unwrap().new_tokens
        );
    }

    #[test]
    fn the_keep_window_covers_the_previous_human_turn() {
        // 6 uses before the latest prompt and 3 after: K = 9 within 4..24.
        let c = census(&body(convo(6, 3)), BETAS, false);
        assert_eq!(c.uses_since_prev_human, 9);
        assert_eq!(plan_advance(&c, &CtxState::default(), &CtxConfig::default(), true), None, "K 9 >= n");
        let c = census(&body(convo(30, 3)), BETAS, false);
        assert_eq!(c.uses_since_prev_human, 33);
        assert_eq!(plan_advance(&c, &CtxState::default(), &CtxConfig::default(), true).unwrap().cleared, 33 - 24);
        let w = KeepWindow::default();
        assert_eq!((w.k(0), w.k(12), w.k(40)), (8, 12, 24));
        let pos = use_positions(&convo(2, 2));
        assert_eq!(pos["a1"], UsePos { pos: 1, since_prev_human: 2 });
        assert_eq!(pos["b1"], UsePos { pos: 3, since_prev_human: 4 });
    }

    /// A complete response to a 50,100-token prompt: `cache_read` of it from
    /// cache and the rest written.
    fn usage(cache_read: u64, applied: Option<(u64, u64)>) -> Usage {
        Usage {
            input_tokens: 100,
            cache_read_input_tokens: cache_read,
            cache_creation_input_tokens: 50_000 - cache_read,
            stop_reason: Some("tool_use".into()),
            applied_edits: applied.map(|(t, th)| AppliedEdits {
                cleared_tool_uses: t,
                cleared_thinking_turns: th,
                cleared_input_tokens: 1000,
            }),
            ..Default::default()
        }
    }

    const OPUS: Option<&str> = Some("claude-opus-5-5");

    #[test]
    fn breakers() {
        let (pinned, e) = advanced(&mut body(convo(6, 3)), 4);
        let e = e.unwrap();
        let ok = Some((5, 1));
        let none: Vec<&str> = Vec::new();

        let mut s = pinned.clone();
        assert_eq!(settle(&mut s, e, false, OPUS, Some(&usage(50_000, ok)), 200), none);
        assert_eq!(s.last_prompt, 50_100);
        assert_eq!(settle(&mut s, e, true, OPUS, Some(&usage(50_000, ok)), 200), none, "the cache held");
        assert_eq!(settle(&mut s, e, false, OPUS, Some(&usage(0, Some((4, 1)))), 200), [CTX_MISMATCH]);
        assert_eq!(settle(&mut s.clone(), e, false, OPUS, Some(&usage(0, None)), 200), [CTX_MISMATCH], "none reported");
        assert_eq!(s.frozen.as_deref(), Some(CTX_MISMATCH));
        assert_eq!(s.cleared, 5, "a frozen thread stays pinned");
        let c = census(&body(convo(6, 20)), BETAS, false);
        assert_eq!(step(&mut s, &c, &cfg(4), true), None, "and stops advancing");

        let mut s = pinned.clone();
        s.last_prompt = 50_000;
        assert_eq!(settle(&mut s, e, true, OPUS, Some(&usage(1000, ok)), 200), [CACHE_REGRESSED]);
        assert_eq!(settle(&mut s, e, true, OPUS, Some(&usage(1000, ok)), 200), [CACHE_REGRESSED]);
        assert_eq!((s.regressions, s.cleared), (2, 5));
        let c = census(&body(convo(6, 4)), BETAS, false);
        step(&mut s, &c, &cfg(4), false);
        assert_eq!(s.cleared, 5, "pinned until the cache is cold anyway");
        step(&mut s, &c, &cfg(4), true);
        assert_eq!((s.cleared, s.frozen.as_deref()), (0, Some(CACHE_REGRESSED)), "then the client's own setting");

        let mut s = pinned.clone();
        assert_eq!(settle(&mut s, e, true, OPUS, None, 529), none, "an overloaded request says nothing");
        assert_eq!(s, pinned);
        assert_eq!(settle(&mut s, e, true, OPUS, None, 400), [REJECTED]);
        assert_eq!((s.cleared, s.frozen.as_deref()), (0, Some(REJECTED)));
    }

    #[test]
    fn a_response_cut_short_says_nothing() {
        let (pinned, e) = advanced(&mut body(convo(6, 3)), 4);
        let mut s = pinned.clone();
        s.last_prompt = 50_000;
        // Esc, or an error event after message_start: usage, but no end, so no edits reported.
        let cut = Usage { stop_reason: None, applied_edits: None, ..usage(0, None) };
        let before = s.clone();
        assert!(settle(&mut s, e.unwrap(), true, OPUS, Some(&cut), 200).is_empty());
        assert_eq!(s, before, "neither breaker runs, and the previous prompt still counts");
        let c = census(&body(convo(6, 20)), BETAS, false);
        assert!(step(&mut s, &c, &cfg(4), true).is_some(), "the thread still advances");
    }

    #[test]
    fn a_mismatching_thread_still_counts_cache_misses() {
        // A server that counts otherwise may move its cut each request: every
        // response mismatches and misses the cache, and the thread must still
        // go back to the client's own setting.
        let (pinned, e) = advanced(&mut body(convo(6, 3)), 4);
        let e = e.unwrap();
        let mut s = pinned.clone();
        s.last_prompt = 50_000;
        for _ in 0..2 {
            let tripped = settle(&mut s, e, true, OPUS, Some(&usage(1000, Some((4, 1)))), 200);
            assert_eq!(tripped, [CTX_MISMATCH, CACHE_REGRESSED]);
        }
        assert_eq!((s.regressions, s.frozen.as_deref()), (2, Some(CTX_MISMATCH)));
        let c = census(&body(convo(6, 5)), BETAS, false);
        step(&mut s, &c, &cfg(4), true);
        assert_eq!(s.cleared, 0);
        assert_eq!(apply(&mut body(convo(6, 5)), &c, &s), None, "the client's own setting");
    }

    #[test]
    fn a_model_switch_is_no_cache_regression() {
        let (pinned, e) = advanced(&mut body(convo(6, 3)), 4);
        let e = e.unwrap();
        let mut s = pinned.clone();
        let sonnet = Some("claude-sonnet-5-5");
        assert!(settle(&mut s, e, false, OPUS, Some(&usage(50_000, Some((5, 1)))), 200).is_empty());
        // The other model has nothing of this thread cached, and switching
        // back may find the first one's expired: neither is a regression.
        assert!(settle(&mut s, e, true, sonnet, Some(&usage(0, Some((5, 1)))), 200).is_empty());
        assert!(settle(&mut s, e, true, OPUS, Some(&usage(0, Some((5, 1)))), 200).is_empty());
        assert_eq!((s.regressions, s.frozen.as_deref(), s.model.as_deref()), (0, None, OPUS));
        assert_eq!(settle(&mut s, e, true, OPUS, Some(&usage(1000, Some((5, 1)))), 200), [CACHE_REGRESSED]);
    }

    #[test]
    fn cold_warm_is_reserved() {
        assert_eq!("cold".parse::<ContextEdits>().unwrap(), ContextEdits::Cold);
        assert!("cold+warm".parse::<ContextEdits>().unwrap_err().to_string().contains("reserved"));
        assert!("warm".parse::<ContextEdits>().is_err());
    }
}
