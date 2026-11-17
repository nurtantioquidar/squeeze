//! Compaction at pauses (`squeeze proxy --compact-pauses`, API keys only).
//!
//! When a turn has ended (`end_turn`) and its thread then stays idle until
//! shortly before the prompt cache would expire, the proxy has the
//! conversation summarized in the background, with the credentials of the
//! request that ended the turn, held in memory only until then. From the next
//! request on, the messages between the first one and the turn's final reply
//! go out as that summary: the first message (Claude Code's instructions and
//! context), any `system`-role messages (Claude Code's environment and tool
//! announcements) and everything from the final reply on go verbatim, except
//! that the final reply loses its thinking, which belongs to the summarized
//! context. A later pause summarizes again, the summary included.
//!
//! The summarizing request is squeeze's own, so it leaves out Claude Code's
//! `claude-code-20250219` beta. Claude Code's cache entries for messages are
//! keyed apart from other requests' (measured 2026-11-17), so the summary
//! reads the system prompt and tools from the cache and the conversation as
//! uncached input; it puts no cache markers on the conversation, which would
//! only add cache writes nobody reads.
//!
//! With `--compact-warm` the proxy instead takes that beta out of the requests
//! it forwards for Claude Code (it removes Claude Code's own marker; it never
//! adds it to a request of its own), so Claude Code's cache entries are the
//! ones any client with the key reads. The summarizing request is then the
//! forwarded request itself with the instruction added at its end: same
//! system prompt, tools, messages, cache markers, thinking settings and betas,
//! so it reads the whole conversation from the cache ([`warm_summary_request`]).

use crate::auth;
use crate::util::short_hash;
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// `--compact-pauses`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactConfig {
    /// Seconds after the start of a turn's last request before summarizing;
    /// `None`: [`IDLE_LEAD_SECS`] before that request's cache would expire.
    pub idle_secs: Option<u64>,
    /// Summarize only a conversation (messages, not the system prompt and
    /// tools) of at least about this many tokens.
    pub min_tokens: u64,
    /// `--compact-warm`: forward Claude Code's requests without its beta and
    /// summarize from the cache ([`warm_summary_request`]).
    pub warm: bool,
}

impl Default for CompactConfig {
    fn default() -> Self {
        Self { idle_secs: None, min_tokens: 20_000, warm: false }
    }
}

/// The summarizing request goes out this long before the cache would expire,
/// so it still reads the system prompt and tools from it.
pub const IDLE_LEAD_SECS: u64 = 60;
/// Room for the summary and the thinking before it.
const SUMMARY_MAX_TOKENS: u64 = 16_000;
/// A summary shorter than this is taken for a failed one.
const MIN_SUMMARY_CHARS: usize = 200;
/// Claude Code's own beta, which squeeze's requests leave out.
pub const CLAUDE_CODE_BETA: &str = "claude-code-20250219";
/// The API's thinking-binding controls. A thinking block is bound to the
/// conversation before it, so the summarizing request, which cuts earlier
/// content to excerpts ([`skeleton`]), asks for mismatched blocks to be
/// dropped (`prefix_mismatch_behavior: drop_block`) instead of refused.
pub const THINKING_BINDING_BETA: &str = "thinking-binding-controls-2026-08-01";
/// The newest tool results and tool inputs that go to the summarizer whole;
/// older ones longer than [`SKELETON_MIN_CHARS`] go as excerpts.
const SKELETON_KEEP_RECENT: usize = 4;
const SKELETON_MIN_CHARS: usize = 800;
const SKELETON_HEAD_CHARS: usize = 300;
const SKELETON_TAIL_CHARS: usize = 150;
/// Fixed text the summarizer needn't read because it stays in the
/// conversation: the first message's long `<system-reminder>` blocks (project
/// instructions) and long `system`-role messages (the environment).
const FIXED_MIN_CHARS: usize = 1_500;
const FIXED_STUB: &str = "[left out of the summary's input: it stays in the conversation]";

/// What the summarizing request asks for, appended to the conversation.
pub const INSTRUCTION: &str = "Stop working on the task for a moment. Your context is about to be compacted: \
everything after the first message of this conversation will be replaced by a summary you write now, and the work \
continues from that summary alone (your final reply of this turn, which isn't shown here, stays after it). Write the \
summary now, in at most about 1,500 words, without calling any tool. Include:
1. Every user request and instruction so far, in order: quote the latest one in full, and any constraints word for \
word (for example, files not to modify).
2. The state of the work: what is done, what remains, and what you were doing last.
3. Files read, created or changed: their paths, what changed and why, and line ranges worth revisiting.
4. Commands that build, run or test the work, with their latest results, and errors still unresolved.
5. Decisions made and approaches ruled out, with the reasons.
6. The next steps.
Leave out file contents, diffs and raw tool output: they can be read again.";

/// Opens the summary where it stands in for the conversation.
const SUMMARY_OPEN: &str = "<compacted-conversation>\nThe conversation between this first message and the reply that \
follows was compacted to save context. It is summarized here; files and command output it mentions can be read \
again.\n\n";
const SUMMARY_CLOSE: &str = "\n</compacted-conversation>";

/// A thread's summaries, kept across restarts: a thread whose later turns
/// were generated after a summary must keep getting it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CompactState {
    /// The summary in use.
    pub active: Option<Summary>,
    /// One made at the latest pause, waiting for the request right after it:
    /// used from it on, or dropped (the active one stays).
    pub pending: Option<Summary>,
    /// Summaries made in this thread.
    pub compactions: u64,
    /// A summary in use stopped fitting the conversation (Claude Code rewound
    /// or compacted it): the history may hold thinking made after the summary,
    /// bound to it, so from then on requests ask the API to drop thinking
    /// whose conversation changed instead of refusing it ([`guard`]).
    pub guard: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Summary {
    /// The client's messages it was made over: it replaces
    /// `messages[1..covered]`, and `messages[covered]` is the turn's final reply.
    pub covered: usize,
    /// [`messages_hash`] of the client's first `covered` messages.
    pub prefix_hash: String,
    pub text: String,
}

impl Summary {
    /// The request continues the conversation this summary was made over; and
    /// with `fresh`, it's the request right after: the final reply, then user
    /// input only (and Claude Code's `system`-role notes).
    fn fits(&self, messages: &[Value], fresh: bool) -> bool {
        let n = self.covered;
        n >= 2
            && messages.len() > n + 1
            && role(&messages[n]) == Some("assistant")
            && (!fresh || messages[n + 1..].iter().all(|m| matches!(role(m), Some("user" | "system"))))
            && messages_hash(&messages[..n]) == self.prefix_hash
    }
}

fn role(m: &Value) -> Option<&str> {
    m.get("role").and_then(Value::as_str)
}

/// A hash of messages as the client sends them, cache markers aside (Claude
/// Code moves them from request to request).
pub fn messages_hash(messages: &[Value]) -> String {
    let parts: Vec<String> = messages
        .iter()
        .map(|m| {
            let mut m = m.clone();
            if let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut) {
                for b in blocks.iter_mut().filter_map(Value::as_object_mut) {
                    b.remove("cache_control");
                }
            }
            m.to_string()
        })
        .collect();
    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
    short_hash(&parts)
}

/// Put a thread's summary into a request's messages, if the request continues
/// the conversation the summary was made over. A pending summary is used only
/// by the request right after its pause (the turn's final reply, then the
/// user's next message) and from then on; otherwise it's dropped and the active
/// one, if any, still applies. With `update` false (count_tokens), the state is
/// only read. Returns whether the body changed.
pub fn apply(body: &mut Value, state: &mut CompactState, update: bool) -> bool {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else { return false };
    if let Some(p) = &state.pending {
        if p.fits(messages, true) {
            let (covered, text) = (p.covered, p.text.clone());
            if update {
                state.active = state.pending.take();
            }
            substitute(body, covered, &text);
            return true;
        }
        if update {
            state.pending = None;
        }
    }
    match &state.active {
        Some(a) if a.fits(messages, false) => {
            let (covered, text) = (a.covered, a.text.clone());
            substitute(body, covered, &text);
            true
        }
        Some(_) => {
            if update {
                state.active = None;
                state.guard = true;
            }
            false
        }
        None => false,
    }
}

/// For a thread with [`CompactState::guard`] set: ask the API to drop thinking
/// whose conversation changed (needs [`THINKING_BINDING_BETA`]). Returns
/// whether the body changed.
pub fn guard(body: &mut Value, state: &CompactState) -> bool {
    if !state.guard {
        return false;
    }
    let Some(t) = body.get_mut("thinking").and_then(Value::as_object_mut) else { return false };
    t.insert("block_binding".into(), json!({ "prefix_mismatch_behavior": "drop_block" }));
    true
}

/// `messages[1..covered]` out but for their `system`-role messages, the
/// summary appended to the first message, and the thinking taken out of the
/// final reply at `covered`.
fn substitute(body: &mut Value, covered: usize, summary: &str) {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else { return };
    let mut first = messages[0].clone();
    let mut content = match first.get("content") {
        Some(Value::String(t)) => vec![json!({ "type": "text", "text": t })],
        Some(Value::Array(blocks)) => blocks.clone(),
        _ => Vec::new(),
    };
    content.push(json!({ "type": "text", "text": format!("{SUMMARY_OPEN}{summary}{SUMMARY_CLOSE}") }));
    first["content"] = Value::Array(content);
    let mut reply = messages[covered].clone();
    if let Some(blocks) = reply.get_mut("content").and_then(Value::as_array_mut) {
        blocks.retain(|b| !matches!(b.get("type").and_then(Value::as_str), Some("thinking" | "redacted_thinking")));
        if blocks.is_empty() {
            blocks.push(json!({ "type": "text", "text": "(continued)" }));
        }
    }
    let system: Vec<Value> = messages[1..covered].iter().filter(|m| role(m) == Some("system")).cloned().collect();
    let rest = messages.split_off(covered + 1);
    *messages = std::iter::once(first).chain(system).chain([reply]).chain(rest).collect();
}

/// The first and last characters of a long text, with a note of what's left out.
fn excerpt(s: &str) -> Option<String> {
    let n = s.chars().count();
    if n <= SKELETON_MIN_CHARS {
        return None;
    }
    let head: String = s.chars().take(SKELETON_HEAD_CHARS).collect();
    let tail: String = s.chars().skip(n - SKELETON_TAIL_CHARS).collect();
    let left_out = n - SKELETON_HEAD_CHARS - SKELETON_TAIL_CHARS;
    Some(format!("{head}\n[... {left_out} characters left out of the summary's input ...]\n{tail}"))
}

/// What the summarizer reads: the conversation with older tool output and
/// large tool inputs (whole files written, long edits) cut to excerpts. The
/// summary is asked to leave these out anyway, and they are most of the
/// tokens it would otherwise read at full price.
fn skeleton(messages: &mut [Value]) {
    let (mut results, mut uses) = (Vec::new(), Vec::new());
    for (i, m) in messages.iter().enumerate() {
        for (j, b) in m.get("content").and_then(Value::as_array).into_iter().flatten().enumerate() {
            match b.get("type").and_then(Value::as_str) {
                Some("tool_result") => results.push((i, j)),
                Some("tool_use") => uses.push((i, j)),
                _ => {}
            }
        }
    }
    let cut = |s: &mut String| {
        if let Some(e) = excerpt(s) {
            *s = e;
        }
    };
    for &(i, j) in &results[..results.len().saturating_sub(SKELETON_KEEP_RECENT)] {
        match messages[i]["content"][j].get_mut("content") {
            Some(Value::String(s)) => cut(s),
            Some(Value::Array(items)) => {
                for it in items.iter_mut() {
                    match it.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            if let Some(Value::String(s)) = it.get_mut("text") {
                                cut(s);
                            }
                        }
                        Some(_) => *it = json!({ "type": "text", "text": "[left out of the summary's input]" }),
                        None => {}
                    }
                }
            }
            _ => {}
        }
    }
    for &(i, j) in &uses[..uses.len().saturating_sub(SKELETON_KEEP_RECENT)] {
        if let Some(input) = messages[i]["content"][j].get_mut("input").and_then(Value::as_object_mut) {
            for v in input.values_mut() {
                if let Value::String(s) = v {
                    cut(s);
                }
            }
        }
    }
}

/// Stub the fixed text in the summarizer's input ([`FIXED_MIN_CHARS`]).
fn stub_fixed(messages: &mut [Value]) {
    let long = |t: &str, reminder_only: bool| {
        t.chars().count() > FIXED_MIN_CHARS
            && (!reminder_only || (t.starts_with("<system-reminder>") && t.trim_end().ends_with("</system-reminder>")))
    };
    for (i, m) in messages.iter_mut().enumerate() {
        let reminder_only = match role(m) {
            Some("system") => false,
            _ if i == 0 => true,
            _ => continue,
        };
        match m.get_mut("content") {
            Some(Value::String(t)) if long(t, reminder_only) => *t = FIXED_STUB.into(),
            Some(Value::Array(blocks)) => {
                for blk in blocks.iter_mut().filter(|b| b.get("type").and_then(Value::as_str) == Some("text")) {
                    if let Some(Value::String(t)) = blk.get_mut("text")
                        && long(t, reminder_only)
                    {
                        *t = FIXED_STUB.into();
                    }
                }
            }
            _ => {}
        }
    }
}

/// The summarizing request: the request that ended the turn, as forwarded,
/// with older tool output cut to excerpts ([`skeleton`]) and fixed text
/// stubbed ([`stub_fixed`]), the instruction added to its last user message
/// (before any `system`-role messages that end it, which the API allows only
/// last or before an assistant message), no cache markers on the messages (the
/// system prompt's stay), mismatched thinking dropped rather than refused, low
/// effort, and no streaming. It needs [`THINKING_BINDING_BETA`].
pub fn summary_request(forwarded: &Value) -> Option<Value> {
    let mut b = forwarded.clone();
    let messages = b.get_mut("messages")?.as_array_mut()?;
    for m in messages.iter_mut() {
        if let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut) {
            for blk in blocks.iter_mut().filter_map(Value::as_object_mut) {
                blk.remove("cache_control");
            }
        }
    }
    skeleton(messages);
    stub_fixed(messages);
    let instruction = json!({ "type": "text", "text": INSTRUCTION });
    let at = messages.iter().rposition(|m| role(m) != Some("system")).map_or(0, |i| i + 1);
    match at.checked_sub(1).map(|i| &mut messages[i]) {
        Some(last) if role(last) == Some("user") => match last.get_mut("content") {
            Some(Value::Array(blocks)) => blocks.push(instruction),
            Some(Value::String(t)) => last["content"] = json!([{ "type": "text", "text": t.clone() }, instruction]),
            _ => return None,
        },
        _ => messages.insert(at, json!({ "role": "user", "content": [instruction] })),
    }
    b["stream"] = json!(false);
    b["max_tokens"] = json!(SUMMARY_MAX_TOKENS);
    if let Some(t) = b.get_mut("thinking").and_then(Value::as_object_mut) {
        t.insert("block_binding".into(), json!({ "prefix_mismatch_behavior": "drop_block" }));
    }
    if let Some(oc) = b.get_mut("output_config").and_then(Value::as_object_mut) {
        oc.insert("effort".into(), json!("low"));
    }
    Some(b)
}

/// The summarizing request for `--compact-warm`: the request that ended the
/// turn exactly as forwarded, with the instruction added to its last message
/// (a user message, or the `system`-role note Claude Code may end with; a
/// user message would not be allowed after that note) and no streaming.
/// Nothing that keys the prompt cache changes, so the conversation up to the
/// last cache marker is read from the cache Claude Code's request left.
pub fn warm_summary_request(forwarded: &Value) -> Option<Value> {
    let mut b = forwarded.clone();
    let messages = b.get_mut("messages")?.as_array_mut()?;
    let instruction = json!({ "type": "text", "text": INSTRUCTION });
    let last = messages.last_mut()?;
    if matches!(role(last), Some("user" | "system")) {
        match last.get_mut("content") {
            Some(Value::Array(blocks)) => blocks.push(instruction),
            Some(Value::String(t)) => last["content"] = json!([{ "type": "text", "text": t.clone() }, instruction]),
            _ => return None,
        }
    } else {
        messages.push(json!({ "role": "user", "content": [instruction] }));
    }
    b["stream"] = json!(false);
    b["max_tokens"] = json!(SUMMARY_MAX_TOKENS);
    Some(b)
}

/// An `anthropic-beta` value without Claude Code's beta, normalized the way
/// [`Credentials`] keeps it (trimmed, comma-joined); `None` if nothing is left.
pub fn without_claude_code_beta(value: &str) -> Option<String> {
    let betas: Vec<&str> =
        value.split(',').map(str::trim).filter(|b| !b.is_empty() && *b != CLAUDE_CODE_BETA).collect();
    (!betas.is_empty()).then(|| betas.join(","))
}

/// The summary in a non-streaming response, if it finished and says enough.
pub fn summary_text(response: &Value) -> Option<String> {
    if response.get("stop_reason").and_then(Value::as_str) != Some("end_turn") {
        return None;
    }
    let text: Vec<&str> = response
        .get("content")?
        .as_array()?
        .iter()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|b| b.get("text").and_then(Value::as_str))
        .collect();
    let text = text.join("\n").trim().to_string();
    (text.len() >= MIN_SUMMARY_CHARS).then_some(text)
}

/// What the summarizing request authenticates with: the API key of the
/// request that ended the turn. Kept in memory only, never written or logged.
#[derive(Clone)]
pub struct Credentials {
    api_key: String,
    version: Option<String>,
    betas: Option<String>,
}

impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials").field("api_key", &"<redacted>").field("betas", &self.betas).finish()
    }
}

impl Credentials {
    /// An API key in `x-api-key`, and nothing else: no OAuth token, and no
    /// gateway credential in `authorization`.
    pub fn from_headers(h: &HeaderMap) -> Option<Self> {
        if h.contains_key(axum::http::header::AUTHORIZATION) {
            return None;
        }
        let key = h.get("x-api-key")?.to_str().ok()?.trim().to_string();
        if key.is_empty() || auth::is_oauth_token(&key) {
            return None;
        }
        let header = |n: &str| h.get(n).and_then(|v| v.to_str().ok()).map(str::to_string);
        let all: Vec<&str> = h.get_all("anthropic-beta").iter().filter_map(|v| v.to_str().ok()).collect();
        let betas = without_claude_code_beta(&all.join(","));
        Some(Self { api_key: key, version: header("anthropic-version"), betas })
    }

    /// The request authenticated, with the client's betas (but Claude Code's)
    /// and `extra` ones.
    pub fn apply(&self, req: reqwest::RequestBuilder, extra: &[&str]) -> reqwest::RequestBuilder {
        let req = req
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", self.version.as_deref().unwrap_or("2023-06-01"));
        let betas: Vec<&str> =
            self.betas.as_deref().into_iter().flat_map(|b| b.split(',')).chain(extra.iter().copied()).collect();
        if betas.is_empty() { req } else { req.header("anthropic-beta", betas.join(",")) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(text: &str) -> Value {
        json!({ "role": "user", "content": [{ "type": "text", "text": text }] })
    }

    fn turn(i: usize) -> [Value; 2] {
        [
            json!({ "role": "assistant", "content": [
                { "type": "thinking", "thinking": "", "signature": format!("sig{i}") },
                { "type": "tool_use", "id": format!("t{i}"), "name": "Bash", "input": { "command": "ls" } }
            ] }),
            json!({ "role": "user", "content": [{ "type": "tool_result", "tool_use_id": format!("t{i}"), "content": "x" }] }),
        ]
    }

    fn system(text: &str) -> Value {
        json!({ "role": "system", "content": text })
    }

    /// A first message and Claude Code's environment message, three tool loops
    /// with a tool announcement among them, then (with `after`) the final reply
    /// and the next prompt.
    fn convo(after: bool) -> Vec<Value> {
        let mut m =
            vec![user("<system-reminder>instructions</system-reminder> do the ticket"), system("# Environment")];
        m.extend((0..3).flat_map(turn));
        m.insert(5, system("New deferred tools: Monitor"));
        if after {
            m.push(json!({ "role": "assistant", "content": [
                { "type": "thinking", "thinking": "", "signature": "final" }, { "type": "text", "text": "Done." }
            ] }));
            m.push(user("next ticket"));
        }
        m
    }

    fn summary() -> Summary {
        let before = convo(false);
        Summary { covered: before.len(), prefix_hash: messages_hash(&before), text: "S".repeat(300) }
    }

    fn pending() -> CompactState {
        CompactState { pending: Some(summary()), ..Default::default() }
    }

    #[test]
    fn the_hash_ignores_cache_markers() {
        let mut a = convo(false);
        let b = a.clone();
        a[8]["content"][0]["cache_control"] = json!({ "type": "ephemeral" });
        assert_eq!(messages_hash(&a), messages_hash(&b));
        a[8]["content"][0]["content"] = json!("y");
        assert_ne!(messages_hash(&a), messages_hash(&b));
    }

    #[test]
    fn the_summary_replaces_the_conversation_after_the_first_message() {
        let mut body = json!({ "messages": convo(true) });
        let mut st = pending();
        assert!(apply(&mut body, &mut st, true));
        assert_eq!((st.active.clone(), st.pending.clone()), (Some(summary()), None), "in use");
        let m = body["messages"].as_array().unwrap();
        assert_eq!(m.len(), 5);
        assert_eq!(m[0]["content"][0]["text"], "<system-reminder>instructions</system-reminder> do the ticket");
        let text = m[0]["content"][1]["text"].as_str().unwrap();
        assert!(text.starts_with(SUMMARY_OPEN) && text.contains(&"S".repeat(300)));
        assert_eq!((m[1].clone(), m[2].clone()), (system("# Environment"), system("New deferred tools: Monitor")));
        assert_eq!(
            m[3],
            json!({ "role": "assistant", "content": [{ "type": "text", "text": "Done." }] }),
            "thinking out"
        );
        assert_eq!(m[4], user("next ticket"));

        // Later requests in the conversation keep it, their own thinking included.
        let mut later = convo(true);
        later.extend(turn(9));
        let mut body = json!({ "messages": later.clone() });
        assert!(apply(&mut body, &mut st, true));
        let m = body["messages"].as_array().unwrap();
        assert_eq!(m.len(), 7);
        assert_eq!(m[5], later[later.len() - 2], "new turns verbatim");
    }

    #[test]
    fn a_pending_summary_is_used_right_after_its_pause_or_dropped() {
        // The request after the pause has more than the reply and the prompt: not this pause's.
        let mut later = convo(true);
        later.extend(turn(9));
        let original = json!({ "messages": later.clone() });
        let mut body = original.clone();
        let mut st = pending();
        assert!(!apply(&mut body, &mut st, true));
        assert_eq!((body, st), (original, CompactState::default()));

        // A dropped pending summary leaves the active one in use.
        let mut st = CompactState {
            active: Some(summary()),
            pending: Some(Summary { covered: 9, ..summary() }),
            compactions: 2,
            guard: false,
        };
        let mut body = json!({ "messages": later.clone() });
        assert!(apply(&mut body, &mut st, true));
        assert_eq!((st.active.is_some(), st.pending.is_none()), (true, true));
        assert_eq!(body["messages"].as_array().unwrap().len(), 7);

        // A history that no longer continues the summarized one drops even an
        // active summary, and from then on mismatched thinking is dropped, not refused.
        let mut st = CompactState { active: Some(summary()), ..Default::default() };
        let mut other = convo(true);
        other[3]["content"][0]["content"] = json!("rewritten");
        let mut body = json!({ "thinking": { "type": "adaptive" }, "messages": other });
        assert!(!apply(&mut body, &mut st, true));
        assert_eq!((st.active.clone(), st.guard), (None, true));
        assert!(guard(&mut body, &st));
        assert_eq!(body["thinking"]["block_binding"]["prefix_mismatch_behavior"], "drop_block");
        assert!(!guard(&mut json!({ "thinking": { "type": "adaptive" } }), &CompactState::default()));

        // count_tokens previews it without changing the state.
        let mut st = pending();
        let mut body = json!({ "messages": convo(true) });
        assert!(apply(&mut body, &mut st, false));
        assert_eq!(st, pending());
    }

    #[test]
    fn the_summarizing_request_reuses_the_turn_without_markers_or_streaming() {
        let mut messages = convo(false);
        messages[8]["content"][0]["cache_control"] = json!({ "type": "ephemeral" });
        let forwarded = json!({
            "model": "claude-opus-5-5", "stream": true, "max_tokens": 32000,
            "system": [{ "type": "text", "text": "sys", "cache_control": { "type": "ephemeral" } }],
            "output_config": { "effort": "high" }, "messages": messages
        });
        let r = summary_request(&forwarded).unwrap();
        assert_eq!((r["stream"].clone(), r["max_tokens"].clone()), (json!(false), json!(SUMMARY_MAX_TOKENS)));
        assert_eq!(r["output_config"]["effort"], "low");
        assert_eq!(r["system"], forwarded["system"], "the system prompt keeps its marker");
        assert!(!r["messages"].to_string().contains("cache_control"));
        let last = r["messages"].as_array().unwrap().last().unwrap();
        assert_eq!(last["content"][1]["text"], INSTRUCTION, "appended to the last user message");
        assert_eq!(r["messages"].as_array().unwrap().len(), 9);
    }

    #[test]
    fn the_summarizer_reads_older_tool_output_as_excerpts() {
        let mut messages = vec![user("go")];
        for i in 0..8 {
            messages.push(json!({ "role": "assistant", "content": [
                { "type": "tool_use", "id": format!("t{i}"), "name": "Write",
                  "input": { "file_path": format!("f{i}"), "content": "w".repeat(3000) } }
            ] }));
            messages.push(json!({ "role": "user", "content": [
                { "type": "tool_result", "tool_use_id": format!("t{i}"), "content": "r".repeat(5000) }
            ] }));
        }
        let r = summary_request(&json!({ "thinking": { "type": "adaptive" }, "messages": messages })).unwrap();
        let m = r["messages"].as_array().unwrap();
        let len = |v: &Value| v.as_str().unwrap().chars().count();
        // The four oldest of eight are cut, the newest four whole.
        assert!(len(&m[8]["content"][0]["content"]) < 600 && len(&m[7]["content"][0]["input"]["content"]) < 600);
        assert!(m[2]["content"][0]["content"].as_str().unwrap().contains("4550 characters left out"));
        assert_eq!(len(&m[10]["content"][0]["content"]), 5000);
        assert_eq!(len(&m[9]["content"][0]["input"]["content"]), 3000);
        assert_eq!(m[7]["content"][0]["input"]["file_path"], "f3", "short fields whole");
        assert_eq!(r["thinking"]["block_binding"]["prefix_mismatch_behavior"], "drop_block");
    }

    #[test]
    fn the_instruction_goes_before_system_messages_that_end_the_request() {
        // Claude Code often ends a request with a `system`-role note; a user
        // message after it is refused ("role 'system' must precede an
        // 'assistant' message or end the array").
        let mut messages = convo(false);
        messages.push(system("# Environment update\n - Primary working directory: /grafana"));
        let r = summary_request(&json!({ "messages": messages })).unwrap();
        let m = r["messages"].as_array().unwrap();
        assert_eq!(m.len(), 10);
        assert_eq!(m[9], system("# Environment update\n - Primary working directory: /grafana"), "still last");
        assert_eq!(m[8]["content"][1]["text"], INSTRUCTION, "on the last user message");

        // After an assistant message, a new user message goes before them.
        let mut messages = convo(true);
        messages.pop();
        messages.push(system("note"));
        let r = summary_request(&json!({ "messages": messages })).unwrap();
        let m = r["messages"].as_array().unwrap();
        assert_eq!((m.len(), m[10]["role"].clone(), m[11].clone()), (12, json!("user"), system("note")));
    }

    #[test]
    fn the_summarizer_skips_fixed_text_that_stays_in_the_conversation() {
        let reminder = format!("<system-reminder>\n{}\n</system-reminder>", "i".repeat(2000));
        let first = json!({ "role": "user", "content": [
            { "type": "text", "text": reminder },
            { "type": "text", "text": format!("{} ticket", "t".repeat(2000)) },
            { "type": "text", "text": format!("{SUMMARY_OPEN}{}{SUMMARY_CLOSE}", "s".repeat(2000)) }
        ] });
        let messages = vec![
            first,
            system(&format!("# Environment\n{}", "e".repeat(2000))),
            json!({ "role": "assistant", "content": [{ "type": "text", "text": "ok" }] }),
            json!({ "role": "user", "content": format!("<system-reminder>{}</system-reminder>", "u".repeat(2000)) }),
            system("short note"),
        ];
        let r = summary_request(&json!({ "messages": messages })).unwrap();
        let m = r["messages"].as_array().unwrap();
        assert_eq!(m[0]["content"][0]["text"], FIXED_STUB, "project instructions");
        assert!(m[0]["content"][1]["text"].as_str().unwrap().ends_with("ticket"), "the request stays");
        assert!(m[0]["content"][2]["text"].as_str().unwrap().contains("sss"), "the previous summary stays");
        assert_eq!(m[1]["content"], FIXED_STUB, "the environment");
        assert_eq!(m[3]["content"][0]["text"].as_str().unwrap().len(), 2000 + 35, "later reminders stay");
        assert_eq!(m[4], system("short note"));
    }

    #[test]
    fn a_pending_summary_fits_with_a_system_note_after_the_reply() {
        let mut messages = convo(true);
        messages.push(system("# Environment update"));
        let mut body = json!({ "messages": messages });
        let mut st = pending();
        assert!(apply(&mut body, &mut st, true));
        assert_eq!(body["messages"].as_array().unwrap().len(), 6);
    }

    #[test]
    fn the_warm_summarizing_request_changes_nothing_that_keys_the_cache() {
        let mut messages = convo(false);
        messages[8]["content"][0]["cache_control"] = json!({ "type": "ephemeral" });
        let forwarded = json!({
            "model": "claude-opus-5-5", "stream": true, "max_tokens": 32000,
            "thinking": { "type": "adaptive", "display": "omitted" }, "output_config": { "effort": "medium" },
            "system": [{ "type": "text", "text": "sys", "cache_control": { "type": "ephemeral" } }],
            "messages": messages
        });
        let r = warm_summary_request(&forwarded).unwrap();
        let mut expect = forwarded.clone();
        expect["messages"][8]["content"].as_array_mut().unwrap().push(json!({ "type": "text", "text": INSTRUCTION }));
        expect["stream"] = json!(false);
        expect["max_tokens"] = json!(SUMMARY_MAX_TOKENS);
        assert_eq!(r, expect);

        // A request that ends with Claude Code's system note gets the
        // instruction in that note: a user message may not follow it.
        let mut messages = convo(false);
        messages.push(system("# Environment update"));
        let r = warm_summary_request(&json!({ "messages": messages })).unwrap();
        let m = r["messages"].as_array().unwrap();
        assert_eq!(m.len(), 10);
        assert_eq!(
            m[9]["content"],
            json!([{ "type": "text", "text": "# Environment update" }, { "type": "text", "text": INSTRUCTION }])
        );
    }

    #[test]
    fn claude_codes_beta_comes_out_normalized() {
        assert_eq!(
            without_claude_code_beta("claude-code-20250219, context-management-2025-06-27,effort-2025-11-24")
                .as_deref(),
            Some("context-management-2025-06-27,effort-2025-11-24")
        );
        assert_eq!(without_claude_code_beta("claude-code-20250219"), None);
    }

    #[test]
    fn summary_text_needs_a_finished_reply() {
        let resp = |stop: &str, text: &str| {
            json!({ "stop_reason": stop, "content": [
            { "type": "thinking", "thinking": "", "signature": "s" }, { "type": "text", "text": text }
        ] })
        };
        assert_eq!(summary_text(&resp("end_turn", &"a".repeat(250))), Some("a".repeat(250)));
        assert_eq!(summary_text(&resp("max_tokens", &"a".repeat(250))), None);
        assert_eq!(summary_text(&resp("end_turn", "short")), None);
    }

    #[test]
    fn credentials_are_api_keys_only_and_drop_claude_codes_beta() {
        let headers = |pairs: &[(&'static str, &'static str)]| {
            let mut h = HeaderMap::new();
            for (k, v) in pairs {
                h.insert(*k, v.parse().unwrap());
            }
            h
        };
        let c = Credentials::from_headers(&headers(&[
            ("x-api-key", "sk-ant-api03-test"),
            ("anthropic-beta", "claude-code-20250219, context-management-2025-06-27,effort-2025-11-24"),
        ]))
        .unwrap();
        assert_eq!(c.betas.as_deref(), Some("context-management-2025-06-27,effort-2025-11-24"));
        assert!(!format!("{c:?}").contains("sk-ant"), "never printed");
        assert!(Credentials::from_headers(&headers(&[("x-api-key", "sk-ant-oat01-x")])).is_none());
        assert!(
            Credentials::from_headers(&headers(&[("x-api-key", "k"), ("authorization", "Bearer gw")])).is_none(),
            "a gateway credential is not an API key"
        );
    }
}
