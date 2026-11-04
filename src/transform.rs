//! Request rewriting for `/v1/messages`.
//!
//! Only `tool_result` text inside `messages` is ever changed. `system`,
//! `tools`, `cache_control`, thinking blocks and every unknown field pass
//! through untouched, as Claude Code's gateway protocol requires.
//!
//! **Invariant:** the rewrite of message *i* depends only on messages `0..=i`
//! (plus an optional prune boundary that moves only when the prompt cache has
//! already expired). Claude Code resends the whole conversation every turn, so
//! this makes the rewritten prefix identical from request to request:
//! - the prompt cache keeps hitting, and
//! - preserved-thinking checks, which reject a request when earlier
//!   `messages` differ from the request that produced the thinking, keep passing.
//!
//! It also means the proxy needs no per-conversation state to stay consistent.

use crate::compress::{self, BashKind};
use crate::ctxmgmt::{self, KeepWindow};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::HashMap;

/// Stores originals of lossily compressed outputs so they can be expanded later.
pub trait BlobSink {
    fn put(&mut self, text: &str) -> String;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpandHint {
    /// The squeeze MCP server is attached: point at its `expand` tool.
    Mcp,
    /// No MCP server: point at the `squeeze expand` CLI.
    Cli,
}

#[derive(Clone, Debug)]
pub struct PruneConfig {
    /// Prune stale context, but only at a cold-cache moment (see `proxy`).
    pub enabled: bool,
    pub bash_min_bytes: usize,
    pub min_age_messages: usize,
    pub keep_head: usize,
    pub keep_tail: usize,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self { enabled: false, bash_min_bytes: 6000, min_age_messages: 20, keep_head: 15, keep_tail: 25 }
    }
}

#[derive(Clone, Debug)]
pub struct TransformConfig {
    /// Tool outputs smaller than this are left alone.
    pub min_bytes: usize,
    /// A rewrite must save at least this many bytes to be applied.
    pub min_saving_bytes: usize,
    pub log: compress::LogPolicy,
    pub dedup: bool,
    pub minify_json: bool,
    pub expand_hint: ExpandHint,
    pub prune: PruneConfig,
    /// With server-side context editing (`--context-edits`), the window of
    /// recent clearable tool uses an advance keeps. A rewrite then never
    /// points at an older result the server may have cleared: Claude Code's
    /// "file unchanged" Read results whose earlier Read falls outside it are
    /// refilled (see `plan_refills`), and an output is deduplicated only
    /// against a copy inside it.
    pub ctx_window: Option<KeepWindow>,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            min_bytes: 1500,
            min_saving_bytes: 300,
            log: compress::LogPolicy::default(),
            dedup: true,
            minify_json: true,
            expand_hint: ExpandHint::Cli,
            prune: PruneConfig::default(),
            ctx_window: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChangeKind {
    Compressed,
    Deduped,
    Pruned,
    /// An unchanged-file stand-in given its content back (costs `added_bytes`).
    Refilled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    pub tool_use_id: String,
    pub kind: ChangeKind,
    pub saved_bytes: usize,
    /// Bytes the rewrite added (a refill).
    pub added_bytes: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct TransformStats {
    pub tool_results: usize,
    pub compressed: usize,
    pub deduped: usize,
    pub pruned: usize,
    pub refilled: usize,
    pub bytes_before: usize,
    pub bytes_after: usize,
    #[serde(skip)]
    pub changes: Vec<Change>,
}

impl TransformStats {
    pub fn changed(&self) -> bool {
        !self.changes.is_empty()
    }

    fn record(&mut self, id: &str, kind: ChangeKind, before: usize, after: usize) {
        match kind {
            ChangeKind::Compressed => self.compressed += 1,
            ChangeKind::Deduped => self.deduped += 1,
            ChangeKind::Pruned => self.pruned += 1,
            ChangeKind::Refilled => self.refilled += 1,
        }
        self.bytes_before += before;
        self.bytes_after += after;
        self.changes.push(Change {
            tool_use_id: id.to_string(),
            kind,
            saved_bytes: before.saturating_sub(after),
            added_bytes: after.saturating_sub(before),
        });
    }
}

struct ToolUse {
    msg: usize,
    name: String,
    input: Value,
}

struct ResultLoc {
    msg: usize,
    block: usize,
    id: String,
}

const LOG_TOOLS: &[&str] = &["Bash", "BashOutput", "TaskOutput"];
const FILE_WRITE_TOOLS: &[&str] = &["Edit", "MultiEdit", "Write", "NotebookEdit"];
const SQUEEZE_TOOL_PREFIX: &str = "mcp__squeeze__";

pub fn transform(
    body: &mut Value,
    cfg: &TransformConfig,
    sink: &mut dyn BlobSink,
    boundary: Option<&str>,
) -> TransformStats {
    let mut stats = TransformStats::default();
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return stats;
    };
    let uses = index_tool_uses(messages);
    let results = locate_results(messages);
    let prune = match boundary {
        Some(b) if cfg.prune.enabled => plan_prune(messages, &uses, &results, b, &cfg.prune),
        _ => HashMap::new(),
    };
    let positions = cfg.ctx_window.map(|_| ctxmgmt::use_positions(messages)).unwrap_or_default();
    let refills = cfg.ctx_window.map(|w| plan_refills(messages, &uses, &results, &positions, w)).unwrap_or_default();

    // Dedup key -> clearable-use position of the latest copy sent in full.
    let mut seen: HashMap<String, usize> = HashMap::new();
    let pos = |id: &str| positions.get(id).map_or(0, |p| p.pos);
    // With context editing, a copy `K` or more clearable uses back may be
    // cleared by the time the agent looks (see `plan_refills`): the output
    // then goes in full and becomes the latest copy. Tools the server never
    // clears have no position, and always deduplicate.
    let in_reach = |id: &str, latest: usize| {
        let at = cfg.ctx_window.zip(positions.get(id));
        at.is_none_or(|(w, p)| p.pos.saturating_sub(latest) < w.k(p.since_prev_human))
    };
    for (ri, loc) in results.iter().enumerate() {
        stats.tool_results += 1;
        let Some(block) =
            messages.get_mut(loc.msg).and_then(|m| m.get_mut("content")).and_then(|c| c.get_mut(loc.block))
        else {
            continue;
        };
        if block.get("is_error").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        let Some(tu) = uses.get(&loc.id) else { continue };
        let before = text_len(block);
        if let Some(text) = refills.get(&ri) {
            // A copy of its own: never deduplicated against the Read it repeats.
            if set_text(block, text) {
                stats.record(&loc.id, ChangeKind::Refilled, before, text_len(block));
            }
            continue;
        }
        let dedups = cfg.dedup && before >= cfg.min_bytes && !tu.name.starts_with(SQUEEZE_TOOL_PREFIX);

        if let Some(action) = prune.get(&ri) {
            // A later identical call stays a reference to an older output that
            // was cut to its head and tail (the rest is one expand away), so
            // pruning doesn't resend it in full. A stale Read is different: the
            // later Read is now where its content lives.
            if dedups
                && matches!(action, PruneAction::OldOutput)
                && let Some(key) = dedup_key(tu, block)
            {
                seen.insert(key, pos(&loc.id));
            }
            if apply_prune(block, action, cfg, sink) {
                stats.record(&loc.id, ChangeKind::Pruned, before, text_len(block));
            }
            continue;
        }
        if before < cfg.min_bytes {
            continue;
        }
        if dedups && let Some(key) = dedup_key(tu, block) {
            if seen.get(&key).is_some_and(|&latest| in_reach(&loc.id, latest)) {
                if set_text(block, &dedup_stub(tu)) {
                    stats.record(&loc.id, ChangeKind::Deduped, before, text_len(block));
                }
                continue;
            }
            seen.insert(key, pos(&loc.id));
        }
        if compress_result(block, tu, cfg, sink) {
            stats.record(&loc.id, ChangeKind::Compressed, before, text_len(block));
        }
    }
    stats
}

fn index_tool_uses(messages: &[Value]) -> HashMap<String, ToolUse> {
    let mut uses = HashMap::new();
    for (mi, m) in messages.iter().enumerate() {
        if m.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(blocks) = m.get("content").and_then(Value::as_array) else { continue };
        for b in blocks {
            if b.get("type").and_then(Value::as_str) != Some("tool_use") {
                continue;
            }
            if let (Some(id), Some(name)) = (b.get("id").and_then(Value::as_str), b.get("name").and_then(Value::as_str))
            {
                let input = b.get("input").cloned().unwrap_or(Value::Null);
                uses.insert(id.to_string(), ToolUse { msg: mi, name: name.to_string(), input });
            }
        }
    }
    uses
}

fn locate_results(messages: &[Value]) -> Vec<ResultLoc> {
    let mut out = Vec::new();
    for (mi, m) in messages.iter().enumerate() {
        if m.get("role").and_then(Value::as_str) != Some("user") {
            continue;
        }
        let Some(blocks) = m.get("content").and_then(Value::as_array) else { continue };
        for (bi, b) in blocks.iter().enumerate() {
            if b.get("type").and_then(Value::as_str) == Some("tool_result")
                && let Some(id) = b.get("tool_use_id").and_then(Value::as_str)
            {
                out.push(ResultLoc { msg: mi, block: bi, id: id.to_string() });
            }
        }
    }
    out
}

/// Total length of the text parts of a tool_result.
pub(crate) fn text_len(block: &Value) -> usize {
    match block.get("content") {
        Some(Value::String(s)) => s.len(),
        Some(Value::Array(items)) => items
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .map(str::len)
            .sum(),
        _ => 0,
    }
}

fn for_each_text(block: &mut Value, mut f: impl FnMut(&mut String)) {
    match block.get_mut("content") {
        Some(Value::String(s)) => f(s),
        Some(Value::Array(items)) => {
            for item in items {
                if item.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(Value::String(s)) = item.get_mut("text")
                {
                    f(s);
                }
            }
        }
        _ => {}
    }
}

/// Replace a tool_result's text with `text`, keeping its shape: string content
/// stays a string; text-block content stays a single text block carrying the
/// original `cache_control`. Results with images or other blocks are left alone.
fn set_text(block: &mut Value, text: &str) -> bool {
    match block.get_mut("content") {
        Some(Value::String(s)) => {
            *s = text.to_string();
            true
        }
        Some(Value::Array(items)) => {
            if items.is_empty() || items.iter().any(|b| b.get("type").and_then(Value::as_str) != Some("text")) {
                return false;
            }
            let cache_control = items.iter().rev().find_map(|b| b.get("cache_control").cloned());
            let mut tb = json!({ "type": "text", "text": text });
            if let Some(cc) = cache_control {
                tb["cache_control"] = cc;
            }
            *items = vec![tb];
            true
        }
        _ => false,
    }
}

fn full_text(block: &Value) -> Option<String> {
    match block.get("content") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(items)) if items.iter().all(|b| b.get("type").and_then(Value::as_str) == Some("text")) => {
            Some(items.iter().filter_map(|b| b.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n"))
        }
        _ => None,
    }
}

/// Key for "same call, same output". Compares text, not JSON shape, so a
/// string result and a text-block result with the same text match. Results
/// containing images or other blocks are never deduplicated.
fn dedup_key(tu: &ToolUse, block: &Value) -> Option<String> {
    let text = full_text(block)?;
    Some(format!("{}\u{0}{}\u{0}{}", tu.name, tu.input, blake3::hash(text.as_bytes()).to_hex()))
}

fn input_str<'a>(tu: &'a ToolUse, key: &str) -> Option<&'a str> {
    tu.input.get(key).and_then(Value::as_str)
}

fn dedup_stub(tu: &ToolUse) -> String {
    let what = match (tu.name.as_str(), input_str(tu, "file_path"), input_str(tu, "command")) {
        ("Read", Some(path), _) => format!("the earlier Read of {path} with the same arguments"),
        (_, _, Some(cmd)) => format!("the earlier {} call `{}`", tu.name, crate::util::truncate_chars(cmd, 120)),
        _ => format!("an earlier {} call with the same input", tu.name),
    };
    format!("[squeeze: this output is identical to the output of {what}, shown above. Refer to that result.]")
}

fn footer(hint: ExpandHint, id: &str, lines_in: usize, lines_out: usize) -> String {
    let how = match hint {
        ExpandHint::Mcp => format!("call mcp__squeeze__expand with id \"{id}\""),
        ExpandHint::Cli => format!("run `squeeze expand {id}`"),
    };
    format!("[squeeze: output compressed from {lines_in} to {lines_out} lines; for the full output {how}]")
}

/// Compress one text part of a tool's output, or `None` to leave it alone.
/// Shared by the proxy (rewriting requests) and the PostToolUse hook
/// (rewriting the output before Claude Code stores it).
pub fn compress_tool_text(
    tool_name: &str,
    input: &Value,
    text: &str,
    cfg: &TransformConfig,
    sink: &mut dyn BlobSink,
) -> Option<String> {
    if text.len() < cfg.min_bytes {
        return None;
    }
    let new = if LOG_TOOLS.contains(&tool_name) {
        match compress::classify_bash(input.get("command").and_then(Value::as_str).unwrap_or("")) {
            BashKind::FileView => compress::clean(text),
            BashKind::Log => {
                let c = compress::compress_log(text, &cfg.log);
                if c.lossy {
                    let id = sink.put(text);
                    format!("{}\n{}", c.text, footer(cfg.expand_hint, &id, c.lines_in, c.lines_out))
                } else {
                    c.text
                }
            }
        }
    } else if cfg.minify_json && tool_name.starts_with("mcp__") && !tool_name.starts_with(SQUEEZE_TOOL_PREFIX) {
        compress::minify_json(text)?
    } else {
        return None;
    };
    (new.len() + cfg.min_saving_bytes <= text.len()).then_some(new)
}

fn compress_result(block: &mut Value, tu: &ToolUse, cfg: &TransformConfig, sink: &mut dyn BlobSink) -> bool {
    let mut changed = false;
    for_each_text(block, |text| {
        if let Some(new) = compress_tool_text(&tu.name, &tu.input, text, cfg, sink) {
            *text = new;
            changed = true;
        }
    });
    changed
}

/// How Claude Code answers a Read of a file it has already shown, unchanged since.
const UNCHANGED_READ: &[&str] = &[
    "File unchanged since last read.",
    "Wasted call \u{2014} file unchanged since your last Read.",
    "<system-reminder>This file is already in your context",
];

/// With server-side context editing, the earlier Read that an unchanged-file
/// stand-in points to may have been cleared by the time the agent looks. A
/// stand-in whose latest copy of that content (same path, offset and limit)
/// is `K` or more clearable tool uses back, outside the window an advance
/// keeps, gets the earlier Read's text instead. The cut an advance makes (all
/// but the newest `K`) only moves forward as the history grows, so a copy
/// fewer than `K` uses back is still there when the stand-in is first sent.
/// Each choice depends only on the messages up to the stand-in, so it's the
/// same on every request; the server's cleared set can't be consulted, since
/// it moves.
fn plan_refills(
    messages: &[Value],
    uses: &HashMap<String, ToolUse>,
    results: &[ResultLoc],
    positions: &HashMap<String, ctxmgmt::UsePos>,
    window: KeepWindow,
) -> HashMap<usize, String> {
    let block = |loc: &ResultLoc| &messages[loc.msg]["content"][loc.block];
    // Per path and range: where the latest copy of its content is, and the Read that returned it.
    let mut copies: HashMap<String, (usize, &ResultLoc)> = HashMap::new();
    let mut plan = HashMap::new();
    for (ri, loc) in results.iter().enumerate() {
        let (Some(tu), Some(at)) = (uses.get(&loc.id), positions.get(&loc.id)) else { continue };
        if tu.name != "Read" || block(loc).get("is_error").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        let first_text = match block(loc).get("content") {
            Some(Value::String(s)) => Some(s.as_str()),
            Some(Value::Array(items)) => items.first().and_then(|b| b.get("text")).and_then(Value::as_str),
            _ => None,
        };
        let key = format!("{}\u{0}{}\u{0}{}", tu.input["file_path"], tu.input["offset"], tu.input["limit"]);
        if !first_text.is_some_and(|t| UNCHANGED_READ.iter().any(|s| t.starts_with(s))) {
            copies.insert(key, (at.pos, loc));
        } else if let Some((latest, source)) = copies.get_mut(&key)
            && at.pos.saturating_sub(*latest) >= window.k(at.since_prev_human)
            && let Some(content) = full_text(block(source))
        {
            plan.insert(ri, content);
            *latest = at.pos;
        }
    }
    plan
}

#[derive(Debug, Clone)]
enum PruneAction {
    StaleRead { path: String, reason: &'static str },
    OldOutput,
}

/// Decide which results up to the boundary to prune. Only evidence at or
/// before the boundary is considered, so the plan is stable once made.
fn plan_prune(
    messages: &[Value],
    uses: &HashMap<String, ToolUse>,
    results: &[ResultLoc],
    boundary: &str,
    pc: &PruneConfig,
) -> HashMap<usize, PruneAction> {
    let mut plan = HashMap::new();
    let Some(bpos) = results.iter().position(|r| r.id == boundary) else {
        return plan;
    };
    let bmsg = results[bpos].msg;

    // File operations at or before the boundary: (message index, is_write, is_full_read, path).
    let ops: Vec<(usize, bool, bool, &str)> = uses
        .values()
        .filter(|tu| tu.msg <= bmsg)
        .filter_map(|tu| {
            let path = input_str(tu, "file_path").or_else(|| input_str(tu, "notebook_path"))?;
            if tu.name == "Read" {
                let full = tu.input.get("offset").is_none() && tu.input.get("limit").is_none();
                Some((tu.msg, false, full, path))
            } else if FILE_WRITE_TOOLS.contains(&tu.name.as_str()) {
                Some((tu.msg, true, false, path))
            } else {
                None
            }
        })
        .collect();

    for (ri, loc) in results[..=bpos].iter().enumerate() {
        let Some(tu) = uses.get(&loc.id) else { continue };
        let block = &messages[loc.msg]["content"][loc.block];
        if block.get("is_error").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        if tu.name == "Read" {
            let Some(path) = input_str(tu, "file_path") else { continue };
            let later: Vec<_> = ops.iter().filter(|(m, _, _, p)| *m > loc.msg && *p == path).collect();
            let reason = if later.iter().any(|(_, write, _, _)| *write) {
                Some("edited")
            } else if later.iter().any(|(_, _, full, _)| *full) {
                Some("read again")
            } else {
                None
            };
            if let Some(reason) = reason
                && text_len(block) >= 200
            {
                plan.insert(ri, PruneAction::StaleRead { path: path.to_string(), reason });
            }
        } else if LOG_TOOLS.contains(&tu.name.as_str())
            && loc.msg + pc.min_age_messages <= bmsg
            && text_len(block) >= pc.bash_min_bytes
        {
            plan.insert(ri, PruneAction::OldOutput);
        }
    }

    // Preserved-thinking guard: never rewrite content that precedes a thinking
    // block, or the API rejects it as "bound to a different conversation".
    if let Some(earliest) = plan.keys().map(|ri| results[*ri].msg).min() {
        let thinking_after = messages[earliest..=bmsg].iter().any(|m| {
            m.get("role").and_then(Value::as_str) == Some("assistant")
                && m.get("content").and_then(Value::as_array).is_some_and(|blocks| {
                    blocks.iter().any(|b| {
                        matches!(b.get("type").and_then(Value::as_str), Some("thinking" | "redacted_thinking"))
                    })
                })
        });
        if thinking_after {
            plan.clear();
        }
    }
    plan
}

fn apply_prune(block: &mut Value, action: &PruneAction, cfg: &TransformConfig, sink: &mut dyn BlobSink) -> bool {
    match action {
        PruneAction::StaleRead { path, reason } => set_text(
            block,
            &format!(
                "[squeeze: this earlier Read of {path} was pruned because the file was {reason} later in the \
                 conversation. Read it again if you need its contents.]"
            ),
        ),
        PruneAction::OldOutput => {
            let Some(text) = full_text(block) else { return false };
            let lines: Vec<&str> = text.split('\n').collect();
            let (h, t) = (cfg.prune.keep_head, cfg.prune.keep_tail);
            if lines.len() <= h + t + 5 {
                return false;
            }
            let id = sink.put(&text);
            let omitted = lines.len() - h - t;
            let new = format!(
                "{}\n[squeeze: {omitted} lines of this older output pruned]\n{}\n{}",
                lines[..h].join("\n"),
                lines[lines.len() - t..].join("\n"),
                footer(cfg.expand_hint, &id, lines.len(), h + t + 1)
            );
            set_text(block, &new)
        }
    }
}

/// Longest prompt-cache TTL requested anywhere in the body, in seconds. It
/// sets the price of a cache write, not when the conversation goes cold.
pub fn max_cache_ttl_secs(body: &Value) -> u64 {
    fn walk(v: &Value, best: &mut u64) {
        match v {
            Value::Object(map) => {
                if let Some(cc) = map.get("cache_control") {
                    let ttl = cc.get("ttl").and_then(Value::as_str).and_then(parse_ttl).unwrap_or(300);
                    *best = (*best).max(ttl);
                }
                map.values().for_each(|x| walk(x, best));
            }
            Value::Array(items) => items.iter().for_each(|x| walk(x, best)),
            _ => {}
        }
    }
    let mut best = 300;
    for key in ["system", "tools", "messages"] {
        if let Some(v) = body.get(key) {
            walk(v, &mut best);
        }
    }
    best
}

/// TTL of the last `cache_control` marker in `messages`, in seconds (300 when
/// it has none). That marker caches the conversation itself, so this is how
/// long the history stays cached between requests; a 1-hour `system` marker
/// keeps only the shared prefix warm.
pub fn message_cache_ttl_secs(body: &Value) -> u64 {
    fn marker(b: &Value) -> Option<&Value> {
        b.get("cache_control").filter(|c| c.is_object())
    }
    // Markers sit on content blocks, or on the text inside a tool_result.
    let last = body
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .rev()
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flat_map(|blocks| blocks.iter().rev())
        .find_map(|b| marker(b).or_else(|| b.get("content")?.as_array()?.iter().rev().find_map(marker)));
    last.and_then(|cc| cc.get("ttl").and_then(Value::as_str).and_then(parse_ttl)).unwrap_or(300)
}

fn parse_ttl(s: &str) -> Option<u64> {
    let (num, unit) = s.split_at(s.find(|c: char| !c.is_ascii_digit())?);
    let n: u64 = num.parse().ok()?;
    Some(match unit {
        "s" => n,
        "m" => n * 60,
        "h" => n * 3600,
        _ => return None,
    })
}

/// Id of the conversation's first tool call. It never changes once made, so
/// it names the conversation (see `proxy`).
pub fn first_tool_use_id(body: &Value) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;
    let id = messages
        .iter()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flatten()
        .find(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))?
        .get("id")?
        .as_str()?;
    Some(id.to_string())
}

/// Id of the last tool_result in the conversation (used as a prune boundary).
pub fn last_tool_result_id(body: &Value) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;
    locate_results(messages).last().map(|r| r.id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct MemSink(HashMap<String, String>);
    impl BlobSink for MemSink {
        fn put(&mut self, text: &str) -> String {
            let id = blake3::hash(text.as_bytes()).to_hex()[..24].to_string();
            self.0.insert(id.clone(), text.to_string());
            id
        }
    }

    fn tool_use(id: &str, name: &str, input: Value) -> Value {
        json!({ "role": "assistant", "content": [{ "type": "tool_use", "id": id, "name": name, "input": input }] })
    }

    fn tool_result(id: &str, content: Value) -> Value {
        json!({ "role": "user", "content": [{ "type": "tool_result", "tool_use_id": id, "content": content }] })
    }

    fn build_log() -> String {
        let mut s: String = (0..400).map(|i| format!("   Compiling dep{i} v0.{i}.0\n")).collect();
        s.push_str("error[E0425]: cannot find value `x` in this scope\n");
        s.push_str(&(0..300).map(|i| format!("note: trace line {i}\n")).collect::<String>());
        s
    }

    fn file_body(tag: &str) -> String {
        (1..=120).map(|i| format!("{i}\tlet {tag}_{i} = compute({i});\n")).collect()
    }

    /// A synthetic conversation exercising every rewrite path.
    fn conversation() -> Vec<Value> {
        vec![
            json!({ "role": "user", "content": [{ "type": "text", "text": "fix the build" }] }),
            tool_use("t1", "Bash", json!({ "command": "cargo build" })),
            tool_result("t1", json!(build_log())),
            tool_use("t2", "Read", json!({ "file_path": "/w/src/lib.rs" })),
            tool_result("t2", json!(file_body("a"))),
            tool_use("t3", "Bash", json!({ "command": "cat /w/src/lib.rs" })),
            tool_result(
                "t3",
                json!(file_body("a").lines().map(|l| format!("\x1b[38;5;12m{l}\x1b[0m\n")).collect::<String>()),
            ),
            tool_use("t4", "Read", json!({ "file_path": "/w/src/lib.rs" })),
            tool_result(
                "t4",
                json!([{ "type": "text", "text": file_body("a"), "cache_control": { "type": "ephemeral", "ttl": "1h" } }]),
            ),
            tool_use("t5", "mcp__other__query", json!({ "q": "x" })),
            tool_result(
                "t5",
                json!([{ "type": "text", "text": serde_json::to_string_pretty(&json!({ "rows": (0..80).map(|i| json!({ "id": i, "name": format!("row{i}") })).collect::<Vec<_>>() })).unwrap() }]),
            ),
            tool_use("t6", "Bash", json!({ "command": "cargo build" })),
            json!({ "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "t6", "is_error": true, "content": build_log() }] }),
            json!({ "role": "assistant", "content": [{ "type": "text", "text": "Done." }] }),
        ]
    }

    fn run(messages: &[Value], cfg: &TransformConfig, boundary: Option<&str>) -> (Vec<Value>, TransformStats) {
        let mut body = json!({ "model": "m", "system": [{ "type": "text", "text": "sys" }], "messages": messages });
        let stats = transform(&mut body, cfg, &mut MemSink::default(), boundary);
        (body["messages"].as_array().unwrap().clone(), stats)
    }

    fn result_text(msg: &Value) -> String {
        match &msg["content"][0]["content"] {
            Value::String(s) => s.clone(),
            Value::Array(items) => items[0]["text"].as_str().unwrap().to_string(),
            _ => panic!("unexpected content"),
        }
    }

    #[test]
    fn rewrites_each_kind() {
        let cfg = TransformConfig::default();
        let (out, stats) = run(&conversation(), &cfg, None);
        // Build log: compressed, error kept, expand footer added.
        let log = result_text(&out[2]);
        assert!(log.len() < build_log().len() / 4);
        assert!(log.contains("error[E0425]"));
        assert!(log.contains("squeeze expand"));
        // First Read untouched.
        assert_eq!(result_text(&out[4]), file_body("a"));
        // `cat` output: lossless cleanup only (ANSI removed), content intact.
        assert_eq!(result_text(&out[6]), file_body("a"));
        // Second identical Read: deduplicated, shape and cache_control preserved.
        assert!(result_text(&out[8]).contains("identical to the output of the earlier Read of /w/src/lib.rs"));
        assert_eq!(out[8]["content"][0]["content"][0]["cache_control"]["ttl"], "1h");
        // Foreign MCP JSON: minified.
        assert!(!result_text(&out[10]).contains('\n'));
        // Errors are never touched.
        assert_eq!(out[12], conversation()[12]);
        assert_eq!(stats.deduped, 1);
        assert_eq!(stats.compressed, 3);
        assert_eq!(stats.tool_results, 6);
    }

    #[test]
    fn small_outputs_and_non_messages_fields_untouched() {
        let msgs = vec![tool_use("a", "Bash", json!({ "command": "ls" })), tool_result("a", json!("x\ny\n"))];
        let (out, stats) = run(&msgs, &TransformConfig::default(), None);
        assert_eq!(out, msgs);
        assert!(!stats.changed());
    }

    /// The core invariant: transforming a prefix gives exactly the prefix of
    /// transforming the whole conversation.
    #[test]
    fn rewrites_are_prefix_stable() {
        let cfg = TransformConfig::default();
        let full = conversation();
        let (full_out, _) = run(&full, &cfg, None);
        for k in 1..=full.len() {
            let (prefix_out, _) = run(&full[..k], &cfg, None);
            assert_eq!(prefix_out[..], full_out[..k], "prefix of length {k} diverged");
        }
    }

    #[test]
    fn transform_is_idempotent_on_its_output_shape() {
        // Running twice on the original gives identical bytes (determinism).
        let cfg = TransformConfig::default();
        let (a, _) = run(&conversation(), &cfg, None);
        let (b, _) = run(&conversation(), &cfg, None);
        assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b).unwrap());
    }

    #[test]
    fn dedup_stops_when_earlier_copy_is_gone() {
        // If Claude Code clears the earlier result, the later one must be sent in full.
        let cfg = TransformConfig::default();
        let mut convo = conversation();
        convo[4] = tool_result("t2", json!("[Old tool result content cleared]"));
        let (out, stats) = run(&convo, &cfg, None);
        assert_eq!(result_text(&out[8]), file_body("a"));
        assert_eq!(stats.deduped, 0);
    }

    fn prune_cfg() -> TransformConfig {
        let mut cfg = TransformConfig::default();
        cfg.prune.enabled = true;
        cfg.prune.min_age_messages = 2;
        cfg
    }

    fn prune_conversation() -> Vec<Value> {
        vec![
            json!({ "role": "user", "content": "go" }),
            tool_use("r1", "Read", json!({ "file_path": "/w/a.rs" })),
            tool_result("r1", json!(file_body("old"))),
            tool_use("e1", "Edit", json!({ "file_path": "/w/a.rs", "old_string": "x", "new_string": "y" })),
            tool_result("e1", json!("ok")),
            tool_use("b1", "Bash", json!({ "command": "ls -R" })),
            tool_result("b1", json!((0..400).map(|i| format!("./dir/file{i}.txt")).collect::<Vec<_>>().join("\n"))),
        ]
    }

    #[test]
    fn prunes_stale_reads_at_boundary() {
        let (out, stats) = run(&prune_conversation(), &prune_cfg(), Some("e1"));
        assert!(result_text(&out[2]).contains("pruned because the file was edited later"));
        assert_eq!(stats.pruned, 1);
        // Without a boundary nothing is pruned.
        let (out, stats) = run(&prune_conversation(), &prune_cfg(), None);
        assert_eq!(result_text(&out[2]), file_body("old"));
        assert_eq!(stats.pruned, 0);
    }

    #[test]
    fn prune_is_stable_as_conversation_grows() {
        let cfg = prune_cfg();
        let full = prune_conversation();
        let (full_out, _) = run(&full, &cfg, Some("e1"));
        for k in 5..=full.len() {
            let (prefix_out, _) = run(&full[..k], &cfg, Some("e1"));
            assert_eq!(prefix_out[..], full_out[..k]);
        }
    }

    #[test]
    fn prune_skips_when_thinking_follows() {
        let mut convo = prune_conversation();
        convo[3] = json!({ "role": "assistant", "content": [
            { "type": "thinking", "thinking": "", "signature": "sig" },
            { "type": "tool_use", "id": "e1", "name": "Edit", "input": { "file_path": "/w/a.rs", "old_string": "x", "new_string": "y" } }
        ] });
        let (out, stats) = run(&convo, &prune_cfg(), Some("e1"));
        assert_eq!(result_text(&out[2]), file_body("old"));
        assert_eq!(stats.pruned, 0);
    }

    #[test]
    fn pruning_an_old_output_keeps_later_copies_deduplicated() {
        let msgs = vec![
            json!({ "role": "user", "content": "go" }),
            tool_use("b1", "Bash", json!({ "command": "cargo build" })),
            tool_result("b1", json!(build_log())),
            tool_use("x", "Bash", json!({ "command": "ls" })),
            tool_result("x", json!("a\nb")),
            tool_use("b2", "Bash", json!({ "command": "cargo build" })),
            tool_result("b2", json!(build_log())),
        ];
        let (warm, _) = run(&msgs, &prune_cfg(), None);
        let (cold, stats) = run(&msgs, &prune_cfg(), Some("b2"));
        assert!(result_text(&cold[2]).contains("older output pruned"));
        assert!(result_text(&cold[6]).contains("identical to the output of the earlier Bash call"));
        assert_eq!(cold[6], warm[6], "the later copy is sent as before, not in full");
        assert_eq!((stats.pruned, stats.deduped), (1, 1));
    }

    #[test]
    fn a_stale_read_hands_its_content_to_the_later_read() {
        let msgs = vec![
            json!({ "role": "user", "content": "go" }),
            tool_use("r1", "Read", json!({ "file_path": "/w/a.rs" })),
            tool_result("r1", json!(file_body("a"))),
            tool_use("r2", "Read", json!({ "file_path": "/w/a.rs" })),
            tool_result("r2", json!(file_body("a"))),
        ];
        let (cold, _) = run(&msgs, &prune_cfg(), Some("r2"));
        assert!(result_text(&cold[2]).contains("pruned because the file was read again"));
        assert_eq!(result_text(&cold[4]), file_body("a"), "the content must still be somewhere");
    }

    const UNCHANGED: &str = "File unchanged since last read. The content from the earlier Read tool_result in this \
                             conversation is still current \u{2014} refer to that instead of re-reading.";
    const WASTED: &str =
        "Wasted call \u{2014} file unchanged since your last Read. Refer to that earlier tool_result instead.";

    /// Read a.rs, again at once, two Bash calls, two more unchanged Reads, two
    /// more Bash calls and an unchanged Read of another range.
    fn reread_conversation() -> Vec<Value> {
        vec![
            json!({ "role": "user", "content": "go" }),
            tool_use("r1", "Read", json!({ "file_path": "/w/a.rs" })),
            tool_result("r1", json!(file_body("a"))),
            tool_use("r2", "Read", json!({ "file_path": "/w/a.rs" })),
            tool_result("r2", json!(UNCHANGED)),
            tool_use("b1", "Bash", json!({ "command": "ls" })),
            tool_result("b1", json!("x")),
            tool_use("b2", "Bash", json!({ "command": "pwd" })),
            tool_result("b2", json!("y")),
            tool_use("r3", "Read", json!({ "file_path": "/w/a.rs" })),
            tool_result("r3", json!(WASTED)),
            tool_use("r4", "Read", json!({ "file_path": "/w/a.rs" })),
            tool_result("r4", json!(UNCHANGED)),
            tool_use("b3", "Bash", json!({ "command": "date" })),
            tool_result("b3", json!("z")),
            tool_use("b4", "Bash", json!({ "command": "id" })),
            tool_result("b4", json!("w")),
            tool_use("r5", "Read", json!({ "file_path": "/w/a.rs", "offset": 10 })),
            tool_result("r5", json!(UNCHANGED)),
        ]
    }

    fn refill_cfg() -> TransformConfig {
        TransformConfig { ctx_window: Some(KeepWindow { min: 3, max: 3 }), ..Default::default() }
    }

    #[test]
    fn refills_unchanged_reads_whose_copy_is_outside_the_window() {
        let (out, stats) = run(&reread_conversation(), &refill_cfg(), None);
        assert_eq!(result_text(&out[4]), UNCHANGED, "one use back: inside the window");
        assert_eq!(result_text(&out[10]), file_body("a"), "four uses back: refilled, and not deduplicated");
        assert_eq!(result_text(&out[12]), UNCHANGED, "the refill is the latest copy now");
        assert_eq!(result_text(&out[18]), UNCHANGED, "another range has no earlier copy");
        assert_eq!((stats.refilled, stats.deduped), (1, 0));
        let (out, _) = run(&reread_conversation(), &TransformConfig::default(), None);
        assert_eq!(out, reread_conversation(), "only with context editing");
    }

    #[test]
    fn refills_are_prefix_stable() {
        let full = reread_conversation();
        let (full_out, _) = run(&full, &refill_cfg(), None);
        for k in 1..=full.len() {
            let (prefix_out, _) = run(&full[..k], &refill_cfg(), None);
            assert_eq!(prefix_out[..], full_out[..k], "prefix of length {k} diverged");
        }
    }

    /// `cargo build` run three times and a foreign MCP query twice: the second
    /// build four clearable uses after the first (TodoWrite isn't one), the
    /// third right after the second.
    fn rebuild_conversation() -> Vec<Value> {
        let rows =
            json!({ "rows": (0..80).map(|i| json!({ "id": i, "name": format!("row{i}") })).collect::<Vec<_>>() });
        let rows = serde_json::to_string_pretty(&rows).unwrap();
        vec![
            json!({ "role": "user", "content": "go" }),
            tool_use("b1", "Bash", json!({ "command": "cargo build" })),
            tool_result("b1", json!(build_log())),
            tool_use("q1", "mcp__other__query", json!({ "q": "x" })),
            tool_result("q1", json!(rows)),
            tool_use("x1", "Bash", json!({ "command": "ls" })),
            tool_result("x1", json!("a")),
            tool_use("x2", "Read", json!({ "file_path": "/w/a.rs" })),
            tool_result("x2", json!("b")),
            tool_use("t", "TodoWrite", json!({ "todos": [] })),
            tool_result("t", json!("ok")),
            tool_use("x3", "Grep", json!({ "pattern": "x" })),
            tool_result("x3", json!("c")),
            tool_use("b2", "Bash", json!({ "command": "cargo build" })),
            tool_result("b2", json!(build_log())),
            tool_use("b3", "Bash", json!({ "command": "cargo build" })),
            tool_result("b3", json!(build_log())),
            tool_use("q2", "mcp__other__query", json!({ "q": "x" })),
            tool_result("q2", json!(rows)),
        ]
    }

    #[test]
    fn with_context_edits_dedup_points_only_at_copies_inside_the_window() {
        let stub = "this output is identical to the output of";
        let (out, stats) = run(&rebuild_conversation(), &refill_cfg(), None);
        assert!(!result_text(&out[14]).contains(stub), "four uses back: the server may have cleared it");
        assert!(result_text(&out[14]).contains("error[E0425]"));
        assert!(result_text(&out[16]).contains(stub), "one use after the copy sent in full");
        assert!(result_text(&out[18]).contains(stub), "an excluded tool's output is never cleared");
        assert_eq!(stats.deduped, 2);
        let (out, stats) = run(&rebuild_conversation(), &TransformConfig::default(), None);
        assert!(result_text(&out[14]).contains(stub), "without context editing, as before");
        assert_eq!(stats.deduped, 3);
        let full = rebuild_conversation();
        let (full_out, _) = run(&full, &refill_cfg(), None);
        for k in 1..=full.len() {
            assert_eq!(run(&full[..k], &refill_cfg(), None).0[..], full_out[..k], "prefix of length {k} diverged");
        }
    }

    #[test]
    fn a_refill_records_what_it_adds() {
        let (_, stats) = run(&reread_conversation(), &refill_cfg(), None);
        let c = stats.changes.iter().find(|c| c.kind == ChangeKind::Refilled).unwrap();
        assert_eq!((c.saved_bytes, c.added_bytes), (0, file_body("a").len() - WASTED.len()));
    }

    #[test]
    fn parses_cache_ttl() {
        let body = json!({ "system": [{ "type": "text", "text": "s", "cache_control": { "type": "ephemeral", "ttl": "1h" } }], "messages": [] });
        assert_eq!(max_cache_ttl_secs(&body), 3600);
        assert_eq!(max_cache_ttl_secs(&json!({ "messages": [] })), 300);
    }

    #[test]
    fn message_ttl_comes_from_the_last_message_marker() {
        let one_hour = json!({ "type": "ephemeral", "ttl": "1h" });
        let mut body = json!({
            "system": [{ "type": "text", "text": "s", "cache_control": one_hour }],
            "messages": [
                { "role": "user", "content": [{ "type": "text", "text": "a", "cache_control": one_hour }] },
                { "role": "assistant", "content": [{ "type": "tool_use", "id": "t", "name": "X", "input": { "cache_control": one_hour } }] },
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "t", "content": "x", "cache_control": { "type": "ephemeral" } },
                    { "type": "text", "text": "more" }
                ] }
            ]
        });
        // A 1-hour system marker keeps only the prefix; the history lives 5 minutes.
        assert_eq!(max_cache_ttl_secs(&body), 3600);
        assert_eq!(message_cache_ttl_secs(&body), 300);
        body["messages"][2]["content"][0] = json!({ "type": "tool_result", "tool_use_id": "t",
            "content": [{ "type": "text", "text": "x", "cache_control": one_hour }] });
        assert_eq!(message_cache_ttl_secs(&body), 3600);
        assert_eq!(message_cache_ttl_secs(&json!({ "messages": [{ "role": "user", "content": "hi" }] })), 300);
    }

    #[test]
    fn first_tool_use_id_is_the_oldest_call() {
        let mut msgs = conversation();
        msgs[3] = json!({ "role": "assistant", "content": [
            { "type": "thinking", "thinking": "", "signature": "s" },
            { "type": "tool_use", "id": "t2", "name": "Read", "input": {} }
        ] });
        assert_eq!(first_tool_use_id(&json!({ "messages": msgs })).as_deref(), Some("t1"));
        assert_eq!(first_tool_use_id(&json!({ "messages": msgs[3..] })).as_deref(), Some("t2"));
        assert_eq!(first_tool_use_id(&json!({ "messages": [{ "role": "user", "content": "hi" }] })), None);
    }
}
