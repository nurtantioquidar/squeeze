//! Offline replay: estimate what squeeze would have saved on past Claude Code
//! sessions, using the transcripts Claude Code keeps under
//! `~/.claude/projects`. Runs locally and sends nothing anywhere.
//!
//! The transcript format is internal to Claude Code and undocumented, so this
//! is best effort: sessions it can't parse are skipped. Subagent (sidechain)
//! threads are not analyzed.

use crate::sse::Usage;
use crate::store::Store;
use crate::transform::{self, BlobSink, TransformConfig};
use crate::util::{approx_tokens, read_price};
use anyhow::Result;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    pub file: PathBuf,
    pub requests: usize,
    pub tool_results: usize,
    pub compressed: usize,
    pub deduped: usize,
    pub bytes_saved: usize,
    /// Estimated input-side savings, in base-input-token equivalents.
    pub saved_input_eq: f64,
    /// Actual input-side cost from the recorded API usage, same units.
    pub actual_input_eq: f64,
}

impl Report {
    pub fn savings_pct(&self) -> Option<f64> {
        (self.actual_input_eq > 0.0).then(|| 100.0 * self.saved_input_eq / self.actual_input_eq)
    }
}

/// Computes blob ids without writing anything.
pub(crate) struct DryRun;
impl BlobSink for DryRun {
    fn put(&mut self, text: &str) -> String {
        Store::blob_id(text)
    }
}

/// Rebuild API-shaped message lists (split at compactions) and per-request usage.
fn parse(path: &Path) -> Result<(Vec<Vec<Value>>, Vec<Usage>)> {
    let mut segments: Vec<Vec<Value>> = vec![Vec::new()];
    let mut usage: Vec<(String, Usage)> = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let Ok(e) = serde_json::from_str::<Value>(&line?) else { continue };
        if e.get("isSidechain").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        let Some(msg) = e.get("message").filter(|m| m.is_object()) else { continue };
        match e.get("type").and_then(Value::as_str) {
            Some("user") => {
                if e.get("isCompactSummary").and_then(Value::as_bool) == Some(true) {
                    segments.push(Vec::new());
                }
                let seg = segments.last_mut().expect("at least one segment");
                push_user(seg, msg.get("content").cloned().unwrap_or(Value::Null));
            }
            Some("assistant") => {
                let id = msg.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                if !id.is_empty()
                    && seen_ids.insert(id.clone())
                    && let Some(u) = msg.get("usage")
                {
                    let mut parsed = Usage::default();
                    absorb(&mut parsed, u);
                    parsed.model = msg.get("model").and_then(Value::as_str).map(str::to_string);
                    usage.push((id.clone(), parsed));
                }
                let blocks = msg.get("content").and_then(Value::as_array).cloned().unwrap_or_default();
                let seg = segments.last_mut().expect("at least one segment");
                match seg.last_mut() {
                    Some(last) if last["role"] == "assistant" && last["_id"] == id.as_str() => {
                        if let Some(arr) = last["content"].as_array_mut() {
                            arr.extend(blocks);
                        }
                    }
                    _ => seg.push(json!({ "role": "assistant", "_id": id, "content": blocks })),
                }
            }
            _ => {}
        }
    }
    Ok((segments, usage.into_iter().map(|(_, u)| u).collect()))
}

fn absorb(u: &mut Usage, v: &Value) {
    let get = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
    u.input_tokens = get("input_tokens");
    u.output_tokens = get("output_tokens");
    u.cache_creation_input_tokens = get("cache_creation_input_tokens");
    u.cache_read_input_tokens = get("cache_read_input_tokens");
    if let Some(cc) = v.get("cache_creation") {
        u.cache_creation_1h = cc.get("ephemeral_1h_input_tokens").and_then(Value::as_u64).unwrap_or(0);
        u.cache_creation_5m = cc.get("ephemeral_5m_input_tokens").and_then(Value::as_u64).unwrap_or(0);
    }
}

fn push_user(seg: &mut Vec<Value>, content: Value) {
    let blocks = match content {
        Value::String(s) => vec![json!({ "type": "text", "text": s })],
        Value::Array(a) => a,
        _ => return,
    };
    match seg.last_mut() {
        Some(last) if last["role"] == "user" => {
            if let Some(arr) = last["content"].as_array_mut() {
                arr.extend(blocks);
            }
        }
        _ => seg.push(json!({ "role": "user", "content": blocks })),
    }
}

pub fn replay_file(path: &Path, cfg: &TransformConfig) -> Result<Option<Report>> {
    let (segments, usage) = parse(path)?;
    if usage.is_empty() {
        return Ok(None);
    }
    let write_mult = if usage.iter().any(|u| u.cache_creation_1h > 0) { 2.0 } else { 1.25 };
    let read = read_price(usage.iter().find_map(|u| u.model.as_deref()).unwrap_or(""));
    let mut report = Report {
        file: path.to_path_buf(),
        requests: usage.len(),
        actual_input_eq: usage.iter().map(|u| u.input_equivalent(read)).sum(),
        ..Default::default()
    };
    for seg in segments {
        if seg.is_empty() {
            continue;
        }
        // Message index of each tool_result, and how many requests follow it.
        let mut result_msg: HashMap<String, usize> = HashMap::new();
        for (i, m) in seg.iter().enumerate() {
            for b in m["content"].as_array().into_iter().flatten() {
                if b["type"] == "tool_result"
                    && let Some(id) = b["tool_use_id"].as_str()
                {
                    result_msg.insert(id.to_string(), i);
                }
            }
        }
        let assistant_idx: Vec<usize> =
            seg.iter().enumerate().filter(|(_, m)| m["role"] == "assistant").map(|(i, _)| i).collect();
        let mut body = json!({ "messages": seg });
        let stats = transform::transform(&mut body, cfg, &mut DryRun, None);
        report.tool_results += stats.tool_results;
        report.compressed += stats.compressed;
        report.deduped += stats.deduped;
        for c in &stats.changes {
            let Some(&mi) = result_msg.get(&c.tool_use_id) else { continue };
            let later = assistant_idx.iter().filter(|&&a| a > mi).count();
            if later == 0 {
                continue;
            }
            report.bytes_saved += c.saved_bytes;
            // Written to the cache once, then read on every later request.
            report.saved_input_eq += approx_tokens(c.saved_bytes) * (write_mult + read * (later - 1) as f64);
        }
    }
    Ok(Some(report))
}

/// All `*.jsonl` files under the given paths (files or directories).
pub fn find_transcripts(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_file() {
            out.push(p.clone());
            continue;
        }
        let walker = ignore::WalkBuilder::new(p).standard_filters(false).build();
        for entry in walker.flatten() {
            let path = entry.path();
            if entry.file_type().is_some_and(|t| t.is_file()) && path.extension().is_some_and(|e| e == "jsonl") {
                out.push(path.to_path_buf());
            }
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn replays_a_synthetic_transcript() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let log: String = (0..500).map(|i| format!("   Compiling dep{i} v1.0.0\n")).collect();
        let usage = json!({ "input_tokens": 10, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 20000, "output_tokens": 50, "cache_creation": { "ephemeral_1h_input_tokens": 1000, "ephemeral_5m_input_tokens": 0 } });
        let lines = vec![
            json!({ "type": "user", "message": { "role": "user", "content": "build it" } }),
            json!({ "type": "assistant", "message": { "id": "m1", "model": "claude-opus-5-5", "role": "assistant", "content": [{ "type": "thinking", "thinking": "" }], "usage": usage } }),
            json!({ "type": "assistant", "message": { "id": "m1", "role": "assistant", "content": [{ "type": "tool_use", "id": "t1", "name": "Bash", "input": { "command": "cargo build" } }], "usage": usage } }),
            json!({ "type": "user", "message": { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "t1", "content": log }] } }),
            json!({ "type": "assistant", "isSidechain": true, "message": { "id": "x", "role": "assistant", "content": [], "usage": usage } }),
            json!({ "type": "assistant", "message": { "id": "m2", "role": "assistant", "content": [{ "type": "text", "text": "ok" }], "usage": usage } }),
            json!({ "type": "assistant", "message": { "id": "m3", "role": "assistant", "content": [{ "type": "text", "text": "more" }], "usage": usage } }),
        ];
        let mut f = File::create(&path).unwrap();
        for l in lines {
            writeln!(f, "{l}").unwrap();
        }
        let r = replay_file(&path, &TransformConfig::default()).unwrap().unwrap();
        assert_eq!(r.requests, 3);
        assert_eq!(r.compressed, 1);
        assert!(r.bytes_saved > log.len() / 2);
        // Written once (2x) then read once (0.05x on Opus 5.5): two later requests.
        let expected = approx_tokens(r.bytes_saved) * 2.05;
        assert!((r.saved_input_eq - expected).abs() < 1e-6);
        assert!(r.savings_pct().unwrap() > 0.0);
    }
}
