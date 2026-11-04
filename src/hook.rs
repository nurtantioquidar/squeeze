//! Claude Code `PostToolUse` hook: compress bulky tool output before Claude
//! sees it, through Claude Code's documented hooks mechanism
//! (<https://code.claude.com/docs/en/hooks>).
//!
//! This is how squeeze works with **any** credential, including a Claude
//! subscription: Claude Code stays unmodified and talks to Anthropic directly,
//! and squeeze only runs as a local hook process. It never sees credentials or
//! network traffic.
//!
//! The hook fires once per tool call, and Claude Code stores the replaced
//! output in its conversation history, so every later request carries the
//! same bytes. Prompt caching and preserved thinking are unaffected.

use crate::store::Store;
use crate::transform::{BlobSink, ExpandHint, TransformConfig, compress_tool_text};
use crate::util::now_secs;
use serde_json::{Value, json};
use std::path::Path;

/// Tools whose output the hook inspects. Matching is a regex in Claude Code.
pub const MATCHER: &str = "Bash|mcp__.*";

/// Compress the tool output in a PostToolUse payload. Returns the hook's JSON
/// reply, or `None` to leave the output unchanged.
pub fn post_tool_use(input: &Value, cfg: &TransformConfig, sink: &mut dyn BlobSink) -> Option<(Value, usize, usize)> {
    let tool = input.get("tool_name")?.as_str()?;
    let tool_input = input.get("tool_input").cloned().unwrap_or(Value::Null);
    let mut response = input.get("tool_response")?.clone();
    let (mut before, mut after) = (0, 0);
    let mut apply = |text: &mut String| {
        if let Some(new) = compress_tool_text(tool, &tool_input, text, cfg, sink) {
            before += text.len();
            after += new.len();
            *text = new;
        }
    };
    match &mut response {
        // Bash: {"stdout": ..., "stderr": ..., "interrupted": ..., "isImage": ...}.
        // Only the text fields change, so the shape always matches.
        Value::Object(map) => {
            for key in ["stdout", "stderr"] {
                if let Some(Value::String(s)) = map.get_mut(key) {
                    apply(s);
                }
            }
            if let Some(Value::Array(items)) = map.get_mut("content") {
                for_text_items(items, &mut apply);
            }
        }
        // MCP tools: a content array or a plain string.
        Value::Array(items) => for_text_items(items, &mut apply),
        Value::String(s) => apply(s),
        _ => {}
    }
    (before > after).then(|| {
        let reply = json!({
            "hookSpecificOutput": { "hookEventName": "PostToolUse", "updatedToolOutput": response }
        });
        (reply, before, after)
    })
}

fn for_text_items(items: &mut [Value], f: &mut impl FnMut(&mut String)) {
    for item in items {
        if item.get("type").and_then(Value::as_str) == Some("text")
            && let Some(Value::String(s)) = item.get_mut("text")
        {
            f(s);
        }
    }
}

/// Entry point for `squeeze hook post-tool-use`. Never fails loudly: on any
/// problem it prints nothing and Claude Code keeps the original output.
pub fn run(stdin: &str, hint: ExpandHint) -> Option<String> {
    let input: Value = serde_json::from_str(stdin).ok()?;
    if let Some(log) = std::env::var_os("SQUEEZE_HOOK_LOG") {
        // Debugging aid only: records raw hook payloads (which contain tool output).
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(log) {
            let _ = writeln!(f, "{stdin}");
        }
    }
    let store = Store::open(Store::default_root()).ok()?;
    let cfg = TransformConfig { expand_hint: hint, ..Default::default() };
    let mut sink = store.clone();
    let (reply, before, after) = post_tool_use(&input, &cfg, &mut sink)?;
    let _ = store.append_record(&json!({
        "ts": now_secs(),
        "mode": "hook",
        "session": input.get("session_id"),
        "tool": input.get("tool_name"),
        "new_compressed": 1,
        "new_bytes_saved": before - after,
    }));
    Some(reply.to_string())
}

fn shell_quote(s: &str) -> String {
    if s.chars().all(|c| c.is_ascii_alphanumeric() || "/._-+:@".contains(c)) {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', r"'\''"))
    }
}

/// Session settings (for `claude --settings`) that register the hook and, when
/// the MCP server is attached, pre-approve squeeze's read-only tools.
pub fn session_settings(exe: &Path, with_mcp: bool) -> Value {
    let mut command = format!("{} hook post-tool-use", shell_quote(&exe.display().to_string()));
    if with_mcp {
        command.push_str(" --hint mcp");
    }
    let mut settings = json!({
        "hooks": {
            "PostToolUse": [{ "matcher": MATCHER, "hooks": [{ "type": "command", "command": command, "timeout": 30 }] }]
        }
    });
    if with_mcp {
        let allow: Vec<String> = crate::mcp::READ_ONLY_TOOLS.iter().map(|t| format!("mcp__squeeze__{t}")).collect();
        settings["permissions"] = json!({ "allow": allow });
    }
    settings
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct Mem(HashMap<String, String>);
    impl BlobSink for Mem {
        fn put(&mut self, text: &str) -> String {
            let id = Store::blob_id(text);
            self.0.insert(id.clone(), text.to_string());
            id
        }
    }

    fn log() -> String {
        let mut s: String = (0..300).map(|i| format!("   Compiling dep{i} v1.0.{i}\n")).collect();
        s.push_str("error: could not compile `app`\n");
        s
    }

    #[test]
    fn compresses_bash_stdout_and_keeps_shape() {
        let input = json!({
            "session_id": "s", "hook_event_name": "PostToolUse", "tool_name": "Bash",
            "tool_input": { "command": "cargo build" },
            "tool_response": { "stdout": log(), "stderr": "", "interrupted": false, "isImage": false, "extra": 1 }
        });
        let mut mem = Mem::default();
        let (reply, before, after) = post_tool_use(&input, &TransformConfig::default(), &mut mem).unwrap();
        let out = &reply["hookSpecificOutput"]["updatedToolOutput"];
        assert_eq!(reply["hookSpecificOutput"]["hookEventName"], "PostToolUse");
        assert!(out["stdout"].as_str().unwrap().contains("error: could not compile"));
        assert!(out["stdout"].as_str().unwrap().contains("[squeeze: output compressed"));
        assert_eq!(out["interrupted"], false);
        assert_eq!(out["extra"], 1, "unknown fields are preserved");
        assert!(after < before);
        assert_eq!(mem.0.values().next().unwrap(), &log(), "original stored for expand");
    }

    #[test]
    fn leaves_file_views_and_small_outputs_alone() {
        let input = json!({ "tool_name": "Bash", "tool_input": { "command": "cat big.log" }, "tool_response": { "stdout": log(), "stderr": "" } });
        assert!(post_tool_use(&input, &TransformConfig::default(), &mut Mem::default()).is_none());
        let input = json!({ "tool_name": "Bash", "tool_input": { "command": "make" }, "tool_response": { "stdout": "ok\n", "stderr": "" } });
        assert!(post_tool_use(&input, &TransformConfig::default(), &mut Mem::default()).is_none());
    }

    #[test]
    fn minifies_foreign_mcp_json() {
        let pretty = serde_json::to_string_pretty(&json!({ "rows": (0..400).collect::<Vec<_>>() })).unwrap();
        let input = json!({ "tool_name": "mcp__db__query", "tool_input": {}, "tool_response": [{ "type": "text", "text": pretty }] });
        let (reply, _, _) = post_tool_use(&input, &TransformConfig::default(), &mut Mem::default()).unwrap();
        assert!(!reply["hookSpecificOutput"]["updatedToolOutput"][0]["text"].as_str().unwrap().contains('\n'));
    }

    #[test]
    fn settings_register_hook_and_permissions() {
        let s = session_settings(Path::new("/opt/my tools/squeeze"), true);
        let cmd = s["hooks"]["PostToolUse"][0]["hooks"][0]["command"].as_str().unwrap();
        assert_eq!(cmd, "'/opt/my tools/squeeze' hook post-tool-use --hint mcp");
        assert_eq!(s["permissions"]["allow"][0], "mcp__squeeze__read_many");
        assert!(session_settings(Path::new("/bin/squeeze"), false).get("permissions").is_none());
    }
}
