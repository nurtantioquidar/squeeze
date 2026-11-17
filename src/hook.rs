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
pub fn run(stdin: &str, hint: ExpandHint, aggressive: bool) -> Option<String> {
    let input: Value = serde_json::from_str(stdin).ok()?;
    if let Some(log) = std::env::var_os("SQUEEZE_HOOK_LOG") {
        // Debugging aid only: records raw hook payloads (which contain tool output).
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(log) {
            let _ = writeln!(f, "{stdin}");
        }
    }
    let store = Store::open(Store::default_root()).ok()?;
    let cfg = if aggressive {
        TransformConfig::aggressive(hint)
    } else {
        TransformConfig { expand_hint: hint, ..Default::default() }
    };
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

/// The PreToolUse hook for squeeze's edit_many: approve it in the permission
/// modes where Claude Code approves its own Edit without asking (acceptEdits,
/// bypassPermissions). In any other mode it says nothing, so Claude Code asks.
pub fn pre_tool_use(stdin: &str) -> Option<String> {
    let input: Value = serde_json::from_str(stdin).ok()?;
    if input.get("tool_name")?.as_str()? != EDIT_TOOL {
        return None;
    }
    let mode = input.get("permission_mode")?.as_str()?;
    matches!(mode, "acceptEdits" | "bypassPermissions").then(|| {
        json!({ "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": format!("squeeze: {mode} accepts file edits"),
        }})
        .to_string()
    })
}

const EDIT_TOOL: &str = "mcp__squeeze__edit_many";

/// Add the PreToolUse hook that lets edit_many follow the permission mode as
/// Edit does (see [`pre_tool_use`]) to session settings.
pub fn add_edit_permission_hook(settings: &mut Value, exe: &Path) {
    let command = format!("{} hook pre-tool-use", shell_quote(&exe.display().to_string()));
    push_hook(
        settings,
        "PreToolUse",
        json!({ "matcher": EDIT_TOOL, "hooks": [{ "type": "command", "command": command, "timeout": 10 }] }),
    );
}

/// What squeeze asks Claude Code's compaction to keep. Every request after a
/// compaction re-reads the summary, so a short, task-focused one costs less;
/// anything left out can be re-read from disk.
pub const COMPACT_GUIDE: &str = "Write the summary for continuing this coding work, in at most about 1,500 words. \
Keep: the user's current request and any earlier requests still open; decisions made and why; files changed or \
created, with what changed; file paths and line ranges to revisit; the commands that build or test the work and \
their latest results; errors still unresolved; the next steps. Leave out raw tool output, file contents, diffs and \
logs (they can be read again) and finished work that no longer matters.";

/// The PreCompact hook: hand Claude Code [`COMPACT_GUIDE`] as custom
/// compaction instructions (merged with any the user gave to /compact).
pub fn pre_compact(stdin: &str) -> Option<String> {
    serde_json::from_str::<Value>(stdin).ok()?;
    Some(json!({ "newCustomInstructions": COMPACT_GUIDE }).to_string())
}

/// Add the PreCompact hook ([`pre_compact`]) to session settings.
pub fn add_compact_hook(settings: &mut Value, exe: &Path) {
    let command = format!("{} hook pre-compact", shell_quote(&exe.display().to_string()));
    push_hook(settings, "PreCompact", json!({ "hooks": [{ "type": "command", "command": command, "timeout": 10 }] }));
}

/// Switch the session's compression hook to `--profile aggressive`.
pub fn set_aggressive(settings: &mut Value) {
    for entry in settings.pointer_mut("/hooks/PostToolUse").and_then(Value::as_array_mut).into_iter().flatten() {
        for h in entry["hooks"].as_array_mut().into_iter().flatten() {
            if let Some(c) = h["command"].as_str()
                && c.contains(" hook post-tool-use")
                && !c.contains("--profile")
            {
                h["command"] = json!(format!("{c} --profile aggressive"));
            }
        }
    }
}

fn push_hook(settings: &mut Value, event: &str, entry: Value) {
    if !settings.is_object() {
        *settings = json!({});
    }
    let hooks = settings.as_object_mut().expect("object").entry("hooks").or_insert_with(|| json!({}));
    match hooks.get_mut(event).and_then(Value::as_array_mut) {
        Some(list) => list.push(entry),
        None => hooks[event] = json!([entry]),
    }
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

    #[test]
    fn compaction_guide_and_profile_settings() {
        let reply: Value = serde_json::from_str(&pre_compact(r#"{"trigger":"auto"}"#).unwrap()).unwrap();
        assert_eq!(reply["newCustomInstructions"], COMPACT_GUIDE);
        assert!(pre_compact("not json").is_none(), "a malformed payload changes nothing");

        let exe = Path::new("/bin/squeeze");
        let mut settings = session_settings(exe, true);
        set_aggressive(&mut settings);
        set_aggressive(&mut settings); // idempotent
        add_compact_hook(&mut settings, exe);
        let post = settings["hooks"]["PostToolUse"][0]["hooks"][0]["command"].as_str().unwrap();
        assert_eq!(post, "/bin/squeeze hook post-tool-use --hint mcp --profile aggressive");
        assert_eq!(settings["hooks"]["PreCompact"][0]["hooks"][0]["command"], "/bin/squeeze hook pre-compact");
    }

    #[test]
    fn edit_many_follows_the_permission_mode() {
        let input = |tool: &str, mode: &str| json!({ "tool_name": tool, "permission_mode": mode }).to_string();
        for mode in ["acceptEdits", "bypassPermissions"] {
            let reply: Value = serde_json::from_str(&pre_tool_use(&input(EDIT_TOOL, mode)).unwrap()).unwrap();
            assert_eq!(reply["hookSpecificOutput"]["permissionDecision"], "allow");
        }
        assert!(pre_tool_use(&input(EDIT_TOOL, "default")).is_none(), "Claude Code asks, as for Edit");
        assert!(pre_tool_use(&input(EDIT_TOOL, "plan")).is_none());
        assert!(pre_tool_use(&input("Bash", "acceptEdits")).is_none(), "only edit_many");

        let mut settings = session_settings(Path::new("/bin/squeeze"), true);
        add_edit_permission_hook(&mut settings, Path::new("/bin/squeeze"));
        assert_eq!(settings["hooks"]["PreToolUse"][0]["matcher"], EDIT_TOOL);
        assert!(settings["hooks"]["PostToolUse"].is_array(), "the compression hook stays");
    }

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
