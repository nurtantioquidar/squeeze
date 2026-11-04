//! A minimal MCP (Model Context Protocol) server over stdio: newline-delimited
//! JSON-RPC 2.0 with `initialize`, `ping`, `tools/list` and `tools/call`.

use crate::tools::Workspace;
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

const SUPPORTED_VERSIONS: &[&str] = &["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"];
const DEFAULT_VERSION: &str = "2025-06-18";

/// Shown to the model in Claude Code's system prompt. Written against what
/// agents actually do in large repos: read and search code through Bash.
pub const INSTRUCTIONS: &str = "squeeze's tools read and search code in fewer turns and fewer tokens than shell \
commands. Use them instead of Bash for reading and searching code: \
(1) Instead of cat, head, tail or sed -n 'a,bp' on source files, call read_many, and put every file and line range \
you need next into that one call instead of reading one per turn. \
(2) Before reading a large file in full, call outline to get its definitions with line ranges, then read only the \
ranges you need with read_many. \
(3) Instead of grep -rn, rg or find | xargs grep, call search: results are grouped by file, with context lines and a \
match budget. \
Keep using Bash for running tests, builds, type checks, git and other commands, and use Read on a file right before \
you Edit it (Edit requires it). Outputs marked [squeeze: ...] were compressed; call expand with the id given there \
to see the full original.";

/// [`INSTRUCTIONS`] when the server also edits files (`--edit-tool`).
pub const INSTRUCTIONS_WITH_EDIT: &str = "squeeze's tools read, search and change code in fewer turns and fewer tokens \
than shell commands or one-file-at-a-time tools. Read with read_many (every file and range you need next, in one \
call), outline a large file before reading it, search instead of grep -rn, and change files with edit_many (every \
edit you have ready, in one call; files must have been read with read_many). Keep using Bash for tests, builds, type \
checks and git. Outputs marked [squeeze: ...] were compressed; call expand with the id given there for the original.";

/// Built-in tools for features a coding session doesn't use (multi-agent
/// workflows, /loop wakeups, review reports, agent teams). Each is loaded in
/// full on every request, about 3.5k tokens together.
pub const TRIMMED_BUILTINS: &[&str] = &["Workflow", "ScheduleWakeup", "ReportFindings", "ListAgents"];

/// Claude Code's built-in file tools that `edit_many`, `read_many`, `search`
/// and `outline` replace when squeeze takes over file access.
pub const REPLACED_BUILTINS: &[&str] = &["Read", "Edit", "Write", "Grep", "Glob", "NotebookEdit", "MultiEdit"];

/// Appended to the system prompt of the agent and its subagents when squeeze
/// takes over file access.
pub const FILE_TOOLS_GUIDE: &str = "# File tools
Claude Code's Read, Edit, Write, Grep and Glob tools are turned off in this session. squeeze's MCP tools replace them:
- Read files with mcp__squeeze__read_many. Put every file and line range you need next into one call. For a large \
file, call mcp__squeeze__outline first and read only the ranges you need.
- Find code with mcp__squeeze__search (regex, grouped by file, with context lines). files_only lists just the \
matching files; without a pattern it lists files.
- Change files with mcp__squeeze__edit_many. Batch every edit you have ready into one call. Omit old_string to \
create a file; set overwrite to replace a whole file. Read a file with read_many before you edit it. Don't re-read a \
file only to check an edit: the reply says where each edit landed.
- Use Bash to run things: tests, builds, type checks, linters, formatters, git, package managers. Don't use Bash to \
read or search files (cat, head, tail, sed -n, awk, grep, rg, find, ls) or to write them (sed -i, cat > file, \
echo >, tee, scripts that rewrite files); use the tools above.";

/// Appended to the main agent's system prompt with [`FILE_TOOLS_GUIDE`].
pub const WORKING_GUIDE: &str = "# Working economically
- Run the narrowest check that answers your question first (the tests of the files you touched, a type check of \
one package), and pipe long output through tail or grep.
- Review your own changes with git diff --stat and the files you changed, not a full diff of the repository.
- Keep replies to the user short: under 120 words after each request, in 2-4 bullets covering what changed, how you \
checked it, and anything left open.";

/// With the scout: added to [`WORKING_GUIDE`].
pub const SCOUT_GUIDE: &str = "- For broad investigation that would take three or more reads or searches across \
files, delegate it to the Explore agent and work from its findings: its exploration stays out of your context.";

/// The scout: Claude Code's Explore subagent redefined to use squeeze's tools
/// on the session's own model (Claude Code's built-in Explore may use a smaller one).
pub fn scout_agents_json() -> String {
    serde_json::json!({ "Explore": {
        "description": "Read-only code scout for questions that would take three or more reads or searches across files. Returns the files, line ranges and findings that matter.",
        "prompt": "You are a read-only code scout. Answer the caller's question about the codebase with squeeze's tools: mcp__squeeze__search to find code, mcp__squeeze__outline for the structure of a large file, and mcp__squeeze__read_many to read the relevant ranges, several per call. Use Bash only for read-only commands such as git log; never change files. Finish in 3 to 6 tool calls unless the caller sets another budget. Reply with FILES (path and line ranges that matter), FINDINGS (what the code does, quoting only the lines the caller needs) and NEXT (what to read or change next).",
        "tools": ["Bash", "mcp__squeeze__read_many", "mcp__squeeze__search", "mcp__squeeze__outline"],
        "model": "inherit"
    }})
    .to_string()
}

/// Read-only squeeze tools, safe to pre-approve.
pub const READ_ONLY_TOOLS: &[&str] = &["read_many", "outline", "search", "expand"];

pub struct Server {
    ws: Workspace,
    edit_tool: bool,
}

impl Server {
    pub fn new(ws: Workspace, edit_tool: bool) -> Self {
        Self { ws, edit_tool }
    }

    /// Handle one line of input; returns the line to write back, if any.
    pub fn handle_line(&mut self, line: &str) -> Option<String> {
        match serde_json::from_str::<Value>(line) {
            Ok(Value::Array(batch)) => {
                let replies: Vec<Value> = batch.iter().filter_map(|m| self.handle(m)).collect();
                (!replies.is_empty()).then(|| Value::Array(replies).to_string())
            }
            Ok(msg) => self.handle(&msg).map(|v| v.to_string()),
            Err(_) => Some(error(Value::Null, -32700, "parse error").to_string()),
        }
    }

    fn handle(&mut self, msg: &Value) -> Option<Value> {
        // Responses to our own requests (we send none) and malformed messages are ignored.
        let method = msg.get("method").and_then(Value::as_str)?;
        let id = msg.get("id").cloned();
        let params = msg.get("params").cloned().unwrap_or_else(|| json!({}));
        let result = match method {
            "initialize" => Ok(self.initialize(&params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": self.tool_definitions() })),
            "tools/call" => Ok(self.call(&params)),
            m if m.starts_with("notifications/") => return None,
            m => Err(format!("method not found: {m}")),
        };
        let id = id?; // notifications get no reply
        Some(match result {
            Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
            Err(m) => error(id, -32601, &m),
        })
    }

    fn initialize(&self, params: &Value) -> Value {
        let requested = params.get("protocolVersion").and_then(Value::as_str).unwrap_or(DEFAULT_VERSION);
        let version = if SUPPORTED_VERSIONS.contains(&requested) { requested } else { DEFAULT_VERSION };
        json!({
            "protocolVersion": version,
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": { "name": "squeeze", "version": crate::VERSION },
            "instructions": if self.edit_tool { INSTRUCTIONS_WITH_EDIT } else { INSTRUCTIONS },
        })
    }

    fn call(&mut self, params: &Value) -> Value {
        let name = params.get("name").and_then(Value::as_str).unwrap_or("");
        let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
        let result = match name {
            "read_many" => self.ws.read_many(&args),
            "outline" => self.ws.outline(&args),
            "search" => self.ws.search(&args),
            "expand" => self.ws.expand(&args),
            "edit_many" if self.edit_tool => self.ws.edit_many(&args),
            other => Err(anyhow::anyhow!("unknown tool {other:?}")),
        };
        match result {
            Ok(text) => json!({ "content": [{ "type": "text", "text": text }] }),
            Err(e) => json!({ "content": [{ "type": "text", "text": format!("error: {e:#}") }], "isError": true }),
        }
    }

    fn tool_definitions(&self) -> Vec<Value> {
        let read_only = json!({ "readOnlyHint": true, "openWorldHint": false });
        let mut tools = vec![
            json!({
                "name": "read_many",
                "title": "Read many files",
                "description": "Read several files, or line ranges of them, in one call. Prefer this over separate Read calls whenever you need two or more files or ranges. Output uses the same `N<TAB>line` numbering as Read. Paths are relative to the project root or absolute. Claude Code's Edit tool still requires a Read of a file before editing it, so Read the specific file you are about to edit.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array", "minItems": 1, "maxItems": 50,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": { "type": "string" },
                                    "offset": { "type": "integer", "minimum": 1, "description": "First line to show (1-based)." },
                                    "limit": { "type": "integer", "minimum": 1, "maximum": 5000, "description": "Maximum lines to show (default 2000)." }
                                },
                                "required": ["path"]
                            }
                        }
                    },
                    "required": ["files"]
                },
                "annotations": read_only,
            }),
            json!({
                "name": "outline",
                "title": "Outline files",
                "description": "List the definitions (functions, classes, types, impls, headings) in one or more files with their line ranges, nested by scope. Use it before reading a large file, then read only the ranges you need with read_many. Supports Rust, Python, JS/TS, Go, Java/Kotlin/C#/Scala/Swift, C/C++, Ruby and Markdown.",
                "inputSchema": {
                    "type": "object",
                    "properties": { "paths": { "type": "array", "items": { "type": "string" }, "minItems": 1, "maxItems": 50 } },
                    "required": ["paths"]
                },
                "annotations": read_only,
            }),
            json!({
                "name": "search",
                "title": "Search code",
                "description": "Regex search across the project, honoring .gitignore. Results are grouped by file (path printed once) with optional context lines and a match budget, so output stays small. Use glob to restrict files (for example \"*.rs\" or \"src/**/*.ts\"). files_only lists matching files with match counts (like grep -l). Without a pattern it lists files (like find or ls -R).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Regular expression (Rust regex syntax). Omit to list files." },
                        "path": { "type": "string", "description": "Directory or file to search (default: project root)." },
                        "glob": { "type": "string" },
                        "ignore_case": { "type": "boolean" },
                        "fixed_string": { "type": "boolean", "description": "Treat pattern as a literal string." },
                        "context": { "type": "integer", "minimum": 0, "maximum": 10 },
                        "files_only": { "type": "boolean", "description": "List matching files with match counts instead of lines." },
                        "max_matches": { "type": "integer", "minimum": 1, "maximum": 2000, "description": "Default 100 matches (200 files when listing)." }
                    }
                },
                "annotations": read_only,
            }),
            json!({
                "name": "expand",
                "title": "Expand compressed output",
                "description": "Return the full original of a tool output that squeeze compressed. Use the id from the [squeeze: ...] note. Supports offset/limit to page through long outputs.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "offset": { "type": "integer", "minimum": 1 },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 10000 }
                    },
                    "required": ["id"]
                },
                "annotations": read_only,
            }),
        ];
        if self.edit_tool {
            tools.push(json!({
                "name": "edit_many",
                "title": "Edit many",
                "description": "Change files in one call: exact-string replacements across one or more files, new files, and full rewrites. Batch every edit you have ready into one call. Every edit is validated before any file is written. An existing file must have been read with read_many in this session; if it changed since (a formatter, a script), edits still apply when old_string matches exactly. Each old_string must match exactly once unless replace_all is true. To create a file, omit old_string. To replace a whole file, set overwrite. The reply says where each edit landed, so there is no need to re-read a file to check an edit.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "edits": {
                            "type": "array", "minItems": 1, "maxItems": 100,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": { "type": "string" },
                                    "old_string": { "type": "string", "description": "Exact text to replace. Omit to create a new file." },
                                    "new_string": { "type": "string" },
                                    "replace_all": { "type": "boolean" },
                                    "overwrite": { "type": "boolean", "description": "Replace the whole file with new_string." }
                                },
                                "required": ["path", "new_string"]
                            }
                        }
                    },
                    "required": ["edits"]
                },
                "annotations": { "readOnlyHint": false, "destructiveHint": true, "idempotentHint": false, "openWorldHint": false },
            }));
        }
        // Claude Code defers MCP tools behind tool search by default: the model
        // sees only their names and rarely loads them. These tools only pay off
        // if the model picks them over Bash and Read, so they are always loaded.
        for t in &mut tools {
            t["_meta"] = json!({ "anthropic/alwaysLoad": true });
        }
        tools
    }
}

fn error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

pub fn serve_stdio(mut server: Server) -> anyhow::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(reply) = server.handle_line(&line) {
            writeln!(stdout, "{reply}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    fn server(edit: bool) -> (tempfile::TempDir, Server) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello\n").unwrap();
        let ws = Workspace::new(dir.path(), false, Store::open(dir.path().join(".home")).unwrap()).unwrap();
        (dir, Server::new(ws, edit))
    }

    fn rpc(s: &mut Server, msg: Value) -> Value {
        serde_json::from_str(&s.handle_line(&msg.to_string()).expect("reply")).unwrap()
    }

    #[test]
    fn handshake_and_tools() {
        let (_d, mut s) = server(false);
        let init = rpc(
            &mut s,
            json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": { "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": { "name": "t", "version": "1" } } }),
        );
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(init["result"]["serverInfo"]["name"], "squeeze");
        assert!(s.handle_line(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).is_none());

        let list = rpc(&mut s, json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }));
        let names: Vec<&str> =
            list["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, READ_ONLY_TOOLS);
        assert!(list["result"]["tools"].as_array().unwrap().iter().all(|t| t["_meta"]["anthropic/alwaysLoad"] == true));

        let call = rpc(
            &mut s,
            json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": { "name": "read_many", "arguments": { "files": ["a.txt"] } } }),
        );
        assert!(call["result"]["content"][0]["text"].as_str().unwrap().contains("1\thello"));

        let bad = rpc(
            &mut s,
            json!({ "jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": { "name": "edit_many", "arguments": {} } }),
        );
        assert_eq!(bad["result"]["isError"], true);

        let unknown = rpc(&mut s, json!({ "jsonrpc": "2.0", "id": 5, "method": "resources/list" }));
        assert_eq!(unknown["error"]["code"], -32601);
    }

    #[test]
    fn negotiates_unknown_version_and_lists_edit_tool() {
        let (_d, mut s) = server(true);
        let init = rpc(
            &mut s,
            json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": { "protocolVersion": "1999-01-01" } }),
        );
        assert_eq!(init["result"]["protocolVersion"], DEFAULT_VERSION);
        let list = rpc(&mut s, json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }));
        assert_eq!(list["result"]["tools"].as_array().unwrap().len(), 5);
    }
}
