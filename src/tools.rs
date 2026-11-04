//! Implementations of squeeze's MCP tools. They target fewer turns (several
//! files per call) and fewer tokens (outlines, grouped search, match budgets).

use crate::outline;
use crate::store::Store;
use crate::util::{is_binary, truncate_chars};
use anyhow::{Context, Result, anyhow, bail};
use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;
use regex::RegexBuilder;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_FILE_BYTES: u64 = 20 * 1024 * 1024;
const MAX_SEARCH_FILE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_OUTPUT_CHARS: usize = 250_000;
const MAX_LINE_CHARS: usize = 2000;

pub struct Workspace {
    root: PathBuf,
    allow_outside_root: bool,
    store: Store,
    /// Content hash of each file as last shown by `read_many`, for `edit_many`.
    reads: HashMap<PathBuf, blake3::Hash>,
}

fn arg_str<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::as_str)
}

fn arg_usize(v: &Value, k: &str) -> Option<usize> {
    v.get(k).and_then(Value::as_u64).map(|n| n as usize)
}

fn arg_bool(v: &Value, k: &str) -> bool {
    v.get(k).and_then(Value::as_bool).unwrap_or(false)
}

impl Workspace {
    pub fn new(root: &Path, allow_outside_root: bool, store: Store) -> Result<Self> {
        let root = root.canonicalize().with_context(|| format!("workspace root {} not found", root.display()))?;
        Ok(Self { root, allow_outside_root, store, reads: HashMap::new() })
    }

    fn resolve(&self, path: &str) -> Result<PathBuf> {
        let p = Path::new(path);
        let joined = if p.is_absolute() { p.to_path_buf() } else { self.root.join(p) };
        let canon = joined.canonicalize().map_err(|_| anyhow!("{path}: not found"))?;
        if !self.allow_outside_root && !canon.starts_with(&self.root) {
            bail!("{path}: outside the project root {}", self.root.display());
        }
        Ok(canon)
    }

    /// Resolve a path for a file that may not exist yet: its nearest existing
    /// ancestor must be inside the root, and the rest may not contain `..`.
    fn resolve_new(&self, path: &str) -> Result<PathBuf> {
        if let Ok(p) = self.resolve(path) {
            return Ok(p);
        }
        let p = Path::new(path);
        let joined = if p.is_absolute() { p.to_path_buf() } else { self.root.join(p) };
        let mut base = joined.as_path();
        let mut rest = Vec::new();
        while !base.exists() {
            let name = base.file_name().with_context(|| format!("{path}: invalid path"))?;
            rest.push(name.to_os_string());
            base = base.parent().with_context(|| format!("{path}: invalid path"))?;
        }
        if joined.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
            bail!("{path}: `..` is not allowed in a new file's path");
        }
        let mut canon = base.canonicalize()?;
        if !self.allow_outside_root && !canon.starts_with(&self.root) {
            bail!("{path}: outside the project root {}", self.root.display());
        }
        for part in rest.into_iter().rev() {
            canon.push(part);
        }
        Ok(canon)
    }

    fn display(&self, p: &Path) -> String {
        match p.strip_prefix(&self.root) {
            Ok(rel) if rel.as_os_str().is_empty() => ".".to_string(),
            Ok(rel) => rel.display().to_string(),
            Err(_) => p.display().to_string(),
        }
    }

    fn load(&self, path: &Path) -> Result<Vec<u8>> {
        let len = fs::metadata(path)?.len();
        if len > MAX_FILE_BYTES {
            bail!("{}: file is too large ({} MB); use search or outline", self.display(path), len / (1024 * 1024));
        }
        Ok(fs::read(path)?)
    }

    pub fn read_many(&mut self, args: &Value) -> Result<String> {
        let files = args.get("files").and_then(Value::as_array).context("`files` (an array) is required")?;
        if files.is_empty() || files.len() > 50 {
            bail!("`files` must contain 1 to 50 entries");
        }
        let mut out = String::new();
        for (n, f) in files.iter().enumerate() {
            let (path, offset, limit) = match f {
                Value::String(p) => (p.as_str(), 1, 2000),
                _ => (
                    arg_str(f, "path").context("each file needs a `path`")?,
                    arg_usize(f, "offset").unwrap_or(1).max(1),
                    arg_usize(f, "limit").unwrap_or(2000).clamp(1, 5000),
                ),
            };
            if out.len() > MAX_OUTPUT_CHARS {
                let _ = writeln!(
                    out,
                    "[squeeze: output budget reached; {} file(s) not shown, request them separately]",
                    files.len() - n
                );
                break;
            }
            if n > 0 {
                out.push('\n');
            }
            match self.read_one(path, offset, limit) {
                Ok(s) => out.push_str(&s),
                Err(e) => {
                    let _ = writeln!(out, "==> {path} <==\n[error: {e}]");
                }
            }
        }
        Ok(out)
    }

    fn read_one(&mut self, path: &str, offset: usize, limit: usize) -> Result<String> {
        let canon = self.resolve(path)?;
        if canon.is_dir() {
            bail!("is a directory");
        }
        let bytes = self.load(&canon)?;
        let shown = self.display(&canon);
        if is_binary(&bytes) {
            return Ok(format!("==> {shown} <==\n[binary file, {} bytes, not shown]\n", bytes.len()));
        }
        self.reads.insert(canon, blake3::hash(&bytes));
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<&str> = text.lines().collect();
        if lines.is_empty() {
            return Ok(format!("==> {shown} (empty file) <==\n"));
        }
        let start = (offset - 1).min(lines.len());
        let end = (start + limit).min(lines.len());
        let mut out = format!("==> {shown} (lines {}-{} of {}) <==\n", start + 1, end, lines.len());
        for (i, line) in lines[start..end].iter().enumerate() {
            let _ = writeln!(out, "{}\t{}", start + i + 1, truncate_chars(line, MAX_LINE_CHARS));
        }
        if end < lines.len() {
            let _ = writeln!(out, "[squeeze: {} more lines; use offset {} to continue]", lines.len() - end, end + 1);
        }
        Ok(out)
    }

    pub fn outline(&self, args: &Value) -> Result<String> {
        let paths: Vec<&str> = match args.get("paths") {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
            Some(Value::String(s)) => vec![s.as_str()],
            _ => bail!("`paths` (an array of file paths) is required"),
        };
        if paths.is_empty() || paths.len() > 50 {
            bail!("`paths` must contain 1 to 50 entries");
        }
        let mut out = String::new();
        for path in paths {
            let res = (|| -> Result<String> {
                let canon = self.resolve(path)?;
                let bytes = self.load(&canon)?;
                let shown = self.display(&canon);
                if is_binary(&bytes) {
                    return Ok(format!("==> {shown} <==\n[binary file]\n"));
                }
                let text = String::from_utf8_lossy(&bytes);
                let total = text.lines().count();
                let Some(items) = outline::outline(&canon, &text) else {
                    return Ok(format!(
                        "==> {shown} ({total} lines) <==\n[no outline support for this file type; use read_many with offset/limit]\n"
                    ));
                };
                let mut s = format!("==> {shown} ({total} lines) <==\n");
                if items.is_empty() {
                    s.push_str("[no definitions found]\n");
                }
                for it in items {
                    let range =
                        if it.start == it.end { it.start.to_string() } else { format!("{}-{}", it.start, it.end) };
                    let _ = writeln!(s, "{}{range}  {}", "  ".repeat(it.depth), it.text);
                }
                Ok(s)
            })();
            match res {
                Ok(s) => out.push_str(&s),
                Err(e) => {
                    let _ = writeln!(out, "==> {path} <==\n[error: {e}]");
                }
            }
        }
        Ok(out)
    }

    pub fn search(&self, args: &Value) -> Result<String> {
        let Some(pattern) = arg_str(args, "pattern").filter(|p| !p.is_empty()) else {
            return self.list_files(args);
        };
        let ctx = arg_usize(args, "context").unwrap_or(0).min(10);
        let max = arg_usize(args, "max_matches").unwrap_or(100).clamp(1, 1000);
        let files_only = arg_bool(args, "files_only");
        let source = if arg_bool(args, "fixed_string") { regex::escape(pattern) } else { pattern.to_string() };
        let re = RegexBuilder::new(&source)
            .case_insensitive(arg_bool(args, "ignore_case"))
            .size_limit(10 << 20)
            .build()
            .with_context(|| format!("invalid regex {pattern:?}"))?;
        let (base, walk) = self.walker(args)?;

        let mut out = String::new();
        let (mut shown, mut total, mut files_with) = (0usize, 0usize, 0usize);
        for entry in walk.flatten() {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            if entry.metadata().map_or(true, |m| m.len() > MAX_SEARCH_FILE_BYTES) {
                continue;
            }
            let Ok(bytes) = fs::read(entry.path()) else { continue };
            if is_binary(&bytes) {
                continue;
            }
            let text = String::from_utf8_lossy(&bytes);
            let lines: Vec<&str> = text.lines().collect();
            let hits: Vec<usize> = (0..lines.len()).filter(|&i| re.is_match(lines[i])).collect();
            if hits.is_empty() {
                continue;
            }
            files_with += 1;
            total += hits.len();
            if shown >= max {
                continue;
            }
            if files_only {
                shown += 1;
                let _ = writeln!(out, "{} ({} matches)", self.display(entry.path()), hits.len());
                continue;
            }
            let take: Vec<usize> = hits.iter().copied().take(max - shown).collect();
            shown += take.len();
            let hit_set: HashSet<usize> = take.iter().copied().collect();
            let _ = writeln!(out, "{}", self.display(entry.path()));
            let mut printed_to: Option<usize> = None;
            for &h in &take {
                let start = h.saturating_sub(ctx);
                let end = (h + ctx).min(lines.len() - 1);
                if let Some(p) = printed_to
                    && start > p + 1
                {
                    out.push_str("--\n");
                }
                let from = printed_to.map_or(start, |p| start.max(p + 1));
                for (i, line) in lines.iter().enumerate().take(end + 1).skip(from) {
                    let sep = if hit_set.contains(&i) { ':' } else { '-' };
                    let _ = writeln!(out, "{}{sep}{}", i + 1, truncate_chars(line.trim_end(), 300));
                }
                printed_to = Some(printed_to.map_or(end, |p| p.max(end)));
            }
            out.push('\n');
        }
        if total == 0 {
            return Ok(format!("No matches for /{pattern}/ under {}.", self.display(&base)));
        }
        let _ = write!(out, "{total} matches in {files_with} files");
        if files_only && files_with > shown {
            let _ = write!(out, " (listing the first {shown} files)");
        } else if !files_only && total > shown {
            let _ = write!(out, " (showing the first {shown}; narrow the pattern, path or glob to see more)");
        }
        out.push('\n');
        Ok(out)
    }

    /// A gitignore-aware walk under `path`, filtered by `glob`.
    fn walker(&self, args: &Value) -> Result<(PathBuf, ignore::Walk)> {
        let base = self.resolve(arg_str(args, "path").unwrap_or("."))?;
        let mut wb = WalkBuilder::new(&base);
        wb.sort_by_file_name(|a, b| a.cmp(b)).require_git(false);
        if let Some(glob) = arg_str(args, "glob") {
            let mut ob = OverrideBuilder::new(&base);
            ob.add(glob).with_context(|| format!("invalid glob {glob:?}"))?;
            wb.overrides(ob.build()?);
        }
        Ok((base, wb.build()))
    }

    /// `search` without a pattern: list files (like `find`/`ls -R`), honoring .gitignore.
    fn list_files(&self, args: &Value) -> Result<String> {
        let max = arg_usize(args, "max_matches").unwrap_or(200).clamp(1, 2000);
        let (base, walk) = self.walker(args)?;
        let mut out = String::new();
        let mut total = 0usize;
        for entry in walk.flatten() {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            total += 1;
            if total <= max {
                let _ = writeln!(out, "{}", self.display(entry.path()));
            }
        }
        if total == 0 {
            return Ok(format!("No files under {}.", self.display(&base)));
        }
        let _ = write!(out, "{total} files");
        if total > max {
            let _ = write!(out, " (showing the first {max}; narrow the path or glob)");
        }
        out.push('\n');
        Ok(out)
    }

    pub fn expand(&self, args: &Value) -> Result<String> {
        let id = arg_str(args, "id").context("`id` is required")?;
        let text = self.store.get_blob(id)?;
        let lines: Vec<&str> = text.split('\n').collect();
        let offset = arg_usize(args, "offset").unwrap_or(1).max(1);
        let limit = arg_usize(args, "limit").unwrap_or(2000).clamp(1, 10_000);
        let start = (offset - 1).min(lines.len());
        let end = (start + limit).min(lines.len());
        let mut out = format!("[squeeze: original output {id}, lines {}-{} of {}]\n", start + 1, end, lines.len());
        out.push_str(&lines[start..end].join("\n"));
        if end < lines.len() {
            let _ = write!(out, "\n[squeeze: {} more lines; call expand with offset {}]", lines.len() - end, end + 1);
        }
        Ok(out)
    }

    /// Apply several edits (exact-string replacements, new files, full
    /// rewrites), validated in full before any file is written.
    pub fn edit_many(&mut self, args: &Value) -> Result<String> {
        let edits = args.get("edits").and_then(Value::as_array).context("`edits` (an array) is required")?;
        if edits.is_empty() || edits.len() > 100 {
            bail!("`edits` must contain 1 to 100 entries");
        }
        struct Edit<'a> {
            old: &'a str,
            new: &'a str,
            all: bool,
            overwrite: bool,
        }
        let mut order: Vec<PathBuf> = Vec::new();
        let mut by_path: HashMap<PathBuf, Vec<Edit>> = HashMap::new();
        for (i, e) in edits.iter().enumerate() {
            let path = arg_str(e, "path").with_context(|| format!("edit {}: `path` is required", i + 1))?;
            let new = arg_str(e, "new_string").with_context(|| format!("edit {}: `new_string` is required", i + 1))?;
            let edit = Edit {
                old: arg_str(e, "old_string").unwrap_or(""),
                new,
                all: arg_bool(e, "replace_all"),
                overwrite: arg_bool(e, "overwrite"),
            };
            if !edit.overwrite && !edit.old.is_empty() && edit.old == edit.new {
                bail!("edit {}: `old_string` and `new_string` are identical", i + 1);
            }
            let canon = self.resolve_new(path)?;
            if canon.is_dir() {
                bail!("edit {}: {path} is a directory", i + 1);
            }
            if !by_path.contains_key(&canon) {
                order.push(canon.clone());
            }
            by_path.entry(canon).or_default().push(edit);
        }

        // (path, new content, existed, what was done)
        let mut staged: Vec<(PathBuf, String, bool, String)> = Vec::new();
        for path in &order {
            let shown = self.display(path);
            let existed = path.exists();
            let (mut text, mut note) = if existed {
                let bytes = self.load(path)?;
                let stale = match self.reads.get(path) {
                    None => bail!("{shown}: read it with read_many before editing it"),
                    Some(h) => *h != blake3::hash(&bytes),
                };
                let text = String::from_utf8(bytes).map_err(|_| anyhow!("{shown}: not valid UTF-8"))?;
                (text, if stale { " (the file had changed since you read it; matched exactly)" } else { "" })
            } else {
                (String::new(), "")
            };
            let mut lines = Vec::new();
            let mut action = None;
            for (i, e) in by_path[path].iter().enumerate() {
                let n = i + 1;
                if e.overwrite || (e.old.is_empty() && !existed && i == 0) {
                    text = e.new.to_string();
                    action = Some(if existed { "rewritten" } else { "created" });
                    note = "";
                    continue;
                }
                if e.old.is_empty() {
                    bail!(
                        "{shown}: edit {n} of this file: `old_string` is empty; set overwrite to replace the whole file"
                    );
                }
                let count = text.matches(e.old).count();
                if count == 0 {
                    bail!("{shown}: edit {n} of this file: old_string not found");
                }
                if count > 1 && !e.all {
                    bail!(
                        "{shown}: edit {n} of this file: old_string matches {count} times; add surrounding context or set replace_all"
                    );
                }
                let at = text.find(e.old).expect("counted above");
                lines.push(text[..at].matches('\n').count() + 1);
                text = if e.all { text.replace(e.old, e.new) } else { text.replacen(e.old, e.new, 1) };
            }
            let mut what = match action {
                Some(a) => format!("{a} ({} lines)", text.lines().count()),
                None => String::new(),
            };
            if !lines.is_empty() {
                let at: Vec<String> = lines.iter().map(usize::to_string).collect();
                if !what.is_empty() {
                    what.push_str(", then ");
                }
                let _ = write!(what, "{} edit(s) applied at line(s) {}", lines.len(), at.join(", "));
            }
            what.push_str(note);
            staged.push((path.clone(), text, existed, what));
        }

        let mut out = String::new();
        for (path, text, existed, what) in staged {
            if existed {
                write_preserving_permissions(&path, text.as_bytes())?;
            } else {
                if let Some(dir) = path.parent() {
                    fs::create_dir_all(dir)?;
                }
                fs::write(&path, text.as_bytes())?;
            }
            self.reads.insert(path.clone(), blake3::hash(text.as_bytes()));
            let _ = writeln!(out, "{}: {what}", self.display(&path));
        }
        Ok(out)
    }
}

fn write_preserving_permissions(path: &Path, data: &[u8]) -> Result<()> {
    let perms = fs::metadata(path)?.permissions();
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = path.with_file_name(format!(".{name}.squeeze-tmp"));
    fs::write(&tmp, data)?;
    fs::set_permissions(&tmp, perms)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn setup() -> (tempfile::TempDir, Workspace) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("proj");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/a.rs"), "fn main() {\n    let x = 1;\n    println!(\"{x}\");\n}\n").unwrap();
        fs::write(root.join("src/b.py"), "def f():\n    return 'needle'\n").unwrap();
        fs::write(root.join(".gitignore"), "target/\n").unwrap();
        fs::create_dir_all(root.join("target")).unwrap();
        fs::write(root.join("target/junk.rs"), "needle").unwrap();
        fs::write(dir.path().join("secret.txt"), "outside").unwrap();
        let store = Store::open(dir.path().join("home")).unwrap();
        let ws = Workspace::new(&root, false, store).unwrap();
        (dir, ws)
    }

    #[test]
    fn read_many_reads_ranges_and_reports_errors() {
        let (_d, mut ws) = setup();
        let out = ws
            .read_many(&json!({ "files": [{ "path": "src/a.rs", "offset": 2, "limit": 1 }, "src/b.py", "missing.rs", "../secret.txt"] }))
            .unwrap();
        assert!(out.contains("==> src/a.rs (lines 2-2 of 4) <==\n2\t    let x = 1;\n"));
        assert!(out.contains("[squeeze: 2 more lines; use offset 3 to continue]"));
        assert!(out.contains("==> src/b.py (lines 1-2 of 2) <==\n1\tdef f():"));
        assert!(out.contains("missing.rs: not found"));
        assert!(out.contains("outside the project root"));
    }

    #[test]
    fn search_respects_gitignore_and_groups() {
        let (_d, ws) = setup();
        let out = ws.search(&json!({ "pattern": "needle" })).unwrap();
        assert!(out.contains("src/b.py\n2:    return 'needle'"));
        assert!(!out.contains("target/junk.rs"));
        assert!(out.ends_with("1 matches in 1 files\n"));
        let out = ws.search(&json!({ "pattern": "let", "glob": "*.rs", "context": 1 })).unwrap();
        assert!(out.contains("1-fn main() {\n2:    let x = 1;\n3-"));
        assert!(ws.search(&json!({ "pattern": "zzz" })).unwrap().starts_with("No matches"));
    }

    #[test]
    fn edit_many_requires_fresh_read_and_is_all_or_nothing() {
        let (d, mut ws) = setup();
        let a = d.path().join("proj/src/a.rs");
        let edit = json!({ "edits": [{ "path": "src/a.rs", "old_string": "let x = 1;", "new_string": "let x = 2;" }] });
        assert!(ws.edit_many(&edit).unwrap_err().to_string().contains("read it with read_many"));

        ws.read_many(&json!({ "files": ["src/a.rs", "src/b.py"] })).unwrap();
        // Second edit fails, so the first must not be written.
        let bad = json!({ "edits": [
            { "path": "src/a.rs", "old_string": "let x = 1;", "new_string": "let x = 2;" },
            { "path": "src/b.py", "old_string": "nope", "new_string": "x" }
        ] });
        assert!(ws.edit_many(&bad).is_err());
        assert!(fs::read_to_string(&a).unwrap().contains("let x = 1;"));

        let out = ws.edit_many(&edit).unwrap();
        assert_eq!(out, "src/a.rs: 1 edit(s) applied at line(s) 2\n");
        assert!(fs::read_to_string(&a).unwrap().contains("let x = 2;"));

        // Changed since the read (say, by a formatter): applied by exact match, with a note.
        fs::write(&a, "changed elsewhere\n").unwrap();
        let again = json!({ "edits": [{ "path": "src/a.rs", "old_string": "changed", "new_string": "x" }] });
        let out = ws.edit_many(&again).unwrap();
        assert!(out.contains("had changed since you read it"), "{out}");
        assert_eq!(fs::read_to_string(&a).unwrap(), "x elsewhere\n");
    }

    #[test]
    fn edit_many_creates_and_rewrites_files() {
        let (d, mut ws) = setup();
        let out = ws
            .edit_many(&json!({ "edits": [
                { "path": "src/new/mod.rs", "new_string": "pub fn f() {}\n" },
                { "path": "src/new/mod.rs", "old_string": "f()", "new_string": "g()" }
            ] }))
            .unwrap();
        assert_eq!(out, "src/new/mod.rs: created (1 lines), then 1 edit(s) applied at line(s) 1\n");
        assert_eq!(fs::read_to_string(d.path().join("proj/src/new/mod.rs")).unwrap(), "pub fn g() {}\n");

        // Existing files: empty old_string needs overwrite, and overwrite needs a read.
        let rewrite = json!({ "edits": [{ "path": "src/b.py", "new_string": "x = 1\n", "overwrite": true }] });
        assert!(ws.edit_many(&rewrite).unwrap_err().to_string().contains("read it with read_many"));
        ws.read_many(&json!({ "files": ["src/b.py"] })).unwrap();
        let empty = json!({ "edits": [{ "path": "src/b.py", "old_string": "", "new_string": "y" }] });
        assert!(ws.edit_many(&empty).unwrap_err().to_string().contains("set overwrite"));
        assert_eq!(ws.edit_many(&rewrite).unwrap(), "src/b.py: rewritten (1 lines)\n");
        assert_eq!(fs::read_to_string(d.path().join("proj/src/b.py")).unwrap(), "x = 1\n");

        // New files stay inside the root.
        let escape = json!({ "edits": [{ "path": "../evil.txt", "new_string": "x" }] });
        assert!(ws.edit_many(&escape).is_err());
        let sneaky = json!({ "edits": [{ "path": "src/nope/../../../evil.txt", "new_string": "x" }] });
        assert!(ws.edit_many(&sneaky).is_err());
        assert!(!d.path().join("evil.txt").exists());
    }

    #[test]
    fn search_lists_files_without_a_pattern() {
        let (_d, ws) = setup();
        let out = ws.search(&json!({ "glob": "*.rs" })).unwrap();
        assert_eq!(out, "src/a.rs\n1 files\n", "gitignored target/ is skipped");
        let out = ws.search(&json!({ "pattern": "e", "files_only": true })).unwrap();
        assert!(out.contains("src/a.rs (1 matches)\nsrc/b.py (2 matches)\n"), "{out}");
        assert!(!out.contains("println"), "files_only prints no lines");
    }

    #[test]
    fn outline_and_expand() {
        let (_d, ws) = setup();
        let out = ws.outline(&json!({ "paths": ["src/a.rs"] })).unwrap();
        assert_eq!(out, "==> src/a.rs (4 lines) <==\n1-4  fn main()\n");
        let id = ws.store.put_blob("l1\nl2\nl3");
        let out = ws.expand(&json!({ "id": id, "offset": 2, "limit": 1 })).unwrap();
        assert!(out.ends_with("l2\n[squeeze: 1 more lines; call expand with offset 3]"));
    }
}
