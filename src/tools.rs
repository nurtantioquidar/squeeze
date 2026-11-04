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
        let pattern = arg_str(args, "pattern").context("`pattern` is required")?;
        let ctx = arg_usize(args, "context").unwrap_or(0).min(10);
        let max = arg_usize(args, "max_matches").unwrap_or(100).clamp(1, 1000);
        let source = if arg_bool(args, "fixed_string") { regex::escape(pattern) } else { pattern.to_string() };
        let re = RegexBuilder::new(&source)
            .case_insensitive(arg_bool(args, "ignore_case"))
            .size_limit(10 << 20)
            .build()
            .with_context(|| format!("invalid regex {pattern:?}"))?;
        let base = self.resolve(arg_str(args, "path").unwrap_or("."))?;
        let mut wb = WalkBuilder::new(&base);
        wb.sort_by_file_name(|a, b| a.cmp(b)).require_git(false);
        if let Some(glob) = arg_str(args, "glob") {
            let mut ob = OverrideBuilder::new(&base);
            ob.add(glob).with_context(|| format!("invalid glob {glob:?}"))?;
            wb.overrides(ob.build()?);
        }

        let mut out = String::new();
        let (mut shown, mut total, mut files_with) = (0usize, 0usize, 0usize);
        for entry in wb.build().flatten() {
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
        if total > shown {
            let _ = write!(out, " (showing the first {shown}; narrow the pattern, path or glob to see more)");
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

    /// Apply several exact-string edits, validated in full before any file is written.
    pub fn edit_many(&mut self, args: &Value) -> Result<String> {
        let edits = args.get("edits").and_then(Value::as_array).context("`edits` (an array) is required")?;
        if edits.is_empty() || edits.len() > 100 {
            bail!("`edits` must contain 1 to 100 entries");
        }
        let mut order: Vec<PathBuf> = Vec::new();
        let mut by_path: HashMap<PathBuf, Vec<(&str, &str, bool)>> = HashMap::new();
        for (i, e) in edits.iter().enumerate() {
            let path = arg_str(e, "path").with_context(|| format!("edit {}: `path` is required", i + 1))?;
            let old = arg_str(e, "old_string").with_context(|| format!("edit {}: `old_string` is required", i + 1))?;
            let new = arg_str(e, "new_string").with_context(|| format!("edit {}: `new_string` is required", i + 1))?;
            if old.is_empty() {
                bail!("edit {}: `old_string` must not be empty", i + 1);
            }
            if old == new {
                bail!("edit {}: `old_string` and `new_string` are identical", i + 1);
            }
            let canon = self.resolve(path)?;
            if !by_path.contains_key(&canon) {
                order.push(canon.clone());
            }
            by_path.entry(canon).or_default().push((old, new, arg_bool(e, "replace_all")));
        }

        let mut staged: Vec<(PathBuf, String, Vec<usize>)> = Vec::new();
        for path in &order {
            let shown = self.display(path);
            let bytes = self.load(path)?;
            match self.reads.get(path) {
                None => bail!("{shown}: read it with read_many before editing"),
                Some(h) if *h != blake3::hash(&bytes) => {
                    bail!("{shown}: the file changed since you last read it with read_many; read it again")
                }
                Some(_) => {}
            }
            let mut text = String::from_utf8(bytes).map_err(|_| anyhow!("{shown}: not valid UTF-8"))?;
            let mut lines = Vec::new();
            for (i, (old, new, all)) in by_path[path].iter().enumerate() {
                let count = text.matches(old).count();
                if count == 0 {
                    bail!("{shown}: edit {} of this file: old_string not found", i + 1);
                }
                if count > 1 && !all {
                    bail!(
                        "{shown}: edit {} of this file: old_string matches {count} times; add surrounding context or set replace_all",
                        i + 1
                    );
                }
                let at = text.find(old).expect("counted above");
                lines.push(text[..at].matches('\n').count() + 1);
                text = if *all { text.replace(old, new) } else { text.replacen(old, new, 1) };
            }
            staged.push((path.clone(), text, lines));
        }

        let mut out = String::new();
        for (path, text, lines) in staged {
            write_preserving_permissions(&path, text.as_bytes())?;
            self.reads.insert(path.clone(), blake3::hash(text.as_bytes()));
            let at: Vec<String> = lines.iter().map(usize::to_string).collect();
            let _ =
                writeln!(out, "{}: {} edit(s) applied at line(s) {}", self.display(&path), lines.len(), at.join(", "));
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

        fs::write(&a, "changed elsewhere\n").unwrap();
        let again = json!({ "edits": [{ "path": "src/a.rs", "old_string": "changed", "new_string": "x" }] });
        assert!(ws.edit_many(&again).unwrap_err().to_string().contains("changed since you last read it"));
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
