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
/// Per-call output budget, under Claude Code's default MCP result limit
/// (MAX_MCP_OUTPUT_TOKENS = 25,000 tokens, about 100,000 characters): past it,
/// Claude Code truncates the result or moves it to a file.
const MAX_OUTPUT_CHARS: usize = 80_000;
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
    /// ancestor must be a directory inside the root, and the rest may not
    /// contain `..`. An existing entry that can't be resolved (a dangling or
    /// looping symlink) is refused: writing through it could land anywhere.
    fn resolve_new(&self, path: &str) -> Result<PathBuf> {
        if let Ok(p) = self.resolve(path) {
            return Ok(p);
        }
        let p = Path::new(path);
        let joined = if p.is_absolute() { p.to_path_buf() } else { self.root.join(p) };
        if joined.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
            bail!("{path}: `..` is not allowed in a new file's path");
        }
        let mut base = joined.as_path();
        let mut rest = Vec::new();
        // symlink_metadata, unlike exists(), sees a symlink whose target is missing.
        while fs::symlink_metadata(base).is_err() {
            let name = base.file_name().with_context(|| format!("{path}: invalid path"))?;
            rest.push(name.to_os_string());
            base = base.parent().with_context(|| format!("{path}: invalid path"))?;
        }
        let mut canon = base
            .canonicalize()
            .map_err(|_| anyhow!("{path}: {} is a symlink whose target does not exist", self.display(base)))?;
        if !self.allow_outside_root && !canon.starts_with(&self.root) {
            bail!("{path}: outside the project root {}", self.root.display());
        }
        if !canon.is_dir() {
            bail!("{path}: {} is not a directory", self.display(&canon));
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
            let budget = MAX_OUTPUT_CHARS.saturating_sub(out.len());
            if budget < 500 {
                let rest: Vec<&str> =
                    files[n..].iter().map(|f| f.as_str().or_else(|| arg_str(f, "path")).unwrap_or("?")).collect();
                let _ = writeln!(
                    out,
                    "[squeeze: output budget reached; not shown, request separately: {}]",
                    rest.join(", ")
                );
                break;
            }
            if n > 0 {
                out.push('\n');
            }
            match self.read_one(path, offset, limit, budget) {
                Ok(s) => out.push_str(&s),
                Err(e) => {
                    let _ = writeln!(out, "==> {path} <==\n[error: {e}]");
                }
            }
        }
        Ok(out)
    }

    fn read_one(&mut self, path: &str, offset: usize, limit: usize, budget: usize) -> Result<String> {
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
        let mut body = String::new();
        let mut shown_to = start;
        for (i, line) in lines[start..end].iter().enumerate() {
            let entry = format!("{}\t{}\n", start + i + 1, truncate_chars(line, MAX_LINE_CHARS));
            if body.len() + entry.len() > budget {
                break;
            }
            body.push_str(&entry);
            shown_to = start + i + 1;
        }
        let mut out = format!("==> {shown} (lines {}-{} of {}) <==\n{body}", start + 1, shown_to, lines.len());
        if shown_to < end {
            let _ = writeln!(out, "[squeeze: output budget reached; use offset {} to continue]", shown_to + 1);
        } else if end < lines.len() {
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
        let (base, files) = self.walk_files(args)?;

        let mut out = String::new();
        let (mut shown, mut total, mut files_with) = (0usize, 0usize, 0usize);
        let mut over_budget = false;
        for path in &files {
            if fs::metadata(path).map_or(true, |m| m.len() > MAX_SEARCH_FILE_BYTES) {
                continue;
            }
            let Ok(bytes) = fs::read(path) else { continue };
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
            if shown >= max || over_budget {
                continue;
            }
            if out.len() > MAX_OUTPUT_CHARS {
                over_budget = true;
                continue;
            }
            if files_only {
                shown += 1;
                let _ = writeln!(out, "{} ({} matches)", self.display(path), hits.len());
                continue;
            }
            let take: Vec<usize> = hits.iter().copied().take(max - shown).collect();
            shown += take.len();
            let hit_set: HashSet<usize> = take.iter().copied().collect();
            let _ = writeln!(out, "{}", self.display(path));
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
        if over_budget {
            let _ = write!(out, " (output budget reached after {shown}; narrow the pattern, path or glob to see more)");
        } else if files_only && files_with > shown {
            let _ = write!(out, " (listing the first {shown} files)");
        } else if !files_only && total > shown {
            let _ = write!(out, " (showing the first {shown}; narrow the pattern, path or glob to see more)");
        }
        out.push('\n');
        Ok(out)
    }

    /// Files under `path`, as Claude Code's Grep and Glob see them: .gitignore
    /// honored, dotfiles included, version-control directories skipped. `glob`
    /// is relative to the project root, and one without `/` matches file names
    /// at any depth.
    fn walk_files(&self, args: &Value) -> Result<(PathBuf, Vec<PathBuf>)> {
        let base = self.resolve(arg_str(args, "path").unwrap_or("."))?;
        let glob = match arg_str(args, "glob").filter(|g| !g.is_empty()) {
            Some(g) => {
                let anchor = if base.starts_with(&self.root) { self.root.as_path() } else { base.as_path() };
                let mut ob = OverrideBuilder::new(anchor);
                ob.add(g).with_context(|| format!("invalid glob {g:?}"))?;
                Some(ob.build()?)
            }
            None => None,
        };
        let mut wb = WalkBuilder::new(&base);
        wb.sort_by_file_name(|a, b| a.cmp(b)).require_git(false).hidden(false).filter_entry(|e| {
            !(e.file_type().is_some_and(|t| t.is_dir())
                && matches!(e.file_name().to_str(), Some(".git" | ".hg" | ".svn" | ".jj" | ".bzr" | ".sl")))
        });
        let files = wb
            .build()
            .flatten()
            .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
            .map(ignore::DirEntry::into_path)
            // Matched against files only: as a walker override, a glob that
            // matched a directory name would let it past .gitignore.
            .filter(|p| glob.as_ref().is_none_or(|ov| !ov.matched(p, false).is_ignore()))
            .collect();
        Ok((base, files))
    }

    /// `search` without a pattern: list files (like `find`/`ls -R`), honoring .gitignore.
    fn list_files(&self, args: &Value) -> Result<String> {
        let max = arg_usize(args, "max_matches").unwrap_or(200).clamp(1, 1000);
        let (base, files) = self.walk_files(args)?;
        let mut out = String::new();
        let total = files.len();
        for path in files.iter().take(max) {
            let _ = writeln!(out, "{}", self.display(path));
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
        // On a case-insensitive volume (macOS by default) two spellings of one new
        // file would both be "created" and the second would replace the first.
        let mut new_names = HashSet::new();
        for path in order.iter().filter(|p| !p.exists()) {
            if !new_names.insert(path.to_string_lossy().to_lowercase()) {
                bail!("{}: the same new file appears twice with different capitalization", self.display(path));
            }
        }

        struct Staged {
            path: PathBuf,
            text: String,
            existed: bool,
            stale: bool,
            what: String,
        }
        let mut staged: Vec<Staged> = Vec::new();
        for path in &order {
            let shown = self.display(path);
            let existed = path.exists();
            let (mut text, stale) = if existed {
                let bytes = self.load(path)?;
                let stale = match self.reads.get(path) {
                    None => bail!("{shown}: read it with read_many before editing it"),
                    Some(h) => *h != blake3::hash(&bytes),
                };
                (String::from_utf8(bytes).map_err(|_| anyhow!("{shown}: not valid UTF-8"))?, stale)
            } else {
                (String::new(), false)
            };
            let changed = if stale { "; the file changed since you read it, so read it again" } else { "" };
            // Byte offset and length of each replacement in the current text, kept
            // up to date as later edits shift them, for the line numbers reported.
            let mut spots: Vec<(usize, usize)> = Vec::new();
            let mut action = None;
            for (i, e) in by_path[path].iter().enumerate() {
                let n = i + 1;
                if e.overwrite || (e.old.is_empty() && !existed) {
                    if i > 0 {
                        bail!(
                            "{shown}: edit {n} of this file: a new file or overwrite must be the first edit of its file"
                        );
                    }
                    if stale {
                        bail!("{shown}: the file changed since you read it; read it again before overwriting it");
                    }
                    text = e.new.to_string();
                    action = Some(if existed { "rewritten" } else { "created" });
                    continue;
                }
                if e.old.is_empty() {
                    bail!(
                        "{shown}: edit {n} of this file: `old_string` is empty; set overwrite to replace the whole file"
                    );
                }
                if e.all && stale {
                    bail!("{shown}: the file changed since you read it; read it again before using replace_all");
                }
                let at: Vec<usize> = text.match_indices(e.old).map(|(k, _)| k).collect();
                if at.is_empty() {
                    bail!("{shown}: edit {n} of this file: old_string not found{changed}");
                }
                if at.len() > 1 && !e.all {
                    bail!(
                        "{shown}: edit {n} of this file: old_string matches {} times; add surrounding context or set replace_all{changed}",
                        at.len()
                    );
                }
                let delta = e.new.len() as isize - e.old.len() as isize;
                for spot in &mut spots {
                    let before = at.iter().filter(|&&k| k < spot.0).count() as isize;
                    spot.0 = (spot.0 as isize + before * delta) as usize;
                }
                for (j, &k) in at.iter().enumerate() {
                    spots.push(((k as isize + j as isize * delta) as usize, e.new.len()));
                }
                text = if e.all { text.replace(e.old, e.new) } else { text.replacen(e.old, e.new, 1) };
            }
            let mut what = match action {
                Some(a) => format!("{a} ({} lines)", text.lines().count()),
                None => String::new(),
            };
            if !spots.is_empty() {
                let bytes = text.as_bytes();
                let mut lines: Vec<usize> =
                    spots.iter().map(|&(k, _)| bytecount_newlines(&bytes[..k.min(bytes.len())]) + 1).collect();
                lines.sort_unstable();
                let at: Vec<String> = lines.iter().map(usize::to_string).collect();
                if !what.is_empty() {
                    what.push_str(", then ");
                }
                let _ = write!(what, "{} replacement(s) at line(s) {}", lines.len(), at.join(", "));
            }
            if stale {
                what.push_str(" (the file had changed since you read it; matched exactly)");
            }
            staged.push(Staged { path: path.clone(), text, existed, stale, what });
        }

        // Write everything to temporary files next to their targets first, and
        // move them into place only once all of them were written.
        let mut temps = Vec::new();
        for st in &staged {
            let shown = self.display(&st.path);
            let dir = st.path.parent().with_context(|| format!("{shown}: invalid path"))?;
            fs::create_dir_all(dir).with_context(|| format!("{shown}: could not create its directory"))?;
            temps.push(
                write_temp(&st.path, st.existed, st.text.as_bytes())
                    .with_context(|| format!("{shown}: write failed"))?,
            );
        }
        let mut out = String::new();
        for (st, tmp) in staged.into_iter().zip(temps) {
            let shown = self.display(&st.path);
            // A rename replaces a symlink at the target instead of following it; a
            // new file must still not exist (no clobbering a file created meanwhile).
            let moved = if st.existed {
                tmp.persist(&st.path).map(|_| ())
            } else {
                tmp.persist_noclobber(&st.path).map(|_| ())
            };
            if let Err(e) = moved {
                bail!("{shown}: could not be written ({}); already written:\n{out}", e.error);
            }
            // After a stale edit the agent still hasn't seen the whole file.
            if !st.stale {
                self.reads.insert(st.path.clone(), blake3::hash(st.text.as_bytes()));
            }
            let _ = writeln!(out, "{shown}: {}", st.what);
        }
        Ok(out)
    }
}

fn bytecount_newlines(b: &[u8]) -> usize {
    b.iter().filter(|&&c| c == b'\n').count()
}

/// A temporary file next to `path` holding `data`, with `path`'s permissions
/// (or the usual 0644 for a new file). Created exclusively, so it never
/// follows a symlink that happens to sit at the temporary name.
fn write_temp(path: &Path, existed: bool, data: &[u8]) -> Result<tempfile::NamedTempFile> {
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    let dir = path.parent().context("invalid path")?;
    let mut tmp = tempfile::Builder::new().prefix(".squeeze-edit-").tempfile_in(dir)?;
    tmp.write_all(data)?;
    tmp.flush()?;
    let perms = if existed { fs::metadata(path)?.permissions() } else { fs::Permissions::from_mode(0o644) };
    fs::set_permissions(tmp.path(), perms)?;
    Ok(tmp)
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
        assert_eq!(out, "src/a.rs: 1 replacement(s) at line(s) 2\n");
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
        assert_eq!(out, "src/new/mod.rs: created (1 lines), then 1 replacement(s) at line(s) 1\n");
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
    fn edit_many_never_writes_outside_the_root() {
        let (d, mut ws) = setup();
        let outside = d.path().join("outside");
        fs::create_dir_all(&outside).unwrap();
        // Dangling symlinks inside the root, absolute and relative, pointing outside it.
        std::os::unix::fs::symlink(outside.join("pwned.txt"), d.path().join("proj/notes.md")).unwrap();
        std::os::unix::fs::symlink("../../outside/rel.txt", d.path().join("proj/src/rel.md")).unwrap();
        for p in ["notes.md", "src/rel.md"] {
            let err = ws.edit_many(&json!({ "edits": [{ "path": p, "new_string": "owned\n" }] })).unwrap_err();
            assert!(err.to_string().contains("symlink whose target does not exist"), "{p}: {err}");
        }
        assert!(!outside.join("pwned.txt").exists() && !outside.join("rel.txt").exists());
        // A new file under an existing file is refused before anything is written.
        ws.read_many(&json!({ "files": ["src/a.rs"] })).unwrap();
        let mixed = json!({ "edits": [
            { "path": "src/a.rs", "old_string": "let x = 1;", "new_string": "let x = 9;" },
            { "path": "src/a.rs/oops.rs", "new_string": "x" }
        ] });
        assert!(ws.edit_many(&mixed).unwrap_err().to_string().contains("is not a directory"));
        assert!(fs::read_to_string(d.path().join("proj/src/a.rs")).unwrap().contains("let x = 1;"), "nothing written");
        // No temporary files are left behind.
        let leftovers: Vec<_> = fs::read_dir(d.path().join("proj/src"))
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with(".squeeze-edit-"))
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn edit_many_refuses_blind_rewrites_of_changed_files() {
        let (d, mut ws) = setup();
        let a = d.path().join("proj/src/a.rs");
        ws.read_many(&json!({ "files": ["src/a.rs"] })).unwrap();
        fs::write(&a, "fn main() {\n    let x = 1;\n    let x = 1;\n}\n").unwrap(); // changed behind the agent's back
        let overwrite = json!({ "edits": [{ "path": "src/a.rs", "new_string": "gone\n", "overwrite": true }] });
        assert!(ws.edit_many(&overwrite).unwrap_err().to_string().contains("read it again before overwriting"));
        let all = json!({ "edits": [{ "path": "src/a.rs", "old_string": "let x = 1;", "new_string": "y", "replace_all": true }] });
        assert!(ws.edit_many(&all).unwrap_err().to_string().contains("read it again before using replace_all"));
        let missing = json!({ "edits": [{ "path": "src/a.rs", "old_string": "nope", "new_string": "y" }] });
        assert!(ws.edit_many(&missing).unwrap_err().to_string().contains("changed since you read it"));
        // An exact match still applies, and the file stays "changed" for blind rewrites.
        let one = json!({ "edits": [{ "path": "src/a.rs", "old_string": "fn main()", "new_string": "fn start()" }] });
        assert!(ws.edit_many(&one).unwrap().contains("matched exactly"));
        assert!(ws.edit_many(&overwrite).is_err());
        assert!(fs::read_to_string(&a).unwrap().starts_with("fn start()"));
    }

    #[test]
    fn edit_many_reports_final_line_numbers_and_rejects_bad_batches() {
        let (d, mut ws) = setup();
        fs::write(d.path().join("proj/src/l.txt"), "a\nb\nc\nb\n").unwrap();
        ws.read_many(&json!({ "files": ["src/l.txt"] })).unwrap();
        // The second edit adds a line above the first: the first is reported where it ends up.
        let out = ws
            .edit_many(&json!({ "edits": [
                { "path": "src/l.txt", "old_string": "c", "new_string": "c1\nc2" },
                { "path": "src/l.txt", "old_string": "a", "new_string": "a0\na1" },
                { "path": "src/l.txt", "old_string": "b", "new_string": "B", "replace_all": true }
            ] }))
            .unwrap();
        assert_eq!(out, "src/l.txt: 4 replacement(s) at line(s) 1, 3, 4, 6\n");
        assert_eq!(fs::read_to_string(d.path().join("proj/src/l.txt")).unwrap(), "a0\na1\nB\nc1\nc2\nB\n");

        let late = json!({ "edits": [
            { "path": "src/l.txt", "old_string": "B", "new_string": "x", "replace_all": true },
            { "path": "src/l.txt", "new_string": "all new\n", "overwrite": true }
        ] });
        assert!(ws.edit_many(&late).unwrap_err().to_string().contains("must be the first edit"));
        let twice = json!({ "edits": [
            { "path": "src/New.rs", "new_string": "1" },
            { "path": "src/new.rs", "new_string": "2" }
        ] });
        assert!(ws.edit_many(&twice).unwrap_err().to_string().contains("different capitalization"));
    }

    #[test]
    fn read_many_stays_within_the_output_budget() {
        let (d, mut ws) = setup();
        let big: String = (0..5000).map(|i| format!("line {i} {}\n", "x".repeat(40))).collect();
        fs::write(d.path().join("proj/src/big.txt"), &big).unwrap();
        let out = ws.read_many(&json!({ "files": ["src/big.txt", "src/a.rs"] })).unwrap();
        assert!(out.len() <= MAX_OUTPUT_CHARS + 500, "{}", out.len());
        assert!(out.contains("[squeeze: output budget reached; use offset"), "a cut file says where to continue");
        assert!(out.contains("not shown, request separately: src/a.rs"), "and which files were left out");
    }

    #[test]
    fn search_sees_dotfiles_and_keeps_globs_from_bypassing_gitignore() {
        let (d, ws) = setup();
        fs::create_dir_all(d.path().join("proj/.github/workflows")).unwrap();
        fs::write(d.path().join("proj/.github/workflows/ci.yml"), "needle: true\n").unwrap();
        fs::create_dir_all(d.path().join("proj/.git")).unwrap();
        fs::write(d.path().join("proj/.git/config"), "needle").unwrap();
        assert_eq!(ws.search(&json!({ "glob": "*.yml" })).unwrap(), ".github/workflows/ci.yml\n1 files\n");
        let hits = ws.search(&json!({ "pattern": "needle", "files_only": true })).unwrap();
        assert!(hits.contains(".github/workflows/ci.yml") && !hits.contains(".git/config"), "{hits}");
        // "**/*" matches the gitignored target/ directory's name, but stays a file filter.
        let all = ws.search(&json!({ "glob": "**/*" })).unwrap();
        assert!(!all.contains("target/junk.rs") && !all.contains(".git/"), "{all}");
        // Globs are relative to the project root even when path narrows the search.
        let anchored = ws.search(&json!({ "path": "src", "glob": "src/*.rs" })).unwrap();
        assert_eq!(anchored, "src/a.rs\n1 files\n");
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
