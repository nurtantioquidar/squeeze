//! Text compression primitives for tool outputs.
//!
//! Everything here is a pure function of its input. That determinism is what
//! keeps rewritten requests byte-stable from one turn to the next, so the
//! prompt cache and preserved-thinking checks keep working.

use regex::Regex;
use std::borrow::Cow;
use std::sync::LazyLock;

static ANSI: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]").expect("ansi regex")
});

/// Lines worth keeping when the middle of a long log is dropped.
static IMPORTANT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:error|errors|fail|fails|failed|failure|failures|failing|panic|panicked|exception|traceback|fatal|warning|warnings|assert|assertion|expected|undefined|denied|segfault|segmentation|abort|aborted|timeout|timed out|cannot|unable)\b|✗|✘|ERR!",
    )
    .expect("important regex")
});

/// A "word" first token, e.g. `Compiling`, `Downloading`, `test`, `npm`.
static WORD_KEY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Za-z][A-Za-z_\-]{2,}:?$").expect("word regex"));

/// A passing-test line from common runners: unittest/cargo (`... ok`), pytest
/// (`PASSED [ 12%]`), go (`--- PASS:` / `ok  pkg`), jest/mocha (`✓ name`).
static PASS_LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:\.\.\.\s*ok\s*$|\bPASSED\b\s*(?:\[\s*\d+%\])?\s*$|^\s*--- PASS:|^\s*=== RUN\b|^\s*(?:✓|✔|√)\s)")
        .expect("pass regex")
});

const PASS_KEY: &str = "\u{0}pass";

static GIT_VIEW: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\bgit\b(?:\s+-C\s+\S+)?(?:\s+--no-pager)?\s+(?:diff|show|format-patch|blame|cat-file)\b|\bgit\b.*\blog\b.*(?:\s-p\b|--patch)|\bsqueeze\s+expand\b",
    )
    .expect("git regex")
});

static REDIRECT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d*>&\d*|&>>?").expect("redirect regex"));

/// Commands whose output is file content the model asked to see verbatim.
const VIEW_COMMANDS: &[&str] = &[
    "cat",
    "head",
    "tail",
    "sed",
    "awk",
    "nl",
    "bat",
    "batcat",
    "less",
    "more",
    "diff",
    "colordiff",
    "jq",
    "yq",
    "xxd",
    "od",
    "hexdump",
    "base64",
    "cut",
    "column",
    "sort",
    "uniq",
    "tr",
    "wc",
    "strings",
    "file",
    "stat",
    "readlink",
    "realpath",
];

/// Commands that neither produce logs nor file content (`cd x && ...`, `...; echo "exit $?"`).
const NEUTRAL_COMMANDS: &[&str] = &["cd", "export", "set", "true", "echo", "printf", "pwd", "sleep", "wait", ":"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BashKind {
    /// Output is file content (cat, sed -n, git diff, ...): only lossless cleanup.
    FileView,
    /// Output is a log (builds, tests, installs, ...): may be summarized.
    Log,
}

/// File content only if every command in the chain is a viewer or neutral;
/// anything else (a build, a test run, a loop) makes the output a log.
pub fn classify_bash(command: &str) -> BashKind {
    if GIT_VIEW.is_match(command) {
        return BashKind::FileView;
    }
    let cleaned = REDIRECT.replace_all(command, " ");
    let mut saw_view = false;
    for segment in cleaned.split(['|', ';', '&', '\n']).map(str::trim).filter(|s| !s.is_empty()) {
        let first = segment.split_whitespace().find(|w| !w.contains('=')).unwrap_or("");
        let program = first.rsplit('/').next().unwrap_or(first);
        if VIEW_COMMANDS.contains(&program) {
            saw_view = true;
        } else if !NEUTRAL_COMMANDS.contains(&program) {
            return BashKind::Log;
        }
    }
    if saw_view { BashKind::FileView } else { BashKind::Log }
}

#[derive(Clone, Debug)]
pub struct LogPolicy {
    /// Logs longer than this (after collapsing) get their middle dropped.
    pub max_lines: usize,
    pub head: usize,
    pub tail: usize,
    /// How many important lines (with one line of context) to keep from a dropped middle.
    pub important_budget: usize,
    /// Minimum run of lines sharing a first word before the run is collapsed.
    pub run_min: usize,
    pub run_keep_head: usize,
    pub run_keep_tail: usize,
    /// Minimum run of identical lines before it is collapsed to one line plus a count.
    pub repeat_min: usize,
}

impl Default for LogPolicy {
    fn default() -> Self {
        Self {
            max_lines: 250,
            head: 60,
            tail: 100,
            important_budget: 60,
            run_min: 10,
            run_keep_head: 3,
            run_keep_tail: 2,
            repeat_min: 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compressed {
    pub text: String,
    /// True when information was dropped (the original should be kept for `expand`).
    pub lossy: bool,
    pub lines_in: usize,
    pub lines_out: usize,
}

pub fn strip_ansi(s: &str) -> Cow<'_, str> {
    if s.contains('\x1b') { ANSI.replace_all(s, "") } else { Cow::Borrowed(s) }
}

/// Lossless cleanup: strip ANSI escapes and resolve carriage-return overwrites
/// (progress bars), keeping what a terminal would finally show.
pub fn clean(s: &str) -> String {
    let s = strip_ansi(s);
    let s = if s.contains('\r') { s.replace("\r\n", "\n") } else { s.into_owned() };
    if !s.contains('\r') {
        return s;
    }
    s.split('\n').map(|line| line.rsplit('\r').find(|seg| !seg.is_empty()).unwrap_or("")).collect::<Vec<_>>().join("\n")
}

/// Re-serialize pretty-printed JSON compactly. Key order and number
/// formatting are preserved (serde_json `preserve_order` + `arbitrary_precision`).
pub fn minify_json(text: &str) -> Option<String> {
    let t = text.trim();
    if !(t.starts_with('{') || t.starts_with('[')) || !t.contains('\n') {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(t).ok()?;
    serde_json::to_string(&v).ok()
}

enum Item {
    Line(String),
    Note(String),
}

impl Item {
    fn text(&self) -> &str {
        match self {
            Item::Line(s) | Item::Note(s) => s,
        }
    }
}

pub fn compress_log(input: &str, p: &LogPolicy) -> Compressed {
    let cleaned = clean(input);
    let raw: Vec<&str> = cleaned.split('\n').collect();
    let lines_in = raw.len();
    let mut lossy = false;

    // Collapse runs of identical lines and of blank lines.
    let mut items = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        let line = raw[i].trim_end();
        let mut j = i + 1;
        while j < raw.len() && raw[j].trim_end() == line {
            j += 1;
        }
        let n = j - i;
        if line.is_empty() {
            items.push(Item::Line(String::new()));
        } else if n >= p.repeat_min {
            items.push(Item::Line(line.to_string()));
            items.push(Item::Note(format!("[squeeze: previous line repeated {} more times]", n - 1)));
        } else {
            items.extend((0..n).map(|_| Item::Line(line.to_string())));
        }
        i = j;
    }

    let items = collapse_similar(items, p, &mut lossy);
    let items = if items.len() > p.max_lines {
        lossy = true;
        truncate_middle(items, p)
    } else {
        items
    };
    let lines_out = items.len();
    let text = items.iter().map(Item::text).collect::<Vec<_>>().join("\n");
    Compressed { text, lossy, lines_in, lines_out }
}

fn run_key(line: &str) -> Option<String> {
    if line.is_empty() || IMPORTANT.is_match(line) {
        return None;
    }
    let body = line.trim_start();
    let indent = &line[..line.len() - body.len()];
    // Passing tests collapse as one run whatever their names; failures are
    // IMPORTANT and break the run, so they are always kept.
    if PASS_LINE.is_match(line) {
        return Some(PASS_KEY.to_string());
    }
    let token = body.split_whitespace().next()?;
    WORD_KEY.is_match(token).then(|| format!("{indent}{token}"))
}

/// Collapse long runs of lines that start with the same word
/// (`Compiling ...`, `Downloading ...`, `test ... ok`), keeping a few at each end.
fn collapse_similar(items: Vec<Item>, p: &LogPolicy, lossy: &mut bool) -> Vec<Item> {
    let keys: Vec<Option<String>> = items
        .iter()
        .map(|it| match it {
            Item::Line(l) => run_key(l),
            Item::Note(_) => None,
        })
        .collect();
    let n = items.len();
    let mut slots: Vec<Option<Item>> = items.into_iter().map(Some).collect();
    let mut take = |k: usize| slots[k].take().expect("each slot is taken once");
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let Some(key) = keys[i].as_ref() else {
            out.push(take(i));
            i += 1;
            continue;
        };
        let mut j = i + 1;
        while j < n && keys[j].as_ref() == Some(key) {
            j += 1;
        }
        let len = j - i;
        if len >= p.run_min && len > p.run_keep_head + p.run_keep_tail {
            *lossy = true;
            out.extend((i..i + p.run_keep_head).map(&mut take));
            let omitted = len - p.run_keep_head - p.run_keep_tail;
            out.push(Item::Note(if key == PASS_KEY {
                format!("[squeeze: {omitted} more passing-test lines omitted]")
            } else {
                format!("[squeeze: {omitted} similar lines starting with \"{}\" omitted]", key.trim_start())
            }));
            out.extend((j - p.run_keep_tail..j).map(&mut take));
        } else {
            out.extend((i..j).map(&mut take));
        }
        i = j;
    }
    out
}

/// Keep the head and tail of a long log plus important lines from the middle.
fn truncate_middle(items: Vec<Item>, p: &LogPolicy) -> Vec<Item> {
    let n = items.len();
    let mut keep = vec![false; n];
    keep.iter_mut().take(p.head).for_each(|k| *k = true);
    keep.iter_mut().skip(n.saturating_sub(p.tail)).for_each(|k| *k = true);
    let mut budget = p.important_budget;
    let middle = p.head.min(n)..n.saturating_sub(p.tail);
    for (k, item) in items.iter().enumerate().take(middle.end).skip(middle.start) {
        if budget == 0 {
            break;
        }
        if let Item::Line(l) = item
            && IMPORTANT.is_match(l)
        {
            // The line plus one line of context on each side.
            for flag in keep.iter_mut().take((k + 2).min(n)).skip(k.saturating_sub(1)) {
                if !*flag && budget > 0 {
                    *flag = true;
                    budget -= 1;
                }
            }
        }
    }
    let mut out = Vec::new();
    let mut gap = 0usize;
    for (k, item) in items.into_iter().enumerate() {
        if keep[k] {
            if gap > 0 {
                out.push(Item::Note(format!("[squeeze: {gap} lines omitted]")));
                gap = 0;
            }
            out.push(item);
        } else {
            gap += 1;
        }
    }
    if gap > 0 {
        out.push(Item::Note(format!("[squeeze: {gap} lines omitted]")));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_commands() {
        assert_eq!(classify_bash("cat src/main.rs"), BashKind::FileView);
        assert_eq!(classify_bash("cd app && sed -n '1,40p' x.py"), BashKind::FileView);
        assert_eq!(classify_bash("cat foo 2>&1"), BashKind::FileView);
        assert_eq!(classify_bash("git diff HEAD~1 -- src"), BashKind::FileView);
        assert_eq!(classify_bash("git -C repo show abc123"), BashKind::FileView);
        assert_eq!(classify_bash("squeeze expand 0123abcd"), BashKind::FileView);
        assert_eq!(classify_bash("cat a.txt; echo ---; sed -n '5,9p' b.txt"), BashKind::FileView);
        assert_eq!(classify_bash("cargo test 2>&1 | tail -50"), BashKind::Log);
        assert_eq!(classify_bash("cargo build; echo \"exit: $?\""), BashKind::Log);
        assert_eq!(classify_bash("for i in $(seq 1 3); do echo $i; done"), BashKind::Log);
        assert_eq!(classify_bash("cargo test 2>&1"), BashKind::Log);
        assert_eq!(classify_bash("RUST_LOG=debug cargo build"), BashKind::Log);
        assert_eq!(classify_bash("npm install"), BashKind::Log);
    }

    #[test]
    fn clean_strips_ansi_and_progress() {
        let s = "\x1b[32mok\x1b[0m\r\nDownloading 10%\rDownloading 55%\rDownloading 100%\ndone";
        assert_eq!(clean(s), "ok\nDownloading 100%\ndone");
    }

    #[test]
    fn collapses_identical_runs() {
        let input = "start\nretry\nretry\nretry\nretry\nretry\nend";
        let c = compress_log(input, &LogPolicy::default());
        assert_eq!(c.text, "start\nretry\n[squeeze: previous line repeated 4 more times]\nend");
        assert!(!c.lossy);
    }

    #[test]
    fn collapses_similar_runs_but_keeps_errors() {
        let mut lines: Vec<String> = (0..30).map(|i| format!("   Compiling crate{i} v1.0.{i}")).collect();
        lines.push("error[E0308]: mismatched types".into());
        lines.push("   Compiling last v0.1.0".into());
        let c = compress_log(&lines.join("\n"), &LogPolicy::default());
        assert!(c.lossy);
        assert!(c.text.contains("crate0"));
        assert!(c.text.contains("crate29"));
        assert!(!c.text.contains("crate10"));
        assert!(c.text.contains("[squeeze: 25 similar lines starting with \"Compiling\" omitted]"));
        assert!(c.text.contains("error[E0308]"));
    }

    #[test]
    fn collapses_passing_test_lines_but_keeps_failures() {
        // unittest -v, the shape a real agent run produced.
        let mut lines: Vec<String> = (1..=78)
            .map(|n| format!("test_median_of_{n}_values (tests.test_generated.GeneratedMedianTest.test_median_of_{n}_values) ... ok"))
            .collect();
        lines.insert(40, "test_mean (tests.test_calc.CalcTest.test_mean) ... FAIL".into());
        lines.push("Ran 303 tests in 0.010s".into());
        let c = compress_log(&lines.join("\n"), &LogPolicy::default());
        assert!(c.lossy);
        assert!(c.text.contains("test_mean (tests.test_calc.CalcTest.test_mean) ... FAIL"));
        assert!(c.text.contains("more passing-test lines omitted]"));
        assert!(c.text.contains("Ran 303 tests"));
        assert!(c.lines_out < 20, "{}", c.text);
        // pytest, go test and jest styles.
        for (pass, n) in [
            ("tests/test_x.py::test_case_{} PASSED          [ 50%]", 30),
            ("--- PASS: TestCase{} (0.00s)", 30),
            ("  ✓ renders case {} (3 ms)", 30),
        ] {
            let text: Vec<String> = (0..n).map(|i| pass.replace("{}", &i.to_string())).collect();
            assert!(
                compress_log(&text.join("\n"), &LogPolicy::default()).text.contains("passing-test lines omitted"),
                "{pass}"
            );
        }
    }

    #[test]
    fn data_rows_that_differ_only_in_numbers_are_kept() {
        // Printed data (e.g. a DataFrame) must not be treated like test output.
        let text: Vec<String> =
            (0..40).map(|i| format!("2024-01-{:02}  {}.5  {}", i % 28 + 1, 100 + i, i * 3)).collect();
        let c = compress_log(&text.join("\n"), &LogPolicy::default());
        assert!(!c.lossy);
    }

    #[test]
    fn similar_runs_ignore_non_word_prefixes() {
        // File listings and grep output must not be collapsed by first token.
        let lines: Vec<String> = (0..30).map(|i| format!("-rw-r--r--  1 u  g  {i} file{i}.txt")).collect();
        let c = compress_log(&lines.join("\n"), &LogPolicy::default());
        assert!(!c.lossy);
        assert_eq!(c.lines_out, 30);
    }

    #[test]
    fn truncates_long_logs_keeping_important_middle_lines() {
        let mut lines: Vec<String> = (0..1000).map(|i| format!("{i}: value={}", i * 7)).collect();
        lines[500] = "FAILED tests/test_api.py::test_login - AssertionError".into();
        let c = compress_log(&lines.join("\n"), &LogPolicy::default());
        assert!(c.lossy);
        assert!(c.lines_out < 250);
        assert!(c.text.starts_with("0: value=0"));
        assert!(c.text.contains("999: value=6993"));
        assert!(c.text.contains("FAILED tests/test_api.py::test_login"));
        assert!(c.text.contains("499: value="), "keeps one line of context before");
        assert!(c.text.contains("lines omitted]"));
    }

    #[test]
    fn compression_is_deterministic() {
        let input: String = (0..800).map(|i| format!("   Compiling c{i}\n")).collect();
        assert_eq!(compress_log(&input, &LogPolicy::default()), compress_log(&input, &LogPolicy::default()));
    }

    #[test]
    fn minifies_pretty_json_only() {
        assert_eq!(minify_json("{\n  \"b\": 1,\n  \"a\": [1.50, 2]\n}").as_deref(), Some(r#"{"b":1,"a":[1.50,2]}"#));
        assert_eq!(minify_json(r#"{"a":1}"#), None);
        assert_eq!(minify_json("not json\n"), None);
    }
}
