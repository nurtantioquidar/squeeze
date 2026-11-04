//! Lightweight structural outlines (definitions with line ranges) so the agent
//! can read only the parts of a file it needs. Regex-based and approximate by
//! design: no parser dependencies, and good enough to navigate by.

use crate::util::truncate_chars;
use regex::Regex;
use std::path::Path;
use std::sync::LazyLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    /// 1-based, inclusive.
    pub start: usize,
    pub end: usize,
    pub depth: usize,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lang {
    Rust,
    Python,
    Js,
    Go,
    Jvm,
    C,
    Ruby,
    Markdown,
}

fn lang_for(path: &Path) -> Option<Lang> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "rs" => Lang::Rust,
        "py" | "pyi" => Lang::Python,
        "js" | "jsx" | "mjs" | "cjs" | "ts" | "tsx" | "mts" | "cts" => Lang::Js,
        "go" => Lang::Go,
        "java" | "kt" | "kts" | "cs" | "scala" | "swift" => Lang::Jvm,
        "c" | "h" | "cc" | "cpp" | "cxx" | "hpp" | "hh" | "hxx" => Lang::C,
        "rb" => Lang::Ruby,
        "md" | "markdown" => Lang::Markdown,
        _ => return None,
    })
}

fn compile(patterns: &[&str]) -> Vec<Regex> {
    patterns.iter().map(|p| Regex::new(p).expect("outline regex")).collect()
}

static RUST: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    compile(&[
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\S+\s+)?fn\s+(?P<name>\w+)",
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|union|trait|type|mod)\s+(?P<name>\w+)",
        r"^\s*(?:unsafe\s+)?impl\b",
        r"^\s*macro_rules!\s*(?P<name>\w+)",
    ])
});
static PYTHON: LazyLock<Vec<Regex>> =
    LazyLock::new(|| compile(&[r"^\s*(?:async\s+)?def\s+(?P<name>\w+)", r"^\s*class\s+(?P<name>\w+)"]));
static JS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    compile(&[
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>\w+)",
        r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(?P<name>\w+)",
        r"^\s*(?:export\s+)?(?:declare\s+)?(?:interface|enum|namespace)\s+(?P<name>\w+)",
        r"^\s*(?:export\s+)?(?:declare\s+)?type\s+(?P<name>\w+)\b[^=]*=",
        r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*(?::[^=]*)?=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*(?::[^=]*)?=>|\w+\s*=>)",
        r"^\s+(?:(?:public|private|protected|static|readonly|async|override|abstract|get|set)\s+)*\*?(?P<name>[A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?::\s*[^{]+)?\{\s*$",
    ])
});
static GO: LazyLock<Vec<Regex>> =
    LazyLock::new(|| compile(&[r"^func\s+(?:\([^)]*\)\s*)?(?P<name>\w+)", r"^type\s+(?P<name>\w+)"]));
static JVM: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    compile(&[
        r"^\s*(?:@\w+\s+)*(?:(?:public|private|protected|internal|static|final|abstract|sealed|open|data|inner|partial|override|export|fileprivate)\s+)*(?:class|interface|enum|record|object|struct|trait|protocol|extension)\s+(?P<name>\w+)",
        r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|override|open|suspend|inline|fileprivate|mutating)\s+)*(?:fun|func|def)\s+(?P<name>[\w.<>]+)",
        r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|synchronized|native|override|virtual|async|sealed|default)\s+)+[\w<>\[\],.?]+(?:\s*<[^>]*>)?\s+(?P<name>\w+)\s*\([^;]*$",
    ])
});
static C: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    compile(&[
        r"^\s*(?:typedef\s+)?(?:struct|class|enum|union|namespace)\s+(?P<name>\w+)[^;]*$",
        r"^(?:[A-Za-z_][\w:<>,]*[\s*&]+)+\**(?P<name>[A-Za-z_][\w:~]*)\s*\([^;]*$",
    ])
});
static RUBY: LazyLock<Vec<Regex>> =
    LazyLock::new(|| compile(&[r"^\s*def\s+(?P<name>[\w.?!=]+)", r"^\s*(?:class|module)\s+(?P<name>[\w:]+)"]));
static MARKDOWN: LazyLock<Vec<Regex>> = LazyLock::new(|| compile(&[r"^(?P<hashes>#{1,6})\s+\S"]));

const KEYWORDS: &[&str] = &[
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "else",
    "do",
    "try",
    "new",
    "typeof",
    "sizeof",
    "function",
    "match",
    "loop",
    "elif",
    "with",
    "await",
    "yield",
    "throw",
    "case",
    "delete",
    "constructor",
];

fn patterns(lang: Lang) -> &'static [Regex] {
    match lang {
        Lang::Rust => &RUST,
        Lang::Python => &PYTHON,
        Lang::Js => &JS,
        Lang::Go => &GO,
        Lang::Jvm => &JVM,
        Lang::C => &C,
        Lang::Ruby => &RUBY,
        Lang::Markdown => &MARKDOWN,
    }
}

/// Outline a file, or `None` if its language is not supported.
pub fn outline(path: &Path, text: &str) -> Option<Vec<Item>> {
    let lang = lang_for(path)?;
    let lines: Vec<&str> = text.lines().collect();
    let mut items = Vec::new();
    let mut in_fence = false;
    for (i, line) in lines.iter().enumerate() {
        if lang == Lang::Markdown {
            if line.trim_start().starts_with("```") {
                in_fence = !in_fence;
                continue;
            }
            if in_fence {
                continue;
            }
        }
        let Some(caps) = patterns(lang).iter().find_map(|re| re.captures(line)) else { continue };
        // Control-flow lines (`if (x) {`) look like methods in C-family syntax.
        if matches!(lang, Lang::Js | Lang::C | Lang::Jvm)
            && caps.name("name").is_some_and(|n| KEYWORDS.contains(&n.as_str()))
        {
            continue;
        }
        let end = match lang {
            Lang::Python | Lang::Ruby => indent_end(&lines, i, lang),
            Lang::Markdown => heading_end(&lines, i),
            _ => brace_end(&lines, i),
        };
        items.push(Item { start: i + 1, end: end + 1, depth: 0, text: signature(line) });
    }
    for k in 0..items.len() {
        let (s, e) = (items[k].start, items[k].end);
        let depth = items[..k].iter().filter(|o| o.start <= s && o.end >= e && (o.start, o.end) != (s, e)).count();
        items[k].depth = depth;
    }
    Some(items)
}

fn signature(line: &str) -> String {
    let t = line.trim();
    let t = t.strip_suffix('{').unwrap_or(t).trim_end();
    truncate_chars(t, 160)
}

fn indent_of(line: &str) -> usize {
    line.chars().take_while(|c| c.is_whitespace()).map(|c| if c == '\t' { 4 } else { 1 }).sum()
}

fn indent_end(lines: &[&str], start: usize, lang: Lang) -> usize {
    let base = indent_of(lines[start]);
    let mut last = start;
    for (k, line) in lines.iter().enumerate().skip(start + 1) {
        let t = line.trim_start();
        if t.is_empty() {
            continue;
        }
        let ind = indent_of(line);
        if ind <= base {
            if lang == Lang::Ruby && ind == base && t.starts_with("end") {
                return k;
            }
            // Closing bracket of a multi-line signature belongs to the definition.
            if t.starts_with(')') || t.starts_with(']') || t.starts_with('}') {
                last = k;
                continue;
            }
            return last;
        }
        last = k;
    }
    last
}

fn heading_level(line: &str) -> Option<usize> {
    let hashes = line.chars().take_while(|c| *c == '#').count();
    (1..=6).contains(&hashes).then_some(hashes).filter(|_| line[hashes..].starts_with([' ', '\t']))
}

fn heading_end(lines: &[&str], start: usize) -> usize {
    let level = heading_level(lines[start]).unwrap_or(1);
    let mut in_fence = false;
    for (k, line) in lines.iter().enumerate().skip(start + 1) {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        }
        if !in_fence && heading_level(line).is_some_and(|l| l <= level) {
            return k - 1;
        }
    }
    lines.len().saturating_sub(1)
}

/// End of a brace-delimited definition: where its braces balance. Strings and
/// `//` comments are skipped roughly; declarations without a body end on their
/// own line.
fn brace_end(lines: &[&str], start: usize) -> usize {
    let mut depth: i64 = 0;
    let mut opened = false;
    for (k, line) in lines.iter().enumerate().skip(start).take(20_000) {
        let mut in_str: Option<char> = None;
        let mut escaped = false;
        let mut prev = '\0';
        for ch in line.chars() {
            if let Some(q) = in_str {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == q {
                    in_str = None;
                }
                continue;
            }
            match ch {
                '"' | '`' => in_str = Some(ch),
                '/' if prev == '/' => break,
                '{' => {
                    depth += 1;
                    opened = true;
                }
                '}' => {
                    depth -= 1;
                    if opened && depth <= 0 {
                        return k;
                    }
                }
                ';' if !opened && depth == 0 => return k,
                _ => {}
            }
            prev = ch;
        }
        if !opened && k >= start + 5 {
            return start;
        }
    }
    if opened { lines.len().saturating_sub(1) } else { start }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans(path: &str, src: &str) -> Vec<(usize, usize, usize, String)> {
        outline(Path::new(path), src).unwrap().into_iter().map(|i| (i.start, i.end, i.depth, i.text)).collect()
    }

    #[test]
    fn rust_outline() {
        let src = "use x;\n\npub struct A {\n    x: u32,\n}\n\nimpl A {\n    pub fn new() -> Self {\n        Self { x: 0 }\n    }\n\n    fn s(&self) -> &str { \"}\" }\n}\n\nstruct Unit;\n";
        let got = spans("a.rs", src);
        assert_eq!(got[0], (3, 5, 0, "pub struct A".into()));
        assert_eq!(got[1], (7, 13, 0, "impl A".into()));
        assert_eq!(got[2], (8, 10, 1, "pub fn new() -> Self".into()));
        assert_eq!(got[3].0..=got[3].1, 12..=12);
        assert_eq!(got[4], (15, 15, 0, "struct Unit;".into()));
    }

    #[test]
    fn python_outline() {
        let src = "import os\n\nclass A:\n    def f(\n        self,\n    ):\n        return 1\n\n    async def g(self):\n        pass\n\ndef top():\n    pass\n";
        let got = spans("a.py", src);
        assert_eq!(got[0], (3, 10, 0, "class A:".into()));
        assert_eq!(got[1].0..=got[1].1, 4..=7);
        assert_eq!(got[1].2, 1);
        assert_eq!(got[2].0..=got[2].1, 9..=10);
        assert_eq!(got[3].0..=got[3].1, 12..=13);
    }

    #[test]
    fn typescript_outline_skips_control_flow() {
        let src = "export function a(x: number): number {\n  if (x) {\n    return 1;\n  }\n  return 2;\n}\nexport const b = async (y: string) => {\n  return y;\n};\nclass C {\n  method(z) {\n    for (const q of z) {\n    }\n  }\n}\n";
        let got = spans("a.ts", src);
        let names: Vec<_> = got.iter().map(|g| g.3.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "export function a(x: number): number",
                "export const b = async (y: string) =>",
                "class C",
                "method(z)"
            ]
        );
        assert_eq!((got[0].0, got[0].1), (1, 6));
        assert_eq!((got[3].0, got[3].1, got[3].2), (11, 14, 1));
    }

    #[test]
    fn go_and_markdown() {
        let go =
            "package x\n\ntype T struct {\n\tA int\n}\n\nfunc (t *T) M() error {\n\treturn nil\n}\n\ntype ID int\n";
        let got = spans("x.go", go);
        assert_eq!(got.iter().map(|g| (g.0, g.1)).collect::<Vec<_>>(), vec![(3, 5), (7, 9), (11, 11)]);
        let md = "# Title\nintro\n## A\ntext\n```\n# not a heading\n```\n## B\nmore\n";
        let got = spans("r.md", md);
        assert_eq!(got.iter().map(|g| (g.0, g.1, g.2)).collect::<Vec<_>>(), vec![(1, 9, 0), (3, 7, 1), (8, 9, 1)]);
    }

    #[test]
    fn unsupported_language() {
        assert!(outline(Path::new("x.unknown"), "abc").is_none());
    }
}
