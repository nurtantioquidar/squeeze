//! `squeeze bench`: paired A/B benchmark of Claude Code with and without squeeze.
//!
//! It follows the structure of the SessionBench methodology published for
//! Fermat (github.com/akuik/sessionbench-swe): bundles of related tickets run
//! as one resumed session, a simulated user that reviews each ticket and can
//! push back, per-ticket reward = tests x rubric judge, K paired repeats. It
//! adds ablation arms and a bootstrap confidence interval.
//!
//! Terms-compliant by construction:
//! - Claude Code is run unmodified and authenticates itself: your subscription
//!   login, a `claude setup-token` token (CLAUDE_CODE_OAUTH_TOKEN, which
//!   Anthropic documents for scripts and CI), or an API key.
//! - squeeze arms use only Claude Code's documented extension points: a
//!   PostToolUse hook passed with `--settings` and an MCP server passed with
//!   `--mcp-config`. With subscription auth no proxy is involved anywhere.
//! - The `squeeze-proxy` arm sees API traffic, so it requires an API key.

use anyhow::{Context as _, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

pub const DEFAULT_TOOLS: &[&str] = &["Read", "Edit", "Write", "Glob", "Grep", "Bash"];

#[derive(Debug, Deserialize, Default, Clone)]
pub struct Defaults {
    pub model: Option<String>,
    pub effort: Option<String>,
    /// Model for the simulated user; omit to disable review rounds.
    pub usersim_model: Option<String>,
    /// Model for the rubric judge; omit to score on tests only.
    pub judge_model: Option<String>,
    pub max_pushbacks: Option<u32>,
    pub max_turns: Option<u32>,
    pub timeout_secs: Option<u64>,
    /// Idle time between tickets. Above the prompt-cache TTL it reproduces a
    /// "workday" regime where caches expire between tickets.
    pub pause_between_tickets_secs: Option<u64>,
    pub allowed_tools: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Ticket {
    pub id: String,
    pub prompt: String,
    /// Shell command, exit status 0 = pass. `$SQUEEZE_BENCH_HIDDEN` points at the
    /// bundle's hidden-tests directory, which the agent never sees.
    pub test: Option<String>,
    /// What a good solution looks like, for the judge.
    pub rubric: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Bundle {
    pub name: String,
    /// Directory copied fresh for every session (relative to the task file).
    pub repo: PathBuf,
    /// Hidden tests directory (relative to the task file), outside the agent's copy.
    pub hidden: Option<PathBuf>,
    pub setup: Option<String>,
    pub allowed_tools: Option<Vec<String>>,
    #[serde(rename = "ticket")]
    pub tickets: Vec<Ticket>,
}

#[derive(Debug, Deserialize)]
struct BenchFile {
    #[serde(default)]
    defaults: Defaults,
    #[serde(default, rename = "bundle")]
    bundles: Vec<Bundle>,
}

pub struct Plan {
    pub dir: PathBuf,
    pub defaults: Defaults,
    pub bundles: Vec<Bundle>,
}

pub fn load(path: &Path) -> Result<Plan> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let file: BenchFile = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    if file.bundles.is_empty() || file.bundles.iter().any(|b| b.tickets.is_empty()) {
        bail!("{} needs at least one [[bundle]] with at least one [[bundle.ticket]]", path.display());
    }
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    let dir = dir.canonicalize().unwrap_or(dir);
    Ok(Plan { dir, defaults: file.defaults, bundles: file.bundles })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Arm {
    Vanilla,
    /// Hook compression + MCP tools.
    Squeeze,
    /// Ablation: hook compression only.
    SqueezeHook,
    /// Ablation: MCP tools only.
    SqueezeMcp,
    /// Proxy compression + MCP tools (API key only).
    SqueezeProxy,
    /// Hook compression + MCP tools that replace Claude Code's file tools
    /// (Read/Edit/Write/Grep/Glob turned off), working guidance, and unused
    /// built-in tools trimmed (`squeeze run --replace-file-tools --trim-tools`).
    SqueezeFull,
}

impl Arm {
    pub fn name(self) -> &'static str {
        match self {
            Arm::Vanilla => "vanilla",
            Arm::Squeeze => "squeeze",
            Arm::SqueezeHook => "squeeze-hook",
            Arm::SqueezeMcp => "squeeze-mcp",
            Arm::SqueezeProxy => "squeeze-proxy",
            Arm::SqueezeFull => "squeeze-full",
        }
    }
    pub(crate) fn uses_mcp(self) -> bool {
        matches!(self, Arm::Squeeze | Arm::SqueezeMcp | Arm::SqueezeProxy | Arm::SqueezeFull)
    }
    pub(crate) fn uses_hook(self) -> bool {
        matches!(self, Arm::Squeeze | Arm::SqueezeHook | Arm::SqueezeFull)
    }
    /// squeeze's tools replace Claude Code's built-in file tools.
    pub(crate) fn replaces_file_tools(self) -> bool {
        self == Arm::SqueezeFull
    }
    /// squeeze tools this arm allows.
    pub(crate) fn tool_names(self) -> Vec<String> {
        let mut t = if self.uses_mcp() { squeeze_tool_names() } else { Vec::new() };
        if self.replaces_file_tools() {
            t.push("mcp__squeeze__edit_many".into());
        }
        t
    }
    /// Built-in tools this arm turns off (on top of the bench's own list).
    pub(crate) fn disallowed_tools(self) -> Vec<String> {
        if self.replaces_file_tools() {
            crate::mcp::REPLACED_BUILTINS.iter().chain(crate::mcp::TRIMMED_BUILTINS).map(|s| s.to_string()).collect()
        } else {
            Vec::new()
        }
    }
    /// Extra `claude` arguments for this arm, other than tools.
    pub(crate) fn extra_args(self) -> Vec<String> {
        if self.replaces_file_tools() {
            vec!["--append-system-prompt".into(), crate::mcp::FILE_TOOLS_GUIDE.into()]
        } else {
            Vec::new()
        }
    }
}

impl std::str::FromStr for Arm {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "vanilla" => Arm::Vanilla,
            "squeeze" => Arm::Squeeze,
            "squeeze-hook" => Arm::SqueezeHook,
            "squeeze-mcp" => Arm::SqueezeMcp,
            "squeeze-proxy" => Arm::SqueezeProxy,
            "squeeze-full" => Arm::SqueezeFull,
            other => {
                bail!(
                    "unknown arm {other:?} (vanilla, squeeze, squeeze-full, squeeze-hook, squeeze-mcp, squeeze-proxy)"
                )
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Auth {
    /// Claude Code's own login (subscription or `claude setup-token`). API-key
    /// and base-URL variables are removed from the child environment so the
    /// session goes straight to Anthropic with that login.
    Subscription,
    ApiKey,
}

pub struct Context {
    pub claude_bin: String,
    pub exe: PathBuf,
    pub store_root: PathBuf,
    pub auth: Auth,
    pub proxy_url: Option<String>,
    pub model_override: Option<String>,
    pub include_user_settings: bool,
    pub keep_dirs: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    /// Claude Code's own cost figure for the invocation (list prices).
    pub cost_usd: f64,
    pub input_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub output_tokens: u64,
    pub turns: u64,
    pub invocations: u64,
    pub duration_ms: u64,
}

impl Usage {
    pub(crate) fn add_result(&mut self, r: &Value) {
        let n = |v: &Value, k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
        self.cost_usd += r.get("total_cost_usd").and_then(Value::as_f64).unwrap_or(0.0);
        self.turns += n(r, "num_turns");
        self.duration_ms += n(r, "duration_ms");
        self.invocations += 1;
        if let Some(u) = r.get("usage") {
            self.input_tokens += n(u, "input_tokens");
            self.cache_creation_input_tokens += n(u, "cache_creation_input_tokens");
            self.cache_read_input_tokens += n(u, "cache_read_input_tokens");
            self.output_tokens += n(u, "output_tokens");
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TicketResult {
    pub id: String,
    pub tests_passed: Option<bool>,
    pub rubric: Option<f64>,
    /// tests (1/0, or 1 when there is no test) x rubric (or 1 without a judge).
    pub reward: f64,
    pub pushbacks: u32,
    pub error: Option<String>,
    /// SWE-Atlas: fractional test score (new relevant tests passing, regressions counted).
    #[serde(default)]
    pub tests_frac: Option<f64>,
    /// Why the ticket is left out of the session mean (e.g. vacuous tests), as SessionBench does.
    #[serde(default)]
    pub excluded: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionResult {
    pub bundle: String,
    pub arm: Arm,
    pub repeat: usize,
    pub model: Option<String>,
    /// The coding agent's usage: the headline cost.
    pub agent: Usage,
    /// Simulated user and judge usage, reported separately.
    pub helpers: Usage,
    pub tickets: Vec<TicketResult>,
    pub reward: f64,
    pub completed: bool,
    pub error: Option<String>,
    /// Claude Code session id (its transcript is `<id>.jsonl`).
    #[serde(default)]
    pub session_id: String,
    /// Outputs the squeeze hook compressed in this session, and bytes saved.
    #[serde(default)]
    pub hook_compressions: u64,
    #[serde(default)]
    pub hook_bytes_saved: u64,
    /// Tool calls made by the agent, by tool name (from the session transcript).
    #[serde(default)]
    pub tool_calls: BTreeMap<String, u64>,
}

impl SessionResult {
    /// Calls to squeeze's own MCP tools.
    pub fn squeeze_tool_calls(&self) -> u64 {
        self.tool_calls.iter().filter(|(k, _)| k.starts_with("mcp__squeeze__")).map(|(_, v)| v).sum()
    }
}

/// Raised when Claude Code reports a usage or rate limit: the run stops and
/// can be resumed later with the same command.
#[derive(Debug)]
pub struct LimitReached(pub String);
impl std::fmt::Display for LimitReached {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "usage or rate limit reached: {}", self.0)
    }
}
impl std::error::Error for LimitReached {}

/// Raised when a `claude -p` process exits without printing a result (it
/// crashed or was killed), as opposed to reporting an API error.
#[derive(Debug)]
pub struct NoResult(pub String);
impl std::fmt::Display for NoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for NoResult {}

pub(crate) fn uuid_v4(seed: &str) -> String {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let h = blake3::hash(format!("{seed}:{nanos}:{}", std::process::id()).as_bytes());
    let mut b: [u8; 16] = h.as_bytes()[..16].try_into().expect("16 bytes");
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!("{}-{}-{}-{}-{}", &hex[0..8], &hex[8..12], &hex[12..16], &hex[16..20], &hex[20..32])
}

fn mcp_config(ctx: &Context, root: &Path, edit_tool: bool) -> String {
    let mut args = vec!["mcp".to_string(), "--root".into(), root.display().to_string()];
    if edit_tool {
        args.push("--edit-tool".into());
    }
    json!({ "mcpServers": { "squeeze": {
        "type": "stdio",
        "command": ctx.exe,
        "args": args,
        "env": { "SQUEEZE_HOME": ctx.store_root },
        "alwaysLoad": true
    }}})
    .to_string()
}

pub(crate) fn squeeze_tool_names() -> Vec<String> {
    crate::mcp::READ_ONLY_TOOLS.iter().map(|t| format!("mcp__squeeze__{t}")).collect()
}

fn apply_auth(cmd: &mut Command, ctx: &Context) {
    if ctx.auth == Auth::Subscription {
        for var in ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"] {
            cmd.env_remove(var);
        }
    }
    cmd.env("SQUEEZE_HOME", &ctx.store_root);
}

/// Run `claude -p` once on this machine and return its JSON result.
pub(crate) async fn claude(
    ctx: &Context,
    cwd: &Path,
    args: &[String],
    prompt: &str,
    timeout: u64,
    base_url: Option<&str>,
) -> Result<Value> {
    let mut cmd = Command::new(&ctx.claude_bin);
    cmd.current_dir(cwd).args(args);
    apply_auth(&mut cmd, ctx);
    if let Some(url) = base_url {
        cmd.env("ANTHROPIC_BASE_URL", url);
    }
    // Keep one Claude Code version for the whole run.
    cmd.env("DISABLE_AUTOUPDATER", "1");
    run_claude_command(cmd, prompt, timeout).await
}

const HELPER_RETRY_PAUSE: Duration = if cfg!(test) { Duration::ZERO } else { Duration::from_secs(10) };

/// [`claude`] for the simulated user and the judge. Their calls are stateless,
/// so a process that crashed without a result is retried (twice).
pub(crate) async fn helper(ctx: &Context, cwd: &Path, args: &[String], prompt: &str, timeout: u64) -> Result<Value> {
    let mut attempt = 0;
    loop {
        match claude(ctx, cwd, args, prompt, timeout, None).await {
            Err(e) if e.is::<NoResult>() && attempt < 2 => {
                attempt += 1;
                eprintln!("      helper claude exited without a result; retrying ({attempt}/2)");
                tokio::time::sleep(HELPER_RETRY_PAUSE).await;
            }
            r => return r,
        }
    }
}

/// Run a prepared `claude -p` command (local, or `container exec ... claude`),
/// feeding the prompt on stdin, and parse its JSON result. Usage and rate
/// limits surface as [`LimitReached`].
pub(crate) async fn run_claude_command(mut cmd: Command, prompt: &str, timeout: u64) -> Result<Value> {
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    let mut child = cmd.spawn().context("starting claude")?;
    let mut stdin = child.stdin.take().context("claude stdin")?;
    stdin.write_all(prompt.as_bytes()).await?;
    drop(stdin);
    let out = tokio::time::timeout(Duration::from_secs(timeout), child.wait_with_output())
        .await
        .map_err(|_| anyhow!("claude timed out after {timeout}s"))??;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let result: Value = stdout
        .lines()
        .rev()
        .find_map(|l| serde_json::from_str::<Value>(l).ok().filter(|v| v.get("type").is_some()))
        .or_else(|| serde_json::from_str(stdout.trim()).ok())
        .ok_or_else(|| {
            let err = String::from_utf8_lossy(&out.stderr);
            NoResult(format!(
                "no JSON result from claude (exit {:?}): {}",
                out.status.code(),
                crate::util::truncate_chars(err.trim(), 400)
            ))
        })?;
    if result.get("is_error").and_then(Value::as_bool) == Some(true) {
        let text = result.get("result").and_then(Value::as_str).unwrap_or("").to_string();
        let status = result.get("api_error_status").and_then(Value::as_u64);
        let lower = text.to_lowercase();
        if status == Some(429)
            || lower.contains("usage limit")
            || lower.contains("rate limit")
            || lower.contains("limit reached")
        {
            return Err(LimitReached(crate::util::truncate_chars(&text, 300)).into());
        }
    }
    Ok(result)
}

async fn sh(script: &str, dir: &Path, hidden: Option<&Path>) -> Result<(bool, String)> {
    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(script).current_dir(dir).stdin(Stdio::null());
    if let Some(h) = hidden {
        cmd.env("SQUEEZE_BENCH_HIDDEN", h);
    }
    let out = cmd.output().await?;
    let mut text = String::from_utf8_lossy(&out.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    Ok((out.status.success(), text))
}

async fn git(dir: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .args(["-c", "user.name=squeeze-bench", "-c", "user.email=bench@squeeze.invalid", "-c", "commit.gpgsign=false"])
        .args(args)
        .current_dir(dir)
        .output()
        .await?;
    if !out.status.success() {
        bail!("git {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

pub(crate) fn last_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// Parse the first JSON object in a model reply.
pub(crate) fn parse_json_reply(text: &str) -> Option<Value> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    serde_json::from_str(&text[start..=end]).ok()
}

const USERSIM_PROMPT: &str = "You simulate the developer who filed a ticket for a coding agent working in their \
repository. You are shown the ticket, the agent's final reply, the diff it produced, and the result of the ticket's \
tests. Decide whether you would accept the work, as a demanding but fair reviewer: push back if the tests fail, if part \
of the ticket is ignored, or if tests were weakened to make them pass. Reply with only a JSON object: \
{\"approve\": true or false, \"message\": \"what you would tell the agent; if not approving, a concise and specific request\"}";

const JUDGE_PROMPT: &str = "You grade a coding agent's change against a rubric. You are shown the ticket, the rubric, \
the diff and the test result. Judge only what the diff does. Reply with only a JSON object: \
{\"score\": a number from 0 to 1, \"reason\": \"one sentence\"}";

pub(crate) fn helper_args(model: &str, system: &str) -> Vec<String> {
    let mut a: Vec<String> =
        ["-p", "--output-format", "json", "--model", model, "--tools", ""].iter().map(|s| s.to_string()).collect();
    a.extend(
        ["--no-session-persistence", "--strict-mcp-config", "--mcp-config", r#"{"mcpServers":{}}"#].map(String::from),
    );
    a.extend(["--setting-sources", "project", "--system-prompt", system].map(String::from));
    a
}

pub struct Run<'a> {
    pub plan: &'a Plan,
    pub bundle: &'a Bundle,
    pub arm: Arm,
    pub repeat: usize,
    pub ctx: &'a Context,
}

/// Run one bundle as one resumed session. `Err` only for a usage/rate limit
/// (the session is then discarded so it reruns on resume); other failures are
/// recorded in the result.
pub async fn run_session(r: Run<'_>) -> Result<SessionResult> {
    let d = &r.plan.defaults;
    let model = r.ctx.model_override.clone().or_else(|| d.model.clone());
    let mut res = SessionResult {
        bundle: r.bundle.name.clone(),
        arm: r.arm,
        repeat: r.repeat,
        model: model.clone(),
        agent: Usage::default(),
        helpers: Usage::default(),
        tickets: Vec::new(),
        reward: 0.0,
        completed: false,
        error: None,
        session_id: String::new(),
        hook_compressions: 0,
        hook_bytes_saved: 0,
        tool_calls: BTreeMap::new(),
    };
    match session_inner(&r, model.as_deref(), &mut res).await {
        Ok(()) => res.completed = res.tickets.iter().all(|t| t.error.is_none()),
        Err(e) if e.is::<LimitReached>() => return Err(e),
        Err(e) => res.error = Some(format!("{e:#}")),
    }
    res.reward = session_reward(&res.tickets);
    Ok(res)
}

async fn session_inner(r: &Run<'_>, model: Option<&str>, res: &mut SessionResult) -> Result<()> {
    let (d, ctx, bundle) = (&r.plan.defaults, r.ctx, r.bundle);
    if r.arm == Arm::SqueezeProxy && (ctx.auth != Auth::ApiKey || ctx.proxy_url.is_none()) {
        bail!("the squeeze-proxy arm needs --auth api-key");
    }
    let tmp = tempfile::Builder::new().prefix("squeeze-bench-").tempdir()?;
    let work = tmp.path().join("repo");
    let repo = r.plan.dir.join(&bundle.repo);
    if !Command::new("cp").arg("-R").arg(&repo).arg(&work).status().await?.success() {
        bail!("copying {} failed", repo.display());
    }
    let work = work.canonicalize()?;
    let hidden = bundle.hidden.as_ref().map(|h| r.plan.dir.join(h));
    if let Some(setup) = &bundle.setup {
        let (ok, out) = sh(setup, &work, hidden.as_deref()).await?;
        if !ok {
            bail!("setup failed: {}", last_lines(&out, 20));
        }
    }
    if !work.join(".git").exists() {
        git(&work, &["init", "-q"]).await?;
    }
    std::fs::write(work.join(".git/info/exclude"), "__pycache__/\n*.pyc\nnode_modules/\ntarget/\n")?;
    git(&work, &["add", "-A"]).await?;
    git(&work, &["commit", "-qm", "baseline", "--allow-empty"]).await?;

    let timeout = d.timeout_secs.unwrap_or(1800);
    let mut tools: Vec<String> = bundle
        .allowed_tools
        .clone()
        .or_else(|| d.allowed_tools.clone())
        .unwrap_or_else(|| DEFAULT_TOOLS.iter().map(|s| s.to_string()).collect());
    tools.extend(r.arm.tool_names());
    let mcp = if r.arm.uses_mcp() {
        mcp_config(ctx, &work, r.arm.replaces_file_tools())
    } else {
        r#"{"mcpServers":{}}"#.to_string()
    };
    let settings = if r.arm.uses_hook() {
        Some(crate::hook::session_settings(&ctx.exe, r.arm.uses_mcp()))
    } else if r.arm.uses_mcp() {
        Some(json!({ "permissions": { "allow": r.arm.tool_names() } }))
    } else {
        None
    };
    let base_url = (r.arm == Arm::SqueezeProxy).then(|| ctx.proxy_url.clone()).flatten();
    let session_id = uuid_v4(&format!("{}/{}/{}", bundle.name, r.arm.name(), r.repeat));
    res.session_id = session_id.clone();
    let mut started = false;

    let agent_args = |resume: bool| -> Vec<String> {
        let mut a: Vec<String> =
            ["-p", "--output-format", "json", "--permission-mode", "acceptEdits"].map(String::from).to_vec();
        if let Some(m) = model {
            a.extend(["--model".into(), m.to_string()]);
        }
        if let Some(e) = &d.effort {
            a.extend(["--effort".into(), e.clone()]);
        }
        if let Some(t) = d.max_turns {
            a.extend(["--max-turns".into(), t.to_string()]);
        }
        if !ctx.include_user_settings {
            a.extend(["--setting-sources".into(), "project,local".into()]);
        }
        a.extend(["--strict-mcp-config".into(), "--mcp-config".into(), mcp.clone()]);
        if let Some(s) = &settings {
            a.extend(["--settings".into(), s.to_string()]);
        }
        a.extend(r.arm.extra_args());
        let disallowed = r.arm.disallowed_tools();
        if !disallowed.is_empty() {
            a.push("--disallowedTools".into());
            a.extend(disallowed);
        }
        a.extend([if resume { "--resume" } else { "--session-id" }.to_string(), session_id.clone()]);
        a.push("--allowedTools".into()); // variadic: keep last
        a.extend(tools.iter().cloned());
        a
    };

    for (i, ticket) in bundle.tickets.iter().enumerate() {
        if i > 0
            && let Some(p) = d.pause_between_tickets_secs.filter(|p| *p > 0)
        {
            tokio::time::sleep(Duration::from_secs(p)).await;
        }
        let mut tr = TicketResult {
            id: ticket.id.clone(),
            tests_passed: None,
            rubric: None,
            reward: 0.0,
            pushbacks: 0,
            error: None,
            tests_frac: None,
            excluded: None,
        };
        let mut message = ticket.prompt.clone();
        let (mut passed, mut test_out) = (None, String::new());
        loop {
            let result = claude(ctx, &work, &agent_args(started), &message, timeout, base_url.as_deref()).await?;
            started = true;
            res.agent.add_result(&result);
            let reply = result.get("result").and_then(Value::as_str).unwrap_or("").to_string();
            if result.get("is_error").and_then(Value::as_bool) == Some(true) {
                tr.error = Some(crate::util::truncate_chars(&reply, 300));
            }
            if let Some(test) = &ticket.test {
                let (ok, out) = sh(test, &work, hidden.as_deref()).await?;
                passed = Some(ok);
                test_out = out;
            }
            let Some(usersim) = d.usersim_model.as_deref() else { break };
            if tr.pushbacks >= d.max_pushbacks.unwrap_or(1) {
                break;
            }
            git(&work, &["add", "-A"]).await?;
            let diff = git(&work, &["diff", "--cached", "HEAD"]).await?;
            let review = format!(
                "Ticket:\n{}\n\nAgent's final reply:\n{}\n\nDiff:\n{}\n\nTest result: {}\n{}",
                ticket.prompt,
                crate::util::truncate_chars(&reply, 4000),
                crate::util::truncate_chars(&diff, 20_000),
                match passed {
                    Some(true) => "passed",
                    Some(false) => "FAILED",
                    None => "no tests",
                },
                last_lines(&test_out, 40),
            );
            let verdict_reply = helper(ctx, tmp.path(), &helper_args(usersim, USERSIM_PROMPT), &review, 600).await?;
            res.helpers.add_result(&verdict_reply);
            let verdict = verdict_reply.get("result").and_then(Value::as_str).and_then(parse_json_reply);
            let approve = verdict.as_ref().and_then(|v| v.get("approve")).and_then(Value::as_bool).unwrap_or(true);
            if approve {
                break;
            }
            message = verdict
                .and_then(|v| v.get("message").and_then(Value::as_str).map(str::to_string))
                .unwrap_or_else(|| "Please review the ticket again; the work is not complete.".into());
            tr.pushbacks += 1;
        }

        git(&work, &["add", "-A"]).await?;
        let diff = git(&work, &["diff", "--cached", "HEAD"]).await?;
        if let (Some(judge), Some(rubric)) = (d.judge_model.as_deref(), ticket.rubric.as_deref()) {
            let prompt = format!(
                "Ticket:\n{}\n\nRubric:\n{rubric}\n\nDiff:\n{}\n\nTest result: {}",
                ticket.prompt,
                crate::util::truncate_chars(&diff, 30_000),
                passed.map_or("no tests", |p| if p { "passed" } else { "FAILED" })
            );
            let reply = helper(ctx, tmp.path(), &helper_args(judge, JUDGE_PROMPT), &prompt, 600).await?;
            res.helpers.add_result(&reply);
            tr.rubric = reply
                .get("result")
                .and_then(Value::as_str)
                .and_then(parse_json_reply)
                .and_then(|v| v.get("score").and_then(Value::as_f64))
                .map(|s| s.clamp(0.0, 1.0));
        }
        tr.tests_passed = passed;
        tr.reward = passed.map_or(1.0, |p| p as u8 as f64) * tr.rubric.unwrap_or(1.0);
        git(&work, &["commit", "-qm", &format!("ticket {}", ticket.id), "--allow-empty"]).await?;
        res.tickets.push(tr);
    }
    record_activity(ctx, res);
    if ctx.keep_dirs {
        eprintln!("  kept work dir: {}", tmp.keep().display());
    }
    Ok(())
}

/// What squeeze actually did in the session: hook compressions (from the hook's
/// own records) and the agent's tool calls (from Claude Code's transcript,
/// whose format is undocumented, so this is best effort).
fn record_activity(ctx: &Context, res: &mut SessionResult) {
    let config = std::env::var_os("CLAUDE_CONFIG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| crate::util::home_dir().join(".claude"));
    record_activity_in(&ctx.store_root, &config, res);
}

/// [`record_activity`] for an explicit squeeze store and Claude config directory.
pub(crate) fn record_activity_in(store_root: &Path, config: &Path, res: &mut SessionResult) {
    res.hook_compressions = 0;
    res.hook_bytes_saved = 0;
    res.tool_calls.clear();
    if let Ok(store) = crate::store::Store::open(store_root) {
        for rec in store.records().unwrap_or_default() {
            if rec["mode"] == "hook" && rec["session"] == res.session_id.as_str() {
                res.hook_compressions += 1;
                res.hook_bytes_saved += rec["new_bytes_saved"].as_u64().unwrap_or(0);
            }
        }
    }
    let name = format!("{}.jsonl", res.session_id);
    let transcript = std::fs::read_dir(config.join("projects"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|d| d.path().join(&name))
        .find(|p| p.exists());
    let Some(text) = transcript.and_then(|p| std::fs::read_to_string(p).ok()) else { return };
    let mut seen = HashSet::new();
    for line in text.lines() {
        let Ok(e) = serde_json::from_str::<Value>(line) else { continue };
        if e.get("isSidechain").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        for b in e.pointer("/message/content").and_then(Value::as_array).into_iter().flatten() {
            if b["type"] == "tool_use"
                && let (Some(id), Some(name)) = (b["id"].as_str(), b["name"].as_str())
                && seen.insert(id.to_string())
            {
                *res.tool_calls.entry(name.to_string()).or_default() += 1;
            }
        }
    }
}

/// Mean reward over the tickets that count (excluded ones are left out, as in SessionBench).
pub(crate) fn session_reward(tickets: &[TicketResult]) -> f64 {
    let graded: Vec<f64> = tickets.iter().filter(|t| t.excluded.is_none()).map(|t| t.reward).collect();
    mean(&graded).unwrap_or(0.0)
}

/// Sessions already recorded in an existing results file (for resuming).
pub fn completed_keys(results: &[SessionResult]) -> HashSet<(String, Arm, usize)> {
    results.iter().map(|r| (r.bundle.clone(), r.arm, r.repeat)).collect()
}

pub fn read_results(path: &Path) -> Vec<SessionResult> {
    std::fs::read_to_string(path).unwrap_or_default().lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
}

fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

fn sd(v: &[f64]) -> f64 {
    let Some(m) = mean(v) else { return 0.0 };
    if v.len() < 2 {
        return 0.0;
    }
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/// Paired bootstrap 95% interval for mean(arm cost) / mean(vanilla cost),
/// resampling repeat indices. Deterministic (fixed seed).
fn bootstrap_ratio(pairs: &[(f64, f64)]) -> Option<(f64, f64)> {
    if pairs.len() < 2 {
        return None;
    }
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut ratios: Vec<f64> = (0..2000)
        .filter_map(|_| {
            let (mut a, mut b) = (0.0, 0.0);
            for _ in 0..pairs.len() {
                let (x, y) = pairs[(next() % pairs.len() as u64) as usize];
                a += x;
                b += y;
            }
            (b > 0.0).then(|| a / b)
        })
        .collect();
    ratios.sort_by(f64::total_cmp);
    (!ratios.is_empty()).then(|| (ratios[ratios.len() * 25 / 1000], ratios[ratios.len() * 975 / 1000]))
}

pub fn summarize(results: &[SessionResult]) -> String {
    let mut groups: BTreeMap<(String, Arm), Vec<&SessionResult>> = BTreeMap::new();
    for r in results {
        groups.entry((r.bundle.clone(), r.arm)).or_default().push(r);
        groups.entry(("(all bundles)".into(), r.arm)).or_default().push(r);
    }
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:<18} {:<14} {:>3} {:>10} {:>7} {:>7} {:>9} {:>9} {:>8} {:>24}",
        "bundle",
        "arm",
        "n",
        "agent $",
        "turns",
        "reward",
        "reward sd",
        "squeezed",
        "sq tools",
        "cost vs vanilla (95% CI)"
    );
    let mut bundles: Vec<String> = groups.keys().map(|(b, _)| b.clone()).collect();
    bundles.dedup();
    bundles.sort_by_key(|b| b == "(all bundles)");
    for b in bundles {
        let vanilla: BTreeMap<(String, usize), f64> = groups
            .get(&(b.clone(), Arm::Vanilla))
            .map(|v| v.iter().map(|r| ((r.bundle.clone(), r.repeat), r.agent.cost_usd)).collect())
            .unwrap_or_default();
        for ((gb, arm), rs) in groups.iter().filter(|((gb, _), _)| *gb == b) {
            let costs: Vec<f64> = rs.iter().map(|r| r.agent.cost_usd).collect();
            let rewards: Vec<f64> = rs.iter().map(|r| r.reward).collect();
            let turns: Vec<f64> = rs.iter().map(|r| r.agent.turns as f64).collect();
            let squeezed: Vec<f64> = rs.iter().map(|r| r.hook_compressions as f64).collect();
            let sq_tools: Vec<f64> = rs.iter().map(|r| r.squeeze_tool_calls() as f64).collect();
            let cmp = if *arm == Arm::Vanilla {
                String::new()
            } else {
                let pairs: Vec<(f64, f64)> = rs
                    .iter()
                    .filter_map(|r| vanilla.get(&(r.bundle.clone(), r.repeat)).map(|v| (r.agent.cost_usd, *v)))
                    .collect();
                let (a, v) = pairs.iter().fold((0.0, 0.0), |acc, p| (acc.0 + p.0, acc.1 + p.1));
                match ((v > 0.0).then(|| a / v), bootstrap_ratio(&pairs)) {
                    (Some(x), Some((lo, hi))) => format!("{x:.2}x ({lo:.2}-{hi:.2})"),
                    (Some(x), None) => format!("{x:.2}x"),
                    _ => "-".into(),
                }
            };
            let _ = writeln!(
                out,
                "{:<18} {:<14} {:>3} {:>10} {:>7} {:>7} {:>9} {:>9} {:>8} {:>24}",
                crate::util::truncate_chars(gb, 18),
                arm.name(),
                rs.len(),
                mean(&costs).map_or("-".into(), |c| format!("{c:.4}")),
                mean(&turns).map_or("-".into(), |t| format!("{t:.1}")),
                mean(&rewards).map_or("-".into(), |r| format!("{r:.3}")),
                format!("{:.3}", sd(&rewards)),
                mean(&squeezed).map_or("-".into(), |v| format!("{v:.1}")),
                mean(&sq_tools).map_or("-".into(), |v| format!("{v:.1}")),
                cmp
            );
        }
    }
    let helper: f64 = results.iter().map(|r| r.helpers.cost_usd).sum();
    let agent: f64 = results.iter().map(|r| r.agent.cost_usd).sum();
    let _ = writeln!(
        out,
        "\nsqueezed = outputs the hook compressed per session; sq tools = calls to squeeze's MCP tools per session.\n\
         agent $ is Claude Code's own list-price figure for the coding agent; on a subscription it measures usage,\n\
         not a bill. Simulated user + judge used another ${helper:.4} (agent total ${agent:.4}). Sessions with errors: {}.",
        results.iter().filter(|r| !r.completed).count()
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in `claude` that dies by SIGSEGV until its `crash_first` runs are
    /// used up, then prints `reply`. Returns the context and its call counter.
    fn fake_claude(dir: &Path, crash_first: u32, reply: &str) -> (Context, PathBuf) {
        use std::os::unix::fs::PermissionsExt;
        let (bin, count) = (dir.join("claude"), dir.join("count"));
        std::fs::write(
            &bin,
            format!(
                "#!/bin/sh\ncat >/dev/null\nn=$(cat '{c}' 2>/dev/null || echo 0); n=$((n+1)); echo $n > '{c}'\n\
                 if [ $n -le {crash_first} ]; then kill -SEGV $$; fi\necho '{reply}'\n",
                c = count.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        let ctx = Context {
            claude_bin: bin.to_string_lossy().into(),
            exe: PathBuf::from("squeeze"),
            store_root: dir.join("store"),
            auth: Auth::ApiKey,
            proxy_url: None,
            model_override: None,
            include_user_settings: false,
            keep_dirs: false,
        };
        (ctx, count)
    }

    #[tokio::test]
    async fn helper_retries_a_crashed_claude_but_not_an_api_error() {
        let calls = |p: &Path| std::fs::read_to_string(p).unwrap().trim().to_string();

        let dir = tempfile::tempdir().unwrap();
        let (ctx, count) = fake_claude(dir.path(), 2, r#"{"type":"result","result":"ok"}"#);
        let r = helper(&ctx, dir.path(), &[], "hi", 30).await.unwrap();
        assert_eq!(r["result"], "ok");
        assert_eq!(calls(&count), "3", "two crashes, then a result");

        let dir = tempfile::tempdir().unwrap();
        let (ctx, count) = fake_claude(dir.path(), 3, r#"{"type":"result","result":"ok"}"#);
        let e = helper(&ctx, dir.path(), &[], "hi", 30).await.unwrap_err();
        assert!(e.is::<NoResult>(), "{e:#}");
        assert_eq!(calls(&count), "3", "gives up after two retries");

        let dir = tempfile::tempdir().unwrap();
        let (ctx, count) = fake_claude(dir.path(), 0, r#"{"type":"result","is_error":true,"result":"overloaded"}"#);
        let r = helper(&ctx, dir.path(), &[], "hi", 30).await.unwrap();
        assert_eq!(r["is_error"], true);
        assert_eq!(calls(&count), "1", "an API error is a result, not a crash");
    }

    fn session(arm: Arm, repeat: usize, cost: f64, reward: f64) -> SessionResult {
        SessionResult {
            bundle: "b".into(),
            arm,
            repeat,
            model: None,
            agent: Usage { cost_usd: cost, turns: 10, ..Default::default() },
            helpers: Usage::default(),
            tickets: vec![],
            reward,
            completed: true,
            error: None,
            session_id: String::new(),
            hook_compressions: 0,
            hook_bytes_saved: 0,
            tool_calls: BTreeMap::new(),
        }
    }

    #[test]
    fn summary_has_paired_ratio_and_ci() {
        let rs = vec![
            session(Arm::Vanilla, 0, 2.0, 0.8),
            session(Arm::Vanilla, 1, 3.0, 0.9),
            session(Arm::Squeeze, 0, 1.0, 0.8),
            session(Arm::Squeeze, 1, 2.0, 1.0),
        ];
        let s = summarize(&rs);
        assert!(s.contains("0.60x ("), "{s}");
        assert!(s.lines().any(|l| l.starts_with("(all bundles)")));
    }

    #[test]
    fn bootstrap_is_deterministic_and_brackets_the_ratio() {
        let pairs = vec![(1.0, 2.0), (2.0, 3.0), (1.5, 2.5), (1.2, 2.2)];
        let a = bootstrap_ratio(&pairs).unwrap();
        assert_eq!(a, bootstrap_ratio(&pairs).unwrap());
        assert!(a.0 <= 5.7 / 9.7 && 5.7 / 9.7 <= a.1);
    }

    #[test]
    fn parses_bundle_file_and_replies() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("b.toml");
        std::fs::write(
            &p,
            "[defaults]\nmodel = \"m\"\nusersim_model = \"u\"\n\n[[bundle]]\nname = \"x\"\nrepo = \"r\"\n\n[[bundle.ticket]]\nid = \"t1\"\nprompt = \"do\"\ntest = \"true\"\n",
        )
        .unwrap();
        let plan = load(&p).unwrap();
        assert_eq!(plan.bundles[0].tickets[0].id, "t1");
        assert_eq!(plan.defaults.usersim_model.as_deref(), Some("u"));
        assert_eq!(parse_json_reply("Sure:\n{\"approve\": false, \"message\": \"x\"}\n").unwrap()["approve"], false);
        assert_eq!(uuid_v4("a").chars().nth(14), Some('4'));
        assert_eq!("squeeze-hook".parse::<Arm>().unwrap(), Arm::SqueezeHook);
    }
}
