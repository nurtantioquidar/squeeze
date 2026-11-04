use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use serde_json::{Value, json};
use squeeze::auth::{self, Credential};
use squeeze::bench::{self, Arm};
use squeeze::mcp;
use squeeze::proxy::{DEFAULT_UPSTREAM, Proxy, ProxyConfig, Totals};
use squeeze::replay;
use squeeze::store::Store;
use squeeze::tools::Workspace;
use squeeze::transform::{ExpandHint, TransformConfig};
use squeeze::util::now_secs;
use std::collections::{BTreeMap, HashSet};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Parser)]
#[command(
    name = "squeeze",
    version,
    about = "Cut Claude Code token spend with a local, cache-aware compression proxy and token-efficient MCP tools.",
    after_help = "squeeze works with API-key or gateway credentials only. Claude subscription (OAuth) traffic is refused.\nNot affiliated with or endorsed by Anthropic."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Launch Claude Code through squeeze (put claude's own arguments after `--`)
    Run(RunArgs),
    /// Claude Code hook entry point (registered for you by `squeeze run`)
    #[command(hide = true)]
    Hook {
        /// Hook event; only post-tool-use is supported
        event: String,
        /// Where compressed-output notes point: mcp or cli
        #[arg(long, default_value = "cli")]
        hint: String,
    },
    /// Run only the proxy, for use with ANTHROPIC_BASE_URL (API-key or gateway credentials only)
    Proxy(ProxyArgs),
    /// Serve squeeze's MCP tools over stdio (Claude Code starts this for you)
    Mcp(McpArgs),
    /// Print the full original of an output squeeze compressed
    Expand { id: String },
    /// Show estimated savings from recent sessions
    Stats {
        #[arg(long, default_value_t = 10)]
        sessions: usize,
    },
    /// Estimate savings offline from past Claude Code transcripts (free, sends nothing)
    Replay(ReplayArgs),
    /// A/B benchmark tasks with and without squeeze (needs ANTHROPIC_API_KEY; costs money)
    Bench(BenchArgs),
    /// SessionBench-style runs over SWE-Atlas tasks in containers (see bench/swe-atlas-rf.toml)
    BenchAtlas(AtlasArgs),
    /// Delete stored originals older than N days
    Gc {
        #[arg(long, default_value_t = 30)]
        days: u64,
    },
}

#[derive(Args, Clone)]
struct CompressFlags {
    /// Forward requests unchanged (usage is still recorded)
    #[arg(long)]
    no_compress: bool,
    /// Don't replace repeated identical tool outputs with a reference
    #[arg(long)]
    no_dedup: bool,
    /// Experimental: prune stale context, only when the prompt cache has already expired
    #[arg(long)]
    prune_cold_cache: bool,
    /// Upstream API base URL [default: $ANTHROPIC_BASE_URL, else https://api.anthropic.com]
    #[arg(long)]
    upstream: Option<String>,
}

impl CompressFlags {
    fn transform(&self, hint: ExpandHint) -> Option<TransformConfig> {
        if self.no_compress || std::env::var("SQUEEZE_DISABLE").is_ok_and(|v| v == "1") {
            return None;
        }
        let mut t = TransformConfig { expand_hint: hint, dedup: !self.no_dedup, ..Default::default() };
        t.prune.enabled = self.prune_cold_cache;
        Some(t)
    }

    fn upstream(&self) -> String {
        self.upstream
            .clone()
            .or_else(|| std::env::var("ANTHROPIC_BASE_URL").ok().filter(|u| !u.trim().is_empty()))
            .unwrap_or_else(|| DEFAULT_UPSTREAM.to_string())
    }
}

#[derive(Args)]
struct RunArgs {
    #[command(flatten)]
    compress: CompressFlags,
    /// Route requests through the local proxy instead of using a hook (adds cross-call
    /// dedup; API-key or gateway credentials only)
    #[arg(long)]
    proxy: bool,
    /// Don't attach squeeze's MCP tools
    #[arg(long)]
    no_mcp: bool,
    /// Also expose the edit_many tool (Claude Code asks before each use)
    #[arg(long)]
    edit_tool: bool,
    /// Let squeeze's tools replace Claude Code's Read, Edit, Write, Grep and Glob, with
    /// guidance on using them (implies --edit-tool). Read can no longer show images or PDFs.
    #[arg(long, conflicts_with = "no_mcp")]
    replace_file_tools: bool,
    /// Turn off built-in tools a coding session rarely needs (Workflow, ScheduleWakeup,
    /// ReportFindings, ListAgents): each is sent in full with every request
    #[arg(long)]
    trim_tools: bool,
    /// With --proxy: Claude Code authenticates through an apiKeyHelper in your settings
    #[arg(long)]
    api_key_helper: bool,
    /// Don't print the savings summary on exit
    #[arg(long, short)]
    quiet: bool,
    /// The claude executable
    #[arg(long, default_value = "claude")]
    claude_bin: String,
    /// Arguments for claude
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    claude_args: Vec<String>,
}

#[derive(Args)]
struct ProxyArgs {
    #[command(flatten)]
    compress: CompressFlags,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8787)]
    port: u16,
}

#[derive(Args)]
struct McpArgs {
    /// Project root the tools may read [default: current directory]
    #[arg(long)]
    root: Option<PathBuf>,
    /// Allow reading files outside the project root
    #[arg(long)]
    allow_outside_root: bool,
    /// Expose the edit_many tool
    #[arg(long)]
    edit_tool: bool,
}

#[derive(Args)]
struct ReplayArgs {
    /// Transcript files or directories [default: ~/.claude/projects]
    paths: Vec<PathBuf>,
    /// Show at most this many sessions
    #[arg(long, default_value_t = 15)]
    limit: usize,
    /// Print JSON instead of a table
    #[arg(long)]
    json: bool,
    /// Try a different log line limit (default 250)
    #[arg(long)]
    max_lines: Option<usize>,
    /// Try a different minimum output size to touch, in bytes (default 1500)
    #[arg(long)]
    min_bytes: Option<usize>,
}

#[derive(Args)]
struct BenchArgs {
    /// Bundle file (TOML), see bench/example.toml
    tasks: PathBuf,
    /// How Claude Code authenticates: `subscription` (its own login, or a token from
    /// `claude setup-token` in CLAUDE_CODE_OAUTH_TOKEN) or `api-key` (ANTHROPIC_API_KEY)
    #[arg(long)]
    auth: String,
    /// Repeats per bundle and arm (SessionBench used 5)
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    /// Arms: vanilla, squeeze, squeeze-hook, squeeze-mcp, squeeze-proxy (api-key only)
    #[arg(long, value_delimiter = ',', default_value = "vanilla,squeeze")]
    arms: Vec<Arm>,
    /// Only these bundles
    #[arg(long, value_delimiter = ',')]
    bundles: Vec<String>,
    /// Agent model (overrides the bundle file)
    #[arg(long)]
    model: Option<String>,
    /// Simulated-user model (overrides the bundle file; `none` disables review rounds)
    #[arg(long)]
    usersim_model: Option<String>,
    /// Judge model (overrides the bundle file; `none` scores on tests only)
    #[arg(long)]
    judge_model: Option<String>,
    /// Results file (JSON lines). If it exists, finished sessions are skipped (resume).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Don't ask for confirmation
    #[arg(long)]
    yes: bool,
    /// Keep each session's work directory
    #[arg(long)]
    keep: bool,
    /// Also load your user-level Claude Code settings (plugins, hooks) in every arm
    #[arg(long)]
    include_user_settings: bool,
    #[arg(long, default_value = "claude")]
    claude_bin: String,
}

#[derive(Args)]
struct AtlasArgs {
    /// Bundle file (TOML), see bench/swe-atlas-rf.toml
    bundles_file: PathBuf,
    /// Skip grading during sessions (10-20 minutes per ticket); grade afterwards with --regrade
    #[arg(long)]
    defer_grading: bool,
    /// `subscription` (needs --token-env-file) or `api-key`; with --regrade, only to judge missing rubrics
    #[arg(long)]
    auth: Option<String>,
    /// Grade what's missing in --out from the saved patches: tests without a score, and with
    /// --auth, rubrics without a score (judge calls). Existing scores are kept. Writes <out>.regraded.jsonl
    #[arg(long)]
    regrade: bool,
    /// Env file holding CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`; passed to the
    /// container runtime by path, never read by squeeze
    #[arg(long)]
    token_env_file: Option<PathBuf>,
    /// SWE-Atlas checkout [default: ~/.squeeze/cache/SWE-Atlas]
    #[arg(long)]
    atlas_root: Option<PathBuf>,
    /// Official Claude Code linux-x64 binary [default: ~/.squeeze/cache/claude-code/<your claude version>/linux-x64/claude]
    #[arg(long)]
    claude_linux: Option<PathBuf>,
    /// squeeze built for linux x86_64 [default: ~/.squeeze/cache/bin/squeeze-linux-x64]
    #[arg(long)]
    squeeze_linux: Option<PathBuf>,
    /// Container runtime: `container` (Apple, as used by opossum) or `docker`
    #[arg(long, default_value = "container")]
    runtime: String,
    #[arg(long, default_value_t = 1)]
    repeats: usize,
    #[arg(long, value_delimiter = ',', default_value = "vanilla,squeeze")]
    arms: Vec<Arm>,
    /// Only these bundles
    #[arg(long, value_delimiter = ',')]
    bundles: Vec<String>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    usersim_model: Option<String>,
    #[arg(long)]
    judge_model: Option<String>,
    /// Judge all rubric items of a ticket in one call (SWE-Atlas uses one call per item)
    #[arg(long)]
    judge_batch: bool,
    #[arg(long, default_value_t = 10)]
    cpus: u32,
    #[arg(long, default_value = "16g")]
    memory: String,
    /// DNS servers for the task containers (e.g. 1.1.1.1 if your Mac uses a local resolver)
    #[arg(long, value_delimiter = ',')]
    dns: Vec<String>,
    /// Extra KEY=VALUE environment for the agent inside the container
    #[arg(long = "agent-env")]
    agent_env: Vec<String>,
    /// Results file (JSON lines); finished sessions are skipped on rerun
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    yes: bool,
    /// Host claude for the simulated user and judge
    #[arg(long, default_value = "claude")]
    claude_bin: String,
}

const SQUEEZE_TOOLS: &[&str] = mcp::READ_ONLY_TOOLS;

fn tool_names() -> Vec<String> {
    SQUEEZE_TOOLS.iter().map(|t| format!("mcp__squeeze__{t}")).collect()
}

fn init_logging(file: Option<&Path>, default_level: &str) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_env("SQUEEZE_LOG").unwrap_or_else(|_| EnvFilter::new(default_level));
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    match file.and_then(|p| std::fs::OpenOptions::new().create(true).append(true).open(p).ok()) {
        Some(f) => builder.with_ansi(false).with_writer(std::sync::Mutex::new(f)).init(),
        None => builder.with_writer(std::io::stderr).init(),
    }
}

fn mcp_config(store: &Store, root: &Path, edit_tool: bool) -> Result<String> {
    let exe = std::env::current_exe().context("locating the squeeze executable")?;
    let mut args = vec!["mcp".to_string(), "--root".to_string(), root.display().to_string()];
    if edit_tool {
        args.push("--edit-tool".into());
    }
    Ok(json!({
        "mcpServers": {
            "squeeze": {
                "type": "stdio",
                "command": exe,
                "args": args,
                "env": { "SQUEEZE_HOME": store.root() },
                // Load the tools in full (not behind tool search) and connect
                // before the first turn, so the tool list never changes mid-session.
                "alwaysLoad": true
            }
        }
    })
    .to_string())
}

const NO_CREDENTIAL: &str = "squeeze needs API-key or gateway credentials, and none were found.\n\
Anthropic's terms do not allow third-party tools to route Claude subscription (Pro/Max) credentials,\n\
so squeeze cannot run on a subscription login. Set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN for a\n\
gateway), or pass --api-key-helper if your Claude Code settings use an apiKeyHelper.\n\
See https://code.claude.com/docs/en/legal-and-compliance";

fn preflight(api_key_helper: bool) -> Result<()> {
    match auth::detect_credential(|k| std::env::var(k).ok()) {
        Credential::ApiKey | Credential::AuthToken => Ok(()),
        Credential::None if api_key_helper => Ok(()),
        Credential::None => bail!("{NO_CREDENTIAL}"),
        Credential::OAuthEnv => bail!(
            "the credential in your environment is a Claude subscription OAuth token.\n{}",
            auth::refusal_message(auth::Refusal::SubscriptionOAuth)
        ),
        Credential::CloudProvider(p) => {
            bail!("squeeze currently supports the Anthropic API only, and {p} is configured. Run claude directly.")
        }
    }
}

fn print_summary(t: &Totals) {
    let mut line = format!(
        "squeeze: {} requests, {} tool outputs rewritten ({} compressed, {} deduplicated{})",
        t.requests,
        t.rewritten(),
        t.compressed,
        t.deduped,
        if t.pruned > 0 { format!(", {} pruned", t.pruned) } else { String::new() }
    );
    if let Some(pct) = t.savings_pct() {
        line.push_str(&format!(
            ", ~{} input-token equivalents saved (est. {pct:.0}% of input-side cost)",
            human(t.saved_input_eq)
        ));
    }
    eprintln!("{line}");
    if t.refused > 0 {
        eprintln!("squeeze: refused {} request(s) that carried subscription credentials; use an API key", t.refused);
    }
}

fn human(n: f64) -> String {
    match n {
        n if n >= 1e6 => format!("{:.1}M", n / 1e6),
        n if n >= 1e3 => format!("{:.1}k", n / 1e3),
        n => format!("{n:.0}"),
    }
}

/// Merge `ours` into the user's own `--settings` argument if they passed one
/// (claude accepts a single `--settings`), otherwise append it.
fn merge_settings_arg(args: &mut Vec<String>, ours: Value) -> Result<()> {
    fn merge(dst: &mut Value, src: Value) {
        match (dst, src) {
            (Value::Object(d), Value::Object(s)) => {
                for (k, v) in s {
                    match d.get_mut(&k) {
                        Some(dv) => merge(dv, v),
                        None => {
                            d.insert(k, v);
                        }
                    }
                }
            }
            (Value::Array(d), Value::Array(s)) => d.extend(s),
            _ => {} // the user's own value wins
        }
    }
    let Some(i) = args.iter().position(|x| x == "--settings" || x.starts_with("--settings=")) else {
        args.push("--settings".into());
        args.push(ours.to_string());
        return Ok(());
    };
    let (raw, value_at) = match args[i].strip_prefix("--settings=") {
        Some(v) => (v.to_string(), None),
        None => (args.get(i + 1).cloned().context("--settings needs a value")?, Some(i + 1)),
    };
    let text = if raw.trim_start().starts_with('{') {
        raw
    } else {
        std::fs::read_to_string(&raw).with_context(|| format!("reading --settings file {raw}"))?
    };
    let mut theirs: Value = serde_json::from_str(&text).context("parsing your --settings JSON")?;
    merge(&mut theirs, ours);
    match value_at {
        Some(j) => args[j] = theirs.to_string(),
        None => args[i] = format!("--settings={theirs}"),
    }
    Ok(())
}

async fn cmd_run(a: RunArgs) -> Result<i32> {
    let store = Store::open(Store::default_root())?;
    let exe = std::env::current_exe().context("locating the squeeze executable")?;
    let started = now_secs();
    let mut args = a.claude_args.clone();
    let mut cmd = tokio::process::Command::new(&a.claude_bin);

    let proxy = if a.proxy {
        // Proxy mode sees API traffic, so it is limited to API-key credentials.
        preflight(a.api_key_helper)?;
        init_logging(Some(&store.root().join("logs/squeeze.log")), "info");
        let hint = if a.no_mcp { ExpandHint::Cli } else { ExpandHint::Mcp };
        let proxy = Proxy::new(ProxyConfig {
            upstream: a.compress.upstream(),
            transform: a.compress.transform(hint),
            store: store.clone(),
        })?;
        let addr = proxy.spawn("127.0.0.1:0").await?;
        tracing::info!("proxy on {addr}, upstream {}", a.compress.upstream());
        cmd.env("ANTHROPIC_BASE_URL", format!("http://{addr}"));
        Some(proxy)
    } else {
        // Hook mode: Claude Code talks to Anthropic itself, with whatever login the
        // user has. squeeze never sees credentials or traffic.
        None
    };

    let settings = if !a.proxy && !a.compress.no_compress {
        squeeze::hook::session_settings(&exe, !a.no_mcp)
    } else if !a.no_mcp {
        json!({ "permissions": { "allow": tool_names() } })
    } else {
        json!({})
    };
    if settings.as_object().is_some_and(|m| !m.is_empty()) {
        merge_settings_arg(&mut args, settings)?;
    }
    // Appended after the user's arguments: --disallowedTools and --mcp-config are
    // variadic and would otherwise swallow a trailing prompt.
    let mut off: Vec<&str> = Vec::new();
    if a.replace_file_tools {
        off.extend(squeeze::mcp::REPLACED_BUILTINS);
        args.push("--append-system-prompt".into());
        args.push(squeeze::mcp::FILE_TOOLS_GUIDE.into());
    }
    if a.trim_tools {
        off.extend(squeeze::mcp::TRIMMED_BUILTINS);
    }
    if !off.is_empty() {
        args.push("--disallowedTools".into());
        args.extend(off.iter().map(|s| s.to_string()));
    }
    if !a.no_mcp {
        args.push("--mcp-config".into());
        args.push(mcp_config(&store, &std::env::current_dir()?, a.edit_tool || a.replace_file_tools)?);
    }
    cmd.args(&args).env("SQUEEZE_HOME", store.root());
    let mut child = cmd.spawn().with_context(|| format!("could not start `{}`", a.claude_bin))?;
    // Ctrl-C belongs to claude (same process group); keep squeeze alive until it exits.
    let status = loop {
        tokio::select! {
            s = child.wait() => break s?,
            _ = tokio::signal::ctrl_c() => {}
        }
    };
    if !a.quiet && std::io::stderr().is_terminal() {
        match &proxy {
            Some(p) => print_summary(&p.totals()),
            None => print_hook_summary(&store, started),
        }
    }
    Ok(status.code().unwrap_or(1))
}

fn print_hook_summary(store: &Store, since: u64) {
    let recs: Vec<Value> = store
        .records()
        .unwrap_or_default()
        .into_iter()
        .filter(|r| r["mode"] == "hook" && r["ts"].as_u64().is_some_and(|t| t >= since))
        .collect();
    if recs.is_empty() {
        return;
    }
    let bytes: u64 = recs.iter().filter_map(|r| r["new_bytes_saved"].as_u64()).sum();
    eprintln!(
        "squeeze: compressed {} tool output(s), {} bytes (~{} tokens) kept out of context",
        recs.len(),
        human(bytes as f64),
        human(squeeze::util::approx_tokens(bytes as usize))
    );
}

async fn cmd_proxy(a: ProxyArgs) -> Result<()> {
    init_logging(None, "info");
    let store = Store::open(Store::default_root())?;
    if !matches!(a.host.as_str(), "127.0.0.1" | "localhost" | "::1") {
        eprintln!("squeeze: warning: listening on {} exposes the proxy to your network", a.host);
    }
    let proxy = Proxy::new(ProxyConfig {
        upstream: a.compress.upstream(),
        transform: a.compress.transform(ExpandHint::Cli),
        store,
    })?;
    let addr = proxy.spawn(&format!("{}:{}", a.host, a.port)).await?;
    eprintln!(
        "squeeze proxy on http://{addr} -> {}\n\n  export ANTHROPIC_BASE_URL=http://{addr}\n\nAPI-key or gateway credentials only; subscription (OAuth) requests are refused. Ctrl-C to stop.",
        a.compress.upstream()
    );
    tokio::signal::ctrl_c().await?;
    print_summary(&proxy.totals());
    Ok(())
}

fn cmd_mcp(a: McpArgs) -> Result<()> {
    init_logging(None, "warn");
    let root = match a.root {
        Some(r) => r,
        None => std::env::current_dir()?,
    };
    let ws = Workspace::new(&root, a.allow_outside_root, Store::open(Store::default_root())?)?;
    mcp::serve_stdio(mcp::Server::new(ws, a.edit_tool))
}

fn ago(ts: u64) -> String {
    let d = now_secs().saturating_sub(ts);
    match d {
        d if d < 120 => format!("{d}s ago"),
        d if d < 7200 => format!("{}m ago", d / 60),
        d if d < 172_800 => format!("{}h ago", d / 3600),
        d => format!("{}d ago", d / 86_400),
    }
}

fn cmd_stats(limit: usize) -> Result<()> {
    let store = Store::open(Store::default_root())?;
    #[derive(Default)]
    struct S {
        last: u64,
        requests: u64,
        rewritten: u64,
        bytes: u64,
        saved: f64,
        actual: f64,
    }
    let mut by: BTreeMap<String, S> = BTreeMap::new();
    for r in store.records()? {
        let key = r["session"].as_str().map(|s| s.chars().take(8).collect()).unwrap_or_else(|| "-".to_string());
        let s = by.entry(key).or_default();
        let n = |k: &str| r[k].as_u64().unwrap_or(0);
        s.last = s.last.max(n("ts"));
        s.requests += 1;
        s.rewritten += n("new_compressed") + n("new_deduped") + n("new_pruned");
        s.bytes += n("new_bytes_saved");
        s.saved += r["saved_input_eq"].as_f64().unwrap_or(0.0);
        s.actual += r["actual_input_eq"].as_f64().unwrap_or(0.0);
    }
    if by.is_empty() {
        println!("No squeeze sessions recorded yet. Run `squeeze run` to start one.");
        return Ok(());
    }
    let mut rows: Vec<_> = by.into_iter().collect();
    rows.sort_by_key(|(_, s)| std::cmp::Reverse(s.last));
    println!(
        "{:<10} {:>9} {:>9} {:>10} {:>12} {:>12}",
        "session", "last", "requests", "rewritten", "bytes saved", "est. saving"
    );
    let (mut ts, mut ta) = (0.0, 0.0);
    for (k, s) in rows.iter().take(limit) {
        ts += s.saved;
        ta += s.actual;
        let pct = if s.saved + s.actual > 0.0 {
            format!("{:.0}%", 100.0 * s.saved / (s.saved + s.actual))
        } else {
            "-".into()
        };
        println!(
            "{k:<10} {:>9} {:>9} {:>10} {:>12} {:>12}",
            ago(s.last),
            s.requests,
            s.rewritten,
            human(s.bytes as f64),
            pct
        );
    }
    if ts + ta > 0.0 {
        println!(
            "\nEstimated input-side saving across these sessions: {:.0}% (token counts are approximate)",
            100.0 * ts / (ts + ta)
        );
    }
    Ok(())
}

fn cmd_replay(a: ReplayArgs) -> Result<()> {
    let paths = if a.paths.is_empty() { vec![squeeze::util::home_dir().join(".claude/projects")] } else { a.paths };
    let files = replay::find_transcripts(&paths);
    if files.is_empty() {
        bail!(
            "no transcripts (*.jsonl) found under {}",
            paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
        );
    }
    let mut cfg = TransformConfig { expand_hint: ExpandHint::Mcp, ..Default::default() };
    if let Some(n) = a.max_lines {
        cfg.log.max_lines = n.max(cfg.log.head + cfg.log.tail + 10);
        cfg.log.head = cfg.log.head.min(n * 2 / 5);
        cfg.log.tail = cfg.log.tail.min(n * 3 / 5);
        cfg.log.max_lines = n.max(cfg.log.head + cfg.log.tail + 1);
    }
    if let Some(n) = a.min_bytes {
        cfg.min_bytes = n;
    }
    let mut reports = Vec::new();
    let mut skipped = 0;
    for f in &files {
        match replay::replay_file(f, &cfg) {
            Ok(Some(r)) => reports.push(r),
            Ok(None) | Err(_) => skipped += 1,
        }
    }
    if a.json {
        println!("{}", serde_json::to_string_pretty(&reports)?);
        return Ok(());
    }
    reports.sort_by(|x, y| y.actual_input_eq.total_cmp(&x.actual_input_eq));
    println!(
        "{:<44} {:>8} {:>8} {:>10} {:>11} {:>10}",
        "session (largest first)", "requests", "outputs", "rewritten", "bytes saved", "est. saving"
    );
    for r in reports.iter().take(a.limit) {
        let name = r
            .file
            .parent()
            .and_then(|p| p.file_name())
            .map(|p| p.to_string_lossy().trim_start_matches('-').to_string())
            .unwrap_or_default();
        let sess =
            r.file.file_stem().map(|s| s.to_string_lossy().chars().take(8).collect::<String>()).unwrap_or_default();
        let label = squeeze::util::truncate_chars(&format!("{sess} {name}"), 43);
        let pct = r.savings_pct().map_or("-".into(), |p| format!("{p:.1}%"));
        println!(
            "{label:<44} {:>8} {:>8} {:>10} {:>11} {:>10}",
            r.requests,
            r.tool_results,
            r.compressed + r.deduped,
            human(r.bytes_saved as f64),
            pct
        );
    }
    let saved: f64 = reports.iter().map(|r| r.saved_input_eq).sum();
    let actual: f64 = reports.iter().map(|r| r.actual_input_eq).sum();
    println!(
        "\n{} sessions analyzed ({} skipped). Estimated input-side saving overall: {:.1}%",
        reports.len(),
        skipped,
        if actual > 0.0 { 100.0 * saved / actual } else { 0.0 }
    );
    println!(
        "Estimate only: compression rules applied offline to recorded tool outputs, ~4 bytes/token, priced\n\
         with cache-write/read multipliers against the recorded API usage. Output tokens and subagent\n\
         threads are not included, and it says nothing about answer quality (use `squeeze bench` for that)."
    );
    Ok(())
}

async fn cmd_bench(a: BenchArgs) -> Result<()> {
    let auth = match a.auth.as_str() {
        "subscription" => bench::Auth::Subscription,
        "api-key" => bench::Auth::ApiKey,
        other => bail!("--auth must be `subscription` or `api-key`, not {other:?}"),
    };
    let cred = auth::detect_credential(|k| std::env::var(k).ok());
    let auth_note = match auth {
        bench::Auth::ApiKey => {
            if !matches!(cred, Credential::ApiKey | Credential::AuthToken) {
                bail!("--auth api-key needs ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN for a gateway)");
            }
            "billed to your API key".to_string()
        }
        bench::Auth::Subscription => {
            if a.arms.contains(&Arm::SqueezeProxy) {
                bail!("the squeeze-proxy arm routes API traffic through squeeze, so it needs --auth api-key");
            }
            let how = if std::env::var("CLAUDE_CODE_OAUTH_TOKEN").is_ok_and(|v| !v.is_empty()) {
                "the CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`"
            } else {
                "your Claude Code login"
            };
            format!(
                "unmodified Claude Code with {how}; squeeze only adds a hook and MCP tools and never sees credentials.\n\
                 This uses your subscription's usage limits (API-key variables are removed for these runs)"
            )
        }
    };
    let mut plan = bench::load(&a.tasks)?;
    if let Some(m) = &a.usersim_model {
        plan.defaults.usersim_model = (m != "none").then(|| m.clone());
    }
    if let Some(m) = &a.judge_model {
        plan.defaults.judge_model = (m != "none").then(|| m.clone());
    }
    if !a.bundles.is_empty() {
        plan.bundles.retain(|b| a.bundles.contains(&b.name));
        if plan.bundles.is_empty() {
            bail!("none of the requested bundles are in {}", a.tasks.display());
        }
    }
    let out_path = a.out.clone().unwrap_or_else(|| PathBuf::from(format!("squeeze-bench-{}.jsonl", now_secs())));
    let mut results = bench::read_results(&out_path);
    let done = bench::completed_keys(&results);
    let todo: Vec<(usize, usize, Arm)> = (0..a.repeats)
        .flat_map(|k| (0..plan.bundles.len()).map(move |b| (k, b)))
        .flat_map(|(k, b)| {
            let mut arms = a.arms.clone();
            if k % 2 == 1 {
                arms.reverse(); // alternate order to spread out time effects
            }
            arms.into_iter().map(move |arm| (k, b, arm))
        })
        .filter(|(k, b, arm)| !done.contains(&(plan.bundles[*b].name.clone(), *arm, *k)))
        .collect();
    let tickets: usize = todo.iter().map(|(_, b, _)| plan.bundles[*b].tickets.len()).sum();
    let d = &plan.defaults;
    eprintln!(
        "{} session(s) to run ({} ticket(s)); {} already recorded in {}.\nagent model: {}, simulated user: {}, judge: {}.\nAuth: {auth_note}.",
        todo.len(),
        tickets,
        results.len(),
        out_path.display(),
        a.model.as_deref().or(d.model.as_deref()).unwrap_or("Claude Code default"),
        d.usersim_model.as_deref().unwrap_or("off"),
        d.judge_model.as_deref().unwrap_or("off (tests only)"),
    );
    if todo.is_empty() {
        println!("{}", bench::summarize(&results));
        return Ok(());
    }
    if !a.yes {
        if !std::io::stdin().is_terminal() {
            bail!("pass --yes to confirm");
        }
        eprint!("Continue? [y/N] ");
        std::io::stderr().flush()?;
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        if !matches!(answer.trim(), "y" | "Y" | "yes") {
            bail!("cancelled");
        }
    }

    let store = Store::open(Store::default_root())?;
    init_logging(Some(&store.root().join("logs/squeeze.log")), "info");
    let proxy_url = if a.arms.contains(&Arm::SqueezeProxy) {
        let flags = CompressFlags { no_compress: false, no_dedup: false, prune_cold_cache: false, upstream: None };
        let proxy = Proxy::new(ProxyConfig {
            upstream: flags.upstream(),
            transform: flags.transform(ExpandHint::Mcp),
            store: store.clone(),
        })?;
        Some(format!("http://{}", proxy.spawn("127.0.0.1:0").await?))
    } else {
        None
    };
    let ctx = bench::Context {
        claude_bin: a.claude_bin.clone(),
        exe: std::env::current_exe()?,
        store_root: store.root().to_path_buf(),
        auth,
        proxy_url,
        model_override: a.model.clone(),
        include_user_settings: a.include_user_settings,
        keep_dirs: a.keep,
    };
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&out_path)?;
    for (n, (k, b, arm)) in todo.iter().enumerate() {
        let bundle = &plan.bundles[*b];
        eprintln!("[{}/{}] {} / {} / repeat {}", n + 1, todo.len(), bundle.name, arm.name(), k + 1);
        match bench::run_session(bench::Run { plan: &plan, bundle, arm: *arm, repeat: *k, ctx: &ctx }).await {
            Ok(r) => {
                if let Some(e) = &r.error {
                    eprintln!("  error: {e}");
                }
                writeln!(out, "{}", serde_json::to_string(&r)?)?;
                results.push(r);
            }
            Err(e) if e.is::<bench::LimitReached>() => {
                eprintln!(
                    "stopped: {e}\nRerun the same command with --out {} to resume where it stopped.",
                    out_path.display()
                );
                break;
            }
            Err(e) => return Err(e),
        }
    }
    println!("\n{}", bench::summarize(&results));
    println!("Results: {}", out_path.display());
    Ok(())
}

fn host_claude_version(bin: &str) -> Option<String> {
    let out = std::process::Command::new(bin).arg("--version").output().ok()?;
    String::from_utf8_lossy(&out.stdout).split_whitespace().next().map(str::to_string)
}

async fn cmd_bench_atlas(a: AtlasArgs) -> Result<()> {
    use squeeze::atlas;
    let auth = match a.auth.as_deref() {
        Some("subscription") => bench::Auth::Subscription,
        Some("api-key") => bench::Auth::ApiKey,
        Some(other) => bail!("--auth must be `subscription` or `api-key`, not {other:?}"),
        None if a.regrade => bench::Auth::ApiKey, // unused: regrading calls no model
        None => bail!("--auth is required (`subscription` or `api-key`)"),
    };
    let runtime = match a.runtime.as_str() {
        "container" => atlas::Runtime::Apple,
        "docker" => atlas::Runtime::Docker,
        other => bail!("--runtime must be `container` or `docker`, not {other:?}"),
    };
    match auth {
        _ if a.regrade => {}
        bench::Auth::Subscription => {
            let f = a.token_env_file.as_ref().context(
                "subscription runs need --token-env-file: a file with CLAUDE_CODE_OAUTH_TOKEN=... from `claude setup-token`",
            )?;
            // Checked by metadata only; squeeze never opens the file.
            let meta = std::fs::metadata(f).with_context(|| format!("--token-env-file {}", f.display()))?;
            use std::os::unix::fs::PermissionsExt;
            if meta.permissions().mode() & 0o077 != 0 {
                bail!("{} is readable by other users; run `chmod 600` on it", f.display());
            }
        }
        bench::Auth::ApiKey => {
            if !matches!(auth::detect_credential(|k| std::env::var(k).ok()), Credential::ApiKey | Credential::AuthToken)
            {
                bail!("--auth api-key needs ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN for a gateway)");
            }
        }
    }
    let cache = Store::default_root().join("cache");
    let root = a.atlas_root.clone().unwrap_or_else(|| cache.join("SWE-Atlas"));
    let claude_linux = match a.claude_linux.clone() {
        Some(p) => p,
        None => {
            let v =
                host_claude_version(&a.claude_bin).context("could not read `claude --version`; pass --claude-linux")?;
            cache.join("claude-code").join(v).join("linux-x64/claude")
        }
    };
    let squeeze_linux = a.squeeze_linux.clone().unwrap_or_else(|| cache.join("bin/squeeze-linux-x64"));
    for (what, p) in [
        ("SWE-Atlas checkout", &root),
        ("Claude Code linux-x64 binary", &claude_linux),
        ("squeeze linux binary", &squeeze_linux),
    ] {
        // Regrading runs only the tasks' tests, not the agent.
        if !p.exists() && (!a.regrade || what == "SWE-Atlas checkout") {
            bail!("{what} not found at {} (see README: SWE-Atlas runs)", p.display());
        }
    }

    let mut file = atlas::load(&a.bundles_file)?;
    if let Some(m) = &a.usersim_model {
        file.defaults.usersim_model = (m != "none").then(|| m.clone());
    }
    if let Some(m) = &a.judge_model {
        file.defaults.judge_model = (m != "none").then(|| m.clone());
    }
    if !a.bundles.is_empty() {
        file.bundles.retain(|b| a.bundles.contains(&b.name));
        if file.bundles.is_empty() {
            bail!("none of the requested bundles are in {}", a.bundles_file.display());
        }
    }
    let out_path = a.out.clone().unwrap_or_else(|| PathBuf::from(format!("squeeze-atlas-{}.jsonl", now_secs())));
    let runs_dir = PathBuf::from(format!("{}.d", out_path.display()));
    let mut results = bench::read_results(&out_path);
    if a.regrade {
        if results.is_empty() {
            bail!("--regrade needs an existing results file, given with --out");
        }
        let store = Store::open(Store::default_root())?;
        let host = bench::Context {
            claude_bin: a.claude_bin.clone(),
            exe: std::env::current_exe()?,
            store_root: store.root().to_path_buf(),
            auth,
            proxy_url: None,
            model_override: None,
            include_user_settings: false,
            keep_dirs: false,
        };
        let ctx = atlas::AtlasCtx {
            host: &host,
            runtime,
            root,
            claude_linux,
            squeeze_linux,
            token_env_file: None,
            agent_env: Vec::new(),
            cpus: a.cpus,
            memory: a.memory.clone(),
            dns: a.dns.clone(),
            judge_batch: a.judge_batch,
            defer_grading: false,
            runs_dir,
        };
        // The judge calls a model, so it needs an explicit --auth.
        let unjudged = results.iter().flat_map(|r| &r.tickets).filter(|t| t.rubric.is_none()).count();
        let judge = file.defaults.judge_model.as_deref().filter(|_| a.auth.is_some());
        if unjudged > 0 && file.defaults.judge_model.is_some() && judge.is_none() {
            eprintln!("note: {unjudged} ticket(s) have no rubric score; pass --auth to judge them");
        }
        let regraded = atlas::regrade(&ctx, &file, &results, judge).await?;
        let path = PathBuf::from(format!("{}.regraded.jsonl", out_path.display()));
        let mut f = std::fs::File::create(&path)?;
        for r in &regraded {
            writeln!(f, "{}", serde_json::to_string(r)?)?;
        }
        println!("{}", bench::summarize(&regraded));
        println!("Regraded results: {}", path.display());
        return Ok(());
    }
    // A session recorded with an error is run again, continuing from its checkpoint.
    let key = |r: &bench::SessionResult| (r.bundle.clone(), r.arm, r.repeat);
    let done: HashSet<_> = results.iter().filter(|r| r.error.is_none()).map(key).collect();
    let todo: Vec<(usize, usize, Arm)> = (0..a.repeats)
        .flat_map(|k| (0..file.bundles.len()).map(move |b| (k, b)))
        .flat_map(|(k, b)| {
            let mut arms = a.arms.clone();
            if k % 2 == 1 {
                arms.reverse();
            }
            arms.into_iter().map(move |arm| (k, b, arm))
        })
        .filter(|(k, b, arm)| !done.contains(&(file.bundles[*b].name.clone(), *arm, *k)))
        .collect();
    let d = &file.defaults;
    let errored_rows: Vec<bench::SessionResult> = results.iter().filter(|r| r.error.is_some()).cloned().collect();
    let errored = |k: usize, b: usize, arm: Arm| {
        errored_rows.iter().find(|r| key(r) == (file.bundles[b].name.clone(), arm, k)).cloned()
    };
    eprintln!(
        "{} session(s) to run ({} ticket(s); {} continuing after an error); {} already recorded in {}.\nagent: {}, simulated user: {}, judge: {}{}.",
        todo.len(),
        todo.iter().map(|(_, b, _)| file.bundles[*b].tasks.len()).sum::<usize>(),
        todo.iter().filter(|(k, b, arm)| errored(*k, *b, *arm).is_some()).count(),
        done.len(),
        out_path.display(),
        a.model.as_deref().or(d.model.as_deref()).unwrap_or("Claude Code default"),
        d.usersim_model.as_deref().unwrap_or("off"),
        d.judge_model.as_deref().unwrap_or("off"),
        if a.judge_batch { " (batched)" } else { " (one call per rubric item)" },
    );
    eprintln!(
        "Auth: {}. Unmodified Claude Code {} runs inside each task container; squeeze adds only a hook and MCP tools.",
        match auth {
            bench::Auth::Subscription => "your subscription via the setup-token env file (passed by path, never read)",
            bench::Auth::ApiKey => "API key (billed)",
        },
        claude_linux
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.file_name())
            .map(|v| v.to_string_lossy().to_string())
            .unwrap_or_default()
    );
    if todo.is_empty() {
        println!("{}", bench::summarize(&results));
        return Ok(());
    }
    if !a.yes {
        if !std::io::stdin().is_terminal() {
            bail!("pass --yes to confirm");
        }
        eprint!("Continue? [y/N] ");
        std::io::stderr().flush()?;
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        if !matches!(answer.trim(), "y" | "Y" | "yes") {
            bail!("cancelled");
        }
    }
    let store = Store::open(Store::default_root())?;
    let host = bench::Context {
        claude_bin: a.claude_bin.clone(),
        exe: std::env::current_exe()?,
        store_root: store.root().to_path_buf(),
        auth,
        proxy_url: None,
        model_override: a.model.clone(),
        include_user_settings: false,
        keep_dirs: false,
    };
    let ctx = atlas::AtlasCtx {
        host: &host,
        runtime,
        root,
        claude_linux,
        squeeze_linux,
        token_env_file: a.token_env_file.clone(),
        agent_env: a.agent_env.clone(),
        cpus: a.cpus,
        memory: a.memory.clone(),
        dns: a.dns.clone(),
        judge_batch: a.judge_batch,
        defer_grading: a.defer_grading,
        runs_dir,
    };
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&out_path)?;
    for (n, (k, b, arm)) in todo.iter().enumerate() {
        let bundle = &file.bundles[*b];
        eprintln!("[{}/{}] {} / {} / repeat {}", n + 1, todo.len(), bundle.name, arm.name(), k + 1);
        let previous = errored(*k, *b, *arm);
        match atlas::run_session(&ctx, &file.defaults, bundle, *arm, *k, previous.as_ref()).await {
            Ok(r) => {
                if let Some(e) = &r.error {
                    eprintln!("  error: {e}");
                }
                if previous.is_some() {
                    // Replace the errored row.
                    results.retain(|x| key(x) != key(&r));
                    results.push(r);
                    write_results(&out_path, &results)?;
                } else {
                    writeln!(out, "{}", serde_json::to_string(&r)?)?;
                    results.push(r);
                }
            }
            Err(e) if e.is::<bench::LimitReached>() => {
                eprintln!("stopped: {e}\nRerun the same command with --out {} to resume.", out_path.display());
                break;
            }
            Err(e) => return Err(e),
        }
    }
    println!("\n{}", bench::summarize(&results));
    println!("Results: {}", out_path.display());
    Ok(())
}

/// Rewrite a results file in place (atomically).
fn write_results(path: &Path, results: &[bench::SessionResult]) -> Result<()> {
    let mut text = String::new();
    for r in results {
        text.push_str(&serde_json::to_string(r)?);
        text.push('\n');
    }
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    let result: Result<i32> = match cli.cmd {
        Cmd::Mcp(a) => cmd_mcp(a).map(|_| 0),
        Cmd::Expand { id } => Store::open(Store::default_root()).and_then(|s| s.get_blob(&id)).map(|text| {
            println!("{text}");
            0
        }),
        Cmd::Hook { event, hint } => {
            // Hooks must never break the session: any problem means "no change".
            let mut input = String::new();
            let _ = std::io::Read::read_to_string(&mut std::io::stdin(), &mut input);
            let hint = if hint == "mcp" { ExpandHint::Mcp } else { ExpandHint::Cli };
            if event == "post-tool-use"
                && let Some(reply) = squeeze::hook::run(&input, hint)
            {
                println!("{reply}");
            }
            Ok(0)
        }
        Cmd::Stats { sessions } => cmd_stats(sessions).map(|_| 0),
        Cmd::Replay(a) => cmd_replay(a).map(|_| 0),
        Cmd::Gc { days } => {
            Store::open(Store::default_root()).and_then(|s| s.gc(Duration::from_secs(days * 86_400))).map(|(n, b)| {
                println!("removed {n} file(s), {}B", human(b as f64));
                0
            })
        }
        cmd => {
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!("squeeze: {e}");
                    std::process::exit(1);
                }
            };
            rt.block_on(async move {
                match cmd {
                    Cmd::Run(a) => cmd_run(a).await,
                    Cmd::Proxy(a) => cmd_proxy(a).await.map(|_| 0),
                    Cmd::Bench(a) => cmd_bench(a).await.map(|_| 0),
                    Cmd::BenchAtlas(a) => cmd_bench_atlas(a).await.map(|_| 0),
                    _ => unreachable!("handled above"),
                }
            })
        }
    };
    match result {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("squeeze: {e:#}");
            std::process::exit(1);
        }
    }
}
