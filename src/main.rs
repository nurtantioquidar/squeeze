use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use serde_json::{Value, json};
use squeeze::auth::{self, Credential};
use squeeze::bench::{self, Arm};
use squeeze::cachesim::{self, Regime, SimConfig};
use squeeze::ctxmgmt::{ContextEdits, CtxConfig, KeepWindow};
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
        /// Hook event: post-tool-use (compression), pre-tool-use (edit_many permissions)
        /// or pre-compact (compaction guidance)
        event: String,
        /// Where compressed-output notes point: mcp or cli
        #[arg(long, default_value = "cli")]
        hint: String,
        /// Compression profile: default or aggressive
        #[arg(long, default_value = "default")]
        profile: String,
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
    BenchAtlas(Box<AtlasArgs>),
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
    #[arg(
        long,
        conflicts_with_all = ["context_edits", "keep_tool_uses", "keep_tool_uses_max", "clear_tool_inputs", "clear_thinking"]
    )]
    prune_cold_cache: bool,
    /// Experimental, proxy only: have the API clear old tool results and thinking
    /// (server-side context editing). `cold` moves what's cleared forward only when
    /// the prompt cache has already expired, so the cache keeps hitting in between.
    /// `off` sends Claude Code's own setting unchanged
    #[arg(long, value_name = "off|cold", default_value = "off", conflicts_with = "no_compress")]
    context_edits: ContextEdits,
    /// With --context-edits: keep at least this many recent clearable tool uses
    #[arg(long, value_name = "N", default_value_t = KeepWindow::default().min)]
    keep_tool_uses: usize,
    /// With --context-edits: keep the tool uses since the start of the previous
    /// human turn, up to this many
    #[arg(long, value_name = "N", default_value_t = KeepWindow::default().max)]
    keep_tool_uses_max: usize,
    /// With --context-edits: clear tool inputs (the agent's older Edit and Write text) too
    #[arg(long)]
    clear_tool_inputs: bool,
    /// Experimental, API keys only: when a turn has ended and its conversation stays idle
    /// until shortly before the prompt cache would expire, have it summarized in the
    /// background, and send the summary in place of the earlier conversation from the next
    /// request on. The summary is one extra request the proxy makes itself, with the API key
    /// of the request that ended the turn, held in memory only until then. Not with
    /// --context-edits cold
    #[arg(long, conflicts_with_all = ["prune_cold_cache", "no_compress"])]
    compact_pauses: bool,
    /// With --compact-pauses: summarize after this many idle seconds instead of a minute
    /// before the cache expires (for testing)
    #[arg(long, hide = true, requires = "compact_pauses")]
    compact_idle_secs: Option<u64>,
    /// With --context-edits: also clear thinking from before the latest human prompt.
    /// Off by default: Claude Code sends old thinking without its text (about 30-50
    /// tokens a turn), and the API counts a thinking turn at every reminder Claude Code
    /// adds, which the proxy can't always see, so a pinned count can cost cache misses
    #[arg(long, value_name = "BOOL", default_value_t = false, action = clap::ArgAction::Set)]
    clear_thinking: bool,
    /// Upstream API base URL [default: $ANTHROPIC_BASE_URL, else https://api.anthropic.com]
    #[arg(long)]
    upstream: Option<String>,
}

/// A proxied bench arm's `squeeze proxy` flags, parsed as the command line would.
#[derive(Parser)]
struct ArmProxyFlags {
    #[command(flatten)]
    compress: CompressFlags,
}

impl CompressFlags {
    fn for_arm(arm: Arm, upstream: Option<String>) -> Result<Self> {
        let args = arm.proxy_args().with_context(|| format!("the {} arm runs without the proxy", arm.name()))?;
        let mut flags = ArmProxyFlags::try_parse_from(std::iter::once("squeeze").chain(args.iter().copied()))?.compress;
        flags.upstream = upstream;
        Ok(flags)
    }

    /// `SQUEEZE_DISABLE=1` turns every rewrite off.
    fn disabled(&self) -> bool {
        self.no_compress || std::env::var("SQUEEZE_DISABLE").is_ok_and(|v| v == "1")
    }

    fn ctx(&self) -> Result<Option<CtxConfig>> {
        if self.context_edits == ContextEdits::Off || self.disabled() {
            return Ok(None);
        }
        Ok(Some(CtxConfig {
            window: KeepWindow::new(self.keep_tool_uses, self.keep_tool_uses_max)?,
            clear_inputs: self.clear_tool_inputs,
            clear_thinking: self.clear_thinking,
        }))
    }

    fn transform(&self, hint: ExpandHint) -> Result<Option<TransformConfig>> {
        if self.disabled() {
            return Ok(None);
        }
        let mut t = TransformConfig { expand_hint: hint, dedup: !self.no_dedup, ..Default::default() };
        t.prune.enabled = self.prune_cold_cache;
        t.ctx_window = self.ctx()?.map(|c| c.window);
        Ok(Some(t))
    }

    fn compact(&self) -> Result<Option<squeeze::compact::CompactConfig>> {
        if !self.compact_pauses || self.disabled() {
            return Ok(None);
        }
        if self.context_edits != ContextEdits::Off {
            bail!("--compact-pauses and --context-edits cold don't go together");
        }
        Ok(Some(squeeze::compact::CompactConfig { idle_secs: self.compact_idle_secs, ..Default::default() }))
    }

    fn proxy_config(&self, hint: ExpandHint, store: Store) -> Result<ProxyConfig> {
        Ok(ProxyConfig {
            upstream: self.upstream(),
            transform: self.transform(hint)?,
            store,
            ctx: self.ctx()?,
            compact: self.compact()?,
        })
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
    /// Have Claude Code compact the conversation within this many tokens of context
    /// (it sets CLAUDE_CODE_AUTO_COMPACT_WINDOW; compaction starts somewhat below it), and
    /// ask compaction for a short, task-focused summary. Every request re-reads the whole
    /// context, so a smaller one costs less.
    #[arg(long, value_name = "TOKENS")]
    compact_window: Option<u64>,
    /// Compress logs, diffs, searches and test output harder (from 1 KB, to about a third;
    /// the original stays available through expand). File views stay verbatim.
    #[arg(long)]
    aggressive: bool,
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
    /// Simulate the prompt cache instead: what `--context-edits cold` would have cost,
    /// against the same sessions sent unchanged (needs transcripts from Claude Code
    /// 2.1.317 or later, which log each request)
    #[arg(long, conflicts_with_all = ["max_lines", "min_bytes"])]
    cache_sim: bool,
    /// With --cache-sim: `5m-fermat` (5-minute cache, with a 330 s pause before each new
    /// ticket and every third human turn, as SessionBench ran Fermat) or `1h` (1-hour
    /// cache, the recorded timing)
    #[arg(long, value_name = "5m-fermat|1h", default_value = "5m-fermat", requires = "cache_sim")]
    regime: Regime,
    /// With --cache-sim: cache read price as a multiple of base input [default: the
    /// model's, 0.05 on Opus 5.5 and Sonnet 5.5, else 0.1]
    #[arg(long, value_name = "X", requires = "cache_sim")]
    read_price: Option<f64>,
    /// With --cache-sim: share of each cleared tool result the agent reads back later
    #[arg(long, value_name = "FRACTION", default_value_t = 0.0, requires = "cache_sim")]
    reread: f64,
    /// With --cache-sim: tokens the server leaves in place of a cleared tool result or input
    #[arg(long, value_name = "TOKENS", default_value_t = 60.0, requires = "cache_sim")]
    placeholder: f64,
    /// With --cache-sim: as `squeeze proxy --keep-tool-uses`
    #[arg(long, value_name = "N", default_value_t = KeepWindow::default().min, requires = "cache_sim")]
    keep_tool_uses: usize,
    /// With --cache-sim: as `squeeze proxy --keep-tool-uses-max`
    #[arg(long, value_name = "N", default_value_t = KeepWindow::default().max, requires = "cache_sim")]
    keep_tool_uses_max: usize,
    /// With --cache-sim: as `squeeze proxy --clear-tool-inputs`
    #[arg(long, requires = "cache_sim")]
    clear_tool_inputs: bool,
    /// With --cache-sim: as `squeeze proxy --clear-thinking`
    #[arg(long, value_name = "BOOL", default_value_t = true, action = clap::ArgAction::Set, requires = "cache_sim")]
    clear_thinking: bool,
    /// With --cache-sim: model compaction at pauses instead of context edits: during each
    /// pause before a human prompt, while the cache is warm, the history is summarized into
    /// this many tokens, which stand in for it from then on ("advances" counts compactions)
    #[arg(long, value_name = "TOKENS", requires = "cache_sim")]
    compact_summary: Option<f64>,
    /// With --compact-summary: compact only a history at least this many tokens long
    #[arg(long, value_name = "TOKENS", default_value_t = 20_000.0, requires = "compact_summary")]
    compact_min: f64,
    /// With --compact-summary: thinking the summarizing request spends, as a share of the summary
    #[arg(long, value_name = "FRACTION", default_value_t = 0.5, requires = "compact_summary")]
    compact_thinking: f64,
    /// With --compact-summary: share of the summarized tool results the agent reads back later
    #[arg(long, value_name = "FRACTION", default_value_t = 0.1, requires = "compact_summary")]
    compact_reread: f64,
    /// With --compact-summary: compact only at pauses before a new ticket
    #[arg(long, requires = "compact_summary")]
    compact_tickets_only: bool,
    /// With --compact-summary: what the summarizing request reads from cache: `warm` (all
    /// of it, in the background), `system` (in the background under another cache key for
    /// messages: only the system prompt and tools), `cold` (when the next prompt arrives)
    #[arg(long, value_name = "warm|system|cold", default_value = "warm", requires = "compact_summary")]
    compact_cache: String,
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
    /// Arms: vanilla, squeeze, squeeze-hook, squeeze-mcp, squeeze-proxy and
    /// squeeze-proxy-cm (api-key only)
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
    /// API key for the proxied arms' agent: a chmod 600 file holding ANTHROPIC_API_KEY=...
    /// Passed to the containers by path, never read. Other arms, the simulated user and the
    /// judge keep --auth, so only the proxied agent's tokens are billed to the key.
    #[arg(long)]
    api_key_env_file: Option<PathBuf>,
    /// Run up to this many sessions at once, each in its own containers (size --cpus and
    /// --memory so they fit together). Sessions start in order, so with 2 the arms of a
    /// repeat run side by side.
    #[arg(long, default_value_t = 1)]
    parallel: usize,
    /// Every arm's agent uses --api-key-env-file, not only the proxied arms', so all arms
    /// run at API prices and the same cache lifetime. The simulated user and the judge
    /// keep --auth, and --token-env-file is not needed.
    #[arg(long, requires = "api_key_env_file")]
    api_key_all_arms: bool,
    /// Where the proxied arms' proxies forward to [default: Anthropic's API]
    #[arg(long)]
    proxy_upstream: Option<String>,
    /// Where the simulated user's opening messages are shared between arms
    /// [default: <out>.d/openings]; point several results files at one directory to pair them
    #[arg(long)]
    openings: Option<PathBuf>,
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
    /// The agent's prompt-cache lifetime: `client` (Claude Code's own: 1 hour on a
    /// subscription, 5 minutes on an API key) or `1h` (every arm). Compare arms run
    /// at the same lifetime.
    #[arg(long, default_value = "client")]
    cache_ttl: squeeze::atlas::CacheTtl,
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
    if t.ctx_advances > 0 || t.refilled > 0 {
        eprintln!(
            "squeeze: context editing moved forward {} time(s) and cleared ~{} input tokens across requests; {} unchanged-file Read(s) refilled",
            t.ctx_advances,
            human(t.cleared_input_tokens as f64),
            t.refilled
        );
    }
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
    if !a.proxy && a.compress.context_edits != ContextEdits::Off {
        bail!("--context-edits works on API requests, so it needs --proxy");
    }

    let proxy = if a.proxy {
        // Proxy mode sees API traffic, so it is limited to API-key credentials.
        preflight(a.api_key_helper)?;
        init_logging(Some(&store.root().join("logs/squeeze.log")), "info");
        let hint = if a.no_mcp { ExpandHint::Cli } else { ExpandHint::Mcp };
        let proxy = Proxy::new(a.compress.proxy_config(hint, store.clone())?)?;
        let addr = proxy.spawn("127.0.0.1:0").await?;
        tracing::info!("proxy on {addr}, upstream {}", a.compress.upstream());
        cmd.env("ANTHROPIC_BASE_URL", format!("http://{addr}"));
        Some(proxy)
    } else {
        // Hook mode: Claude Code talks to Anthropic itself, with whatever login the
        // user has. squeeze never sees credentials or traffic.
        None
    };

    let mut settings = if !a.proxy && !a.compress.no_compress {
        squeeze::hook::session_settings(&exe, !a.no_mcp)
    } else if !a.no_mcp {
        json!({ "permissions": { "allow": tool_names() } })
    } else {
        json!({})
    };
    if !a.no_mcp && (a.edit_tool || a.replace_file_tools) {
        // edit_many is approved without asking exactly when Edit would be.
        squeeze::hook::add_edit_permission_hook(&mut settings, &exe);
    }
    if a.aggressive {
        squeeze::hook::set_aggressive(&mut settings);
    }
    if a.compact_window.is_some() {
        squeeze::hook::add_compact_hook(&mut settings, &exe);
    }
    if settings.as_object().is_some_and(|m| !m.is_empty()) {
        merge_settings_arg(&mut args, settings)?;
    }
    // Appended after the user's arguments: --disallowedTools and --mcp-config are
    // variadic and would otherwise swallow a trailing prompt.
    let mut off: Vec<&str> = Vec::new();
    if a.replace_file_tools {
        off.extend(squeeze::mcp::REPLACED_BUILTINS);
        // Claude Code keeps only the last of each of these flags, and refuses a
        // flag together with its -file form: merge the user's text into ours.
        let guide = format!("{}\n\n{}", squeeze::mcp::FILE_TOOLS_GUIDE, squeeze::mcp::WORKING_GUIDE);
        let main_prompt = merged_prompt(&mut args, "--append-system-prompt", &guide)?;
        let sub_prompt =
            merged_prompt(&mut args, "--append-subagent-system-prompt", squeeze::mcp::SUBAGENT_FILE_TOOLS_GUIDE)?;
        args.extend([
            "--append-system-prompt".into(),
            main_prompt,
            "--append-subagent-system-prompt".into(),
            sub_prompt,
        ]);
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
    if let Some(w) = a.compact_window {
        cmd.env("CLAUDE_CODE_AUTO_COMPACT_WINDOW", w.to_string());
    }
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
    let proxy = Proxy::new(a.compress.proxy_config(ExpandHint::Cli, store)?)?;
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
    let paths =
        if a.paths.is_empty() { vec![squeeze::util::home_dir().join(".claude/projects")] } else { a.paths.clone() };
    let files = replay::find_transcripts(&paths);
    if files.is_empty() {
        bail!(
            "no transcripts (*.jsonl) found under {}",
            paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
        );
    }
    if a.cache_sim {
        return cmd_cache_sim(&a, &files);
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
        let label = session_label(&r.file);
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

/// A transcript's session id prefix and project.
fn session_label(file: &Path) -> String {
    let name = file
        .parent()
        .and_then(|p| p.file_name())
        .map(|p| p.to_string_lossy().trim_start_matches('-').to_string())
        .unwrap_or_default();
    let sess = file.file_stem().map(|s| s.to_string_lossy().chars().take(8).collect::<String>()).unwrap_or_default();
    squeeze::util::truncate_chars(&format!("{sess} {name}"), 43)
}

fn cmd_cache_sim(a: &ReplayArgs, files: &[PathBuf]) -> Result<()> {
    if !(0.0..=1.0).contains(&a.reread) {
        bail!("--reread is a fraction from 0 to 1");
    }
    if a.read_price.is_some_and(|r| !(r > 0.0 && r <= 1.0)) || !(0.0..).contains(&a.placeholder) {
        bail!("--read-price must be above 0 and at most 1, and --placeholder at least 0");
    }
    let summary_cache = match a.compact_cache.as_str() {
        "warm" => cachesim::SummaryCache::Warm,
        "system" => cachesim::SummaryCache::SystemOnly,
        "cold" => cachesim::SummaryCache::Cold,
        other => bail!("--compact-cache must be warm, system or cold, not {other:?}"),
    };
    let cfg = SimConfig {
        regime: a.regime,
        read_price: a.read_price,
        reread: a.reread,
        placeholder: a.placeholder,
        ctx: CtxConfig {
            window: KeepWindow::new(a.keep_tool_uses, a.keep_tool_uses_max)?,
            clear_inputs: a.clear_tool_inputs,
            clear_thinking: a.clear_thinking,
        },
        compact: a.compact_summary.map(|summary| cachesim::Compaction {
            summary,
            min_history: a.compact_min,
            thinking: a.compact_thinking,
            reread: a.compact_reread,
            tickets_only: a.compact_tickets_only,
            cache: summary_cache,
        }),
    };
    let sim = cachesim::simulate_files(files, &cfg)?;
    if a.json {
        println!("{}", serde_json::to_string_pretty(&sim)?);
        return Ok(());
    }
    let regime = match a.regime {
        Regime::FiveMinFermat => "5m-fermat: 5-minute cache, 330 s pauses",
        Regime::OneHour => "1h: 1-hour cache, recorded timing",
    };
    if let Some(c) = &cfg.compact {
        println!(
            "Compaction at pauses: {:.0}-token summaries of histories from {:.0} tokens, {:.0}% thinking on top,\n\
             {:.0}% of summarized tool results read back. (The context-edit settings below don't apply.)",
            c.summary,
            c.min_history,
            100.0 * c.thinking,
            100.0 * c.reread
        );
    }
    println!(
        "Cache simulation ({regime}). --context-edits cold keeping {}-{} tool uses, inputs {}, thinking {};\n\
         placeholder {} tokens, {:.0}% of each cleared result read back.",
        a.keep_tool_uses,
        a.keep_tool_uses_max,
        if a.clear_tool_inputs { "cleared" } else { "kept" },
        if a.clear_thinking { "cleared" } else { "kept" },
        a.placeholder,
        100.0 * a.reread
    );
    let c = &sim.calibration;
    println!(
        "Token fit per 4 bytes: text {:.3}, tool input {:.3}, tool result {:.3}, other {:.3}, +{:.1} per block\n\
         (thinking billed as recorded; fitted weight {:.3}).\n",
        c.text, c.input, c.result, c.other, c.per_block, c.thinking
    );
    println!(
        "{:<44} {:>8} {:>7} {:>5} {:>8} {:>8} {:>7} {:>5} {:>10} {:>10} {:>6}",
        "session",
        "requests",
        "fit err",
        "cold",
        "advances",
        "breakers",
        "refills",
        "read",
        "unchanged",
        "with edits",
        "ratio"
    );
    for s in sim.sessions.iter().take(a.limit) {
        println!(
            "{:<44} {:>8} {:>6.1}% {:>5} {:>8} {:>8} {:>7} {:>5} {:>10} {:>10} {:>6.3}",
            session_label(&s.file),
            s.requests,
            100.0 * s.fit_error,
            s.cold,
            s.advances,
            if s.breakers.is_empty() { "-".to_string() } else { s.breakers.join(",") },
            s.refilled,
            s.read_price,
            human(s.unchanged.cost),
            human(s.context_edits.cost),
            s.ratio
        );
    }
    let unresolved: usize = sim.sessions.iter().map(|s| s.unresolved).sum();
    println!(
        "\nMean ratio over {} sessions: {:.3}{}{}",
        sim.sessions.len(),
        sim.mean_ratio,
        if sim.skipped > 0 { format!(" ({} transcripts skipped: no request log)", sim.skipped) } else { String::new() },
        if unresolved > 0 { format!("; {unresolved} messages not found, billed as empty") } else { String::new() }
    );
    println!(
        "Costs are in base-input-token equivalents (cache reads and writes, input, and output at 5x). Estimate\n\
         only: token counts are fitted, the server is emulated, and the agent is assumed to behave the same."
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
            if let Some(arm) = a.arms.iter().find(|a| a.is_proxied()) {
                bail!("the {} arm routes API traffic through squeeze, so it needs --auth api-key", arm.name());
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
    let mut proxy_urls = BTreeMap::new();
    for arm in a.arms.iter().filter(|a| a.is_proxied()) {
        let proxy = Proxy::new(CompressFlags::for_arm(*arm, None)?.proxy_config(ExpandHint::Mcp, store.clone())?)?;
        proxy_urls.insert(*arm, format!("http://{}", proxy.spawn("127.0.0.1:0").await?));
    }
    let ctx = bench::Context {
        claude_bin: a.claude_bin.clone(),
        exe: std::env::current_exe()?,
        store_root: store.root().to_path_buf(),
        auth,
        proxy_urls,
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
        None if a.regrade => bench::Auth::ApiKey, // unused: without --auth, regrading calls no model
        None => bail!("--auth is required (`subscription` or `api-key`)"),
    };
    let runtime = match a.runtime.as_str() {
        "container" => atlas::Runtime::Apple,
        "docker" => atlas::Runtime::Docker,
        other => bail!("--runtime must be `container` or `docker`, not {other:?}"),
    };
    match auth {
        _ if a.regrade && a.auth.is_none() => {}
        // The judge runs on the host's own Claude Code login; no token file needed.
        bench::Auth::Subscription if a.regrade => {}
        // Every agent uses the API key file (checked below); the simulated user and the
        // judge run on the host's own Claude Code login.
        bench::Auth::Subscription if a.api_key_all_arms => {}
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
            proxy_urls: BTreeMap::new(),
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
            api_key_env_file: None,
            api_key_all_arms: false,
            openings_lock: Default::default(),
            proxy_urls: BTreeMap::new(),
            agent_env: Vec::new(),
            cache_ttl: a.cache_ttl,
            cpus: a.cpus,
            memory: a.memory.clone(),
            dns: a.dns.clone(),
            judge_batch: a.judge_batch,
            defer_grading: false,
            openings_dir: runs_dir.join("openings"),
            runs_dir,
        };
        // Build on earlier regrades: a session already in <out>.regraded.jsonl is
        // taken from there, so finished grading (and paid judge calls) is kept.
        let regraded_path = PathBuf::from(format!("{}.regraded.jsonl", out_path.display()));
        let earlier = bench::read_results(&regraded_path);
        let same = |x: &bench::SessionResult, y: &bench::SessionResult| {
            x.session_id == y.session_id && x.bundle == y.bundle && x.arm == y.arm && x.repeat == y.repeat
        };
        let mut rows: Vec<bench::SessionResult> =
            results.iter().map(|r| earlier.iter().find(|e| same(e, r)).cloned().unwrap_or_else(|| r.clone())).collect();
        let selected = |r: &bench::SessionResult| file.bundles.iter().any(|b| b.name == r.bundle);
        let unjudged =
            rows.iter().filter(|r| selected(r)).flat_map(|r| &r.tickets).filter(|t| t.rubric.is_none()).count();
        // A ticket the configured judge hasn't scored must not be counted with
        // the rubric taken as 1.0, so judging it needs a decision.
        let judge = match (file.defaults.judge_model.as_deref(), a.auth.is_some()) {
            (Some(m), true) => Some(m),
            (Some(_), false) if unjudged > 0 => bail!(
                "{unjudged} ticket(s) have no rubric score yet: pass --auth to judge them, or --judge-model none to \
                 score their tests only"
            ),
            _ => None,
        };
        if !earlier.is_empty() {
            eprintln!("continuing from {}", regraded_path.display());
        }
        for i in 0..rows.len() {
            // Rows of bundles not selected with --bundles pass through unchanged.
            let Some(bundle) = file.bundles.iter().find(|b| b.name == rows[i].bundle) else { continue };
            let mut r = rows[i].clone();
            let outcome = atlas::regrade_session(&ctx, bundle, &mut r, judge).await;
            rows[i] = r;
            write_results(&regraded_path, &rows)?;
            if let Err(e) = outcome {
                eprintln!("stopped: {e:#}\nGrading so far is saved; rerun the same command to continue.",);
                break;
            }
        }
        println!("{}", bench::summarize(&rows));
        println!("Regraded results: {}", regraded_path.display());
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
            _ if a.api_key_all_arms => {
                "every agent on the API key in --api-key-env-file (billed; passed by path, never read); \
                 simulated user and judge per --auth"
            }
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
    if a.api_key_all_arms
        && let Some(f) = &a.api_key_env_file
    {
        use std::os::unix::fs::PermissionsExt;
        let meta = std::fs::metadata(f).with_context(|| format!("--api-key-env-file {}", f.display()))?;
        if meta.permissions().mode() & 0o077 != 0 {
            bail!("{} is readable by other users; run `chmod 600` on it", f.display());
        }
    }
    let store = Store::open(Store::default_root())?;
    // Proxied arms: a proxy each, reachable from the task containers. Apple's
    // runtime reaches the host at the bridge address, so bind there, not to every
    // interface. The proxy refuses subscription credentials, as everywhere.
    let mut proxy_urls = BTreeMap::new();
    // Held until the run ends: the proxies listen on the container network's host address.
    let _bridge = if runtime == atlas::Runtime::Apple && a.arms.iter().any(|a| a.is_proxied()) {
        Some(atlas::BridgeAnchor::start().await?)
    } else {
        None
    };
    for arm in a.arms.iter().filter(|a| a.is_proxied()) {
        if runtime != atlas::Runtime::Apple {
            bail!("the {} arm needs --runtime container for now", arm.name());
        }
        if let Some(f) = &a.api_key_env_file {
            use std::os::unix::fs::PermissionsExt;
            let meta = std::fs::metadata(f).with_context(|| format!("--api-key-env-file {}", f.display()))?;
            if meta.permissions().mode() & 0o077 != 0 {
                bail!("{} is readable by other users; run `chmod 600` on it", f.display());
            }
        } else if auth != bench::Auth::ApiKey {
            bail!("the {} arm needs an API key: --api-key-env-file (or --auth api-key)", arm.name());
        }
        let flags = CompressFlags::for_arm(*arm, a.proxy_upstream.clone())?;
        let proxy = Proxy::new(flags.proxy_config(ExpandHint::Mcp, store.clone())?)?;
        let addr = proxy.spawn(&format!("{}:0", atlas::APPLE_HOST_ADDR)).await?;
        eprintln!("squeeze proxy for the {} arm at http://{addr}", arm.name());
        proxy_urls.insert(*arm, format!("http://{addr}"));
    }
    let host = bench::Context {
        claude_bin: a.claude_bin.clone(),
        exe: std::env::current_exe()?,
        store_root: store.root().to_path_buf(),
        auth,
        proxy_urls: BTreeMap::new(),
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
        api_key_env_file: a.api_key_env_file.clone(),
        api_key_all_arms: a.api_key_all_arms,
        openings_lock: Default::default(),
        proxy_urls,
        agent_env: a.agent_env.clone(),
        cache_ttl: a.cache_ttl,
        cpus: a.cpus,
        memory: a.memory.clone(),
        dns: a.dns.clone(),
        judge_batch: a.judge_batch,
        defer_grading: a.defer_grading,
        openings_dir: a.openings.clone().unwrap_or_else(|| runs_dir.join("openings")),
        runs_dir,
    };
    // Up to --parallel sessions at once. On a usage limit or a failure no new
    // session starts, and the running ones finish (and checkpoint) first.
    use futures_util::stream::{FuturesUnordered, StreamExt};
    let mut queue = todo.iter().enumerate();
    let mut running = FuturesUnordered::new();
    let (mut stop, mut failure) = (false, None);
    loop {
        while !stop && running.len() < a.parallel.max(1) {
            let Some((n, &(k, b, arm))) = queue.next() else { break };
            let bundle = &file.bundles[b];
            eprintln!("[{}/{}] {} / {} / repeat {}", n + 1, todo.len(), bundle.name, arm.name(), k + 1);
            let previous = errored(k, b, arm);
            let (ctx, d) = (&ctx, &file.defaults);
            running.push(async move {
                let r = atlas::run_session(ctx, d, bundle, arm, k, previous.as_ref()).await;
                (previous.is_some(), r)
            });
        }
        let Some((replaces, r)) = running.next().await else { break };
        let r = match r {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  failed: {e:#}");
                (stop, failure) = (true, failure.or(Some(e)));
                continue;
            }
        };
        if let Some(e) = &r.error {
            eprintln!("  error ({} / repeat {}): {e}", r.arm.name(), r.repeat + 1);
        }
        if replaces {
            replace_result(&out_path, &r)?;
            results.retain(|x| key(x) != key(&r));
        } else {
            append_result(&out_path, &r)?;
        }
        if r.error.as_deref().is_some_and(bench::is_limit_error) && !stop {
            stop = true;
            eprintln!("stopped by a usage limit. Rerun the same command with --out {} to resume.", out_path.display());
        }
        results.push(r);
    }
    if let Some(e) = failure {
        return Err(e);
    }
    println!("\n{}", bench::summarize(&results));
    println!("Results: {}", out_path.display());
    Ok(())
}

/// Remove every `--flag value` / `--flag=value` from `args` and return the values.
fn take_flag(args: &mut Vec<String>, flag: &str) -> Vec<String> {
    let mut values = Vec::new();
    let prefix = format!("{flag}=");
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag && i + 1 < args.len() {
            values.push(args.remove(i + 1));
            args.remove(i);
        } else if let Some(v) = args[i].strip_prefix(&prefix) {
            values.push(v.to_string());
            args.remove(i);
        } else {
            i += 1;
        }
    }
    values
}

/// squeeze's text for a single-value prompt flag, followed by whatever the user
/// passed with that flag or its `-file` form (whose file is read here).
fn merged_prompt(args: &mut Vec<String>, flag: &str, ours: &str) -> Result<String> {
    let mut parts = vec![ours.to_string()];
    parts.extend(take_flag(args, flag));
    for f in take_flag(args, &format!("{flag}-file")) {
        parts.push(std::fs::read_to_string(&f).with_context(|| format!("{flag}-file {f}"))?);
    }
    Ok(parts.join("\n\n"))
}

/// Append one session's row to a results file.
fn append_result(path: &Path, r: &bench::SessionResult) -> Result<()> {
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{}", serde_json::to_string(r)?)?;
    Ok(())
}

/// Replace the row of `r`'s session (same bundle, arm and repeat) in a results
/// file. Every other line stays as it was, including lines this build can't
/// parse; the previous file is kept as `<path>.bak`.
fn replace_result(path: &Path, r: &bench::SessionResult) -> Result<()> {
    let old = std::fs::read_to_string(path).unwrap_or_default();
    let mut text = String::new();
    for line in old.lines().filter(|l| !l.trim().is_empty()) {
        let same = serde_json::from_str::<bench::SessionResult>(line)
            .is_ok_and(|x| x.bundle == r.bundle && x.arm == r.arm && x.repeat == r.repeat);
        if !same {
            text.push_str(line);
            text.push('\n');
        }
    }
    text.push_str(&serde_json::to_string(r)?);
    text.push('\n');
    if !old.is_empty() {
        std::fs::write(PathBuf::from(format!("{}.bak", path.display())), &old)?;
    }
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Write a results file we derive (`<out>.regraded.jsonl`) atomically.
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
        Cmd::Hook { event, hint, profile } => {
            // Hooks must never break the session: any problem means "no change".
            let mut input = String::new();
            let _ = std::io::Read::read_to_string(&mut std::io::stdin(), &mut input);
            let hint = if hint == "mcp" { ExpandHint::Mcp } else { ExpandHint::Cli };
            let reply = match event.as_str() {
                "post-tool-use" => squeeze::hook::run(&input, hint, profile == "aggressive"),
                "pre-tool-use" => squeeze::hook::pre_tool_use(&input),
                "pre-compact" => squeeze::hook::pre_compact(&input),
                _ => None,
            };
            if let Some(reply) = reply {
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
                    Cmd::BenchAtlas(a) => cmd_bench_atlas(*a).await.map(|_| 0),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_sim_flags_need_cache_sim() {
        let parse = |args: &[&str]| Cli::try_parse_from(["squeeze", "replay"].iter().chain(args));
        let Cmd::Replay(a) = parse(&["--cache-sim", "x.jsonl"]).unwrap().cmd else { panic!("not replay") };
        assert_eq!((a.regime, a.read_price, a.reread, a.placeholder), (Regime::FiveMinFermat, None, 0.0, 60.0));
        assert_eq!(
            (a.keep_tool_uses, a.keep_tool_uses_max, a.clear_tool_inputs, a.clear_thinking),
            (8, 24, false, true)
        );
        let Cmd::Replay(a) =
            parse(&["--cache-sim", "--regime", "1h", "--read-price", "0.1", "--reread", "0.25"]).unwrap().cmd
        else {
            panic!("not replay")
        };
        assert_eq!((a.regime, a.read_price, a.reread), (Regime::OneHour, Some(0.1), 0.25));
        assert!(parse(&["--regime", "1h"]).is_err());
        assert!(parse(&["--clear-tool-inputs"]).is_err());
        assert!(parse(&["--cache-sim", "--max-lines", "100"]).is_err());
        assert!(parse(&["--cache-sim", "--regime", "5m"]).is_err());
        assert!(parse(&["--max-lines", "100"]).is_ok(), "plain replay is unchanged");
    }

    #[test]
    fn proxied_arms_get_their_flags() {
        let cm = CompressFlags::for_arm(Arm::SqueezeProxyCm, Some("http://u".into())).unwrap();
        let ctx = cm.ctx().unwrap().expect("context editing on");
        assert_eq!((ctx.window, ctx.clear_inputs, ctx.clear_thinking), (KeepWindow::default(), false, false));
        let t = cm.transform(ExpandHint::Mcp).unwrap().unwrap();
        assert_eq!((t.ctx_window, t.prune.enabled), (Some(ctx.window), false));
        assert_eq!(cm.upstream(), "http://u");
        let inputs = CompressFlags::for_arm(Arm::SqueezeProxyCmInputs, None).unwrap().ctx().unwrap().unwrap();
        assert_eq!((inputs.window, inputs.clear_inputs, inputs.clear_thinking), (KeepWindow::default(), true, false));
        let plain = CompressFlags::for_arm(Arm::SqueezeProxy, None).unwrap();
        assert_eq!(plain.ctx().unwrap(), None);
        assert_eq!(plain.transform(ExpandHint::Mcp).unwrap().unwrap().ctx_window, None);
        assert!(CompressFlags::for_arm(Arm::Squeeze, None).is_err());
    }

    #[test]
    fn merges_the_users_prompt_flags_into_squeezes() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("extra.md");
        std::fs::write(&file, "FROM FILE").unwrap();
        let mut args: Vec<String> =
            ["-p", "do it", "--append-system-prompt", "MINE", "--append-system-prompt-file"].map(String::from).to_vec();
        args.push(file.display().to_string());
        args.push("--append-subagent-system-prompt=SUB".into());
        let main = merged_prompt(&mut args, "--append-system-prompt", "OURS").unwrap();
        let sub = merged_prompt(&mut args, "--append-subagent-system-prompt", "OURS-SUB").unwrap();
        assert_eq!(main, "OURS\n\nMINE\n\nFROM FILE");
        assert_eq!(sub, "OURS-SUB\n\nSUB");
        assert_eq!(args, ["-p", "do it"], "the user's flags are taken out, everything else stays");
    }

    #[test]
    fn replacing_a_row_keeps_every_other_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.jsonl");
        let row = |arm: &str, cost: f64| {
            json!({ "bundle": "b", "arm": arm, "repeat": 0, "model": null,
                "agent": { "cost_usd": cost, "input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                           "output_tokens": 0, "turns": 0, "invocations": 0, "duration_ms": 0 },
                "helpers": { "cost_usd": 0.0, "input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                             "output_tokens": 0, "turns": 0, "invocations": 0, "duration_ms": 0 },
                "tickets": [], "reward": 0.0, "completed": cost > 0.0, "error": if cost > 0.0 { Value::Null } else { json!("boom") } })
            .to_string()
        };
        let future = r#"{"bundle":"b","arm":"squeeze-future","repeat":0}"#;
        std::fs::write(&path, format!("{}\n{future}\n{}\n", row("vanilla", 1.0), row("squeeze", 0.0))).unwrap();
        let fixed: bench::SessionResult = serde_json::from_str(&row("squeeze", 2.0)).unwrap();
        replace_result(&path, &fixed).unwrap();
        append_result(&path, &serde_json::from_str(&row("squeeze-full", 3.0)).unwrap()).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4, "{text}");
        assert!(lines.contains(&future), "a row this build can't parse survives");
        let rows = bench::read_results(&path);
        assert_eq!(rows.iter().filter(|r| r.arm == Arm::Squeeze).count(), 1, "the errored row was replaced");
        assert!(rows.iter().any(|r| r.arm == Arm::SqueezeFull), "appends after a replace reach the file");
        assert!(std::fs::read_to_string(dir.path().join("results.jsonl.bak")).unwrap().contains("boom"));
    }
}
