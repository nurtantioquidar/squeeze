//! `squeeze bench-atlas`: SessionBench-style sessions over SWE-Atlas refactoring
//! tasks. Bundles come from SessionBench (github.com/akuik/sessionbench-swe,
//! Apache-2.0); tasks, tests and rubrics from SWE-Atlas
//! (github.com/scaleapi/SWE-Atlas, Apache-2.0; please cite SWE-Atlas).
//!
//! - One Claude Code conversation is resumed across a bundle's tickets. Each
//!   ticket runs in a fresh container from that task's image, with the
//!   official, unmodified Claude Code Linux binary inside.
//! - A simulated user on the host writes the opening message from the ticket
//!   (the agent never sees the raw ticket) and reviews the agent's replies.
//! - Each ticket is scored with the task's own `tests/test.sh` in a fresh
//!   container, using SessionBench's fractional test score, times the rubric
//!   score from an LLM judge that uses SWE-Atlas's own prompts.
//! - Subscription runs hand the container runtime the path of an env file that
//!   holds a `claude setup-token` token (`--env-file`). squeeze never reads it.

use crate::bench::{self, Arm, Auth, Defaults, SessionResult, TicketResult, Usage};
use anyhow::{Context as _, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

const TOOLS_DIR: &str = "/opt/sqz";
/// The host as seen from Apple `container`'s default network.
pub const APPLE_HOST_ADDR: &str = "192.168.64.1";
/// A small image for [`BridgeAnchor`].
const ANCHOR_IMAGE: &str = "docker.io/library/alpine:3.20";

/// Apple's runtime brings up the host side of its network ([`APPLE_HOST_ADDR`])
/// only while some container runs. The proxied arms' proxies listen there from
/// before the first task container starts, through the pauses between tickets,
/// until the run ends, so an idle container holds the network up meanwhile.
pub struct BridgeAnchor(String);

impl BridgeAnchor {
    pub async fn start() -> Result<Self> {
        let name = format!("squeeze-bridge-{}", std::process::id());
        let mut c = Command::new(Runtime::Apple.bin());
        // Bounded, in case the run dies without stopping it.
        c.args(["run", "-d", "--rm", "--name", &name, ANCHOR_IMAGE, "sleep", "172800"]);
        run_ok(c, "starting a container to hold the container network up").await?;
        let anchor = Self(name);
        for _ in 0..60 {
            if std::net::TcpListener::bind((APPLE_HOST_ADDR, 0)).is_ok() {
                return Ok(anchor);
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        bail!("the container network's host address {APPLE_HOST_ADDR} did not come up")
    }
}

impl Drop for BridgeAnchor {
    fn drop(&mut self) {
        let _ = std::process::Command::new(Runtime::Apple.bin())
            .args(["stop", &self.0])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}
const AGENT_CONFIG: &str = "/root/.sqz-claude";
const AGENT_STORE: &str = "/root/.squeeze";
const GRADE_TIMEOUT_SECS: u64 = 7200;
const CHECKPOINT: &str = "checkpoint.json";

#[derive(Debug, Deserialize)]
pub struct AtlasFile {
    #[serde(default)]
    pub defaults: Defaults,
    #[serde(default, rename = "atlas_bundle")]
    pub bundles: Vec<AtlasBundle>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AtlasBundle {
    pub name: String,
    /// Repository path inside the task images (SessionBench's `cwd`).
    pub workdir: String,
    pub tasks: Vec<String>,
    /// PATH for the agent's commands, if the image's own PATH must be overridden.
    pub agent_path: Option<String>,
    /// PATH for grading, to work around a broken image environment.
    pub grade_path: Option<String>,
    pub allowed_tools: Option<Vec<String>>,
}

pub fn load(path: &Path) -> Result<AtlasFile> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let file: AtlasFile = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    if file.bundles.is_empty() || file.bundles.iter().any(|b| b.tasks.is_empty()) {
        bail!("{} needs at least one [[atlas_bundle]] with tasks", path.display());
    }
    file.defaults.check().with_context(|| format!("in {}", path.display()))?;
    Ok(file)
}

/// The agent's prompt-cache lifetime for a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTtl {
    /// Claude Code's own: one hour on a subscription, five minutes on an API key.
    Client,
    /// One hour for every arm (`ENABLE_PROMPT_CACHING_1H=1`).
    OneHour,
}

impl std::str::FromStr for CacheTtl {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim() {
            "client" => Ok(CacheTtl::Client),
            "1h" => Ok(CacheTtl::OneHour),
            other => bail!("unknown cache TTL {other:?} (client, 1h)"),
        }
    }
}

pub struct Task {
    pub id: String,
    pub dir: PathBuf,
    pub image: String,
    pub instruction: String,
    pub problem: String,
    pub rubrics: Vec<Value>,
    pub system_prompt: String,
    pub user_template: String,
}

pub fn load_task(root: &Path, id: &str) -> Result<Task> {
    let dir = root.join("data/rf").join(id);
    let read = |rel: &str| std::fs::read_to_string(dir.join(rel)).with_context(|| format!("{id}: reading {rel}"));
    let meta: toml::Value = toml::from_str(&read("task.toml")?)?;
    let image = meta
        .get("environment")
        .and_then(|e| e.get("docker_image"))
        .and_then(|v| v.as_str())
        .with_context(|| format!("{id}: task.toml has no environment.docker_image"))?;
    let rubrics: Value = serde_json::from_str(&read("tests/rubrics.json")?)?;
    Ok(Task {
        id: id.to_string(),
        image: normalize_image(image),
        instruction: read("instruction.md")?,
        problem: read("tests/prompt.txt")?,
        rubrics: rubrics.as_array().cloned().unwrap_or_default(),
        system_prompt: read("tests/rubrics_system_prompt.txt")?,
        user_template: read("tests/rubrics_user_prompt_template.txt")?,
        dir,
    })
}

/// `repo:tag@sha256:x` -> `repo@sha256:x`: still pinned by digest, and
/// accepted by runtimes that reject a tag together with a digest.
pub fn normalize_image(image: &str) -> String {
    let Some((name, digest)) = image.split_once('@') else { return image.to_string() };
    let repo = match name.rfind(':') {
        Some(i) if !name[i..].contains('/') => &name[..i],
        _ => name,
    };
    format!("{repo}@{digest}")
}

/// Python `str.format` for SWE-Atlas's prompt templates: `{{`/`}}` escape
/// braces and `{name}` substitutes, all in one pass, so braces inside the
/// substituted values (a patch, say) are left alone.
pub fn py_format(template: &str, vars: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(i) = rest.find(['{', '}']) {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        if tail.starts_with("{{") || tail.starts_with("}}") {
            out.push_str(&tail[..1]);
            rest = &tail[2..];
        } else if let Some(end) = tail.starts_with('{').then(|| tail.find('}')).flatten()
            && let Some((_, v)) = vars.iter().find(|(k, _)| *k == &tail[1..end])
        {
            out.push_str(v);
            rest = &tail[end + 1..];
        } else {
            out.push_str(&tail[..1]);
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Runtime {
    /// Apple's `container` (what opossum drives on macOS).
    Apple,
    Docker,
}

impl Runtime {
    pub fn bin(self) -> &'static str {
        match self {
            Runtime::Apple => "container",
            Runtime::Docker => "docker",
        }
    }

    fn platform(self) -> [&'static str; 2] {
        match self {
            Runtime::Apple => ["--arch", "amd64"],
            Runtime::Docker => ["--platform", "linux/amd64"],
        }
    }

    /// Pull with retries: multi-GB registry downloads can fail mid-stream.
    pub async fn pull(self, image: &str) -> Result<()> {
        let mut last = None;
        for attempt in 1..=3u64 {
            let mut c = Command::new(self.bin());
            match self {
                Runtime::Apple => c.args(["image", "pull"]),
                Runtime::Docker => c.arg("pull"),
            };
            c.args(self.platform()).arg(image);
            match run_ok(c, &format!("pulling {image}")).await {
                Ok(_) => return Ok(()),
                Err(e) => {
                    eprintln!("      pull attempt {attempt}/3 failed: {e:#}");
                    last = Some(e);
                    tokio::time::sleep(Duration::from_secs(10 * attempt)).await;
                }
            }
        }
        Err(last.expect("at least one attempt"))
    }

    async fn remove(self, name: &str) {
        let args: &[&[&str]] = match self {
            Runtime::Apple => &[&["stop", name], &["rm", name]],
            Runtime::Docker => &[&["rm", "-f", name]],
        };
        for a in args {
            let _ = Command::new(self.bin()).args(*a).stdin(Stdio::null()).output().await;
        }
    }
}

async fn run_ok(mut c: Command, what: &str) -> Result<String> {
    let out = c.stdin(Stdio::null()).output().await.with_context(|| format!("{what}: could not run"))?;
    if !out.status.success() {
        bail!("{what} failed: {}", crate::util::truncate_chars(String::from_utf8_lossy(&out.stderr).trim(), 600));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

pub struct AtlasCtx<'a> {
    /// Host-side settings: claude for the simulated user and judge, auth, models.
    pub host: &'a bench::Context,
    pub runtime: Runtime,
    /// A SWE-Atlas checkout.
    pub root: PathBuf,
    /// The official Claude Code linux-x64 binary.
    pub claude_linux: PathBuf,
    /// squeeze built for linux x86_64.
    pub squeeze_linux: PathBuf,
    /// Env file with CLAUDE_CODE_OAUTH_TOKEN (subscription runs). Passed by path only.
    pub token_env_file: Option<PathBuf>,
    /// Env file with ANTHROPIC_API_KEY for the proxied arms' agent (the proxy
    /// accepts API keys only). Passed by path only; other arms keep `--auth`.
    pub api_key_env_file: Option<PathBuf>,
    /// Every arm's agent uses `api_key_env_file`, so arms compare at the same
    /// prices and cache lifetime. The simulated user and judge keep `--auth`.
    pub api_key_all_arms: bool,
    /// The squeeze proxy each proxied arm reaches from the containers.
    pub proxy_urls: BTreeMap<Arm, String>,
    /// Extra KEY=VALUE environment for the agent (e.g. a gateway URL).
    pub agent_env: Vec<String>,
    /// The agent's prompt-cache lifetime. Arms compare fairly only at the same one.
    pub cache_ttl: CacheTtl,
    pub cpus: u32,
    pub memory: String,
    pub dns: Vec<String>,
    /// Judge all rubric items in one call instead of one call per item.
    pub judge_batch: bool,
    /// Skip grading during the session (run `--regrade` afterwards): grading
    /// takes 10-20 minutes per ticket and doesn't affect the agent.
    pub defer_grading: bool,
    /// Per-session artifacts (patches, grading logs, transcripts).
    pub runs_dir: PathBuf,
    /// The simulated user's opening message for each ticket, per bundle and
    /// repeat, shared by every arm (as SessionBench does): the first arm to
    /// reach a ticket writes it, later arms reuse it.
    pub openings_dir: PathBuf,
}

const USERSIM_OPEN: &str = "You are a developer assigning a ticket to a software engineer who works in your \
repository. The engineer has the repository but NOT the ticket text. Write your opening message to them, conveying the \
ticket faithfully and completely: you may reorganize it, but don't drop requirements. Reply with only the message.";

const USERSIM_REVIEW: &str = "You are the developer who assigned a ticket to a software engineer. You see the ticket \
and the conversation so far, not the code. Decide whether the engineer's latest reply shows the ticket is fully done. \
If something is missing, unclear or unverified, push back with a concise, specific request. Reply with only a JSON \
object: {\"approve\": true or false, \"message\": \"what you tell the engineer\"}";

/// Run one session, or continue `previous` (the same session, recorded with an
/// error) from its last checkpoint.
pub async fn run_session(
    ctx: &AtlasCtx<'_>,
    d: &Defaults,
    bundle: &AtlasBundle,
    arm: Arm,
    repeat: usize,
    previous: Option<&SessionResult>,
) -> Result<SessionResult> {
    let model = ctx.host.model_override.clone().or_else(|| d.model.clone());
    let sid = previous
        .map(|p| p.session_id.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| bench::uuid_v4(&format!("atlas/{}/{}/{}", bundle.name, arm.name(), repeat)));
    let sdir = session_dir(ctx, &bundle.name, arm, repeat, &sid);
    let mut res = SessionResult {
        bundle: bundle.name.clone(),
        arm,
        repeat,
        model: model.clone(),
        agent: Usage::default(),
        helpers: Usage::default(),
        tickets: Vec::new(),
        reward: 0.0,
        completed: false,
        error: None,
        session_id: sid,
        hook_compressions: 0,
        hook_bytes_saved: 0,
        tool_calls: BTreeMap::new(),
        grading_deferred: false,
        agent_all_models: Usage::default(),
    };
    res.grading_deferred = ctx.defer_grading || previous.is_some_and(|p| p.grading_deferred);
    let outcome = session_inner(ctx, d, bundle, arm, model.as_deref(), &sdir, &mut res).await;
    // Counted from the session's own files, so it also covers sessions that stopped early.
    bench::record_activity_in(&sdir.join("squeeze-home"), &sdir.join("claude-config"), &mut res);
    match outcome {
        Ok(()) => {
            res.completed = res.tickets.len() == bundle.tasks.len() && res.tickets.iter().all(|t| t.error.is_none())
        }
        // Recorded like any other error (the caller stops the run), so a rerun
        // continues this session from its checkpoint instead of starting over.
        Err(e) => res.error = Some(format!("{e:#}")),
    }
    // A session that couldn't even resume keeps what its earlier attempt recorded.
    if let Some(p) = previous
        && res.error.is_some()
        && res.tickets.is_empty()
        && res.agent.invocations == 0
    {
        res.agent = p.agent.clone();
        res.agent_all_models = p.agent_all_models.clone();
        res.helpers = p.helpers.clone();
        res.tickets = p.tickets.clone();
    }
    res.reward = bench::session_reward(&res.tickets);
    Ok(res)
}

/// Where a session's artifacts live: `<out>.d/<bundle>-<arm>-r<k>-<sid8>/`.
fn session_dir(ctx: &AtlasCtx<'_>, bundle: &str, arm: Arm, repeat: usize, sid: &str) -> PathBuf {
    ctx.runs_dir.join(format!("{bundle}-{}-r{}-{}", arm.name(), repeat + 1, sid.get(..8).unwrap_or_default()))
}

/// Progress saved at each clean point of a session (a ticket graded, or a
/// ticket's agent work and patch saved), so a session that stops there (a
/// crashed helper, an infrastructure error) continues instead of starting over.
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    /// The session so far: fully graded tickets and usage.
    session: SessionResult,
    /// Whether the agent's Claude Code session exists yet (resume rather than create it).
    started: bool,
    /// The next ticket, when its agent work and patch are saved but grading isn't done.
    pending: Option<TicketResult>,
    /// Size of the agent's transcript at this point. If it has grown since, the
    /// agent ran past the checkpoint (it stopped mid-ticket) and can't resume cleanly.
    transcript_bytes: u64,
}

fn transcript_bytes(cfg_dir: &Path, sid: &str) -> u64 {
    let name = format!("{sid}.jsonl");
    std::fs::read_dir(cfg_dir.join("projects"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|d| std::fs::metadata(d.path().join(&name)).ok())
        .map(|m| m.len())
        .sum()
}

fn save_checkpoint(
    sdir: &Path,
    cfg_dir: &Path,
    res: &SessionResult,
    started: bool,
    pending: Option<&TicketResult>,
) -> Result<()> {
    let cp = Checkpoint {
        session: res.clone(),
        started,
        pending: pending.cloned(),
        transcript_bytes: transcript_bytes(cfg_dir, &res.session_id),
    };
    let tmp = sdir.join(format!("{CHECKPOINT}.tmp"));
    std::fs::write(&tmp, serde_json::to_vec_pretty(&cp)?)?;
    std::fs::rename(&tmp, sdir.join(CHECKPOINT))?;
    Ok(())
}

fn clone_file(src: &Path, dst: &Path) -> Result<()> {
    // `cp -c` makes an APFS clone (instant, no extra space) and falls back to a copy.
    // macOS's own cp: a GNU cp earlier on PATH has no -c.
    let ok = std::process::Command::new("/bin/cp")
        .arg("-c")
        .arg(src)
        .arg(dst)
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if !ok {
        std::fs::copy(src, dst).with_context(|| format!("copying {}", src.display()))?;
    }
    Ok(())
}

async fn session_inner(
    ctx: &AtlasCtx<'_>,
    d: &Defaults,
    bundle: &AtlasBundle,
    arm: Arm,
    model: Option<&str>,
    sdir: &Path,
    res: &mut SessionResult,
) -> Result<()> {
    if arm.is_proxied() {
        if !ctx.proxy_urls.contains_key(&arm) {
            bail!("the {} arm needs its proxy running (bench-atlas starts it when the arm is selected)", arm.name());
        }
        if ctx.api_key_env_file.is_none() && ctx.host.auth != Auth::ApiKey {
            bail!("the {} arm needs an API key: --api-key-env-file, or --auth api-key", arm.name());
        }
    } else if ctx.host.auth == Auth::Subscription && ctx.token_env_file.is_none() && !ctx.api_key_all_arms {
        bail!("subscription runs inside containers need --token-env-file (a `claude setup-token` token)");
    }
    let sid = res.session_id.clone();
    let (cfg_dir, store_dir, tools_dir) = (sdir.join("claude-config"), sdir.join("squeeze-home"), sdir.join("tools"));
    let mut started = false;
    let mut pending: Option<TicketResult> = None;
    let rerun_hint = || format!("remove {} and its line in the results file to rerun it from scratch", sdir.display());
    match std::fs::read(sdir.join(CHECKPOINT)) {
        Ok(bytes) => {
            let cp: Checkpoint = serde_json::from_slice(&bytes).context("reading the session checkpoint")?;
            if transcript_bytes(&cfg_dir, &sid) != cp.transcript_bytes {
                bail!(
                    "the agent ran past this session's last checkpoint, so it can't resume cleanly; {}",
                    rerun_hint()
                );
            }
            res.agent = cp.session.agent;
            res.helpers = cp.session.helpers;
            res.tickets = cp.session.tickets;
            started = cp.started;
            pending = cp.pending;
            eprintln!("    resuming session {sid} after ticket {} of {}", res.tickets.len(), bundle.tasks.len());
        }
        Err(_) if transcript_bytes(&cfg_dir, &sid) > 0 => {
            bail!("this session has an agent transcript but no checkpoint, so it can't resume; {}", rerun_hint());
        }
        Err(_) => {}
    }
    for dir in [&cfg_dir, &store_dir, &tools_dir] {
        std::fs::create_dir_all(dir)?;
    }
    // A resumed session keeps the binaries it started with.
    for (src, name) in [(&ctx.claude_linux, "claude"), (&ctx.squeeze_linux, "squeeze")] {
        if !tools_dir.join(name).exists() {
            clone_file(src, &tools_dir.join(name))?;
        }
    }
    let exe = PathBuf::from(format!("{TOOLS_DIR}/squeeze"));

    let mut tools: Vec<String> = bundle
        .allowed_tools
        .clone()
        .or_else(|| d.allowed_tools.clone())
        .unwrap_or_else(|| bench::DEFAULT_TOOLS.iter().map(|s| s.to_string()).collect());
    tools.extend(arm.tool_names());
    let mut mcp_args = vec!["mcp".to_string(), "--root".into(), bundle.workdir.clone()];
    if arm.replaces_file_tools() {
        mcp_args.push("--edit-tool".into());
    }
    let mcp = if arm.uses_mcp() {
        json!({ "mcpServers": { "squeeze": {
            "type": "stdio", "command": exe, "args": mcp_args, "env": { "SQUEEZE_HOME": AGENT_STORE },
            "alwaysLoad": true
        }}})
        .to_string()
    } else {
        r#"{"mcpServers":{}}"#.to_string()
    };
    let settings = if arm.uses_hook() {
        let mut s = crate::hook::session_settings(&exe, arm.uses_mcp());
        arm.tune_settings(&mut s, &exe);
        Some(s)
    } else if arm.uses_mcp() {
        Some(json!({ "permissions": { "allow": arm.tool_names() } }))
    } else {
        None
    };
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
        a.extend(["--setting-sources".into(), "project,local".into()]);
        a.extend(["--strict-mcp-config".into(), "--mcp-config".into(), mcp.clone()]);
        if let Some(s) = &settings {
            a.extend(["--settings".into(), s.to_string()]);
        }
        a.extend(arm.extra_args());
        // As in SessionBench: no todo or ask-user tools; no web lookups of solutions.
        a.extend(["--disallowedTools", "TodoWrite", "AskUserQuestion", "WebFetch", "WebSearch"].map(String::from));
        a.extend(arm.disallowed_tools());
        a.extend([if resume { "--resume" } else { "--session-id" }.to_string(), sid.clone()]);
        a.push("--allowedTools".into()); // variadic: keep last
        a.extend(tools.iter().cloned());
        a
    };

    let mut tools_checked = false;
    for (i, task_id) in bundle.tasks.iter().enumerate().skip(res.tickets.len()) {
        if i > 0
            && let Some(p) = d.pause_between_tickets_secs.filter(|p| *p > 0)
        {
            tokio::time::sleep(Duration::from_secs(p)).await;
        }
        let task = load_task(&ctx.root, task_id)?;
        let tdir = sdir.join("tickets").join(format!("{}-{}", i + 1, task_id));
        std::fs::create_dir_all(&tdir)?;
        eprintln!("    ticket {}/{}: {task_id}", i + 1, bundle.tasks.len());
        ctx.runtime.pull(&task.image).await?;
        let (mut tr, patch) = match pending.take().filter(|p| p.id == *task_id) {
            Some(tr) => {
                eprintln!("      agent work already done; grading its saved patch");
                let patch = std::fs::read_to_string(tdir.join("agent.patch"))
                    .with_context(|| format!("no saved patch in {}", tdir.display()))?;
                (tr, patch)
            }
            None => {
                let name = format!("sqz-{}-{}", &sid[..8], i + 1);
                start_container(
                    ctx,
                    &name,
                    &task.image,
                    &[(&tools_dir, TOOLS_DIR), (&cfg_dir, AGENT_CONFIG), (&store_dir, AGENT_STORE)],
                )
                .await?;
                if arm.uses_mcp() && !tools_checked {
                    if let Err(e) = check_container_tools(ctx, &name, &bundle.workdir, arm.replaces_file_tools()).await
                    {
                        ctx.runtime.remove(&name).await;
                        return Err(e);
                    }
                    tools_checked = true;
                }
                let outcome = run_ticket(ctx, d, bundle, &task, &name, &tdir, &agent_args, &mut started, res).await;
                let patch = match &outcome {
                    Ok(_) => capture_patch(ctx, &name, &bundle.workdir).await,
                    Err(_) => Ok(String::new()),
                };
                ctx.runtime.remove(&name).await;
                let tr = outcome?;
                let patch = patch?;
                std::fs::write(tdir.join("agent.patch"), &patch)?;
                save_checkpoint(sdir, &cfg_dir, res, started, Some(&tr))?;
                (tr, patch)
            }
        };
        if ctx.defer_grading {
            eprintln!("      grading deferred (run --regrade later)");
            tr.excluded = Some(bench::GRADING_DEFERRED.into());
            res.tickets.push(tr);
            save_checkpoint(sdir, &cfg_dir, res, started, None)?;
            continue;
        }
        grade(ctx, d, bundle, &task, &patch, &tdir, &mut tr, &mut res.helpers).await?;
        eprintln!(
            "      tests {} x rubric {} = {:.3}{}",
            tr.tests_frac.map_or("-".into(), |t| format!("{t:.3}")),
            tr.rubric.map_or("-".into(), |r| format!("{r:.3}")),
            tr.reward,
            tr.excluded.as_ref().map_or(String::new(), |e| format!(" (excluded: {e})"))
        );
        res.tickets.push(tr);
        save_checkpoint(sdir, &cfg_dir, res, started, None)?;
    }
    Ok(())
}

/// The squeeze binary in the containers must serve exactly the tools this
/// build describes: arms that replace Claude Code's file tools tell the agent
/// how those tools behave, and a stale Linux build would contradict that.
async fn check_container_tools(ctx: &AtlasCtx<'_>, name: &str, workdir: &str, edit_tool: bool) -> Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut c = exec_cmd(ctx, name, workdir, None, &[]);
    c.arg(format!("{TOOLS_DIR}/squeeze")).args(["mcp", "--root", workdir]);
    if edit_tool {
        c.arg("--edit-tool");
    }
    c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    let mut child = c.spawn().context("checking the container's squeeze binary")?;
    let mut stdin = child.stdin.take().context("stdin")?;
    let init = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": { "name": "squeeze-bench", "version": "1" } } });
    let list = json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" });
    stdin.write_all(format!("{init}\n{list}\n").as_bytes()).await?;
    drop(stdin);
    let out = tokio::time::timeout(Duration::from_secs(60), child.wait_with_output())
        .await
        .map_err(|_| anyhow!("checking the container's squeeze binary timed out"))??;
    let tools = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|v| v["id"] == 2)
        .and_then(|v| v["result"]["tools"].as_array().cloned());
    if tools.as_deref() != Some(crate::mcp::tool_definitions(edit_tool).as_slice()) {
        bail!(
            "the squeeze binary for the containers ({}) serves different tools than this build; rebuild it from the \
             same source (see README: SWE-Atlas runs)",
            ctx.squeeze_linux.display()
        );
    }
    Ok(())
}

async fn start_container(ctx: &AtlasCtx<'_>, name: &str, image: &str, mounts: &[(&PathBuf, &str)]) -> Result<()> {
    ctx.runtime.remove(name).await;
    let mut c = Command::new(ctx.runtime.bin());
    c.args(["run", "-d", "--name", name]).args(ctx.runtime.platform());
    for dns in &ctx.dns {
        c.args(["--dns", dns]);
    }
    c.args(["--cpus", &ctx.cpus.to_string(), "--memory", &ctx.memory]);
    for (host, inside) in mounts {
        c.arg("-v").arg(format!("{}:{inside}", host.display()));
    }
    c.args(["--entrypoint", "sleep", image, "infinity"]);
    run_ok(c, &format!("starting {name}")).await.map(|_| ())
}

/// `container exec` with the agent's environment. Credentials are passed by
/// reference only: an env-file path, or `-e NAME` to inherit a variable.
fn exec_cmd(
    ctx: &AtlasCtx<'_>,
    name: &str,
    workdir: &str,
    agent_path: Option<&str>,
    env: &[(&str, String)],
) -> Command {
    let mut c = Command::new(ctx.runtime.bin());
    c.args(["exec", "-i"]);
    let set_here = |k: &str| env.iter().any(|(e, _)| *e == k);
    // Going through the squeeze proxy (or every arm on the key): the agent
    // authenticates with the API key file, and never gets the subscription
    // token (the proxy refuses it anyway).
    let proxied = set_here("ANTHROPIC_BASE_URL");
    match (&ctx.api_key_env_file, ctx.host.auth) {
        (Some(f), _) if proxied || ctx.api_key_all_arms => {
            c.arg("--env-file").arg(f);
        }
        (_, Auth::Subscription) => {
            if let Some(f) = &ctx.token_env_file {
                c.arg("--env-file").arg(f);
            }
        }
        (_, Auth::ApiKey) => {
            for k in ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"] {
                let overridden = set_here(k) || ctx.agent_env.iter().any(|kv| kv.starts_with(&format!("{k}=")));
                if std::env::var_os(k).is_some() && !overridden {
                    c.args(["-e", k]);
                }
            }
        }
    }
    for kv in &ctx.agent_env {
        // Variables this command sets itself (the proxy URL) win.
        if !kv.split_once('=').is_some_and(|(k, _)| set_here(k)) {
            c.args(["-e", kv]);
        }
    }
    for (k, v) in env {
        c.args(["-e", &format!("{k}={v}")]);
    }
    if let Some(p) = agent_path {
        c.args(["-e", &format!("PATH={p}")]);
    }
    c.args(["-e", &format!("CLAUDE_CONFIG_DIR={AGENT_CONFIG}"), "-e", &format!("SQUEEZE_HOME={AGENT_STORE}")]);
    c.args(["-e", "DISABLE_AUTOUPDATER=1", "-e", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"]);
    c.args(["-w", workdir, name]);
    c
}

#[allow(clippy::too_many_arguments)]
async fn run_ticket(
    ctx: &AtlasCtx<'_>,
    d: &Defaults,
    bundle: &AtlasBundle,
    task: &Task,
    name: &str,
    tdir: &Path,
    agent_args: &dyn Fn(bool) -> Vec<String>,
    started: &mut bool,
    res: &mut SessionResult,
) -> Result<TicketResult> {
    let mut tr = TicketResult {
        id: task.id.clone(),
        tests_passed: None,
        rubric: None,
        reward: 0.0,
        pushbacks: 0,
        error: None,
        tests_frac: None,
        excluded: None,
    };
    let opening =
        ctx.openings_dir.join(&bundle.name).join(format!("r{}", res.repeat + 1)).join(format!("{}.txt", task.id));
    let mut message = match (d.usersim_model.as_deref(), std::fs::read_to_string(&opening)) {
        (Some(_), Ok(shared)) if !shared.trim().is_empty() => shared,
        (Some(m), _) => {
            let r = bench::helper(ctx.host, tdir, &bench::helper_args(m, USERSIM_OPEN), &task.instruction, 600).await?;
            res.helpers.add_result(&r);
            let text = r
                .get("result")
                .and_then(Value::as_str)
                .filter(|s| !s.trim().is_empty())
                .unwrap_or(&task.instruction)
                .to_string();
            if let Some(dir) = opening.parent() {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(&opening, &text)?;
            text
        }
        (None, _) => task.instruction.clone(),
    };
    let timeout = d.timeout_secs.unwrap_or(10_800);
    let mut conversation: Vec<String> = Vec::new();
    loop {
        if tr.pushbacks > 0
            && let Some(p) = d.turn_pause(res.agent.invocations + 1)
        {
            tokio::time::sleep(p).await;
        }
        let env = agent_env(res.arm, ctx.proxy_urls.get(&res.arm).map(String::as_str), ctx.cache_ttl);
        let mut cmd = exec_cmd(ctx, name, &bundle.workdir, bundle.agent_path.as_deref(), &env);
        cmd.arg(format!("{TOOLS_DIR}/claude")).args(agent_args(*started));
        let result = bench::run_claude_command(cmd, &message, timeout).await?;
        *started = true;
        res.add_agent_result(&result);
        // Each raw result, for auditing usage and cost per model later.
        let n = std::fs::read_dir(tdir).map_or(0, |d| {
            d.flatten().filter(|e| e.file_name().to_string_lossy().starts_with("agent-result-")).count()
        });
        std::fs::write(tdir.join(format!("agent-result-{}.json", n + 1)), serde_json::to_vec_pretty(&result)?)?;
        let reply = result.get("result").and_then(Value::as_str).unwrap_or("").to_string();
        if result.get("is_error").and_then(Value::as_bool) == Some(true) {
            tr.error = Some(crate::util::truncate_chars(&reply, 300));
            break;
        }
        conversation.push(format!(
            "Developer: {}\n\nEngineer: {}",
            crate::util::truncate_chars(&message, 6000),
            crate::util::truncate_chars(&reply, 6000)
        ));
        let Some(usersim) = d.usersim_model.as_deref() else { break };
        if tr.pushbacks >= d.max_pushbacks.unwrap_or(1) {
            break;
        }
        let review = format!(
            "Ticket (only you have this text):\n{}\n\nConversation so far:\n{}",
            task.instruction,
            conversation.join("\n\n---\n\n")
        );
        let v = bench::helper(ctx.host, tdir, &bench::helper_args(usersim, USERSIM_REVIEW), &review, 600).await?;
        res.helpers.add_result(&v);
        let verdict = v.get("result").and_then(Value::as_str).and_then(bench::parse_json_reply);
        if verdict.as_ref().and_then(|v| v.get("approve")).and_then(Value::as_bool).unwrap_or(true) {
            break;
        }
        message = verdict
            .and_then(|v| v.get("message").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_else(|| "Please double-check the ticket; it isn't complete yet.".into());
        tr.pushbacks += 1;
    }
    Ok(tr)
}

/// The agent's environment: the arm's own, the proxy for a proxied arm, and
/// the run's cache lifetime.
fn agent_env(arm: Arm, proxy_url: Option<&str>, ttl: CacheTtl) -> Vec<(&'static str, String)> {
    let mut env = arm.env();
    if arm.is_proxied()
        && let Some(url) = proxy_url
    {
        env.push(("ANTHROPIC_BASE_URL", url.to_string()));
    }
    if ttl == CacheTtl::OneHour {
        env.push(("ENABLE_PROMPT_CACHING_1H", "1".into()));
    }
    env
}

async fn capture_patch(ctx: &AtlasCtx<'_>, name: &str, workdir: &str) -> Result<String> {
    let mut c = exec_cmd(ctx, name, workdir, None, &[]);
    c.args(["bash", "-c", "git add -A -- . ':!.claude' && git diff --cached HEAD; git reset -q"]);
    run_ok(c, "capturing the agent's patch").await
}

#[allow(clippy::too_many_arguments)]
async fn grade(
    ctx: &AtlasCtx<'_>,
    d: &Defaults,
    bundle: &AtlasBundle,
    task: &Task,
    patch: &str,
    tdir: &Path,
    tr: &mut TicketResult,
    helpers: &mut Usage,
) -> Result<()> {
    grade_tests(ctx, bundle, task, patch, tdir, tr).await?;
    // Rubric: SWE-Atlas's own judge prompts, one call per item (or batched).
    tr.rubric = judge(ctx, d.judge_model.as_deref(), task, patch, tdir, helpers).await?;
    tr.reward = ticket_reward(tr);
    Ok(())
}

/// SessionBench's reward_frac: fractional tests x rubric.
fn ticket_reward(tr: &TicketResult) -> f64 {
    tr.tests_frac.unwrap_or(0.0) * tr.rubric.unwrap_or(1.0)
}

/// The test half of grading: rebuild the agent's workspace in a fresh
/// container and run the task's own test.sh, as SWE-Atlas does.
async fn grade_tests(
    ctx: &AtlasCtx<'_>,
    bundle: &AtlasBundle,
    task: &Task,
    patch: &str,
    tdir: &Path,
    tr: &mut TicketResult,
) -> Result<()> {
    tr.tests_frac = None;
    tr.tests_passed = None;
    tr.excluded = None;
    // An agent that changed nothing scores 0 on tests (SWE-Atlas's validator
    // can't apply an empty patch, and excluding the ticket would reward doing nothing).
    if patch.trim().is_empty() {
        tr.tests_frac = Some(0.0);
        tr.tests_passed = Some(false);
        return Ok(());
    }
    let logs = tdir.join("grade-logs");
    let pdir = tdir.join("patch");
    let _ = std::fs::remove_dir_all(&logs);
    std::fs::create_dir_all(&logs)?;
    std::fs::create_dir_all(&pdir)?;
    std::fs::write(pdir.join("agent.patch"), patch)?;
    let path = bundle.grade_path.as_ref().map(|p| format!("export PATH='{p}'; ")).unwrap_or_default();
    // The patch is the agent's workspace relative to HEAD, which includes any
    // tracked files the image itself ships modified (e.g. a symlink in the
    // grafana images). Reset tracked files and drop untracked ones (ignored
    // files such as node_modules stay) so the patch reproduces that workspace.
    let script = format!(
        "{path}cd {wd} && git checkout -q HEAD -- . && git clean -fdq && \
         git apply --whitespace=nowarn /sqz-patch/agent.patch || {{ echo SQUEEZE_PATCH_APPLY_FAILED; exit 3; }}; \
         bash /tests/test.sh",
        wd = bundle.workdir
    );
    let mut c = Command::new(ctx.runtime.bin());
    c.args(["run", "--rm"]).args(ctx.runtime.platform());
    for dns in &ctx.dns {
        c.args(["--dns", dns]);
    }
    c.args(["--cpus", &ctx.cpus.to_string(), "--memory", &ctx.memory]);
    for (host, inside) in [(task.dir.join("tests"), "/tests"), (logs.clone(), "/logs"), (pdir.clone(), "/sqz-patch")] {
        c.arg("-v").arg(format!("{}:{inside}", host.display()));
    }
    c.args(["--entrypoint", "bash", &task.image, "-c", &script]);
    c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    let out = tokio::time::timeout(Duration::from_secs(GRADE_TIMEOUT_SECS), c.output())
        .await
        .map_err(|_| anyhow!("grading timed out"))??;
    let mut log = String::from_utf8_lossy(&out.stdout).to_string();
    log.push_str("\n--- stderr ---\n");
    log.push_str(&String::from_utf8_lossy(&out.stderr));
    std::fs::write(tdir.join("grade-stdout.txt"), log.as_bytes())?;
    if log.contains("SQUEEZE_PATCH_APPLY_FAILED") {
        tr.excluded = Some("patch_apply_failed".into());
        return Ok(());
    }
    match tests_score(&logs.join("verifier")) {
        Ok((frac, binary, ran)) => {
            tr.tests_frac = Some(frac);
            tr.tests_passed = Some(binary >= 1.0);
            if ran == 0 {
                tr.excluded = Some("vacuous_tests".into());
            }
        }
        Err(e) => tr.excluded = Some(format!("tests_error: {e}")),
    }
    Ok(())
}

/// Grade what is missing in one recorded session, from its saved patches: the
/// tests of tickets without a test score (failed, or deferred), and, given a
/// judge model, the rubric of tickets without a rubric score. Existing scores
/// are kept; only the judge calls a model. A ticket that can't be graded is
/// marked excluded with the reason and the rest go on; a usage limit stops.
pub async fn regrade_session(
    ctx: &AtlasCtx<'_>,
    bundle: &AtlasBundle,
    r: &mut SessionResult,
    judge_model: Option<&str>,
) -> Result<()> {
    let sdir = session_dir(ctx, &r.bundle, r.arm, r.repeat, &r.session_id);
    eprintln!("regrading {} / {} / repeat {}", r.bundle, r.arm.name(), r.repeat + 1);
    for i in 0..r.tickets.len() {
        let tr = &r.tickets[i];
        let needs_tests = tr.tests_frac.is_none();
        let needs_rubric = tr.rubric.is_none() && judge_model.is_some();
        if !needs_tests && !needs_rubric {
            eprintln!("    {}: already graded; kept", tr.id);
            continue;
        }
        let tdir = sdir.join("tickets").join(format!("{}-{}", i + 1, tr.id));
        let mut tr = tr.clone();
        let graded: Result<()> = async {
            let task = load_task(&ctx.root, &tr.id)?;
            let patch = std::fs::read_to_string(tdir.join("agent.patch"))
                .with_context(|| format!("no saved patch in {}", tdir.display()))?;
            if needs_tests {
                grade_tests(ctx, bundle, &task, &patch, &tdir, &mut tr).await?;
            }
            if needs_rubric {
                tr.rubric = judge(ctx, judge_model, &task, &patch, &tdir, &mut r.helpers).await?;
            }
            Ok(())
        }
        .await;
        match graded {
            Ok(()) => {}
            Err(e) if e.is::<bench::LimitReached>() => return Err(e),
            Err(e) => {
                tr.excluded = Some(format!("grading_error: {}", crate::util::truncate_chars(&format!("{e:#}"), 200)))
            }
        }
        tr.reward = ticket_reward(&tr);
        eprintln!(
            "    {}: tests {} x rubric {} = {:.3}{}",
            tr.id,
            tr.tests_frac.map_or("-".into(), |t| format!("{t:.3}")),
            tr.rubric.map_or("-".into(), |x| format!("{x:.3}")),
            tr.reward,
            tr.excluded.as_ref().map_or(String::new(), |e| format!(" (excluded: {e})"))
        );
        r.tickets[i] = tr;
        r.reward = bench::session_reward(&r.tickets);
    }
    r.reward = bench::session_reward(&r.tickets);
    // Recount from the session's own files; this also corrects rows recorded
    // before costs were read as running totals.
    bench::record_activity_in(&sdir.join("squeeze-home"), &sdir.join("claude-config"), r);
    Ok(())
}

/// SessionBench's fractional test score from test.sh's outputs: relevant
/// missing->pass over (missing->pass + missing->fail + pass->fail), falling
/// back to the binary score. Also returns how many tests ran at baseline.
pub fn tests_score(verifier: &Path) -> Result<(f64, f64, u64)> {
    let comp: Value = serde_json::from_str(&std::fs::read_to_string(verifier.join("comparison_results.json"))?)?;
    let ran = std::fs::read_to_string(verifier.join("agent_baseline_results.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .map(|b| {
            let c = |k: &str| b.get(k).and_then(Value::as_u64).unwrap_or(0);
            c("passed_test_count") + c("failed_test_count")
        })
        .unwrap_or(0);
    if let Some(e) = comp.get("error") {
        // The baseline ran but the agent's side couldn't be tested, for example
        // because the agent deleted a file the hidden test patch changes, so
        // that patch no longer applies. SWE-Atlas scores that with its own
        // tests_reward (0): an outcome of the agent's work, not a reason to
        // drop the ticket from the mean.
        let own = std::fs::read_to_string(verifier.join("tests_reward.txt")).ok().and_then(|t| t.trim().parse().ok());
        return match own {
            Some(r) if ran > 0 => Ok((r, r, ran)),
            _ => bail!("{e}"),
        };
    }
    let binary = comp.get("tests_reward").and_then(Value::as_f64).unwrap_or(0.0);
    let n = |k: &str| comp.pointer(&format!("/filtered_summary/{k}")).and_then(Value::as_u64).unwrap_or(0);
    let denom = n("missing_to_pass") + n("missing_to_fail") + n("pass_to_fail");
    let frac =
        if denom > 0 { (n("missing_to_pass") as f64 / denom as f64 * 10_000.0).round() / 10_000.0 } else { binary };
    Ok((frac, binary, ran))
}

fn rubric_status(reply: &Value, index: usize) -> Option<String> {
    let text = reply.get("result").and_then(Value::as_str)?;
    let v = bench::parse_json_reply(text)?;
    let item = v.get("ratings").and_then(|r| r.get(index)).unwrap_or(&v);
    Some(item.get("status")?.as_str()?.trim().to_uppercase())
}

async fn judge(
    ctx: &AtlasCtx<'_>,
    model: Option<&str>,
    task: &Task,
    patch: &str,
    tdir: &Path,
    helpers: &mut Usage,
) -> Result<Option<f64>> {
    let Some(model) = model else { return Ok(None) };
    let items: Vec<&Value> = task
        .rubrics
        .iter()
        .filter(|r| r.pointer("/annotations/type").and_then(Value::as_str) != Some("high level intent"))
        .collect();
    if items.is_empty() {
        return Ok(Some(1.0));
    }
    let title = |r: &Value| r.get("title").and_then(Value::as_str).unwrap_or("").to_string();
    let negative =
        |r: &Value| r.pointer("/annotations/type").and_then(Value::as_str).is_some_and(|t| t.contains("negative"));
    let mut statuses: Vec<Option<String>> = Vec::new();
    if ctx.judge_batch {
        let list: Vec<String> = items.iter().enumerate().map(|(i, r)| format!("{}. {}", i + 1, title(r))).collect();
        let criteria = format!(
            "{}\n(Rate each of these {} criteria independently; return one rating per criterion, in order.)",
            list.join("\n"),
            items.len()
        );
        let user = py_format(
            &task.user_template,
            &[("problem_statement", &task.problem), ("model_answer", patch), ("title", &criteria)],
        );
        let r = bench::helper(ctx.host, tdir, &bench::helper_args(model, &task.system_prompt), &user, 1800).await?;
        helpers.add_result(&r);
        statuses = (0..items.len()).map(|i| rubric_status(&r, i)).collect();
    } else {
        for r in &items {
            let t = title(r);
            let user = py_format(
                &task.user_template,
                &[("problem_statement", &task.problem), ("model_answer", patch), ("title", &t)],
            );
            let reply =
                bench::helper(ctx.host, tdir, &bench::helper_args(model, &task.system_prompt), &user, 900).await?;
            helpers.add_result(&reply);
            statuses.push(rubric_status(&reply, 0));
        }
    }
    let mut details = Vec::new();
    let (mut evaluable, mut passed) = (0usize, 0usize);
    for (r, status) in items.iter().zip(&statuses) {
        let ok = match status.as_deref() {
            Some(s @ ("YES" | "NO")) => {
                evaluable += 1;
                let pass = if negative(r) { s == "NO" } else { s == "YES" };
                passed += pass as usize;
                Some(pass)
            }
            _ => None,
        };
        details.push(json!({ "title": title(r), "negative": negative(r), "status": status, "passes": ok }));
    }
    std::fs::write(tdir.join("rubric.json"), serde_json::to_vec_pretty(&details)?)?;
    Ok((evaluable > 0).then(|| passed as f64 / evaluable as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_round_trips_and_tracks_the_transcript() {
        let dir = tempfile::tempdir().unwrap();
        let (sdir, cfg) = (dir.path(), dir.path().join("claude-config"));
        let sid = "37c083e9-0000-4000-8000-000000000000";
        assert_eq!(transcript_bytes(&cfg, sid), 0, "no transcript yet");
        let project = cfg.join("projects").join("-grafana");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(project.join(format!("{sid}.jsonl")), "{\"type\":\"user\"}\n").unwrap();
        std::fs::write(project.join("other-session.jsonl"), "ignored").unwrap();

        let ticket = |id: &str| TicketResult {
            id: id.into(),
            tests_passed: None,
            rubric: Some(0.5),
            reward: 0.0,
            pushbacks: 1,
            error: None,
            tests_frac: None,
            excluded: None,
        };
        let res = SessionResult {
            bundle: "b".into(),
            arm: Arm::Squeeze,
            repeat: 0,
            model: None,
            agent: Usage { invocations: 3, ..Usage::default() },
            helpers: Usage::default(),
            tickets: vec![ticket("t1")],
            reward: 0.0,
            completed: false,
            error: None,
            session_id: sid.into(),
            hook_compressions: 0,
            hook_bytes_saved: 0,
            tool_calls: BTreeMap::new(),
            grading_deferred: false,
            agent_all_models: Usage::default(),
        };
        save_checkpoint(sdir, &cfg, &res, true, Some(&ticket("t2"))).unwrap();
        let cp: Checkpoint = serde_json::from_slice(&std::fs::read(sdir.join(CHECKPOINT)).unwrap()).unwrap();
        assert!(cp.started);
        assert_eq!(cp.session.tickets.len(), 1);
        assert_eq!(cp.session.agent.invocations, 3);
        assert_eq!(cp.pending.unwrap().id, "t2");
        assert_eq!(cp.transcript_bytes, 16, "only this session's transcript counts");
        assert!(!sdir.join(format!("{CHECKPOINT}.tmp")).exists());

        // The agent writing past the checkpoint shows up as a size change.
        std::fs::write(project.join(format!("{sid}.jsonl")), "{\"type\":\"user\"}\n{\"type\":\"assistant\"}\n")
            .unwrap();
        assert_ne!(transcript_bytes(&cfg, sid), cp.transcript_bytes);
    }

    #[test]
    fn agents_get_the_api_key_file_only_when_proxied_or_all_arms() {
        let host = bench::Context {
            claude_bin: "claude".into(),
            exe: PathBuf::from("/bin/squeeze"),
            store_root: PathBuf::from("/tmp/store"),
            auth: Auth::Subscription,
            proxy_urls: BTreeMap::new(),
            model_override: None,
            include_user_settings: false,
            keep_dirs: false,
        };
        let ctx = |all_arms: bool| AtlasCtx {
            host: &host,
            runtime: Runtime::Apple,
            root: PathBuf::new(),
            claude_linux: PathBuf::new(),
            squeeze_linux: PathBuf::new(),
            token_env_file: Some(PathBuf::from("/cfg/token.env")),
            api_key_env_file: Some(PathBuf::from("/cfg/api.env")),
            api_key_all_arms: all_arms,
            proxy_urls: BTreeMap::new(),
            agent_env: Vec::new(),
            cache_ttl: CacheTtl::Client,
            cpus: 1,
            memory: "1g".into(),
            dns: Vec::new(),
            judge_batch: false,
            defer_grading: true,
            runs_dir: PathBuf::new(),
            openings_dir: PathBuf::new(),
        };
        let env_file = |ctx: &AtlasCtx<'_>, env: &[(&str, String)]| {
            let c = exec_cmd(ctx, "box", "/w", None, env);
            let args: Vec<String> = c.as_std().get_args().map(|a| a.to_string_lossy().into_owned()).collect();
            args.iter().position(|a| a == "--env-file").map(|i| args[i + 1].clone())
        };
        let proxied = [("ANTHROPIC_BASE_URL", "http://h:1".to_string())];
        let (some, all) = (ctx(false), ctx(true));
        assert_eq!(env_file(&some, &proxied).as_deref(), Some("/cfg/api.env"));
        assert_eq!(env_file(&some, &[]).as_deref(), Some("/cfg/token.env"), "vanilla stays on the subscription");
        assert_eq!(env_file(&all, &[]).as_deref(), Some("/cfg/api.env"), "every arm on the key");
        assert_eq!(env_file(&all, &proxied).as_deref(), Some("/cfg/api.env"));
    }

    #[test]
    fn cache_ttl_is_the_clients_unless_set_for_the_run() {
        let has = |env: &[(&str, String)], k: &str| env.iter().any(|(e, _)| *e == k);
        let proxied = agent_env(Arm::SqueezeProxy, Some("http://h:1"), "client".parse().unwrap());
        assert!(has(&proxied, "ANTHROPIC_BASE_URL"));
        assert!(!has(&proxied, "ENABLE_PROMPT_CACHING_1H"), "an API key keeps Claude Code's own TTL");
        let vanilla = agent_env(Arm::Vanilla, Some("http://h:1"), "1h".parse().unwrap());
        assert!(has(&vanilla, "ENABLE_PROMPT_CACHING_1H"), "1h applies to every arm");
        assert!(!has(&vanilla, "ANTHROPIC_BASE_URL"));
        assert!("5m".parse::<CacheTtl>().is_err());
    }

    #[test]
    fn normalizes_image_refs() {
        assert_eq!(normalize_image("ghcr.io/x/y:tag_1.1@sha256:abc"), "ghcr.io/x/y@sha256:abc");
        assert_eq!(normalize_image("localhost:5000/y@sha256:abc"), "localhost:5000/y@sha256:abc");
        assert_eq!(normalize_image("alpine:3.20"), "alpine:3.20");
    }

    #[test]
    fn py_format_matches_python_semantics() {
        let t = "# Prompt\n{problem_statement}\n\n# Response\n{model_answer}\n\n#Rubric\n{{\n  \"rubric_statement\": {title}\n}}";
        let out = py_format(t, &[("problem_statement", "P"), ("model_answer", "fn x() { y{{z}} }"), ("title", "T")]);
        assert_eq!(out, "# Prompt\nP\n\n# Response\nfn x() { y{{z}} }\n\n#Rubric\n{\n  \"rubric_statement\": T\n}");
    }

    #[test]
    fn fractional_tests_score() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("comparison_results.json"),
            r#"{"tests_reward": 0.0, "filtered_summary": {"missing_to_pass": 3, "missing_to_fail": 1, "pass_to_fail": 0}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("agent_baseline_results.json"),
            r#"{"passed_test_count": 5, "failed_test_count": 2}"#,
        )
        .unwrap();
        assert_eq!(tests_score(dir.path()).unwrap(), (0.75, 0.0, 7));
        std::fs::write(
            dir.path().join("comparison_results.json"),
            r#"{"tests_reward": 1.0, "filtered_summary": {"missing_to_pass": 0, "missing_to_fail": 0, "pass_to_fail": 0}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("agent_baseline_results.json"),
            r#"{"passed_test_count": 0, "failed_test_count": 0}"#,
        )
        .unwrap();
        assert_eq!(tests_score(dir.path()).unwrap(), (1.0, 1.0, 0), "falls back to binary; 0 ran = vacuous");
    }

    #[test]
    fn a_patch_that_breaks_the_test_patch_scores_zero() {
        let dir = tempfile::tempdir().unwrap();
        let error = r#"{"error": "[Errno 2] No such file or directory: '/tmp_artifact_storage/agent_comparison_results.json'"}"#;
        std::fs::write(dir.path().join("comparison_results.json"), error).unwrap();
        std::fs::write(
            dir.path().join("agent_baseline_results.json"),
            r#"{"passed_test_count": 337, "failed_test_count": 0}"#,
        )
        .unwrap();
        // Without SWE-Atlas's own score it's a grading error, as before.
        assert!(tests_score(dir.path()).is_err());
        std::fs::write(dir.path().join("tests_reward.txt"), "0.0\n").unwrap();
        assert_eq!(tests_score(dir.path()).unwrap(), (0.0, 0.0, 337), "SWE-Atlas's own score: 0, counted");
        // If even the baseline didn't run, the environment is at fault: still an error.
        std::fs::write(
            dir.path().join("agent_baseline_results.json"),
            r#"{"passed_test_count": 0, "failed_test_count": 0}"#,
        )
        .unwrap();
        assert!(tests_score(dir.path()).is_err());
    }

    #[test]
    fn rubric_status_parsing() {
        let r = json!({ "result": "```json\n{\"ratings\": [{\"status\": \"yes\", \"justification\": \"x\"}, {\"status\": \"NO\"}]}\n```" });
        assert_eq!(rubric_status(&r, 0).as_deref(), Some("YES"));
        assert_eq!(rubric_status(&r, 1).as_deref(), Some("NO"));
        assert_eq!(rubric_status(&json!({ "result": "no json" }), 0), None);
    }
}
