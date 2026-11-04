# squeeze

Token compression and token-efficient tools for Claude Code. Works with a Claude
subscription or an API key, within Anthropic's terms.

squeeze shrinks bulky tool output, such as long build and test logs, before the
model sees it. It also gives the agent MCP tools that do more per call.
Everything runs on your machine.

> squeeze is an independent open-source project. It is not affiliated with,
> endorsed by, or supported by Anthropic. "Claude Code" refers to Anthropic's
> product, which squeeze works with.

## Two modes

| | **Hook mode** (default) | **Proxy mode** (`--proxy`) |
|---|---|---|
| How | Claude Code's documented [`PostToolUse` hook](https://code.claude.com/docs/en/hooks) rewrites tool output before Claude sees it | A local proxy on `ANTHROPIC_BASE_URL` rewrites `tool_result` text in API requests |
| Credentials | **Any**: subscription login, `claude setup-token`, API key. squeeze never sees credentials or API traffic | **API key or gateway credential only** |
| Extras | | Replaces repeated identical outputs with references; experimental cold-cache pruning |

Anthropic's terms do not allow third-party tools to route or intermediate Claude
subscription (Free/Pro/Max/Ultra) credentials
([legal and compliance](https://code.claude.com/docs/en/legal-and-compliance),
[usage policy](https://www.anthropic.com/legal/aup)). Hook mode avoids that
entirely. Claude Code runs unmodified, logs in through Anthropic's own flow, and
talks to Anthropic directly. squeeze only adds a hook and an MCP server, the
extension points Claude Code documents for this. Proxy mode refuses to start
without API-key credentials. It also **rejects, never forwards**, any request
that carries subscription OAuth credentials.

```
hook mode:   claude (your login, unmodified) ───────────────▶ api.anthropic.com
               ├─ PostToolUse hook ─▶ squeeze hook   (compresses Bash/MCP output)
               └─ MCP (stdio) ──────▶ squeeze mcp    (read_many, outline, search, expand)

proxy mode:  claude ──ANTHROPIC_BASE_URL──▶ squeeze proxy ──▶ api.anthropic.com   (API key only)
```

## What gets compressed

| Rewrite | Applies to | Lossy? |
|---|---|---|
| Strip ANSI codes and resolve `\r` progress bars | Bash output | no |
| Collapse repeated lines (`retry ×40`) | logs | no |
| Collapse long runs of same-prefix lines (`Compiling …`, `test … ok`), keeping both ends | logs | yes* |
| Keep head and tail of long logs plus error/warning/fail lines from the middle | logs | yes* |
| Minify pretty-printed JSON | other MCP tools | no |
| Replace a repeat of an identical call's identical output with a reference | any tool (proxy mode) | no |

\* Lossy rewrites store the original under `~/.squeeze/blobs` and leave a note
such as `[squeeze: output compressed from 401 to 7 lines; for the full output
call mcp__squeeze__expand with id "…"]`, so nothing is permanently lost.

File content is never summarized. `cat`, `sed -n`, `git diff` and similar output
is file content, and stays verbatim. A command chain counts as a log if any part
of it isn't a file viewer (`cargo build; echo "exit $?"` is a log).

**Prompt-cache safety.** Claude Code resends the whole conversation every turn
and relies on prompt caching. Rewriting earlier context differently from turn to
turn would break the cache. It would also trip the API's preserved-thinking
check, which rejects a request whose earlier `messages` differ from the request
that produced the thinking. Both modes avoid this:

- **Hook mode:** the hook fires once per tool call, and Claude Code stores the
  rewritten output in its history. The end-to-end tests check that it stays
  byte-identical in every later request.
- **Proxy mode:** the rewrite of message *i* depends only on messages `0..=i`,
  so each output is compressed once, before its first cache write, and resent
  identically afterwards. This is covered by property tests and by tests on
  real Claude Code traffic.

**The MCP tools** aim to cut turns, since every turn re-reads the whole cached
context:

- `read_many`: several files or line ranges in one call, in `Read`'s `N<TAB>line` format
- `outline`: definitions with line ranges (Rust, Python, JS/TS, Go, JVM languages, C/C++, Ruby, Markdown)
- `search`: regex search that honors `.gitignore`, grouped by file, with a match budget
- `expand`: recover the original of anything squeeze compressed
- `edit_many` (opt in with `--edit-tool`): batched exact-string edits, validated
  before any write, allowed only on files read via `read_many` and unchanged
  since. Claude Code asks before each use, and `/rewind` checkpoints don't track it.

## Install and use

```sh
cargo install --git https://github.com/nurtantioquidar/squeeze   # or, from a checkout: cargo install --path .
# or download a prebuilt binary (macOS, static Linux; x86_64 and arm64) from the Releases page

squeeze run                          # interactive Claude Code with squeeze (hook mode)
squeeze run -- -p "fix the tests"    # claude's own arguments go after --
squeeze stats                        # what squeeze has compressed
```

`squeeze run` launches `claude` with session-only `--settings` (the hook, plus
pre-approval of squeeze's read-only tools) and `--mcp-config`. It does not edit
your settings files. If you pass your own `--settings`, squeeze merges into it.

To add only the tools to your own setup, set `alwaysLoad: true` on the server:

```json
{ "mcpServers": { "squeeze": { "type": "stdio", "command": "squeeze", "args": ["mcp"], "alwaysLoad": true } } }
```

By default Claude Code defers every MCP tool behind tool search, so the model
sees only the tool names and rarely loads them. In the first grafana_ts run, Opus
made **zero** calls to squeeze's tools. Each squeeze tool also marks itself
always-loaded (`_meta` `anthropic/alwaysLoad`). The server-level flag also makes
Claude Code connect the server before the first turn. That keeps the tool list,
and the prompt cache, stable for the whole session.

API-key users who want proxy mode:

```sh
export ANTHROPIC_API_KEY=sk-ant-api03-...
squeeze run --proxy
squeeze proxy --port 8787            # standalone, for IDE extensions or scripts
```

## Benchmarking

`squeeze bench` follows the structure of SessionBench, the benchmark published
for Fermat ([akuik/sessionbench-swe](https://github.com/akuik/sessionbench-swe)):

- **Bundles** of related tickets run as **one resumed Claude Code session**
  (`--session-id`, then `--resume`).
- A **simulated user** reviews each ticket (ticket, final reply, diff, test
  result) and can push back.
- **Per-ticket reward** = tests (hidden tests the agent never sees) × an LLM
  judge's rubric score.
- **Paired A/B, K repeats**, arm order alternated between repeats.
- **Headline cost** is the coding agent's usage, from Claude Code's own
  `--output-format json` report. Simulated-user and judge usage are reported
  separately.

It goes beyond that methodology in a few ways:

- **Ablation arms.** SessionBench compared only the whole stack against vanilla.
  Here you can choose `vanilla`, `squeeze` (hook + MCP), `squeeze-hook`
  (compression only), `squeeze-mcp` (tools only) and `squeeze-proxy` (API key only).
- **A paired bootstrap 95% CI** on the cost ratio.
- **Resume after a usage limit.** Rerun the same command and finished sessions
  are skipped.
- **Local runs.** No cloud runner or Docker is needed.

```sh
# On your subscription: unmodified Claude Code with your login (or a token from
# `claude setup-token` in CLAUDE_CODE_OAUTH_TOKEN, which Anthropic documents for
# scripts and CI). API-key variables are removed for these runs. Uses your usage limits.
squeeze bench bench/example.toml --auth subscription --repeats 3

# On an API key, with all ablations:
squeeze bench bench/example.toml --auth api-key --repeats 5 \
  --arms vanilla,squeeze,squeeze-hook,squeeze-mcp,squeeze-proxy
```

`--auth` is required, so a run never silently bills an API key or silently uses
a subscription. The example bundle runs every role on Opus 5.5, as SessionBench
ran every role on Opus. `--model`, `--usersim-model` and `--judge-model`
override that (`none` turns the simulated user or judge off).

The summary also shows what squeeze actually did in each arm: `squeezed`
(outputs the hook compressed per session) and `sq tools` (calls to squeeze's MCP
tools). A cost difference with both at zero is noise, not a squeeze effect.

`bench/example.toml` is a small 5-ticket smoke/demo bundle, not a
representative benchmark. Meaningful numbers need real repositories with real
build and test output. SWE-Atlas tasks would be a natural source, but they need
Docker images.

Known limits of the method: a simulated user and an LLM judge add noise, and the
CLI can't set them to temperature 0. Small K gives wide intervals. On a
subscription, "agent $" is Claude Code's list-price figure, which measures usage,
not a bill.

### SWE-Atlas runs (Fermat's own tasks)

`squeeze bench-atlas bench/swe-atlas-rf.toml` runs SessionBench's three
published bundles (grafana Go, grafana TypeScript, scapy; 15 SWE-Atlas
refactoring tickets) the way SessionBench does:

- **Containers:** each ticket runs in its own task container, with the official,
  checksum-verified Claude Code Linux binary inside, and one conversation is
  resumed across the bundle.
- **Simulated user:** writes the opening message from the ticket, so the agent
  never sees the raw text, and reviews the agent's replies.
- **Scoring:** SWE-Atlas's own `test.sh` in a fresh container, scored as
  SessionBench's fractional test score, times the rubric. Rubric items are judged
  one call per item with SWE-Atlas's own prompts, or in one call with
  `--judge-batch`.

Setup, once:

```sh
git clone --depth 1 https://github.com/scaleapi/SWE-Atlas ~/.squeeze/cache/SWE-Atlas
# The official Claude Code linux-x64 binary for your version, checked against
# downloads.claude.ai/claude-code-releases/<version>/manifest.json, at
#   ~/.squeeze/cache/claude-code/<version>/linux-x64/claude
# squeeze built for linux x86_64 (static musl), at ~/.squeeze/cache/bin/squeeze-linux-x64
claude setup-token   # then put CLAUDE_CODE_OAUTH_TOKEN=... in a chmod 600 file
```

```sh
squeeze bench-atlas bench/swe-atlas-rf.toml --auth subscription \
  --token-env-file ~/.config/squeeze/claude-token.env --bundles grafana_ts --repeats 1
```

The token reaches the containers through the runtime's `--env-file`; squeeze
passes the path and never reads the file. `--runtime container` (Apple's
runtime, the default on macOS) or `--runtime docker`. If your Mac uses a local
DNS resolver (a VPN or filter app), containers can't resolve names: pass
`--dns 1.1.1.1`.

Long runs stop sometimes. To continue one, rerun the same command with the same
`--out`:

- **Sessions recorded with an error** continue from their last checkpoint:
  either a graded ticket, or a ticket whose agent work and patch are already
  saved. If the agent stopped mid-ticket, the session can't resume cleanly and
  the error says what to delete to rerun it from scratch.
- **Crashed helper calls** (simulated user, judge) are retried twice, because
  they are stateless.
- **Claude Code version:** `--claude-linux` defaults to the host's Claude Code
  version. If the host updated in the meantime, pass the original binary's
  path so both arms use the same agent version.
- **Grading only:** `--regrade` reruns the tests, from the saved patches, for
  tickets that have no test score. It keeps the rubric scores and makes no model
  calls.

Findings from setting this up:

- The **scapy images** put a Python 3.12 venv first on `PATH`. scapy's bundled
  `six` can't import under 3.12, so the official `test.sh` runs **zero tests**
  and the test score is vacuous. `bench/swe-atlas-rf.toml` grades scapy with the
  image's own Python 3.8 (`grade_path`). With that, the baseline runs 1,059
  tests instead of 0.
- On an M4 Pro under Rosetta, grading a grafana_ts ticket takes about 9 minutes
  (1,585 existing tests plus the new ones). Each grafana image is about 4.2 GB.
- SWE-Atlas's validator can't apply an empty patch, so a ticket where the
  agent changed nothing scores 0 on tests rather than being dropped.
- The **grafana images** ship one tracked file already modified (a symlink under
  `pkg/services/provisioning`). The captured patch includes it, so grading
  resets the fresh container's workspace to `HEAD` before applying the patch, as
  SWE-Atlas's own validator does.

### Free checks

| Command | Cost | What it tells you |
|---|---|---|
| `cargo test` | free | compression rules, cache stability, credential refusal, stream handling, hook output shape, MCP tools, benchmark statistics |
| `cargo test --test e2e_claude -- --include-ignored` | free | the real `claude` CLI against a scripted local mock API: hook mode, proxy mode, and `squeeze bench` across all five arms. Uses a dummy key and an isolated config dir, so nothing reaches Anthropic |
| `squeeze replay` | free, offline | applies the compression rules to your past transcripts (`~/.claude/projects`) and prices them against the recorded usage |

## Results so far

**SWE-Atlas grafana_ts** (SessionBench's TypeScript bundle: 5 refactoring
tickets in one resumed session, Opus 5.5 as agent, simulated user and judge,
Claude Code 2.1.317, one session per arm):

| | vanilla | squeeze |
|---|---|---|
| Agent cost (Claude Code's list-price figure) | $42.66 | $33.39 (0.78x) |
| Turns | 164 | 158 |
| Cache-read tokens | 27.3M | 23.4M |
| Reward (fractional tests x rubric, mean of 5 tickets) | 0.905 | 0.878 |
| Outputs the hook compressed | | 6 (~36 KB) |
| Calls to squeeze's MCP tools | | 0 |

What this does and doesn't show:

- **Little of the 22% gap is squeeze's own doing.**
  - *Compression:* the 6 compressed outputs account for roughly 1M of the 3.9M
    fewer cache-read tokens.
  - *Replay:* `squeeze replay` on vanilla's own transcript estimates **1.6%**.
  - *The rest is a different trajectory:* the squeeze session drew one fewer
    pushback from the simulated user. With one session per arm, variance and
    effect can't be separated.
- **The MCP tools were never used** in that run. Claude Code defers MCP tools
  behind tool search by default, so the model saw only their names. squeeze now
  opts out of deferral (see [Install and use](#install-and-use)). In a one-ticket
  check afterwards, Opus used `read_many` to batch file reads (3 calls, replacing
  most `cat`/`sed` reads), but it kept searching with `grep`. Full sessions with
  the tools visible are in progress.
- For comparison, [Fermat's Last Token](https://www.quotientlabs.com/#how) reports
  $60.62 → $29.79 (0.49x) on the same bundle with its own stack.

On one developer's 38 past sessions (`squeeze replay`), mostly data and infra
work driven by shell commands, the estimated input-side saving was **0.3%** with
default rules and **1.1%** with aggressive ones. In that workload most context
is the agent's own inline scripts and written files, and file content it needs
verbatim. Compression helps most in **log-heavy** work (builds, test suites,
installs). Turn reduction from the MCP tools is what `squeeze bench` measures.
Contributed results are welcome.

## Limitations

- Proxy mode supports the Anthropic API only (not Bedrock, Vertex or Foundry).
  Hook mode works with any provider Claude Code supports.
- Hook mode doesn't do cross-call dedup.
- Hook mode can't touch the output of commands that **exit non-zero**, such as
  failing builds or failing test runs. Claude Code routes those to
  `PostToolUseFailure`, whose output can't be replaced: in testing, both
  `updatedToolOutput` and `additionalContext` left the output unchanged. Claude
  Code itself already trims failing output to about 10k characters. Proxy mode
  (API key) compresses failing output too.
- Token figures in `stats` and `replay` are estimates (~4 bytes per token).
- Claude Code's `Edit` still requires a `Read` of the file first, so reading
  through `read_many` alone is not enough before editing.
- `replay` parses Claude Code's undocumented transcript format. It's best
  effort and skips subagent threads.
- Proxy mode: changing settings in the middle of a resumed conversation changes
  earlier context once. Claude Code recovers by dropping earlier thinking
  blocks, at the cost of one cache rebuild.

## Data and privacy

Everything stays local under `~/.squeeze` (or `$SQUEEZE_HOME`): originals of
compressed outputs (`blobs/`), per-request stats (`stats.jsonl`), and logs.
`squeeze gc --days 30` deletes old originals.

## License

Apache-2.0. See [LICENSE](LICENSE).
