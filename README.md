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
| Extras | | Replaces repeated identical outputs with references; summarizes the conversation at pauses longer than the prompt cache lasts; experimental cold-cache pruning and server-side context editing |

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
- `search`: regex search that honors `.gitignore` (dotfiles included, version-control
  directories skipped), grouped by file, with a match budget; without a pattern it lists files
- `expand`: recover the original of anything squeeze compressed
- `edit_many` (opt in with `--edit-tool`): exact-string replacements, new files
  and full rewrites, batched in one call. Every edit is checked before any file
  is written, and files are replaced atomically.
  - **Read first:** an existing file must have been read with `read_many`.
  - **Changed files:** if the file changed since that read (a formatter, a script),
    replacements still apply where `old_string` matches exactly. Overwrites and
    `replace_all` ask for a fresh read.
  - **Containment:** writes stay inside the project, even through symlinks.
  - **Permissions:** it's approved without asking only where Edit would be
    (`acceptEdits`, `bypassPermissions`); otherwise Claude Code asks.
  - **Rewind:** `/rewind` checkpoints don't track it.

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

Two opt-in flags go further:

- **`--replace-file-tools`** turns off Claude Code's Read, Edit, Write, Grep and
  Glob, and hands file access to squeeze's tools:
  - `read_many` (several files and ranges per call)
  - `search` (grouped results, file lists)
  - `outline`
  - `edit_many` (batched edits, new files, full rewrites)

  It appends short guidance for the agent and its subagents: use these tools
  instead of `cat`/`grep`/`sed` in Bash, run narrow checks, and keep replies brief.
  The trade-offs:
  - Read can no longer show images or PDFs.
  - squeeze's tools only reach the project, so the agent reads files outside it
    (dependency sources, saved tool output) with Bash.

  `edit_many` asks for permission exactly where Edit would.
- **`--trim-tools`** turns off Workflow, ScheduleWakeup, ReportFindings and
  ListAgents. Claude Code sends each of them in full with every request, about
  3.5k tokens together, and a coding session rarely needs them.
- **`--compact-window TOKENS`** makes Claude Code compact the conversation within
  that many tokens, by setting its own `CLAUDE_CODE_AUTO_COMPACT_WINDOW`.
  - **Why it helps:** every request re-reads the whole context. On a 1M-token
    model the default threshold is rarely reached, so long sessions keep paying
    for history they no longer need.
  - **When it fires:** Claude Code keeps a summary buffer, so compaction starts
    somewhat below the window. In testing, a 120k window compacted at 100k but
    not at 80k.
  - **The trade-off:** a summary keeps less detail than the full history.

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

**Compaction at pauses** (`--compact-pauses`, proxy mode, API keys only). On
an API key, Claude Code's prompt cache lasts 5 minutes. After a longer pause
(reading a diff, a meeting, a review), the next request writes the whole
conversation to the cache again, at 1.25x the input price. Late in a long
session that is 100k–300k tokens, or $0.50–1.50 per pause on Opus 5.5.

```sh
squeeze run --proxy --compact-pauses --compact-warm --keep-warm
squeeze proxy --compact-pauses --compact-warm --keep-warm    # standalone
```

When Claude Code finishes a turn and the conversation then sits idle until a
minute before its cache would expire, the proxy has the model summarize it in
the background. From the next request on, the proxy sends the summary in place
of everything between the first message and the turn's final reply. Claude
Code's first message (its instructions and project context), its system-role
messages and the final reply go out unchanged, and Claude Code's own transcript
keeps the full history. If you come back sooner, nothing changes.

- **`--compact-pauses`** turns this on. On its own, the summarizing request
  can't use Claude Code's cached conversation: Claude Code's requests carry its
  `claude-code-20250219` beta, which keeps their cache entries apart. So it
  pays the full input price for the conversation, about $0.15–0.30 a summary.
- **`--compact-warm`** forwards Claude Code's requests without that beta.
  squeeze only removes it; it never adds it to a request of its own. The
  summarizing request is then Claude Code's last request plus the instruction,
  and reads the conversation from the cache: about $0.09–0.13 a summary on
  Opus 5.5, mostly for the summary's own output. What the beta changes on
  Anthropic's side isn't documented. Claude Code worked normally without it in
  our runs: all 748 requests of five benchmark sessions succeeded.
- **`--keep-warm`** (with `--compact-warm`) covers the waits a summary can't:
  a tool call that runs longer than the cache lasts (a long test suite), and a
  pause after a turn too short to summarize. A minute before the cache would
  expire, the proxy sends the last request again for one output token. That
  reads the conversation from the cache, about a 25th of the cost of writing
  it again ($0.01–0.06 on Opus 5.5), at most 6 times per wait.

A summary keeps every user request in order (the latest one quoted in full,
constraints word for word), the state of the work, and the files read or
changed. For code it keeps exact names, signatures, user-visible strings and
what calls what. It also keeps the commands run with their latest results,
decisions and next steps. File contents and raw tool output are left out; the
agent reads them again when it needs them. A summary still holds less than the
full history: see [Results](#proxy-compaction-on-an-api-key) for what that did
to quality.

- **What's summarized:** a conversation of at least about 20k tokens. Not a
  subagent's: its reply ends it.
- **Credentials:** the proxy's own requests use the API key of the request
  that ended the turn, held in memory only until then. Subscription
  credentials never get this far; proxy mode refuses them.
- **Restarts:** summaries are saved per conversation under
  `~/.squeeze/threads/`, so a restarted proxy keeps sending them. If Claude
  Code rewinds or compacts the conversation itself, the summary no longer
  fits, and the proxy stops sending it.
- **Cost records:** each request the proxy makes is a line in
  `~/.squeeze/stats.jsonl` (`compaction` or `keep_warm`), with its usage.
- **Ending a session:** with `squeeze run`, the proxy stops with Claude Code.
  A standalone proxy can't tell a finished session from a pause, so it
  summarizes the last turn too (about $0.10).
- **A 1-hour cache** (`ENABLE_PROMPT_CACHING_1H=1`): the proxy waits for a
  minute before that expiry instead.
- **Doesn't combine with** `--context-edits cold`.

**Server-side context editing** (experimental, `--context-edits cold`, proxy
mode only). The proxy asks the API to clear old tool results and thinking,
through the `context_management` field Claude Code already sends. It never
edits the messages themselves. What's cleared moves forward only when the
prompt cache has already expired, and stays fixed in between, so the other
requests keep reading the history from cache.

- **What's cleared:** results of Read, Grep, Glob, Edit, Write, NotebookEdit,
  Bash and squeeze's own tools. The newest ones stay: those since the start of
  the previous human turn, at least `--keep-tool-uses` (8) and at most
  `--keep-tool-uses-max` (24). Thinking from before the latest prompt is
  cleared too, always keeping the most recent thinking turn
  (`--clear-thinking false` keeps it all). `--clear-tool-inputs` also clears
  those calls' inputs, including the text of the agent's older Edit and Write
  calls.
- **When it helps:** sessions that pause longer than the cache lifetime, such
  as an API key's 5-minute cache. With a 1-hour cache and shorter pauses,
  nothing moves.
- **Unchanged files and repeats:** when Claude Code answers a Read with "File
  unchanged since last read" and the earlier Read is old enough to be cleared,
  the proxy sends the earlier content in its place. That costs tokens even
  when nothing ends up cleared (with a 1-hour cache, say), and is counted as a
  cost in the savings estimate. Likewise squeeze replaces a repeated output
  with a reference only to a copy recent enough not to be cleared.
- **Safeguards:** a conversation stops moving forward if the API reports
  clearing something other than what was asked, or if a request well within
  the cache lifetime didn't read the cache (a response cut short, or the first
  one after a model switch, doesn't count). After the second cache miss it
  goes back to Claude Code's own setting at the next expired-cache moment, and
  right away if a request is refused. Each line in `~/.squeeze/stats.jsonl`
  records what was asked, what the API cleared and any `breakers` tripped
  (`ctx`).
- **Status:** off by default. A two-ticket run on an API key confirmed that
  the API clears what's asked and the cache holds between advances.
  Compaction at pauses (above) saves far more on the same workload. It can't
  be combined with `--prune-cold-cache` or `--compact-pauses`.
- **Estimating it offline:** `squeeze replay --cache-sim <transcripts>`
  rebuilds every request of Claude Code 2.1.317+ transcripts, runs them through
  the proxy's own context-editing logic (unchanged-file refills included;
  compression and dedup left out), and prices them with a prompt-cache model
  against the same session sent unchanged. `--regime 5m-fermat` (default)
  is a 5-minute cache with SessionBench's 330 s pauses before each new ticket
  and every third human turn; `1h` keeps the recorded timing. It takes the
  proxy's `--keep-tool-uses`, `--keep-tool-uses-max`, `--clear-tool-inputs` and
  `--clear-thinking`, plus `--read-price`, `--placeholder` (tokens left per
  cleared item, 60) and `--reread` (share of cleared results the agent reads
  back). On the four grafana_ts sessions below, at 5m-fermat with a read price
  of 0.1 / 0.05: 0.66 / 0.67 of the unchanged cost with the defaults, 0.54 /
  0.56 keeping 8 tool uses with inputs cleared, and 1.00 under a 1-hour cache.

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
  Here you can choose among:
  - `vanilla`
  - `squeeze`: hook + MCP tools, alongside Claude Code's own tools
  - `squeeze-full`: squeeze's tools *replace* Claude Code's file tools, with
    working guidance and trimmed tools (`squeeze run --replace-file-tools --trim-tools`)
  - `squeeze-scout`: `squeeze-full` plus an Explore scout on the same model
  - `squeeze-compact`: `squeeze-full` plus earlier compaction (`--compact-window 130000`)
  - `squeeze-hook`: compression only
  - `squeeze-mcp`: tools only
  - `squeeze-proxy`: proxy compression (API key only)
  - `squeeze-proxy-cm`: `squeeze-proxy` plus `--context-edits cold` (API key only)
  - `squeeze-proxy-cm-inputs`: `squeeze-proxy-cm` plus `--clear-tool-inputs` (API key only)
  - `squeeze-proxy-compact`: `squeeze-proxy` plus `--compact-pauses` (API key only)
  - `squeeze-proxy-compact-warm`: plus `--compact-warm` (API key only)
  - `squeeze-proxy-compact-keepwarm`: plus `--keep-warm` (API key only)
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
  `--judge-batch`. With `--auth api-key` and a plain API key, each item is one
  direct API request (adaptive thinking, medium effort, as `claude -p` sends).
  The ticket and diff, which every item shares, are cached once and read by the
  others. Re-judging one scapy session this way gave the same verdict on all 77
  items, for $0.69 instead of $5.08. The grafana_go session's 51 items cost
  $1.11. Other auth keeps `claude -p`.

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

**Pauses and cache lifetime.** `pause_between_tickets_secs` in `[defaults]`
idles before each ticket; `pause_every_n_turns = 3` also idles that long before
every third user message, as SessionBench did for Fermat (330 s each).
`--cache-ttl` sets the agent's prompt-cache lifetime: `client` (the default) is
Claude Code's own, one hour on a subscription and five minutes on an API key;
`1h` sets `ENABLE_PROMPT_CACHING_1H` for every arm. Compare arms only at the
same lifetime.

Every arm, vanilla included, runs the agent with Claude Code's
`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`. With background commands on,
`claude -p` can return while a command is still running. Its result then misses
the work that follows (in our one-ticket probes, 47–80% of the cache reads), and
the simulated user reviews a reply written mid-task. Each invocation's raw
result is saved next to the ticket (`agent-result-N.json`) for auditing.

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
- **Grading only:** `--regrade` grades what's missing, from the saved patches.
  - **What it grades:** tests for tickets without a test score, and the judge
    for tickets without a rubric score.
  - **The judge needs a decision:** pass `--auth` to run it, or
    `--judge-model none` to score the tests alone. Without one of these, it
    won't count those tickets as rubric 1.0.
  - **Progress is kept:** results go to `<out>.regraded.jsonl` after every session.
    A rerun continues from there, so finished grading and paid judge calls are
    kept. A ticket that can't be graded is marked excluded, with the reason.
- **Faster sessions:** grading takes 10–20 minutes per ticket. `--defer-grading`
  skips it during sessions, and the summary shows those sessions' rewards as
  pending until `--regrade`.
  - **Agent behaviour:** unchanged.
  - **Cost:** the gap between tickets shrinks. That doesn't matter while the
    prompt cache outlives a grading gap: Claude Code's subscription sessions use
    a one-hour cache, and our grafana runs read the full context from cache after
    14–26 minute gaps. With a 5-minute cache, compare only sessions run the same
    way. The summary flags mixed runs.

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
| `cargo test` | free | compression rules, cache stability, credential refusal, stream handling, hook output shape, MCP tools, benchmark statistics, and `--context-edits` on a rebuilt session paced like Fermat's runs, sent through the proxy to an emulated API that applies the edits, caches the edited prompt and checks thinking blocks; its cache numbers must match `--cache-sim`'s |
| `cargo test --test e2e_claude -- --include-ignored` | free | the real `claude` CLI against a scripted local mock API: hook mode, proxy mode with and without context editing (against the emulated API, with 5-minute and 1-hour cache markers), and `squeeze bench` across every arm. Uses a dummy key and an isolated config dir, so nothing reaches Anthropic |
| `squeeze replay` | free, offline | applies the compression rules to your past transcripts (`~/.claude/projects`) and prices them against the recorded usage |
| `squeeze replay --cache-sim` | free, offline | replays past requests through `--context-edits cold` and a prompt-cache model, against the same sessions sent unchanged |
| `squeeze replay --cache-sim --compact-summary TOKENS` | free, offline | the same, modelling compaction at pauses with a summary of that size instead |

## Results so far

### Proxy compaction on an API key

**SWE-Atlas grafana_ts** (SessionBench's TypeScript bundle: 5 refactoring
tickets in one resumed session). Opus 5.5 is the agent, the simulated user and
the judge. Claude Code 2.1.326 runs in the task containers, with the agent on
an API key (5-minute cache). As in SessionBench, the session pauses 330 s
before each ticket and before every third user message. The code improved
between sessions, as noted.

| Session | Build | squeeze | Same session without squeeze | Ratio | Reward |
|---|---|---|---|---|---|
| 1 | `--compact-warm` | $5.39 | $10.44 | 0.516 | 0.908 |
| 2 | same | $6.05 | $12.55 | 0.482 | 0.923 |
| 3 | plus `--keep-warm` | $6.62 | $13.59 | 0.487 | 0.852 |
| 4 | keep-warm fixes | $6.14 | $15.71 | 0.391 | 0.879 |
| 5 | summaries keep exact names and strings | $6.85 | $14.35 | 0.477 | 0.943 |
| **Mean** | | | | **0.471** | **0.901** |

Three earlier sessions, without `--compact-warm`, measured 0.54–0.55.
Vanilla's reward for comparison: 0.905 and 0.872 (mean 0.889), from the two
vanilla sessions below (on a subscription, Claude Code 2.1.317).

- **How the ratio is measured.**
  - *squeeze:* what the session cost, meaning Claude Code's requests as billed
    plus the proxy's own summaries and keep-warm requests.
  - *Same session without squeeze:* every request is rebuilt from Claude Code's
    transcript with its full history and counted with the API's token-counting
    endpoint. It is priced at Opus 5.5's rates with a 5-minute cache, with the
    same output. A request within 300 s of the one before reads that one's
    prompt from the cache; after a longer gap it writes the whole prompt again.
  - *Why this way:* holding the session fixed removes run-to-run noise, about
    30% between the two vanilla sessions here. But it isn't a separate vanilla
    run.
- **Where the saving comes from.** Mostly from the pauses. In session 5, the
  six requests after a pause would have cost $4.99 without squeeze; they cost
  $0.53 with it. The rest comes from every later request reading a shorter
  history.
- **Why it varies.** The ratio depends on how many pauses a session has, and
  how long its history is by then. It ranged from 0.39 (session 4, which
  paused more) to 0.52.
- **Quality.** Within noise of vanilla: a mean of 0.901 against 0.889, with
  only two vanilla sessions to compare. One effect was traced and fixed.
  Ticket 5 rewrites a component from ticket 4 after its repository is reset,
  and its hidden tests check exact labels. Summaries that only described the
  code lost those labels, and passed 21–23 of the 29 tests. Since session 5,
  summaries keep exact names and strings: 25 of 29 (vanilla 23–24).
- **Comparing with Fermat.** Fermat reports 0.49x on this bundle, from five
  sessions per arm. squeeze's mean is 0.47x by the measure above. The methods
  differ, but the level is similar.

**SWE-Atlas scapy_py_scoped** (SessionBench's Python bundle), one session with
the same setup and the session-5 build (`--compact-pauses --compact-warm
--keep-warm`):

| Session | squeeze | Same session without squeeze | Ratio | Reward |
|---|---|---|---|---|
| 1 | $8.59 | $18.48 | 0.465 | 0.901 |

- **Cost.** The $8.59 is $7.88 of Claude Code's requests, plus $0.71 for five
  summaries and one keep-warm request. Every summary was used. Fermat reports
  0.50x on this bundle.
- **Quality.** Rewards for the first four tickets were 0.769, 1.000, 0.833 and
  1.000. The fifth ticket is left out because its tests check nothing (its
  rubric scored 0.957). SessionBench reports 0.824 for vanilla and 0.924 for Fermat on
  this bundle, from a different setup, and there is no vanilla session here.
  Neither loss traces to a summary:
  - Ticket 1 ended before the first summary was written.
  - Ticket 3 failed three hidden tests on exactly the two behaviours the
    simulated user asked the agent to change (raise `TypeError` for different
    key sets; no equality shortcut in `__contains__`). The agent applied that
    request correctly after a mid-ticket summary.
- **Subagents.** One subagent ran. As designed, squeeze neither summarized it
  nor kept it warm.

**SWE-Atlas grafana_go** (SessionBench's Go bundle, where Fermat does worst),
one session with the same setup and build:

| Session | squeeze | Same session without squeeze | Ratio | Reward |
|---|---|---|---|---|
| 1 | $4.47 | $8.04 | 0.556 | 1.000 |

- **Cost.** The $4.47 is $4.15 of Claude Code's requests, plus $0.33 for three
  summaries and three keep-warm requests. Every summary was used. Fermat
  reports 0.66x on this bundle.
- **Why the ratio is higher here.** This session had fewer cold moments.
  Without squeeze, five requests after the first would have rewritten the whole
  history: $2.92, or 36% of the $8.04. squeeze paid $0.47 for them. Two
  came after ticket pauses that squeeze summarized. Two came after `go test`
  runs of 4–6 minutes, which keep-warm requests covered. The rest of the cost
  is warm requests, which squeeze only shortens. The last ticket ran 46
  requests without a pause.
- **Quality.** All five tickets scored 1.000 on both hidden tests and rubric
  (51 rubric items). SessionBench reports 0.824 for vanilla and 0.889 for
  Fermat on this bundle, from a different setup.
- **Across the three bundles**, squeeze cost $44.11 against $93.16 for the same
  sessions without it (0.474). By bundle, squeeze measures 0.471, 0.465 and
  0.556. Fermat reports 0.49x, 0.50x and 0.66x, and 0.53x overall.

### Hook mode and MCP tools on a subscription

**SWE-Atlas grafana_ts** (SessionBench's TypeScript bundle: 5 refactoring
tickets in one resumed session, Opus 5.5 as agent, simulated user and judge,
Claude Code 2.1.317). Two sessions per arm; cost is Claude Code's own figure
at its list prices.

| | vanilla | squeeze (hook + MCP tools beside Claude Code's) |
|---|---|---|
| Agent cost, pair A | $10.32 | $9.18 (0.89x) |
| Agent cost, pair B | $7.88 | $8.25 (1.05x) |
| **Mean** | **$9.10** | **$8.72 (0.96x)** |
| Reward, pair A (fractional tests x rubric) | 0.905 | 0.878 |
| Reward, pair B | 0.872 | 0.921 |

What this shows:

- **No measurable saving from hook mode yet.** Two vanilla sessions of the same setup differed
  by 31%, more than any difference between the arms. A claim needs 3–5 sessions
  per arm.
- **squeeze's own effect is small.** The hook compressed a few outputs, worth
  about 1–4% of cache reads. `squeeze replay` on vanilla's transcript estimates
  1.6%.
- **The MCP tools were rarely used.** Claude Code defers MCP tools behind tool
  search, so the model only saw their names. squeeze now opts out of deferral.
  Even with the tools visible, Opus kept reading and searching through Bash
  (10 `read_many` calls against 135 Bash calls). `squeeze-full` replaces Claude
  Code's file tools outright.
- **A correction.** Earlier versions of this README showed costs about 4x too
  high. Claude Code's `total_cost_usd` is a running total for the session (it
  survives `--resume`), and the benchmark summed it over every call. The
  benchmark now keeps the latest total and reads the session's last cost
  snapshot from its transcript.
- **Comparing with Fermat.** [Fermat's Last Token](https://www.quotientlabs.com/#how)
  reports $60.62 → $29.79 (0.49x) on the same bundle (5 sessions per arm).
  - *How:* a proxy compresses conversation history, which is where most of the
    cost sits.
  - *Their figures:* their harness also sums `total_cost_usd` per turn, so its
    dollars are likely running totals too. Their token counts (cache reads at
    0.43x of vanilla) point to a real effect.

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
  doesn't compress it either: it never rewrites a tool result marked as an error.
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
compressed outputs (`blobs/`), per-request stats (`stats.jsonl`), the proxy's
per-conversation state including compaction summaries (`threads/`), and logs.
`squeeze gc --days 30` deletes old originals.

## License

Apache-2.0. See [LICENSE](LICENSE).
