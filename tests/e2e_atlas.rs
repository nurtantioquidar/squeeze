//! `squeeze bench-atlas` end to end against a scripted mock API (free; nothing
//! reaches Anthropic): a real SWE-Atlas task image, the official Claude Code
//! Linux binary running inside it, squeeze's hook and MCP server inside it,
//! SWE-Atlas's own test.sh for grading, and a mock simulated user and judge.
//!
//! Needs Apple's `container` runtime, the SWE-Atlas checkout and the cached
//! binaries (see README, "SWE-Atlas runs"). Run with:
//!     SQUEEZE_E2E_ATLAS=1 cargo test --test e2e_atlas -- --ignored --nocapture

mod common;

use common::{Recorded, Reply, sse_text, sse_tool_use, tool_result_text};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

const FAKE_KEY: &str = "sk-ant-api03-mock-key-not-real";
const TASK: &str = "task-696719205599a51110d4b476"; // scapy OPC-DA (0.4 GB image)
// Exits 0: Claude Code sends non-zero exits to PostToolUseFailure, whose output hooks can't replace.
const LOG_CMD: &str = r#"for i in $(seq 1 400); do echo "test_case_$i (tests.Suite.test_case_$i) ... ok"; done; echo "Ran 400 tests"; echo OK"#;
// A real change, so grading runs SWE-Atlas's full validator.
const EDIT_CMD: &str = r#"printf '\n# squeeze e2e marker\n' >> scapy/contrib/opc_da.py && git status --short"#;

fn script(r: &Recorded) -> Reply {
    match r.path.as_str() {
        "/v1/messages" => {}
        "/v1/messages/count_tokens" => return Reply::Json(200, json!({ "input_tokens": 1000 })),
        "/v1/models" => {
            return Reply::Json(
                200,
                json!({ "data": [{ "type": "model", "id": "claude-mock", "display_name": "Mock" }], "has_more": false }),
            );
        }
        _ => {
            return Reply::Json(
                404,
                json!({ "type": "error", "error": { "type": "not_found_error", "message": "mock" } }),
            );
        }
    }
    let body = r.json();
    let system = body["system"].to_string();
    if system.contains("assigning a ticket") {
        return Reply::Sse(sse_text(
            90,
            "Please fix the OPC-DA contrib module as the ticket describes, and run its tests.",
        ));
    }
    if system.contains("who assigned a ticket") {
        return Reply::Sse(sse_text(91, &json!({ "approve": true, "message": "thanks" }).to_string()));
    }
    if system.contains("expert evaluator") {
        return Reply::Sse(sse_text(
            92,
            &json!({ "ratings": [{ "status": "YES", "justification": "mock" }] }).to_string(),
        ));
    }
    let is_agent = body["tools"].as_array().is_some_and(|t| t.iter().any(|x| x["name"] == "Bash"));
    if !is_agent {
        return Reply::Sse(sse_text(99, "ok"));
    }
    let turns = body["messages"].as_array().map_or(0, |m| m.iter().filter(|x| x["role"] == "assistant").count());
    match turns {
        0 => Reply::Sse(sse_tool_use(
            0,
            "toolu_atlas_bash",
            "Bash",
            json!({ "command": LOG_CMD, "description": "Run tests" }),
        )),
        1 => Reply::Sse(sse_tool_use(
            1,
            "toolu_atlas_edit",
            "Bash",
            json!({ "command": EDIT_CMD, "description": "Edit" }),
        )),
        _ => Reply::Sse(sse_text(turns, "done")),
    }
}

#[tokio::test]
#[ignore = "needs Apple container, the SWE-Atlas checkout and cached binaries; set SQUEEZE_E2E_ATLAS=1"]
async fn bench_atlas_runs_claude_code_inside_a_real_task_container() {
    if std::env::var("SQUEEZE_E2E_ATLAS").as_deref() != Ok("1") {
        eprintln!("skipped: set SQUEEZE_E2E_ATLAS=1");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let cache = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".squeeze/cache");
    let mock = common::start_on("0.0.0.0:0", Arc::new(script)).await;
    let host_url = format!("http://127.0.0.1:{}", mock.port());
    // The host as seen from Apple container's default network.
    let container_url = format!("http://192.168.64.1:{}", mock.port());
    let bundles = tmp.path().join("atlas.toml");
    std::fs::write(
        &bundles,
        format!(
            r#"[defaults]
model = "claude-mock"
usersim_model = "claude-mock"
judge_model = "claude-mock"
max_pushbacks = 1
allowed_tools = ["Bash"]
timeout_secs = 900

[[atlas_bundle]]
name = "scapy_one"
workdir = "/app"
grade_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
tasks = ["{TASK}"]
"#
        ),
    )
    .unwrap();
    let results = tmp.path().join("results.jsonl");
    let out = tokio::process::Command::new(env!("CARGO_BIN_EXE_squeeze"))
        .arg("bench-atlas")
        .arg(&bundles)
        .args([
            "--auth",
            "api-key",
            "--repeats",
            "1",
            "--arms",
            "vanilla,squeeze",
            "--yes",
            "--cpus",
            "6",
            "--memory",
            "8g",
        ])
        .args(["--dns", "1.1.1.1,8.8.8.8", "--agent-env", &format!("ANTHROPIC_BASE_URL={container_url}")])
        .arg("--out")
        .arg(&results)
        // SQUEEZE_HOME is a temp dir here, so point at the real cache explicitly.
        .arg("--atlas-root")
        .arg(cache.join("SWE-Atlas"))
        .arg("--claude-linux")
        .arg(cache.join("claude-code/2.1.317/linux-x64/claude"))
        .arg("--squeeze-linux")
        .arg(cache.join("bin/squeeze-linux-x64"))
        .env("ANTHROPIC_API_KEY", FAKE_KEY)
        .env("ANTHROPIC_BASE_URL", &host_url)
        .env("SQUEEZE_HOME", tmp.path().join("squeeze-home"))
        .env("CLAUDE_CONFIG_DIR", tmp.path().join("host-claude-config"))
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .env_remove("CLAUDE_CODE_OAUTH_TOKEN")
        .output();
    let out = tokio::time::timeout(Duration::from_secs(1800), out).await.expect("timed out").unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    eprintln!("{stderr}\n{stdout}");
    assert!(out.status.success(), "bench-atlas failed");

    let rows: Vec<Value> =
        std::fs::read_to_string(&results).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(rows.len(), 2);
    for r in &rows {
        assert!(r["error"].is_null(), "{r}");
        let t = &r["tickets"][0];
        assert!(t["excluded"].is_null(), "tests must not be vacuous with grade_path: {t}");
        assert!(t["tests_frac"].as_f64().is_some(), "{t}");
        let rubric = t["rubric"].as_f64().expect("rubric scored");
        assert!(rubric > 0.0 && rubric < 1.0, "negative items rated YES must fail: {rubric}");
        assert!(r["helpers"]["invocations"].as_u64().unwrap() >= 3, "opening + review + judge");
    }
    let by_arm = |arm: &str| rows.iter().find(|r| r["arm"] == arm).unwrap();
    assert_eq!(by_arm("vanilla")["hook_compressions"], 0);
    assert!(by_arm("squeeze")["hook_compressions"].as_u64().unwrap() >= 1, "hook ran inside the container");

    // The agent requests came from the official Claude Code binary inside the container.
    let reqs = mock.requests();
    let agent: Vec<&Recorded> = reqs
        .iter()
        .filter(|r| r.path == "/v1/messages")
        .filter(|r| r.json()["tools"].as_array().is_some_and(|t| t.iter().any(|x| x["name"] == "Bash")))
        .collect();
    assert!(!agent.is_empty());
    for r in &agent {
        assert_eq!(r.headers["x-api-key"], FAKE_KEY);
        assert!(r.headers["user-agent"].to_str().unwrap().starts_with("claude-cli/"));
    }
    // Opening message came from the simulated user, not the raw ticket.
    assert!(agent.iter().any(|r| r.json().to_string().contains("Please fix the OPC-DA contrib module")));
    // squeeze arm: the Bash output reached the API compressed; vanilla: verbatim.
    let outputs: Vec<String> = agent.iter().filter_map(|r| tool_result_text(&r.json(), "toolu_atlas_bash")).collect();
    assert!(outputs.iter().any(|t| t.contains("passing-test lines omitted")), "squeeze arm compressed");
    assert!(outputs.iter().any(|t| t.contains("test_case_200 ")), "vanilla arm verbatim");
}

/// The squeeze-proxy arm in a task container: the agent authenticates with an
/// API key file (never the subscription token) and reaches the API through
/// squeeze's proxy on the container bridge, which compresses its tool output.
#[tokio::test]
#[ignore = "needs Apple container, the SWE-Atlas checkout and cached binaries; set SQUEEZE_E2E_ATLAS=1"]
async fn bench_atlas_proxy_arm_goes_through_the_proxy_on_an_api_key() {
    if std::env::var("SQUEEZE_E2E_ATLAS").as_deref() != Ok("1") {
        eprintln!("skipped: set SQUEEZE_E2E_ATLAS=1");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let cache = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".squeeze/cache");
    let mock = common::start_on("127.0.0.1:0", Arc::new(script)).await;
    let key_file = tmp.path().join("api.env");
    std::fs::write(&key_file, format!("ANTHROPIC_API_KEY={FAKE_KEY}\n")).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&key_file, std::fs::Permissions::from_mode(0o600)).unwrap();
    let bundles = tmp.path().join("atlas.toml");
    std::fs::write(
        &bundles,
        format!(
            r#"[defaults]
model = "claude-mock"
usersim_model = "claude-mock"
judge_model = "claude-mock"
max_pushbacks = 0
allowed_tools = ["Bash"]
timeout_secs = 900

[[atlas_bundle]]
name = "scapy_one"
workdir = "/app"
grade_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
tasks = ["{TASK}"]
"#
        ),
    )
    .unwrap();
    let results = tmp.path().join("results.jsonl");
    let out = tokio::process::Command::new(env!("CARGO_BIN_EXE_squeeze"))
        .arg("bench-atlas")
        .arg(&bundles)
        .args(["--auth", "api-key", "--repeats", "1", "--arms", "squeeze-proxy", "--yes", "--defer-grading"])
        .args(["--cpus", "6", "--memory", "8g", "--dns", "1.1.1.1,8.8.8.8"])
        .arg("--api-key-env-file")
        .arg(&key_file)
        .args(["--proxy-upstream", &mock.url()])
        .arg("--out")
        .arg(&results)
        .arg("--atlas-root")
        .arg(cache.join("SWE-Atlas"))
        .arg("--claude-linux")
        .arg(cache.join("claude-code/2.1.317/linux-x64/claude"))
        .arg("--squeeze-linux")
        .arg(cache.join("bin/squeeze-linux-x64.v4"))
        // The host-side simulated user and judge talk to the mock directly.
        .env("ANTHROPIC_API_KEY", FAKE_KEY)
        .env("ANTHROPIC_BASE_URL", mock.url())
        .env("SQUEEZE_HOME", tmp.path().join("squeeze-home"))
        .env("CLAUDE_CONFIG_DIR", tmp.path().join("host-claude-config"))
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .env_remove("CLAUDE_CODE_OAUTH_TOKEN")
        .output();
    let out = tokio::time::timeout(Duration::from_secs(1800), out).await.expect("timed out").unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    eprintln!("{stderr}");
    assert!(out.status.success(), "bench-atlas failed");
    assert!(stderr.contains("squeeze proxy for the squeeze-proxy arm at http://192.168.64.1:"), "{stderr}");
    let rows: Vec<Value> =
        std::fs::read_to_string(&results).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(rows.len(), 1);
    assert!(rows[0]["error"].is_null(), "{}", rows[0]);

    let reqs = mock.requests();
    let agent: Vec<&Recorded> = reqs
        .iter()
        .filter(|r| r.path == "/v1/messages")
        .filter(|r| r.json()["tools"].as_array().is_some_and(|t| t.iter().any(|x| x["name"] == "Bash")))
        .collect();
    assert!(!agent.is_empty(), "the agent's requests reached the API through the proxy");
    for r in &agent {
        assert_eq!(r.headers["x-api-key"], FAKE_KEY, "the agent used the API key file");
        assert!(r.headers.get("authorization").is_none(), "no subscription bearer");
    }
    let outputs: Vec<String> = agent.iter().filter_map(|r| tool_result_text(&r.json(), "toolu_atlas_bash")).collect();
    assert!(outputs.iter().any(|t| t.contains("passing-test lines omitted")), "the proxy compressed the output");
}
