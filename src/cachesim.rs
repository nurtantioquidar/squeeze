//! `squeeze replay --cache-sim`: what `--context-edits cold` would have cost on
//! recorded Claude Code sessions, priced against the same sessions sent as is.
//!
//! Claude Code (2.1.317 and later) logs every request it sends: `api-request`
//! lists the request's messages as sha256 hashes, delta-encoded against an
//! earlier request, and the messages are the transcript's own entries or
//! `api-request-blob`s. So each request is rebuilt exactly, run through the
//! proxy's real census and advance logic, cleared the way the server would
//! clear it, and billed by a prompt-cache model on a chosen timeline. Of the
//! proxy's own rewrites it applies the one context editing brings with it,
//! unchanged-file Reads given their content back; compression and dedup,
//! which squeeze does with or without it, are left out.
//!
//! Token counts come from a least-squares fit of each request's recorded
//! prompt growth against the bytes it appended, per kind of content, over all
//! the sessions together. Runs locally and sends nothing anywhere.

use crate::ctxmgmt::{self, CLEAR_THINKING, CLEAR_TOOL_USES, CtxConfig};
use crate::proxy::COLD_MARGIN_SECS;
use crate::replay::DryRun;
use crate::sse::{AppliedEdits, Usage};
use crate::store::CtxState;
use crate::transform::{self, ChangeKind, TransformConfig};
use crate::util::read_price;
use anyhow::{Result, bail};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// SessionBench, as Fermat ran it, pauses this long before each new ticket
/// and before every third human turn.
const FERMAT_PAUSE_SECS: f64 = 330.0;
/// A human prompt at least this long opens a new ticket: in the grafana runs
/// ticket prompts are 2,468-4,470 characters and follow-ups 733-1,935.
const TICKET_MIN_CHARS: usize = 2200;
/// Output tokens cost this many base input tokens on current Claude models.
const OUTPUT_PRICE: f64 = 5.0;

/// The cache lifetime, write price and timeline a simulation runs under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Regime {
    /// A 5-minute cache (API keys) with SessionBench's pauses, as Fermat ran it.
    #[serde(rename = "5m-fermat")]
    FiveMinFermat,
    /// A 1-hour cache (subscriptions, or `ENABLE_PROMPT_CACHING_1H`) on the recorded timeline.
    #[serde(rename = "1h")]
    OneHour,
}

impl std::str::FromStr for Regime {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim() {
            "5m-fermat" => Ok(Self::FiveMinFermat),
            "1h" => Ok(Self::OneHour),
            other => bail!("unknown regime {other:?} (5m-fermat, 1h)"),
        }
    }
}

impl Regime {
    fn ttl(self) -> f64 {
        match self {
            Self::FiveMinFermat => 300.0,
            Self::OneHour => 3600.0,
        }
    }

    fn write_price(self) -> f64 {
        match self {
            Self::FiveMinFermat => 1.25,
            Self::OneHour => 2.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SimConfig {
    pub regime: Regime,
    /// Cache read price as a multiple of base input; `None` uses the
    /// session's model's.
    pub read_price: Option<f64>,
    /// Share of each cleared tool result the agent reads back later.
    pub reread: f64,
    /// Tokens the server leaves in place of a cleared tool result or input.
    pub placeholder: f64,
    pub ctx: CtxConfig,
    /// Model compaction at pauses instead of `--context-edits cold`.
    pub compact: Option<Compaction>,
}

/// Compaction at pauses (Stage 3 of the proxy plan), as a proxy could do it:
/// when a request leaves the conversation idle long enough to suggest a pause,
/// the proxy has the history summarized while the prompt cache is still warm
/// (the summarizing request reads it at the read price), and from the next
/// human prompt on sends the summary in its place. The current turn always
/// goes verbatim, and each later pause summarizes again, summary included.
#[derive(Clone, Copy, Debug)]
pub struct Compaction {
    /// Tokens of each summary.
    pub summary: f64,
    /// Compact only a history at least this many tokens long.
    pub min_history: f64,
    /// Thinking the summarizing request spends, as a share of the summary.
    pub thinking: f64,
    /// Share of the summarized tool results the agent reads back later.
    pub reread: f64,
    /// Compact only at pauses before a new ticket, not mid-ticket ones.
    pub tickets_only: bool,
    /// What the summarizing request finds in the prompt cache.
    pub cache: SummaryCache,
}

/// What a summarizing request reads from the prompt cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SummaryCache {
    /// In the background while the cache is warm, sharing the conversation's
    /// cache entries: everything before the instruction is read.
    Warm,
    /// In the background while the cache is warm, but under another cache key
    /// for messages: the system prompt and tools are read, the conversation is
    /// uncached input. Measured (2026-11-17): a request without Claude Code's
    /// `claude-code-20250219` beta shares only the system prompt and tools with
    /// Claude Code's own cache entries.
    SystemOnly,
    /// When the next prompt arrives, after the cache expired: all uncached input.
    Cold,
}

/// The summarizing request goes out this long before the cache would expire.
const COMPACT_LEAD_SECS: f64 = 60.0;
/// Tokens of the instruction the summarizing request appends.
const COMPACT_INSTRUCTION_TOKENS: f64 = 300.0;

/// The hash Claude Code logs a message under: sha256 of compact JSON with
/// sorted keys and non-ASCII text left as is.
pub fn message_hash(m: &Value) -> String {
    fn canonical(v: &Value, out: &mut Vec<u8>) {
        match v {
            Value::Object(map) => {
                let mut entries: Vec<_> = map.iter().collect();
                entries.sort_by(|a, b| a.0.cmp(b.0));
                out.push(b'{');
                for (i, (k, v)) in entries.into_iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    serde_json::to_writer(&mut *out, k).expect("writing to memory");
                    out.push(b':');
                    canonical(v, out);
                }
                out.push(b'}');
            }
            Value::Array(items) => {
                out.push(b'[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    canonical(v, out);
                }
                out.push(b']');
            }
            _ => serde_json::to_writer(&mut *out, v).expect("writing to memory"),
        }
    }
    let mut bytes = Vec::new();
    canonical(m, &mut bytes);
    format!("{:x}", Sha256::digest(&bytes))
}

/// Unix seconds of a transcript timestamp (`2026-11-03T23:31:51.092Z`).
fn unix_secs(ts: &str) -> Option<f64> {
    let num = |r: std::ops::Range<usize>| ts.get(r)?.parse::<i64>().ok();
    let (y, mo, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (h, mi) = (num(11..13)?, num(14..16)?);
    let rest = ts.get(17..)?;
    let end = rest.find(['Z', '+', '-']).unwrap_or(rest.len());
    let s: f64 = rest[..end].parse().ok()?;
    let offset = match rest.get(end..end + 1) {
        Some(sign @ ("+" | "-")) => {
            let mins = rest.get(end + 1..end + 3)?.parse::<i64>().ok()? * 60
                + rest.get(end + 4..end + 6)?.parse::<i64>().ok()?;
            if sign == "+" { mins * 60 } else { -mins * 60 }
        }
        _ => 0,
    };
    // Days from the civil date (Howard Hinnant's algorithm).
    let y = if mo <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (mo + if mo > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let days = era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468;
    Some((days * 86_400 + h * 3600 + mi * 60 - offset) as f64 + s)
}

/// Bytes of a value as compact JSON.
fn json_len(v: &Value) -> usize {
    serde_json::to_string(v).map_or(0, |s| s.len())
}

/// What a kind of content costs in tokens: fitted per byte, except thinking
/// and tool-list changes, whose tokens are known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Thinking,
    Text,
    Input,
    Result,
    /// Human prompts, reminders and any other block.
    Other,
    /// Tokens added by a mid-session change to the tool list.
    ToolDelta,
}

#[derive(Debug)]
struct Block {
    kind: Kind,
    bytes: usize,
    /// Known tokens, for `Thinking` and `ToolDelta`.
    tokens: f64,
    /// tool_use id, for `Input` and `Result`.
    id: Option<String>,
}

impl Block {
    fn new(kind: Kind, bytes: usize) -> Self {
        Self { kind, bytes, tokens: 0.0, id: None }
    }
}

#[derive(Debug)]
struct Msg {
    /// The message as sent; `None` for a tool-list change.
    json: Option<Value>,
    blocks: Vec<Block>,
    /// Characters of human prompt text in it, if it holds a prompt.
    prompt_chars: Option<usize>,
    /// The request whose response this assistant message is.
    src_call: Option<usize>,
}

#[derive(Debug)]
struct Call {
    t: f64,
    /// Recorded prompt tokens (uncached input, cache reads and writes).
    total: f64,
    input: f64,
    output: f64,
    thinking: f64,
    shape: String,
    /// Indices into `Session::msgs`, in request order.
    msgs: Vec<usize>,
    human: bool,
    ticket: bool,
    betas: String,
    /// The request's body parameters (model, thinking, `context_management`, ...).
    params: Value,
    /// Its `cache_control` markers, each a `path` into the body and a `value`.
    markers: Vec<Value>,
}

/// A transcript's requests, rebuilt. Each transcript is one thread.
#[derive(Debug)]
struct Session {
    path: PathBuf,
    model: String,
    calls: Vec<Call>,
    msgs: Vec<Msg>,
    /// `system` and `tools` of each request shape.
    shapes: HashMap<String, Value>,
    unresolved: usize,
    /// Tokens of the system prompt and tools, which always hit the cache.
    sys_tokens: f64,
}

enum Source {
    Blob,
    Assistant(String),
    User,
}

#[derive(Default)]
struct Response {
    request_ref: Option<String>,
    blocks: Vec<Value>,
    usage: Value,
}

/// Rebuild every request of a transcript; `None` when it logs no requests.
fn load(path: &Path) -> Result<Option<Session>> {
    let mut shapes: HashMap<String, Value> = HashMap::new();
    let mut blobs: Vec<(String, Value)> = Vec::new();
    let mut reqs: HashMap<String, Value> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut resps: Vec<(String, Response)> = Vec::new();
    let mut resp_at: HashMap<String, usize> = HashMap::new();
    let mut entries: Vec<Value> = Vec::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let Ok(e) = serde_json::from_str::<Value>(&line?) else { continue };
        let s = |k: &str| e[k].as_str().map(str::to_string);
        match e["type"].as_str() {
            Some("api-request-shape") => {
                if let Some(h) = s("shapeHash") {
                    shapes.insert(h, e["shape"].clone());
                }
            }
            Some("api-request-blob") => {
                if let Some(h) = s("hash") {
                    blobs.push((h, e["message"].clone()));
                }
            }
            Some("api-request") => {
                if let Some(id) = s("id") {
                    order.push(id.clone());
                    reqs.insert(id, e);
                }
            }
            Some("assistant") => {
                let Some(mid) = e["message"]["id"].as_str().map(str::to_string) else { continue };
                let i = *resp_at.entry(mid.clone()).or_insert_with(|| {
                    resps.push((mid, Response { request_ref: s("requestRef"), ..Default::default() }));
                    resps.len() - 1
                });
                // Each content block is its own entry; the last carries the final usage.
                let r = &mut resps[i].1;
                r.usage = e["message"]["usage"].clone();
                r.blocks.extend(e["message"]["content"].as_array().into_iter().flatten().cloned());
                entries.push(e);
            }
            Some("user") => entries.push(e),
            _ => {}
        }
    }
    if order.is_empty() {
        return Ok(None);
    }

    // Every message a request could hold, by hash: blobs, responses (with and
    // without the fields Claude Code drops before sending), user entries
    // (string prompts both as is and as a text block), and parallel tool
    // results merged into one message.
    let mut by_hash: HashMap<String, (Value, Source)> = HashMap::new();
    for (h, m) in blobs {
        by_hash.entry(h).or_insert((m, Source::Blob));
    }
    for (mid, r) in &resps {
        for fields in [&[][..], &["caller"], &["caller", "citations"]] {
            let content: Vec<Value> = r
                .blocks
                .iter()
                .map(|b| {
                    let mut b = b.clone();
                    if let Some(o) = b.as_object_mut() {
                        fields.iter().for_each(|k| _ = o.shift_remove(*k));
                    }
                    b
                })
                .collect();
            let m = json!({ "role": "assistant", "content": content });
            by_hash.entry(message_hash(&m)).or_insert((m, Source::Assistant(mid.clone())));
        }
    }
    let mut human_texts: HashSet<&str> = HashSet::new();
    let mut mid_of_uuid: HashMap<&str, &str> = HashMap::new();
    for e in &entries {
        let content = &e["message"]["content"];
        if e["type"] == "assistant" {
            if let (Some(u), Some(mid)) = (e["uuid"].as_str(), e["message"]["id"].as_str()) {
                mid_of_uuid.insert(u, mid);
            }
            continue;
        }
        let mut forms = vec![json!({ "role": "user", "content": content })];
        if let Some(text) = content.as_str() {
            forms.push(json!({ "role": "user", "content": [{ "type": "text", "text": text }] }));
            if !text.starts_with("<task-notification>") {
                human_texts.insert(text);
            }
        }
        for m in forms {
            by_hash.entry(message_hash(&m)).or_insert((m, Source::User));
        }
    }
    let mut groups: Vec<(&str, Vec<&Value>)> = Vec::new();
    let mut group_at: HashMap<&str, usize> = HashMap::new();
    for e in entries.iter().filter(|e| e["type"] == "user") {
        let Some(list) = e["message"]["content"].as_array() else { continue };
        let parent = e["sourceToolAssistantUUID"].as_str().or_else(|| e["parentUuid"].as_str());
        let Some(&mid) = parent.and_then(|p| mid_of_uuid.get(p)) else { continue };
        if list.iter().any(|b| b["type"] == "tool_result") {
            let i = *group_at.entry(mid).or_insert_with(|| {
                groups.push((mid, Vec::new()));
                groups.len() - 1
            });
            groups[i].1.push(e);
        }
    }
    for (_, es) in groups.iter().filter(|(_, es)| es.len() > 1) {
        let content: Vec<Value> =
            es.iter().flat_map(|e| e["message"]["content"].as_array().into_iter().flatten().cloned()).collect();
        let m = json!({ "role": "user", "content": content });
        by_hash.entry(message_hash(&m)).or_insert((m, Source::User));
    }

    let mut mid_of_ref: HashMap<&str, &str> = HashMap::new();
    for (mid, r) in &resps {
        if let Some(rf) = r.request_ref.as_deref() {
            mid_of_ref.insert(rf, mid);
        }
    }
    // A request without a recorded response can't be billed.
    order.retain(|rid| mid_of_ref.contains_key(rid.as_str()));
    let call_of_mid: HashMap<&str, usize> =
        order.iter().enumerate().map(|(i, rid)| (mid_of_ref[rid.as_str()], i)).collect();

    let mut hashes: HashMap<String, Vec<String>> = HashMap::new();
    let mut msg_at: HashMap<String, usize> = HashMap::new();
    let mut msgs: Vec<Msg> = Vec::new();
    let mut unresolved = 0;
    let mut calls = Vec::new();
    for rid in &order {
        let r = &reqs[rid];
        let mut ids = Vec::new();
        for h in request_hashes(&reqs, &mut hashes, rid) {
            let i = *msg_at.entry(h.clone()).or_insert_with(|| {
                msgs.push(match by_hash.get(&h) {
                    Some((m, src)) => make_msg(m, src, &call_of_mid, &human_texts),
                    // Billed as an empty user message.
                    None => {
                        unresolved += 1;
                        Msg {
                            json: Some(json!({ "role": "user", "content": [] })),
                            blocks: vec![Block::new(Kind::Other, 0)],
                            prompt_chars: None,
                            src_call: None,
                        }
                    }
                });
                msgs.len() - 1
            });
            ids.push(i);
        }
        let u = &resps[resp_at[mid_of_ref[rid.as_str()]]].1.usage;
        let n = |k: &str| u[k].as_f64().unwrap_or(0.0);
        let mut params = r["params"].clone();
        let betas = params.as_object_mut().and_then(|p| p.shift_remove("betas"));
        calls.push(Call {
            t: r["timestamp"].as_str().and_then(unix_secs).unwrap_or(0.0),
            total: n("input_tokens") + n("cache_read_input_tokens") + n("cache_creation_input_tokens"),
            input: n("input_tokens"),
            output: n("output_tokens"),
            thinking: u["output_tokens_details"]["thinking_tokens"].as_f64().unwrap_or(0.0),
            shape: r["shapeHash"].as_str().unwrap_or("").to_string(),
            msgs: ids,
            human: false,
            ticket: false,
            betas: betas
                .as_ref()
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(","),
            params,
            markers: r["cacheControl"].as_array().cloned().unwrap_or_default(),
        });
    }

    // A response's thinking tokens, split over its thinking blocks by size.
    for m in &mut msgs {
        let Some(c) = m.src_call else { continue };
        let total: usize = m.blocks.iter().filter(|b| b.kind == Kind::Thinking).map(|b| b.bytes).sum();
        for b in m.blocks.iter_mut().filter(|b| b.kind == Kind::Thinking) {
            b.tokens = calls[c].thinking * b.bytes as f64 / total.max(1) as f64;
        }
    }
    // A request is a human turn when a message it appends holds a prompt.
    let mut prev = 0;
    for c in &mut calls {
        let prompts = c.msgs.get(prev..).unwrap_or_default().iter().filter_map(|&i| msgs[i].prompt_chars);
        let longest = prompts.max();
        (c.human, c.ticket) = (longest.is_some(), longest.is_some_and(|n| n >= TICKET_MIN_CHARS));
        prev = c.msgs.len();
    }
    let model = reqs[&order[0]]["params"]["model"].as_str().unwrap_or("").to_string();
    Ok(Some(Session { path: path.to_path_buf(), model, calls, msgs, shapes, unresolved, sys_tokens: 0.0 }))
}

/// A request's message hashes: its base request's first `keep`, then its own `tail`.
fn request_hashes(reqs: &HashMap<String, Value>, memo: &mut HashMap<String, Vec<String>>, rid: &str) -> Vec<String> {
    let mut chain = Vec::new();
    let mut cur = Some(rid);
    while let Some(r) = cur.filter(|r| !memo.contains_key(*r) && chain.len() <= reqs.len()) {
        chain.push(r);
        cur = reqs.get(r).and_then(|q| q["base"].as_str()).filter(|b| reqs.contains_key(*b));
    }
    for r in chain.into_iter().rev() {
        let q = &reqs[r];
        let mut list: Vec<String> = q["base"].as_str().and_then(|b| memo.get(b)).cloned().unwrap_or_default();
        list.truncate(q["keep"].as_u64().unwrap_or(0) as usize);
        list.extend(q["tail"].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_string));
        memo.insert(r.to_string(), list);
    }
    memo.get(rid).cloned().unwrap_or_default()
}

fn make_msg(m: &Value, src: &Source, call_of_mid: &HashMap<&str, usize>, human_texts: &HashSet<&str>) -> Msg {
    let text;
    let content: &[Value] = match &m["content"] {
        Value::String(s) => {
            text = [json!({ "type": "text", "text": s })];
            &text
        }
        Value::Array(a) => a,
        _ => &[],
    };
    let mut msg = Msg { json: Some(m.clone()), blocks: Vec::new(), prompt_chars: None, src_call: None };
    if m["role"] == "assistant" {
        if let Source::Assistant(mid) = src {
            msg.src_call = call_of_mid.get(mid.as_str()).copied();
        }
        for b in content {
            msg.blocks.push(match b["type"].as_str() {
                Some("thinking" | "redacted_thinking") => Block::new(Kind::Thinking, json_len(b)),
                Some("text") => Block::new(Kind::Text, b["text"].as_str().map_or(0, str::len)),
                Some("tool_use") => Block {
                    id: b["id"].as_str().map(str::to_string),
                    ..Block::new(Kind::Input, b.get("input").map_or(2, json_len))
                },
                _ => Block::new(Kind::Other, json_len(b)),
            });
        }
        return msg;
    }
    for b in content {
        msg.blocks.push(match b["type"].as_str() {
            Some("tool_result") => Block {
                id: b["tool_use_id"].as_str().map(str::to_string),
                ..Block::new(Kind::Result, result_bytes(&b["content"]))
            },
            Some("text") => {
                let t = b["text"].as_str().unwrap_or("");
                if human_texts.contains(t) {
                    *msg.prompt_chars.get_or_insert(0) += t.chars().count();
                }
                Block::new(Kind::Other, t.len())
            }
            _ => Block::new(Kind::Other, json_len(b)),
        });
    }
    msg
}

/// Bytes of a tool result's content: its text, and other parts (images) as JSON.
fn result_bytes(c: &Value) -> usize {
    match c {
        Value::String(s) => s.len(),
        Value::Array(items) => items
            .iter()
            .map(|x| if x["type"] == "text" { x["text"].as_str().map_or(0, str::len) } else { json_len(x) })
            .sum(),
        _ => 0,
    }
}

/// Tokens per 4 bytes of each kind of content, fitted on the sessions.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Calibration {
    pub text: f64,
    pub input: f64,
    pub result: f64,
    pub other: f64,
    /// Fitted weight of the recorded thinking tokens. They are billed as
    /// recorded; a value near 1 says the fit is sound.
    pub thinking: f64,
    /// Tokens of framing per content block.
    pub per_block: f64,
}

impl Calibration {
    fn tokens(&self, b: &Block) -> f64 {
        let per_4_bytes = match b.kind {
            Kind::Thinking | Kind::ToolDelta => return b.tokens,
            Kind::Text => self.text,
            Kind::Input => self.input,
            Kind::Result => self.result,
            Kind::Other => self.other,
        };
        per_4_bytes * b.bytes as f64 / 4.0 + self.per_block
    }

    fn msg_tokens(&self, m: &Msg) -> f64 {
        m.blocks.iter().map(|b| self.tokens(b)).sum()
    }
}

/// Fit token weights on how much each request's recorded prompt grew over the
/// previous one's, against the content it appended. Requests whose tool list
/// changed are left out.
fn calibrate(sessions: &[Session]) -> Result<Calibration> {
    let (mut rows, mut y) = (Vec::new(), Vec::new());
    for s in sessions {
        for w in s.calls.windows(2) {
            if w[0].shape != w[1].shape {
                continue;
            }
            let mut f = [0.0; 6];
            for b in w[1].msgs.get(w[0].msgs.len()..).unwrap_or_default().iter().flat_map(|&i| &s.msgs[i].blocks) {
                let quarter = b.bytes as f64 / 4.0;
                match b.kind {
                    Kind::Thinking => f[4] += b.tokens,
                    Kind::Text => f[0] += quarter,
                    Kind::Input => f[1] += quarter,
                    Kind::Result => f[2] += quarter,
                    Kind::Other => f[3] += quarter,
                    Kind::ToolDelta => {}
                }
                if b.kind != Kind::Thinking {
                    f[5] += 1.0;
                }
            }
            rows.push(f);
            y.push(w[1].total - w[0].total);
        }
    }
    match least_squares(&rows, &y) {
        Some([text, input, result, other, thinking, per_block]) => {
            Ok(Calibration { text, input, result, other, thinking, per_block })
        }
        None => bail!("too little varied data to fit token counts ({} requests)", rows.len()),
    }
}

/// Least squares by the normal equations, solved by Gauss-Jordan elimination.
fn least_squares(rows: &[[f64; 6]], y: &[f64]) -> Option<[f64; 6]> {
    let mut m = [[0.0; 7]; 6];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().take(6).enumerate() {
            *cell = rows.iter().map(|r| r[i] * r[j]).sum();
        }
        row[6] = rows.iter().zip(y).map(|(r, v)| r[i] * v).sum();
    }
    for i in 0..6 {
        // Partial pivoting: the first row with the largest entry in this column.
        let p = (i..6).fold(i, |p, r| if m[r][i].abs() > m[p][i].abs() { r } else { p });
        m.swap(i, p);
        if m[i][i].abs() < 1e-12 {
            return None;
        }
        let pivot = m[i];
        for (_, row) in m.iter_mut().enumerate().filter(|(r, _)| *r != i) {
            let f = row[i] / pivot[i];
            for (x, z) in row.iter_mut().zip(pivot) {
                *x -= f * z;
            }
        }
    }
    let out: [f64; 6] = std::array::from_fn(|i| m[i][6] / m[i][i]);
    out.iter().all(|x| x.is_finite()).then_some(out)
}

/// Set the session's system and tools tokens from its first request, and
/// put each mid-session change to the tool list in as a message of its own
/// where it happened, so the cache keeps what came before it.
fn prepare(s: &mut Session, cal: &Calibration) {
    s.sys_tokens = s.calls[0].total - s.calls[0].msgs.iter().map(|&i| cal.msg_tokens(&s.msgs[i])).sum::<f64>();
    for i in 1..s.calls.len() {
        if s.calls[i].shape == s.calls[i - 1].shape {
            continue;
        }
        let at = s.calls[i - 1].msgs.len();
        let new: f64 = s.calls[i].msgs.get(at..).unwrap_or_default().iter().map(|&m| cal.msg_tokens(&s.msgs[m])).sum();
        let tokens = (s.calls[i].total - s.calls[i - 1].total - new).max(0.0);
        s.msgs.push(Msg {
            json: None,
            blocks: vec![Block { tokens, ..Block::new(Kind::ToolDelta, 0) }],
            prompt_chars: None,
            src_call: None,
        });
        let id = s.msgs.len() - 1;
        for c in &mut s.calls[i..] {
            let at = at.min(c.msgs.len());
            c.msgs.insert(at, id);
        }
    }
}

/// A request's body as Claude Code sent it: its parameters, its shape's
/// system prompt and tools, its messages, and its cache markers with the
/// regime's lifetime (Claude Code marks every breakpoint 5-minute, or 1-hour
/// with `ENABLE_PROMPT_CACHING_1H`). The billing line Claude Code puts first
/// in `system` isn't logged, so it's left out.
fn request_body(s: &Session, c: &Call, regime: Regime) -> Value {
    let mut body = if c.params.is_object() { c.params.clone() } else { json!({}) };
    for key in ["system", "tools"] {
        if let Some(v) = s.shapes.get(&c.shape).and_then(|shape| shape.get(key)) {
            body[key] = v.clone();
        }
    }
    body["messages"] = c.msgs.iter().filter_map(|&i| s.msgs[i].json.clone()).collect();
    for m in &c.markers {
        let mut marker = m["value"].clone();
        if let Some(o) = marker.as_object_mut() {
            match regime {
                Regime::FiveMinFermat => _ = o.shift_remove("ttl"),
                Regime::OneHour => _ = o.insert("ttl".into(), json!("1h")),
            }
        }
        if let Some(path) = m["path"].as_array() {
            place_marker(&mut body, path, marker);
        }
    }
    body
}

/// Put a `cache_control` marker on the block at `path`, making string content
/// a text block first, as Claude Code does to mark it.
fn place_marker(body: &mut Value, path: &[Value], marker: Value) -> Option<()> {
    let mut v = body;
    for seg in path {
        if let Value::String(text) = v {
            let text = std::mem::take(text);
            *v = json!([{ "type": "text", "text": text }]);
        }
        v = match seg {
            Value::String(key) => v.get_mut(key.as_str())?,
            Value::Number(i) => v.get_mut(usize::try_from(i.as_u64()?).ok()?)?,
            _ => return None,
        };
    }
    v.as_object_mut()?.insert("cache_control".into(), marker);
    Some(())
}

/// When each request is sent. Under `5m-fermat`, as SessionBench paced Fermat:
/// a pause of at least 330 s before each new ticket and every third human turn.
fn timeline(s: &Session, regime: Regime) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(s.calls.len());
    let mut humans = 0;
    for (i, c) in s.calls.iter().enumerate() {
        let Some(prev) = i.checked_sub(1) else {
            out.push(c.t);
            continue;
        };
        let mut gap = c.t - s.calls[prev].t;
        if regime == Regime::FiveMinFermat && c.human {
            humans += 1;
            if c.ticket || (humans + 1) % 3 == 0 {
                gap = gap.max(FERMAT_PAUSE_SECS);
            }
        }
        out.push(out[prev] + gap);
    }
    out
}

fn mix(a: u64, b: u64) -> u64 {
    let mut h = DefaultHasher::new();
    (a, b).hash(&mut h);
    h.finish()
}

/// Prompt cache keyed by prefix: an entry for each written prefix, alive for
/// the TTL after its last write or read. A request reads its longest live
/// prefix (reading refreshes every shorter entry, as the KV is shared) and
/// writes an entry at its end. The system prompt and tools always hit.
struct Cache {
    ttl: f64,
    expiry: HashMap<u64, f64>,
}

impl Cache {
    /// Chain hashes of a request's units, and the last one with a live entry.
    fn lookup(&self, t: f64, keys: &[u64]) -> (Vec<u64>, Option<usize>) {
        let mut h = 0;
        let hs: Vec<u64> = keys
            .iter()
            .map(|&k| {
                h = mix(h, k);
                h
            })
            .collect();
        let hit = hs.iter().rposition(|x| self.expiry.get(x).is_some_and(|&e| e >= t));
        (hs, hit)
    }

    /// Refresh what was read, and write the end of the request and `also`:
    /// where the request leaves the previous one's prefix, as the cache model
    /// does (in an append-only history, the previous request's end).
    fn touch(&mut self, t: f64, hs: &[u64], hit: Option<usize>, also: Option<usize>) {
        let until = t + self.ttl;
        for h in &hs[..hit.map_or(0, |k| k + 1)] {
            if let Some(e) = self.expiry.get_mut(h) {
                *e = until;
            }
        }
        for j in hs.len().checked_sub(1).into_iter().chain(also) {
            self.expiry.insert(hs[j], until);
        }
    }
}

/// Numbers thinking turns the way ctxmgmt counts them: consecutive assistant
/// messages are one turn, which counts once one of them has thinking.
#[derive(Default)]
struct ThinkingTurns {
    n: u64,
}

impl ThinkingTurns {
    /// The thinking turn an assistant message `m` is part of. Turns start at
    /// user messages as the server counts them ([`ctxmgmt::starts_turn`]).
    fn next(&mut self, m: &Value) -> Option<u64> {
        if ctxmgmt::starts_turn(m) {
            self.n += 1;
        }
        (m["role"] == "assistant").then_some(self.n)
    }
}

/// What the server clears for a request.
#[derive(Debug, Default)]
struct Cleared {
    /// tool_use ids whose results (and inputs, with `inputs`) are cleared.
    ids: HashSet<String>,
    tool_uses: u64,
    inputs: bool,
    /// The oldest thinking turns cleared.
    thinking_turns: u64,
}

/// A request's `context_management` edits as the server applies them (the
/// counting measured on the API; see ctxmgmt's module docs).
fn server_clears(body: &Value) -> Cleared {
    let messages = body["messages"].as_array().map_or(&[][..], Vec::as_slice);
    let results: HashMap<&str, &Value> = messages
        .iter()
        .flat_map(|m| m["content"].as_array().into_iter().flatten())
        .filter(|b| b["type"] == "tool_result")
        .filter_map(|b| b["tool_use_id"].as_str().map(|id| (id, b)))
        .collect();
    let mut out = Cleared::default();
    for e in body["context_management"]["edits"].as_array().into_iter().flatten() {
        let keep = e["keep"]["value"].as_u64();
        match e["type"].as_str() {
            Some(CLEAR_THINKING) => {
                let Some(keep) = keep else { continue };
                let mut turns = ThinkingTurns::default();
                messages.iter().for_each(|m| _ = turns.next(m));
                out.thinking_turns = turns.n.saturating_sub(keep);
            }
            Some(CLEAR_TOOL_USES) => {
                let (Some(keep), Some(trigger)) = (keep, e["trigger"]["value"].as_u64()) else { continue };
                let excluded: HashSet<&str> =
                    e["exclude_tools"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
                let inputs = e["clear_tool_inputs"] == true;
                let uses: Vec<&str> = messages
                    .iter()
                    .filter(|m| m["role"] == "assistant")
                    .flat_map(|m| m["content"].as_array().into_iter().flatten())
                    .filter(|b| b["type"] == "tool_use" && !b["name"].as_str().is_some_and(|n| excluded.contains(n)))
                    .filter_map(|b| b["id"].as_str())
                    .filter(|id| ctxmgmt::counts_toward_keep(results.get(id).copied(), inputs))
                    .collect();
                if uses.len() as u64 > trigger {
                    let n = uses.len().saturating_sub(keep as usize);
                    out.ids = uses[..n].iter().map(|s| s.to_string()).collect();
                    out.tool_uses = n as u64;
                    out.inputs = inputs;
                }
            }
            _ => {}
        }
    }
    out
}

/// Tokens billed by kind, in base-input-token equivalents once priced.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Bill {
    pub cache_read: f64,
    pub cache_write: f64,
    pub input: f64,
    pub output: f64,
    pub cost: f64,
}

#[derive(Debug, Default)]
struct Run {
    bill: Bill,
    cold: usize,
    advances: usize,
    breakers: Vec<&'static str>,
    /// Unchanged-file Read results refilled, by tool_use id.
    refilled: HashSet<String>,
}

/// Replay a session's requests through the cache, sent as is or, with
/// `edits`, through the proxy's `--context-edits cold` logic.
fn simulate(s: &Session, cal: &Calibration, cfg: &SimConfig, edits: bool) -> Run {
    let ttl = cfg.regime.ttl();
    let times = timeline(s, cfg.regime);
    // Only the refills: they come before anything else in `transform`, which
    // leaves outputs under `min_bytes` alone.
    let refill_only =
        TransformConfig { min_bytes: usize::MAX, dedup: false, ctx_window: Some(cfg.ctx.window), ..Default::default() };
    let mut cache = Cache { ttl, expiry: HashMap::new() };
    let mut state = CtxState::default();
    let mut run = Run::default();
    let mut last: Option<f64> = None;
    let mut prev_keys: Vec<u64> = Vec::new();
    for (c, &t) in s.calls.iter().zip(&times) {
        // The cache expires at the TTL; the proxy calls it cold a margin later.
        let idle = last.map(|l| t - l);
        run.cold += usize::from(idle.is_some_and(|g| g > ttl));
        let proxy_cold = idle.is_some_and(|g| g > ttl + COLD_MARGIN_SECS as f64);
        let proxy_warm = idle.is_some_and(|g| g + COLD_MARGIN_SECS as f64 <= ttl);
        let (mut cleared, mut pinned) = (Cleared::default(), None);
        let mut advanced = false;
        // Result bytes of each refilled tool result.
        let mut refills: HashMap<String, usize> = HashMap::new();
        if edits {
            // The request as the proxy gets it, through what the proxy does to
            // it, and the edits it sends as the server would apply them.
            let mut body = request_body(s, c, cfg.regime);
            let changes = transform::transform(&mut body, &refill_only, &mut DryRun, None).changes;
            let ids: HashSet<&str> =
                changes.iter().filter(|c| c.kind == ChangeKind::Refilled).map(|c| c.tool_use_id.as_str()).collect();
            for b in body["messages"].as_array().into_iter().flatten().flat_map(|m| m["content"].as_array()).flatten() {
                if let Some(id) = b["tool_use_id"].as_str().filter(|id| ids.contains(id)) {
                    refills.insert(id.to_string(), result_bytes(&b["content"]));
                }
            }
            run.refilled.extend(refills.keys().cloned());
            let census = ctxmgmt::census(&body, &c.betas, cfg.ctx.clear_inputs);
            advanced = ctxmgmt::step(&mut state, &census, &cfg.ctx, proxy_cold).is_some();
            run.advances += usize::from(advanced);
            pinned = ctxmgmt::apply(&mut body, &census, &state);
            cleared = server_clears(&body);
        }

        // One cache unit per message, keyed by the message and what is cleared in it.
        let mut turns = ThinkingTurns::default();
        let units: Vec<(u64, f64)> = c
            .msgs
            .iter()
            .map(|&i| {
                let m = &s.msgs[i];
                let drop_thinking =
                    m.json.as_ref().and_then(|j| turns.next(j)).is_some_and(|n| n <= cleared.thinking_turns);
                let (mut tokens, mut results, mut inputs) = (0.0, 0, 0);
                for b in &m.blocks {
                    let refill = b.id.as_ref().and_then(|id| refills.get(id)).filter(|_| b.kind == Kind::Result);
                    let t = match refill {
                        Some(&bytes) => cal.tokens(&Block::new(Kind::Result, bytes)),
                        None => cal.tokens(b),
                    };
                    let cut = b.id.as_ref().is_some_and(|id| cleared.ids.contains(id));
                    tokens += match b.kind {
                        Kind::Thinking if drop_thinking => 0.0,
                        // The placeholder, plus what the agent reads back.
                        Kind::Result if cut => {
                            results += 1;
                            let kept = t.min(cfg.placeholder);
                            kept + cfg.reread * (t - kept)
                        }
                        Kind::Input if cut && cleared.inputs => {
                            inputs += 1;
                            t.min(cfg.placeholder)
                        }
                        _ => t,
                    };
                }
                let variant = results | (inputs << 24) | (u64::from(drop_thinking) << 48);
                (mix(i as u64, variant), tokens)
            })
            .collect();

        let keys: Vec<u64> = units.iter().map(|u| u.0).collect();
        let (hs, hit) = cache.lookup(t, &keys);
        let shared = prev_keys.iter().zip(&keys).take_while(|(a, b)| a == b).count();
        let split = hit.map_or(0, |k| k + 1);
        let read = s.sys_tokens + units[..split].iter().map(|u| u.1).sum::<f64>();
        let write: f64 = units[split..].iter().map(|u| u.1).sum();
        cache.touch(t, &hs, hit, shared.checked_sub(1));
        run.bill.cache_read += read;
        run.bill.cache_write += write;
        run.bill.input += c.input;
        run.bill.output += c.output;

        // The response, through the proxy's breakers.
        if let Some(e) = pinned {
            let usage = Usage {
                input_tokens: c.input as u64,
                cache_read_input_tokens: read.round() as u64,
                cache_creation_input_tokens: write.round() as u64,
                stop_reason: Some("end_turn".into()),
                applied_edits: Some(AppliedEdits {
                    cleared_tool_uses: cleared.tool_uses,
                    cleared_thinking_turns: cleared.thinking_turns,
                    ..Default::default()
                }),
                ..Default::default()
            };
            let model = c.params["model"].as_str();
            run.breakers.extend(ctxmgmt::settle(&mut state, e, proxy_warm && !advanced, model, Some(&usage), 200));
        }
        last = Some(t);
        prev_keys = keys;
    }
    run
}

/// Replay a session's requests with [`Compaction`] at each pause before a
/// human prompt. `advances` counts the compactions.
fn simulate_compaction(s: &Session, cal: &Calibration, cfg: &SimConfig, comp: &Compaction) -> Run {
    let ttl = cfg.regime.ttl();
    let times = timeline(s, cfg.regime);
    let mut cache = Cache { ttl, expiry: HashMap::new() };
    let mut run = Run::default();
    let unit = |i: usize| (mix(i as u64, 0), cal.msg_tokens(&s.msgs[i]));
    // The summary in force: the messages it stands for, and its cache unit.
    let mut summary: Option<(Vec<usize>, (u64, f64))> = None;
    // A request's messages as the proxy sends them.
    let units = |msgs: &[usize], summary: &Option<(Vec<usize>, (u64, f64))>| -> Vec<(u64, f64)> {
        match summary {
            Some((covered, u)) if msgs.starts_with(covered) => {
                std::iter::once(*u).chain(msgs[covered.len()..].iter().map(|&i| unit(i))).collect()
            }
            _ => msgs.iter().map(|&i| unit(i)).collect(),
        }
    };
    let bill = |cache: &mut Cache, b: &mut Bill, t: f64, us: &[(u64, f64)], prev: &[u64]| {
        let keys: Vec<u64> = us.iter().map(|u| u.0).collect();
        let (hs, hit) = cache.lookup(t, &keys);
        let shared = prev.iter().zip(&keys).take_while(|(a, b)| a == b).count();
        let split = hit.map_or(0, |k| k + 1);
        b.cache_read += s.sys_tokens + us[..split].iter().map(|u| u.1).sum::<f64>();
        b.cache_write += us[split..].iter().map(|u| u.1).sum::<f64>();
        cache.touch(t, &hs, hit, shared.checked_sub(1));
        keys
    };
    let mut prev: Option<(f64, Vec<u64>)> = None;
    for (c, &t) in s.calls.iter().zip(&times) {
        let idle = prev.as_ref().map(|(lt, _)| t - lt);
        run.cold += usize::from(idle.is_some_and(|g| g > ttl));
        if !summary.as_ref().is_none_or(|(covered, _)| c.msgs.starts_with(covered)) {
            summary = None; // Claude Code compacted or rewound the history itself.
        }
        let paused = c.human && idle.is_some_and(|g| g > ttl) && (c.ticket || !comp.tickets_only);
        let at = c.msgs.iter().rposition(|&i| s.msgs[i].prompt_chars.is_some()).unwrap_or(c.msgs.len());
        if paused && let Some((lt, prev_keys)) = &prev {
            let history = units(&c.msgs[..at], &summary);
            if history.iter().map(|u| u.1).sum::<f64>() >= comp.min_history {
                // Summarize while the previous request's prefix is still cached.
                let conversation = history.iter().map(|u| u.1).sum::<f64>() + COMPACT_INSTRUCTION_TOKENS;
                match comp.cache {
                    SummaryCache::Cold => run.bill.input += s.sys_tokens + conversation,
                    SummaryCache::SystemOnly => {
                        run.bill.cache_read += s.sys_tokens;
                        run.bill.input += conversation;
                    }
                    SummaryCache::Warm => {
                        let mut req = history.clone();
                        req.push((mix(u64::MAX, run.advances as u64), COMPACT_INSTRUCTION_TOKENS));
                        bill(&mut cache, &mut run.bill, lt + ttl - COMPACT_LEAD_SECS, &req, prev_keys);
                    }
                }
                run.bill.output += comp.summary * (1.0 + comp.thinking);
                let before = summary.as_ref().map_or(0, |(covered, _)| covered.len());
                let results: f64 = c.msgs[before..at]
                    .iter()
                    .flat_map(|&i| &s.msgs[i].blocks)
                    .filter(|b| b.kind == Kind::Result)
                    .map(|b| cal.tokens(b))
                    .sum();
                let key = mix(u64::MAX - 1, c.msgs[at - 1] as u64);
                summary = Some((c.msgs[..at].to_vec(), (key, comp.summary + comp.reread * results)));
                run.advances += 1;
            }
        }
        let us = units(&c.msgs, &summary);
        let keys = bill(&mut cache, &mut run.bill, t, &us, prev.as_ref().map_or(&[][..], |p| &p.1));
        run.bill.input += c.input;
        run.bill.output += c.output;
        prev = Some((t, keys));
    }
    run
}

/// One session, sent as is and with `--context-edits cold`.
#[derive(Debug, Serialize)]
pub struct SessionSim {
    pub file: PathBuf,
    pub model: String,
    pub requests: usize,
    /// Messages the transcript doesn't hold, billed as empty.
    pub unresolved: usize,
    /// Mean error of the modelled prompt size against the recorded one.
    pub fit_error: f64,
    /// Requests that found the prompt cache expired.
    pub cold: usize,
    /// Cold moments at which the cleared set moved forward.
    pub advances: usize,
    pub breakers: Vec<&'static str>,
    /// Unchanged-file Read results the proxy would give their content back.
    pub refilled: usize,
    pub read_price: f64,
    pub unchanged: Bill,
    pub context_edits: Bill,
    /// Cost with context edits over cost unchanged.
    pub ratio: f64,
}

#[derive(Debug, Serialize)]
pub struct CacheSim {
    pub regime: Regime,
    pub calibration: Calibration,
    pub sessions: Vec<SessionSim>,
    /// Transcripts that log no requests (Claude Code before 2.1.317) or failed to parse.
    pub skipped: usize,
    /// Mean of the per-session ratios.
    pub mean_ratio: f64,
}

/// A request rebuilt from a transcript, and when it goes out on a regime's timeline.
#[derive(Debug, Clone)]
pub struct Rebuilt {
    /// Unix seconds.
    pub at: f64,
    /// Its `anthropic-beta` header.
    pub betas: String,
    pub body: Value,
}

/// A transcript's requests as Claude Code sent them, timed as `regime` times
/// them, for replaying through the proxy; `None` when it logs no requests.
pub fn rebuild(path: &Path, regime: Regime) -> Result<Option<Vec<Rebuilt>>> {
    let Some(s) = load(path)? else { return Ok(None) };
    let times = timeline(&s, regime);
    let requests = s.calls.iter().zip(times);
    Ok(Some(
        requests.map(|(c, at)| Rebuilt { at, betas: c.betas.clone(), body: request_body(&s, c, regime) }).collect(),
    ))
}

/// Rebuild, calibrate and simulate the given transcripts.
pub fn simulate_files(files: &[PathBuf], cfg: &SimConfig) -> Result<CacheSim> {
    let mut sessions = Vec::new();
    let mut skipped = 0;
    for f in files {
        match load(f) {
            Ok(Some(s)) if !s.calls.is_empty() => sessions.push(s),
            _ => skipped += 1,
        }
    }
    if sessions.is_empty() {
        bail!("no transcript logs its API requests (Claude Code 2.1.317 or later does)");
    }
    let calibration = calibrate(&sessions)?;
    let mut out = Vec::new();
    for mut s in sessions {
        prepare(&mut s, &calibration);
        let price = |b: &mut Bill, read: f64| {
            b.cost = read * b.cache_read + cfg.regime.write_price() * b.cache_write + b.input + OUTPUT_PRICE * b.output;
        };
        let read = cfg.read_price.unwrap_or_else(|| read_price(&s.model));
        let mut base = simulate(&s, &calibration, cfg, false);
        let mut cm = match &cfg.compact {
            Some(comp) => simulate_compaction(&s, &calibration, cfg, comp),
            None => simulate(&s, &calibration, cfg, true),
        };
        price(&mut base.bill, read);
        price(&mut cm.bill, read);
        let fit_error = s
            .calls
            .iter()
            .map(|c| {
                let modelled = s.sys_tokens + c.msgs.iter().map(|&i| calibration.msg_tokens(&s.msgs[i])).sum::<f64>();
                (modelled - c.total).abs() / c.total.max(1.0)
            })
            .sum::<f64>()
            / s.calls.len() as f64;
        out.push(SessionSim {
            file: s.path,
            model: s.model,
            requests: s.calls.len(),
            unresolved: s.unresolved,
            fit_error,
            cold: base.cold,
            advances: cm.advances,
            breakers: cm.breakers,
            refilled: cm.refilled.len(),
            read_price: read,
            ratio: cm.bill.cost / base.bill.cost,
            unchanged: base.bill,
            context_edits: cm.bill,
        });
    }
    let mean_ratio = out.iter().map(|s| s.ratio).sum::<f64>() / out.len() as f64;
    Ok(CacheSim { regime: cfg.regime, calibration, sessions: out, skipped, mean_ratio })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctxmgmt::KeepWindow;

    /// Writes a transcript the way Claude Code logs one: entries for prompts,
    /// responses and tool results, and each request as an `api-request` delta
    /// against the one before.
    struct Log {
        lines: Vec<Value>,
        /// What the next request sends.
        messages: Vec<Value>,
        /// Messages the previous request sent.
        sent: usize,
        t: f64,
        requests: usize,
        shape: String,
    }

    impl Log {
        fn new() -> Self {
            let mut log =
                Self { lines: Vec::new(), messages: Vec::new(), sent: 0, t: 0.0, requests: 0, shape: String::new() };
            log.set_tools(&["Bash", "Read", "TodoWrite"]);
            log
        }

        fn set_tools(&mut self, names: &[&str]) {
            self.shape = names.join("+");
            let tools: Vec<Value> = names.iter().map(|n| json!({ "name": n })).collect();
            let system = [json!({ "type": "text", "text": "You are a Claude agent." })];
            self.lines.push(json!({ "type": "api-request-shape", "shapeHash": self.shape, "shape": { "system": system, "tools": tools } }));
        }

        fn prompt(&mut self, text: &str) {
            self.lines.push(json!({ "type": "user", "uuid": format!("u{}", self.lines.len()), "message": { "role": "user", "content": text } }));
            self.messages.push(json!({ "role": "user", "content": [{ "type": "text", "text": text }] }));
        }

        /// Tool results for the latest response, logged as one entry, or one
        /// entry per block as Claude Code logs parallel results.
        fn results(&mut self, blocks: Vec<Value>, split: bool) {
            let parent = format!("a{}", self.requests - 1);
            let entries = if split { blocks.iter().map(|b| vec![b.clone()]).collect() } else { vec![blocks.clone()] };
            for content in entries {
                self.lines.push(json!({ "type": "user", "uuid": format!("u{}", self.lines.len()),
                    "sourceToolAssistantUUID": parent, "message": { "role": "user", "content": content } }));
            }
            self.messages.push(json!({ "role": "user", "content": blocks }));
        }

        /// A request sent `after` seconds after the previous one, and its response.
        fn request(&mut self, after: f64, reply: Vec<Value>, thinking_tokens: u64) {
            self.t += after;
            let (h, m, s) = ((self.t / 3600.0) as u64, (self.t % 3600.0 / 60.0) as u64, self.t % 60.0);
            let id = format!("r{}", self.requests);
            let tail: Vec<String> = self.messages[self.sent..].iter().map(message_hash).collect();
            let prompt = 12_000 + serde_json::to_string(&self.messages).unwrap().len() / 3;
            // Markers as a subscription session logs them: 1-hour, on the system prompt and the last block.
            let last = self.messages.len() - 1;
            let block = self.messages[last]["content"].as_array().map_or(0, |c| c.len() - 1);
            let marker = json!({ "type": "ephemeral", "ttl": "1h" });
            self.lines.push(
                json!({ "type": "api-request", "id": id, "timestamp": format!("2026-11-03T{h:02}:{m:02}:{s:06.3}Z"),
                "shapeHash": self.shape,
                "params": { "model": "claude-opus-5-5", "betas": ["claude-code-20250219", ctxmgmt::BETA],
                            "thinking": { "type": "adaptive" },
                            "context_management": { "edits": [{ "type": CLEAR_THINKING, "keep": "all" }] } },
                "base": self.requests.checked_sub(1).map(|i| format!("r{i}")), "keep": self.sent, "tail": tail,
                "cacheControl": [{ "path": ["system", 0], "value": marker },
                                 { "path": ["messages", last, "content", block], "value": marker }] }),
            );
            self.sent = self.messages.len();
            self.lines.push(json!({ "type": "assistant", "uuid": format!("a{}", self.requests), "requestRef": id,
                "message": { "id": format!("m{}", self.requests), "role": "assistant", "content": reply,
                    "usage": { "input_tokens": 3, "cache_read_input_tokens": prompt, "cache_creation_input_tokens": 0,
                               "output_tokens": 100, "output_tokens_details": { "thinking_tokens": thinking_tokens } } } }));
            self.messages.push(json!({ "role": "assistant", "content": reply }));
            self.requests += 1;
        }

        fn write(&self, dir: &Path, name: &str) -> PathBuf {
            let path = dir.join(name);
            let text: String = self.lines.iter().map(|l| format!("{l}\n")).collect();
            std::fs::write(&path, text).unwrap();
            path
        }
    }

    /// A ticket, then `turns` follow-ups, each with `loops` Bash calls of
    /// varied sizes, `gap` seconds apart.
    fn session_log(turns: usize, loops: usize, gap: f64) -> Log {
        let mut log = Log::new();
        let mut k = 0;
        for turn in 0..=turns {
            log.prompt(&if turn == 0 { "t".repeat(TICKET_MIN_CHARS + 100) } else { format!("follow-up {turn}") });
            for _ in 0..loops {
                k += 1;
                let id = format!("toolu_{k}");
                let mut reply =
                    vec![json!({ "type": "thinking", "thinking": "", "signature": "s".repeat(40 + k % 7 * 13) })];
                if k % 3 == 0 {
                    reply.push(json!({ "type": "text", "text": "n".repeat(30 + k % 5 * 41) }));
                }
                reply.push(json!({ "type": "tool_use", "id": id, "name": "Bash", "input": { "command": "c".repeat(20 + k % 4 * 57) } }));
                log.request(gap, reply, 50 + (k % 6) as u64 * 37);
                let mut result = vec![
                    json!({ "type": "tool_result", "tool_use_id": id, "content": "r".repeat(3000 + k % 9 * 811) }),
                ];
                if k % 4 == 1 {
                    result.push(json!({ "type": "text", "text": "<system-reminder>".repeat(1 + k % 3) }));
                }
                log.results(result, false);
            }
            let done = vec![
                json!({ "type": "thinking", "thinking": "", "signature": "s".repeat(64) }),
                json!({ "type": "text", "text": "done" }),
            ];
            log.request(gap, done, 40);
        }
        log
    }

    fn loaded(log: &Log) -> Session {
        let dir = tempfile::tempdir().unwrap();
        load(&log.write(dir.path(), "s.jsonl")).unwrap().expect("requests logged")
    }

    /// One token per 4 bytes of anything, and no framing.
    const FLAT: Calibration =
        Calibration { text: 1.0, input: 1.0, result: 1.0, other: 1.0, thinking: 1.0, per_block: 0.0 };

    fn cfg(regime: Regime) -> SimConfig {
        SimConfig {
            regime,
            read_price: Some(0.1),
            reread: 0.0,
            placeholder: 60.0,
            ctx: CtxConfig { window: KeepWindow { min: 4, max: 4 }, clear_inputs: true, clear_thinking: true },
            compact: None,
        }
    }

    #[test]
    fn compaction_at_pauses_replaces_the_history_with_the_summary() {
        let mut s = loaded(&session_log(5, 10, 20.0));
        prepare(&mut s, &FLAT);
        let cost = |run: &Run| {
            let b = &run.bill;
            0.1 * b.cache_read + 1.25 * b.cache_write + b.input + OUTPUT_PRICE * b.output
        };
        let cfg = cfg(Regime::FiveMinFermat);
        let base = simulate(&s, &FLAT, &cfg, false);
        let comp = |summary: f64, min_history: f64| Compaction {
            summary,
            min_history,
            thinking: 0.5,
            reread: 0.1,
            tickets_only: false,
            cache: SummaryCache::Warm,
        };
        let small = simulate_compaction(&s, &FLAT, &cfg, &comp(500.0, 1000.0));
        // Pauses before the second follow-up and the fifth (every third human turn).
        assert_eq!((base.cold, small.cold, small.advances), (2, 2, 2));
        assert!(cost(&small) < 0.8 * cost(&base), "{} vs {}", cost(&small), cost(&base));
        assert!(small.bill.output > base.bill.output, "the summaries are output");
        // A summary as big as the history saves nothing; a high minimum never compacts.
        let huge = simulate_compaction(&s, &FLAT, &cfg, &comp(1e7, 1000.0));
        assert!(cost(&huge) > cost(&base));
        let never = simulate_compaction(&s, &FLAT, &cfg, &comp(500.0, 1e9));
        assert_eq!(never.advances, 0);
        assert!((cost(&never) - cost(&base)).abs() < 1e-6, "without compaction it's the session as sent");
        // The pauses here are follow-ups, not new tickets.
        let tickets = simulate_compaction(&s, &FLAT, &cfg, &Compaction { tickets_only: true, ..comp(500.0, 1000.0) });
        assert_eq!(tickets.advances, 0);
        // Summarizing after the cache expired reads the history at full price.
        let late =
            simulate_compaction(&s, &FLAT, &cfg, &Compaction { cache: SummaryCache::Cold, ..comp(500.0, 1000.0) });
        let sys = simulate_compaction(
            &s,
            &FLAT,
            &cfg,
            &Compaction { cache: SummaryCache::SystemOnly, ..comp(500.0, 1000.0) },
        );
        assert!(cost(&small) < cost(&sys) && cost(&sys) < cost(&late) && cost(&late) < cost(&base));
    }

    #[test]
    fn message_hashes_match_claude_codes() {
        // Python: sha256(json.dumps(m, separators=(",", ":"), sort_keys=True, ensure_ascii=False)).
        let m: Value = serde_json::from_str(
            r#"{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "line \"1\"\n\ttab é → \u0001 \\ end", "is_error": false}, {"type": "text", "text": "ok", "cache_control": {"type": "ephemeral", "ttl": "1h"}, "n": 12, "f": 0.5, "z": null}]}"#,
        )
        .unwrap();
        assert_eq!(message_hash(&m), "42b30c3b5fb0b73261763df33433a6a0c8ed7513852b143c38d2ff2b9ae6550a");
    }

    #[test]
    fn parses_transcript_timestamps() {
        assert_eq!(unix_secs("1970-01-01T00:00:00Z"), Some(0.0));
        assert!((unix_secs("2026-11-03T23:31:51.092Z").unwrap() - 1_793_748_711.092).abs() < 1e-3);
        assert_eq!(unix_secs("2024-02-29T12:30:00+01:30"), Some(1_709_204_400.0));
        assert_eq!(unix_secs("yesterday"), None);
    }

    #[test]
    fn rebuilds_each_request_from_its_delta() {
        let mut log = Log::new();
        log.prompt(&"t".repeat(TICKET_MIN_CHARS));
        let uses = [("t1", "Bash"), ("t2", "Read")]
            .map(|(id, name)| json!({ "type": "tool_use", "id": id, "name": name, "input": { "x": id } }));
        let thinking =
            ["abc", "abcdefghijkl"].map(|sig| json!({ "type": "thinking", "thinking": "", "signature": sig }));
        log.request(0.0, vec![thinking[0].clone(), uses[0].clone(), thinking[1].clone(), uses[1].clone()], 30);
        // Claude Code logs a field it doesn't send.
        log.lines.last_mut().unwrap()["message"]["content"][1]["caller"] = json!({ "type": "direct" });
        log.results(
            vec![
                json!({ "type": "tool_result", "tool_use_id": "t1", "content": "one" }),
                json!({ "type": "tool_result", "tool_use_id": "t2", "content": [{ "type": "text", "text": "two" }] }),
            ],
            true,
        );
        log.request(10.0, vec![json!({ "type": "text", "text": "done" })], 0);
        // A message sent in a form only a blob records, one recorded nowhere, and a follow-up.
        let blob = json!({ "role": "user", "content": [{ "type": "text", "text": "hook said hi" }] });
        log.lines.push(json!({ "type": "api-request-blob", "hash": message_hash(&blob), "message": blob }));
        log.messages.push(blob);
        log.messages.push(json!({ "role": "user", "content": "lost" }));
        log.prompt("and the tests?");
        log.request(20.0, vec![json!({ "type": "text", "text": "ok" })], 0);

        let s = loaded(&log);
        assert_eq!(s.calls.len(), 3);
        assert_eq!(s.calls.iter().map(|c| c.msgs.len()).collect::<Vec<_>>(), [1, 3, 7]);
        assert_eq!(s.unresolved, 1);
        assert_eq!(
            s.calls.iter().map(|c| (c.human, c.ticket)).collect::<Vec<_>>(),
            [(true, true), (false, false), (true, false)]
        );
        assert_eq!(s.calls[2].t - s.calls[0].t, 30.0);
        let response = &s.msgs[s.calls[1].msgs[1]];
        assert_eq!(response.src_call, Some(0));
        let (a, b) = (json_len(&thinking[0]) as f64, json_len(&thinking[1]) as f64);
        assert_eq!(
            (response.blocks[0].tokens, response.blocks[2].tokens),
            (30.0 * a / (a + b), 30.0 * b / (a + b)),
            "split by size"
        );
        let results = &s.msgs[s.calls[1].msgs[2]];
        assert_eq!(
            results.blocks.iter().map(|b| (b.kind, b.bytes, b.id.as_deref())).collect::<Vec<_>>(),
            [(Kind::Result, 3, Some("t1")), (Kind::Result, 3, Some("t2"))],
            "parallel results merged"
        );
        assert_eq!(s.msgs[s.calls[2].msgs[4]].json.as_ref().unwrap()["content"][0]["text"], "hook said hi");
        assert_eq!(s.calls[2].params["context_management"]["edits"][0]["keep"], "all");
        assert!(s.calls[2].betas.contains(ctxmgmt::BETA));
    }

    #[test]
    fn rebuilds_requests_as_claude_code_sent_them() {
        let mut log = Log::new();
        // A prompt Claude Code logs, and sends after the first request, as a string.
        log.lines.push(json!({ "type": "user", "uuid": "p0", "message": { "role": "user", "content": "go" } }));
        log.messages.push(json!({ "role": "user", "content": "go" }));
        log.request(0.0, vec![json!({ "type": "text", "text": "ok" })], 0);
        log.prompt("more");
        log.request(400.0, vec![json!({ "type": "text", "text": "done" })], 0);
        let dir = tempfile::tempdir().unwrap();
        let path = log.write(dir.path(), "s.jsonl");

        let five = rebuild(&path, Regime::FiveMinFermat).unwrap().expect("requests logged");
        assert_eq!((five.len(), five[1].at - five[0].at), (2, 400.0));
        assert_eq!(five[0].betas, format!("claude-code-20250219,{}", ctxmgmt::BETA));
        let first = &five[0].body;
        assert!(first.get("betas").is_none(), "betas go in the header");
        assert_eq!(
            (first["model"].clone(), first["thinking"].clone()),
            (json!("claude-opus-5-5"), json!({ "type": "adaptive" }))
        );
        assert_eq!(first["context_management"]["edits"][0]["keep"], "all");
        assert_eq!(first["tools"].as_array().unwrap().len(), 3);
        let five_min = json!({ "type": "ephemeral" });
        assert_eq!(first["system"][0]["cache_control"], five_min, "5-minute markers under 5m-fermat");
        assert_eq!(
            first["messages"][0]["content"],
            json!([{ "type": "text", "text": "go", "cache_control": five_min }]),
            "string content made a block to carry its marker"
        );
        let second = &five[1].body;
        assert_eq!(second["messages"][0]["content"], "go", "unmarked, as logged");
        assert_eq!(second["messages"][2]["content"][0]["cache_control"], five_min);
        let hour = rebuild(&path, Regime::OneHour).unwrap().unwrap();
        assert_eq!(hour[1].at - hour[0].at, 400.0);
        assert_eq!(hour[1].body["system"][0]["cache_control"]["ttl"], "1h");
        assert_eq!(hour[1].body["messages"][2]["content"][0]["cache_control"]["ttl"], "1h");

        let old = dir.path().join("old.jsonl");
        std::fs::write(&old, "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"hi\"}}\n").unwrap();
        assert!(rebuild(&old, Regime::FiveMinFermat).unwrap().is_none());
    }

    #[test]
    fn calibration_recovers_the_token_weights() {
        let mut s = loaded(&session_log(2, 12, 10.0));
        let truth = Calibration { text: 1.3, input: 1.7, result: 1.8, other: 1.75, thinking: 1.0, per_block: 41.0 };
        for c in &mut s.calls {
            c.total = 9000.0 + c.msgs.iter().map(|&i| truth.msg_tokens(&s.msgs[i])).sum::<f64>();
        }
        let fit = calibrate(std::slice::from_ref(&s)).unwrap();
        for (got, want) in [
            (fit.text, truth.text),
            (fit.input, truth.input),
            (fit.result, truth.result),
            (fit.other, truth.other),
            (fit.thinking, 1.0),
            (fit.per_block, truth.per_block),
        ] {
            assert!((got - want).abs() < 1e-6, "{fit:?}");
        }
        prepare(&mut s, &fit);
        assert!((s.sys_tokens - 9000.0).abs() < 1e-6);
        assert!(least_squares(&[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; 3], &[1.0; 3]).is_none(), "too little to fit");
    }

    #[test]
    fn a_tool_list_change_is_a_message_of_its_own() {
        let mut log = session_log(2, 12, 10.0);
        log.set_tools(&["Bash", "Read", "TodoWrite", "mcp__linear__get_issue"]);
        log.prompt("next");
        log.request(10.0, vec![json!({ "type": "text", "text": "ok" })], 0);
        log.prompt("more");
        log.request(10.0, vec![json!({ "type": "text", "text": "ok" })], 0);
        let mut s = loaded(&log);
        let n = s.calls.len();
        let truth = Calibration { text: 1.3, input: 1.7, result: 1.8, other: 1.75, thinking: 1.0, per_block: 41.0 };
        for (i, c) in s.calls.iter_mut().enumerate() {
            let tools = if i >= n - 2 { 500.0 } else { 0.0 };
            c.total = 9000.0 + tools + c.msgs.iter().map(|&m| truth.msg_tokens(&s.msgs[m])).sum::<f64>();
        }
        let fit = calibrate(std::slice::from_ref(&s)).unwrap();
        assert!((fit.per_block - truth.per_block).abs() < 1e-6, "the change is left out of the fit: {fit:?}");
        prepare(&mut s, &fit);
        let at = s.calls[n - 3].msgs.len();
        for c in &s.calls[n - 2..] {
            let delta = &s.msgs[c.msgs[at]];
            assert!(delta.json.is_none());
            assert!((delta.blocks[0].tokens - 500.0).abs() < 1e-6);
        }
        assert!(s.calls[..n - 2].iter().all(|c| c.msgs.iter().all(|&i| s.msgs[i].json.is_some())));
    }

    #[test]
    fn fermat_pauses_before_tickets_and_every_third_human_turn() {
        let mut log = Log::new();
        log.prompt(&"t".repeat(TICKET_MIN_CHARS));
        for i in 1..=15 {
            match i {
                3 | 6 | 9 | 12 => log.prompt(&format!("follow-up {i}")),
                14 => log.prompt(&"t".repeat(TICKET_MIN_CHARS)),
                _ => {}
            }
            log.request(10.0, vec![json!({ "type": "text", "text": "ok" })], 0);
        }
        let s = loaded(&log);
        let gaps = |regime| timeline(&s, regime).windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>();
        let paused: Vec<usize> =
            gaps(Regime::FiveMinFermat).iter().enumerate().filter(|(_, g)| **g > 10.0).map(|(i, _)| i + 1).collect();
        // Human turns at requests 2, 5, 8, 11 (follow-ups) and 13 (a ticket): the
        // second one pauses, and the ticket.
        assert_eq!(paused, [5, 13]);
        assert!(gaps(Regime::OneHour).iter().all(|&g| g == 10.0));
    }

    #[test]
    fn the_cache_reads_the_longest_live_prefix() {
        let mut cache = Cache { ttl: 300.0, expiry: HashMap::new() };
        let (hs, hit) = cache.lookup(0.0, &[1, 2, 3]);
        assert_eq!(hit, None);
        cache.touch(0.0, &hs, hit, None);
        let (hs, hit) = cache.lookup(100.0, &[1, 2, 3, 4]);
        assert_eq!(hit, Some(2), "the previous request's end");
        cache.touch(100.0, &hs, hit, Some(1));
        assert_eq!(cache.lookup(350.0, &[1, 2, 9]).1, Some(1), "an extra entry where requests diverge");
        assert_eq!(cache.lookup(350.0, &[1, 2, 3]).1, Some(2), "reading refreshed it");
        assert_eq!(cache.lookup(401.0, &[1, 2, 3, 4]).1, None, "expired");
        assert_eq!(cache.lookup(100.0, &[9, 2, 3]).1, None, "keyed by prefix");
    }

    #[test]
    fn the_server_clears_the_oldest_uses_and_thinking_turns() {
        let a = |id: &str, name: &str, thinking: bool| {
            let mut content = vec![json!({ "type": "tool_use", "id": id, "name": name, "input": {} })];
            if thinking {
                content.insert(0, json!({ "type": "thinking", "thinking": "", "signature": "s" }));
            }
            json!({ "role": "assistant", "content": content })
        };
        let r = |id: &str| json!({ "role": "user", "content": [{ "type": "tool_result", "tool_use_id": id, "content": "x" }] });
        // Claude Code's results with a reminder next to them: the server starts a turn there.
        let mixed = |id: &str| {
            json!({ "role": "user", "content": [
            { "type": "tool_result", "tool_use_id": id, "content": "x" }, { "type": "text", "text": "<system-reminder>" }
        ] })
        };
        let body = json!({
            "context_management": { "edits": [
                { "type": CLEAR_THINKING, "keep": { "type": "thinking_turns", "value": 1 } },
                { "type": CLEAR_TOOL_USES, "trigger": { "type": "tool_uses", "value": 2 }, "keep": { "type": "tool_uses", "value": 1 },
                  "exclude_tools": ["TodoWrite"], "clear_tool_inputs": true }
            ] },
            "messages": [
                json!({ "role": "user", "content": "go" }),
                a("u1", "Bash", true), r("u1"),
                a("todo", "TodoWrite", false), mixed("todo"),
                a("u2", "Read", true), a("u3", "Grep", true), r("u2"), r("u3"),
                json!({ "role": "system", "content": "notice" }),
                json!({ "role": "user", "content": [{ "type": "text", "text": "next" }] }),
                a("u4", "Bash", true), r("u4"),
            ]
        });
        let c = server_clears(&body);
        let mut ids: Vec<&str> = c.ids.iter().map(String::as_str).collect();
        ids.sort();
        // Three turns (the prompt, the reminder, the next prompt), with or without thinking; one kept.
        assert_eq!((ids, c.tool_uses, c.inputs, c.thinking_turns), (vec!["u1", "u2", "u3"], 3, true, 2));
        let mut turns = ThinkingTurns::default();
        let numbered: Vec<Option<u64>> = body["messages"].as_array().unwrap().iter().map(|m| turns.next(m)).collect();
        assert_eq!(
            numbered,
            [None, Some(1), None, Some(1), None, Some(2), Some(2), None, None, None, None, Some(3), None]
        );
        // Without clear_tool_inputs, 1-character results don't count toward the trigger.
        let mut results_only = body.clone();
        results_only["context_management"]["edits"][1]["clear_tool_inputs"] = json!(false);
        assert!(server_clears(&results_only).ids.is_empty());
        // Under the trigger nothing is cleared.
        let mut few = body.clone();
        few["context_management"]["edits"][1]["trigger"]["value"] = json!(4);
        assert!(server_clears(&few).ids.is_empty());
    }

    #[test]
    fn pinned_edits_advance_at_cold_moments_and_keep_the_cache() {
        let mut s = loaded(&session_log(5, 10, 20.0));
        prepare(&mut s, &FLAT);
        let cfg = cfg(Regime::FiveMinFermat);
        let base = simulate(&s, &FLAT, &cfg, false);
        let cm = simulate(&s, &FLAT, &cfg, true);
        assert_eq!(base.advances, 0);
        assert!(base.cold >= 2, "Fermat pauses make cold moments: {}", base.cold);
        assert!(cm.advances >= 2 && cm.advances <= base.cold, "{} advances", cm.advances);
        assert_eq!(cm.breakers, Vec::<&str>::new(), "warm requests kept reading the cache");
        assert!(cm.bill.cache_read < 0.7 * base.bill.cache_read);
        assert!(cm.bill.cache_write < base.bill.cache_write);
        assert_eq!((cm.bill.input, cm.bill.output), (base.bill.input, base.bill.output));

        // Reading cleared results back, or a bigger placeholder, gives some of it back.
        let reread = simulate(&s, &FLAT, &SimConfig { reread: 0.25, ..cfg }, true);
        let bigger = simulate(&s, &FLAT, &SimConfig { placeholder: 400.0, ..cfg }, true);
        for other in [reread, bigger] {
            assert!(other.bill.cache_read > cm.bill.cache_read && other.bill.cache_read < base.bill.cache_read);
        }
        // Thinking counts too.
        let keep_thinking = SimConfig { ctx: CtxConfig { clear_thinking: false, ..cfg.ctx }, ..cfg };
        assert!(simulate(&s, &FLAT, &keep_thinking, true).bill.cache_read > cm.bill.cache_read);
        // Without the beta there is nothing to merge into.
        let mut off = loaded(&session_log(5, 10, 20.0));
        off.calls.iter_mut().for_each(|c| c.betas = "claude-code-20250219".into());
        prepare(&mut off, &FLAT);
        assert_eq!(simulate(&off, &FLAT, &cfg, true).advances, 0);
    }

    #[test]
    fn a_pause_just_past_the_ttl_trips_no_breaker() {
        let mut s = loaded(&session_log(5, 10, 20.0));
        // Mid-turn, after the advance at the third turn's Fermat pause, 303 s
        // pass: the cache may or may not have expired.
        assert!(s.calls[22].human && !s.calls[25].human);
        s.calls[25..].iter_mut().for_each(|c| c.t += 283.0);
        prepare(&mut s, &FLAT);
        let cm = simulate(&s, &FLAT, &cfg(Regime::FiveMinFermat), true);
        assert!(cm.advances >= 1);
        assert_eq!(cm.breakers, Vec::<&str>::new());
    }

    #[test]
    fn refills_unchanged_reads_as_the_proxy_does() {
        // A Read, five other calls, Claude Code's stand-in for the unchanged
        // file, and three more prompts: under a 1-hour cache nothing is ever
        // cleared, and the refill is pure cost.
        let mut log = Log::new();
        let file = "let x = 1;\n".repeat(1000);
        let stand_in = "File unchanged since last read. Refer to that earlier tool_result instead.";
        let call = |id: &str, name: &str| {
            vec![json!({ "type": "tool_use", "id": id, "name": name, "input": { "file_path": "/w/a.rs" } })]
        };
        let result = |id: &str, text: &str| vec![json!({ "type": "tool_result", "tool_use_id": id, "content": text })];
        log.prompt("look");
        log.request(20.0, call("r1", "Read"), 0);
        log.results(result("r1", &file), false);
        for i in 0..5 {
            log.request(20.0, call(&format!("b{i}"), "Bash"), 0);
            log.results(result(&format!("b{i}"), "x"), false);
        }
        log.request(20.0, call("r2", "Read"), 0);
        log.results(result("r2", stand_in), false);
        for turn in 0..4 {
            log.request(20.0, vec![json!({ "type": "text", "text": "ok" })], 0);
            log.prompt(&format!("next {turn}"));
        }
        let mut s = loaded(&log);
        prepare(&mut s, &FLAT);
        let cfg = cfg(Regime::OneHour);
        let (base, cm) = (simulate(&s, &FLAT, &cfg, false), simulate(&s, &FLAT, &cfg, true));
        assert_eq!(cm.refilled, HashSet::from(["r2".to_string()]));
        assert_eq!(cm.advances, 0);
        let added = (file.len() - stand_in.len()) as f64 / 4.0;
        assert_eq!(cm.bill.cache_write - base.bill.cache_write, added, "written once");
        assert_eq!(cm.bill.cache_read - base.bill.cache_read, 3.0 * added, "then read by each later request");
    }

    #[test]
    fn a_one_hour_cache_with_short_gaps_never_advances() {
        let mut s = loaded(&session_log(5, 10, 20.0));
        prepare(&mut s, &FLAT);
        let cfg = cfg(Regime::OneHour);
        let (base, cm) = (simulate(&s, &FLAT, &cfg, false), simulate(&s, &FLAT, &cfg, true));
        assert_eq!((base.cold, cm.advances), (0, 0));
        assert_eq!((cm.bill.cache_read, cm.bill.cache_write), (base.bill.cache_read, base.bill.cache_write));
    }

    #[test]
    fn simulates_transcript_files() {
        let dir = tempfile::tempdir().unwrap();
        let files = vec![
            session_log(4, 10, 20.0).write(dir.path(), "a.jsonl"),
            session_log(3, 14, 15.0).write(dir.path(), "b.jsonl"),
            {
                let old = dir.path().join("old.jsonl");
                std::fs::write(&old, "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"hi\"}}\n")
                    .unwrap();
                old
            },
        ];
        let sim = simulate_files(&files, &SimConfig { read_price: None, ..cfg(Regime::FiveMinFermat) }).unwrap();
        assert_eq!((sim.sessions.len(), sim.skipped), (2, 1));
        for s in &sim.sessions {
            assert_eq!(s.read_price, 0.05, "Opus 5.5's");
            assert!(s.fit_error < 0.05, "{}", s.fit_error);
            assert!(s.ratio < 1.0 && s.advances > 0);
            assert!((s.ratio - s.context_edits.cost / s.unchanged.cost).abs() < 1e-12);
        }
        let mean = sim.sessions.iter().map(|s| s.ratio).sum::<f64>() / 2.0;
        assert!((sim.mean_ratio - mean).abs() < 1e-12);
        assert!(simulate_files(&files[2..], &cfg(Regime::OneHour)).is_err(), "no request logs");
    }
}
