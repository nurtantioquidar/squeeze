//! Reads token usage out of a response as it streams past, without buffering
//! or altering the stream (buffering would stall Claude Code).

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub cache_creation_1h: u64,
    pub cache_creation_5m: u64,
}

impl Usage {
    fn absorb(&mut self, u: &Value) {
        let get = |k: &str| u.get(k).and_then(Value::as_u64);
        if let Some(v) = get("input_tokens") {
            self.input_tokens = v;
        }
        if let Some(v) = get("output_tokens") {
            self.output_tokens = v;
        }
        if let Some(v) = get("cache_creation_input_tokens") {
            self.cache_creation_input_tokens = v;
        }
        if let Some(v) = get("cache_read_input_tokens") {
            self.cache_read_input_tokens = v;
        }
        if let Some(cc) = u.get("cache_creation") {
            if let Some(v) = cc.get("ephemeral_1h_input_tokens").and_then(Value::as_u64) {
                self.cache_creation_1h = v;
            }
            if let Some(v) = cc.get("ephemeral_5m_input_tokens").and_then(Value::as_u64) {
                self.cache_creation_5m = v;
            }
        }
    }

    /// Input-side cost in "base input token" units: cache writes cost 1.25x
    /// (5-minute TTL) or 2x (1-hour TTL), cache reads 0.1x. Lets savings be
    /// expressed as a share of cost without hard-coding per-model prices.
    pub fn input_equivalent(&self) -> f64 {
        let (w1h, w5m) = if self.cache_creation_1h + self.cache_creation_5m > 0 {
            (self.cache_creation_1h, self.cache_creation_5m)
        } else {
            (0, self.cache_creation_input_tokens)
        };
        self.input_tokens as f64 + 2.0 * w1h as f64 + 1.25 * w5m as f64 + 0.1 * self.cache_read_input_tokens as f64
    }

    pub fn add(&mut self, o: &Usage) {
        self.input_tokens += o.input_tokens;
        self.output_tokens += o.output_tokens;
        self.cache_creation_input_tokens += o.cache_creation_input_tokens;
        self.cache_read_input_tokens += o.cache_read_input_tokens;
        self.cache_creation_1h += o.cache_creation_1h;
        self.cache_creation_5m += o.cache_creation_5m;
    }
}

const MAX_JSON_BUFFER: usize = 8 * 1024 * 1024;

/// Incremental usage parser for SSE (`text/event-stream`) or JSON bodies.
pub struct UsageTap {
    sse: bool,
    buf: Vec<u8>,
    usage: Usage,
    seen: bool,
}

impl UsageTap {
    pub fn new(sse: bool) -> Self {
        Self { sse, buf: Vec::new(), usage: Usage::default(), seen: false }
    }

    pub fn feed(&mut self, chunk: &[u8]) {
        if !self.sse {
            if self.buf.len() + chunk.len() <= MAX_JSON_BUFFER {
                self.buf.extend_from_slice(chunk);
            }
            return;
        }
        self.buf.extend_from_slice(chunk);
        let mut start = 0;
        while let Some(pos) = self.buf[start..].iter().position(|b| *b == b'\n') {
            let end = start + pos;
            let line = self.buf[start..end].to_vec();
            self.line(&line);
            start = end + 1;
        }
        self.buf.drain(..start);
    }

    fn line(&mut self, line: &[u8]) {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(data) = line.strip_prefix(b"data:") else { return };
        let Ok(event) = serde_json::from_slice::<Value>(data) else { return };
        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                if let Some(msg) = event.get("message") {
                    self.message(msg);
                }
            }
            Some("message_delta") => {
                if let Some(u) = event.get("usage") {
                    self.usage.absorb(u);
                    self.seen = true;
                }
            }
            _ => {}
        }
    }

    fn message(&mut self, msg: &Value) {
        if let Some(m) = msg.get("model").and_then(Value::as_str) {
            self.usage.model = Some(m.to_string());
        }
        if let Some(u) = msg.get("usage") {
            self.usage.absorb(u);
            self.seen = true;
        }
    }

    pub fn finish(mut self) -> Option<Usage> {
        if self.sse {
            let rest = std::mem::take(&mut self.buf);
            self.line(&rest);
        } else if let Ok(v) = serde_json::from_slice::<Value>(&self.buf) {
            self.message(&v);
        }
        self.seen.then_some(self.usage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STREAM: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-x\",\"usage\":{\"input_tokens\":12,\"cache_creation_input_tokens\":3000,\"cache_read_input_tokens\":40000,\"cache_creation\":{\"ephemeral_1h_input_tokens\":3000,\"ephemeral_5m_input_tokens\":0},\"output_tokens\":1}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":87}}\n\n",
    );

    #[test]
    fn parses_sse_across_arbitrary_chunk_boundaries() {
        for split in [1, 7, 64, 333] {
            let mut tap = UsageTap::new(true);
            for chunk in STREAM.as_bytes().chunks(split) {
                tap.feed(chunk);
            }
            let u = tap.finish().expect("usage");
            assert_eq!(u.model.as_deref(), Some("claude-x"));
            assert_eq!(u.input_tokens, 12);
            assert_eq!(u.output_tokens, 87);
            assert_eq!(u.cache_read_input_tokens, 40000);
            assert_eq!(u.cache_creation_1h, 3000);
        }
    }

    #[test]
    fn parses_json_body() {
        let mut tap = UsageTap::new(false);
        tap.feed(br#"{"model":"m","usage":{"input_tokens":5,"output_tokens":9}}"#);
        let u = tap.finish().unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (5, 9));
    }

    #[test]
    fn input_equivalent_weights_cache() {
        let u = Usage { input_tokens: 10, cache_creation_1h: 100, cache_read_input_tokens: 1000, ..Default::default() };
        assert!((u.input_equivalent() - (10.0 + 200.0 + 100.0)).abs() < 1e-9);
    }

    #[test]
    fn no_usage_no_result() {
        let mut tap = UsageTap::new(true);
        tap.feed(b"data: {\"type\":\"ping\"}\n\n");
        assert!(tap.finish().is_none());
    }
}
