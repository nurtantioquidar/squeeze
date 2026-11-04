use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Rough token estimate. Real tokenization is model-specific and not available
/// locally; ~4 bytes per token under-counts code and logs, so savings figures
/// derived from it lean conservative.
pub fn approx_tokens(bytes: usize) -> f64 {
    bytes as f64 / 4.0
}

/// Price of a cache read as a multiple of the base input price: Opus 5.5 and
/// Sonnet 5.5 bill cache hits at 0.05x, earlier models at 0.1x.
pub fn read_price(model: &str) -> f64 {
    if model.starts_with("claude-opus-5-5") || model.starts_with("claude-sonnet-5-5") { 0.05 } else { 0.1 }
}

/// Stable short hash of several string parts.
pub fn short_hash(parts: &[&str]) -> String {
    let mut h = blake3::Hasher::new();
    for p in parts {
        h.update(p.as_bytes());
        h.update(&[0]);
    }
    h.finalize().to_hex()[..16].to_string()
}

pub fn truncate_chars(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        None => s.to_string(),
        Some((idx, _)) => format!("{}…", &s[..idx]),
    }
}

pub fn home_dir() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

pub fn is_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8192).any(|b| *b == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_on_char_boundaries() {
        assert_eq!(truncate_chars("héllo", 2), "hé…");
        assert_eq!(truncate_chars("hi", 5), "hi");
    }

    #[test]
    fn read_price_by_model() {
        assert_eq!(read_price("claude-opus-5-5"), 0.05);
        assert_eq!(read_price("claude-sonnet-5-5-20261001"), 0.05);
        assert_eq!(read_price("claude-haiku-5-5"), 0.1);
        assert_eq!(read_price("claude-opus-5"), 0.1);
        assert_eq!(read_price(""), 0.1);
    }

    #[test]
    fn short_hash_separates_parts() {
        assert_ne!(short_hash(&["ab", "c"]), short_hash(&["a", "bc"]));
    }
}
