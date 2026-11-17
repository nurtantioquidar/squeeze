//! Local state under `~/.squeeze` (or `$SQUEEZE_HOME`): originals of
//! compressed outputs, a JSONL log of per-request stats, and per-thread
//! proxy state. Nothing here ever leaves the machine.

use crate::transform::BlobSink;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

#[derive(Clone, Debug)]
pub struct Store {
    root: PathBuf,
}

/// What the proxy keeps per conversation thread, across restarts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ThreadState {
    /// When the thread's latest request arrived (unix seconds): the prompt
    /// cache's TTL runs from there.
    #[serde(alias = "last_seen")]
    pub last_start: Option<u64>,
    /// When the thread's latest response finished.
    pub last_stop: Option<u64>,
    /// Prune boundary (`--prune-cold-cache`).
    pub boundary: Option<String>,
    /// Server-side context editing, once the thread uses it.
    pub ctx: Option<CtxState>,
    /// Compaction at pauses (`--compact-pauses`).
    pub compact: crate::compact::CompactState,
}

/// Server-side context editing pinned at cold-cache moments: the proxy asks the
/// API to clear the same `cleared` tool uses and `cleared_th` thinking turns on
/// every request, so the cleared view stays a stable, cacheable prefix.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CtxState {
    /// Clearable tool uses cleared so far, oldest first.
    pub cleared: u64,
    /// Id of the last cleared tool use, to notice a history that no longer lines up.
    pub cleared_upto_id: Option<String>,
    /// Thinking turns cleared so far.
    pub cleared_th: u64,
    /// Tool inputs are cleared along with results.
    pub clear_inputs: bool,
    /// The client sends its own `clear_tool_uses`, so the proxy stays out.
    pub client_managed: bool,
    /// Why advancing stopped (a breaker tripped), if it did.
    pub frozen: Option<String>,
    /// Times a warm request lost the cache it should have read (`cache_regressed`).
    pub regressions: u64,
    /// Prompt tokens of the thread's latest edited response, which the next
    /// warm one should mostly read from cache.
    pub last_prompt: u64,
    /// The model that response came from: the prompt cache is per model.
    pub model: Option<String>,
    /// Cold moments at which `cleared` moved forward.
    pub advances: u64,
}

impl Store {
    pub fn default_root() -> PathBuf {
        match std::env::var_os("SQUEEZE_HOME") {
            Some(h) if !h.is_empty() => PathBuf::from(h),
            _ => crate::util::home_dir().join(".squeeze"),
        }
    }

    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        for sub in ["blobs", "threads", ANCHORS, "logs"] {
            fs::create_dir_all(root.join(sub)).with_context(|| format!("creating {}", root.join(sub).display()))?;
        }
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn blob_id(text: &str) -> String {
        blake3::hash(text.as_bytes()).to_hex()[..24].to_string()
    }

    /// Content-addressed, so storing the same original again is a no-op.
    pub fn put_blob(&self, text: &str) -> String {
        let id = Self::blob_id(text);
        let path = self.root.join("blobs").join(&id);
        if !path.exists() {
            let tmp = path.with_extension(format!("tmp{}", std::process::id()));
            let res = fs::write(&tmp, text).and_then(|_| fs::rename(&tmp, &path));
            if let Err(e) = res {
                tracing::warn!("could not store original {id}: {e}");
            }
        }
        id
    }

    pub fn get_blob(&self, id: &str) -> Result<String> {
        let id = id.trim();
        if id.len() < 8 || id.len() > 64 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
            bail!("invalid id {id:?}: expected the hex id from a [squeeze: ...] note");
        }
        let path = self.root.join("blobs").join(id.to_ascii_lowercase());
        fs::read_to_string(&path)
            .with_context(|| format!("no stored output with id {id} (it may have been garbage-collected)"))
    }

    pub fn append_record(&self, rec: &Value) -> Result<()> {
        let mut f = OpenOptions::new().create(true).append(true).open(self.root.join("stats.jsonl"))?;
        let mut line = serde_json::to_vec(rec)?;
        line.push(b'\n');
        f.write_all(&line)?;
        Ok(())
    }

    pub fn records(&self) -> Result<Vec<Value>> {
        let path = self.root.join("stats.jsonl");
        if !path.exists() {
            return Ok(Vec::new());
        }
        let reader = BufReader::new(fs::File::open(path)?);
        Ok(reader.lines().map_while(|l| l.ok()).filter_map(|l| serde_json::from_str(&l).ok()).collect())
    }

    fn thread_path(&self, key: &str) -> PathBuf {
        self.root.join("threads").join(format!("{key}.json"))
    }

    pub fn load_thread(&self, key: &str) -> Option<ThreadState> {
        serde_json::from_slice(&fs::read(self.thread_path(key)).ok()?).ok()
    }

    /// Saved before the request is forwarded, and durably: a crash or a
    /// concurrent reader sees the old state or the new one, never a torn file.
    pub fn save_thread(&self, key: &str, state: &ThreadState) {
        let res = serde_json::to_vec(state)
            .map_err(std::io::Error::other)
            .and_then(|b| write_durably(&self.thread_path(key), &b));
        if let Err(e) = res {
            tracing::warn!("could not save thread state: {e}");
        }
    }

    /// Index entry for a thread's first tool_use id. Ids come from requests,
    /// so anything but a plain token is never used as a file name.
    fn anchor_path(&self, anchor: &str) -> Option<PathBuf> {
        let plain = anchor.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
        ((1..=128).contains(&anchor.len()) && plain).then(|| self.root.join(ANCHORS).join(anchor))
    }

    /// Key of the thread that first sent `anchor` as its first tool_use id.
    pub fn thread_by_anchor(&self, anchor: &str) -> Option<String> {
        let key = fs::read_to_string(self.anchor_path(anchor)?).ok()?;
        (!key.is_empty()).then_some(key)
    }

    /// Record `key` as the thread for `anchor`, unless one already is.
    pub fn link_anchor(&self, anchor: &str, key: &str) {
        let Some(path) = self.anchor_path(anchor) else { return };
        if !path.exists()
            && let Err(e) = write_durably(&path, key.as_bytes())
        {
            tracing::warn!("could not index thread {key}: {e}");
        }
    }

    /// Delete stored originals and thread state older than `max_age`.
    pub fn gc(&self, max_age: Duration) -> Result<(usize, u64)> {
        let cutoff = SystemTime::now() - max_age;
        let (mut files, mut bytes) = (0, 0);
        for sub in ["blobs", "threads", ANCHORS] {
            for entry in fs::read_dir(self.root.join(sub))? {
                let entry = entry?;
                let meta = entry.metadata()?;
                if meta.is_file() && meta.modified().is_ok_and(|m| m < cutoff) {
                    fs::remove_file(entry.path())?;
                    files += 1;
                    bytes += meta.len();
                }
            }
        }
        Ok((files, bytes))
    }
}

/// The by-anchor thread index: `threads/by-anchor/<first tool_use id>` holds
/// the key of the thread that id belongs to.
const ANCHORS: &str = "threads/by-anchor";

/// Write `bytes` to `path` so that it survives a crash whole: a temporary file
/// flushed to disk, renamed over `path`, then the directory flushed.
fn write_durably(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    let res = (|| {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        fs::rename(&tmp, path)?;
        match path.parent() {
            Some(dir) => fs::File::open(dir)?.sync_all(),
            None => Ok(()),
        }
    })();
    if res.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    res
}

impl BlobSink for Store {
    fn put(&mut self, text: &str) -> String {
        self.put_blob(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_roundtrip_and_validation() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let id = store.put_blob("hello\nworld");
        assert_eq!(id, store.put_blob("hello\nworld"));
        assert_eq!(store.get_blob(&id).unwrap(), "hello\nworld");
        assert!(store.get_blob("../../etc/passwd").is_err());
        assert!(store.get_blob("abc").is_err());
    }

    #[test]
    fn records_and_threads() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        store.append_record(&serde_json::json!({ "a": 1 })).unwrap();
        store.append_record(&serde_json::json!({ "a": 2 })).unwrap();
        assert_eq!(store.records().unwrap().len(), 2);
        let state = ThreadState {
            last_start: Some(5),
            last_stop: Some(9),
            boundary: Some("t1".into()),
            ctx: Some(CtxState {
                cleared: 10,
                cleared_upto_id: Some("toolu_10".into()),
                frozen: Some("ctx_mismatch".into()),
                model: Some("claude-opus-5-5".into()),
                ..Default::default()
            }),
            compact: crate::compact::CompactState {
                pending: Some(crate::compact::Summary { covered: 7, prefix_hash: "h".into(), text: "summary".into() }),
                compactions: 1,
                ..Default::default()
            },
        };
        store.save_thread("k", &state);
        assert_eq!(store.load_thread("k"), Some(state));
        // save_thread only logs failures; the durable write itself must succeed here.
        write_durably(&dir.path().join("threads/k.json"), b"{}").unwrap();
        assert_eq!(store.load_thread("k"), Some(ThreadState::default()));
        let leftovers: Vec<_> = fs::read_dir(dir.path().join("threads"))
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().contains("tmp"))
            .collect();
        assert!(leftovers.is_empty(), "the temporary file is renamed into place");
    }

    #[test]
    fn older_thread_files_still_load() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        fs::write(dir.path().join("threads/old.json"), r#"{"last_seen":7,"boundary":"t3"}"#).unwrap();
        let t = store.load_thread("old").unwrap();
        assert_eq!((t.last_start, t.last_stop, t.boundary.as_deref(), t.ctx), (Some(7), None, Some("t3"), None));
        // A field this build doesn't know is ignored.
        fs::write(dir.path().join("threads/old.json"), r#"{"ctx":{"cleared":3,"journal":{"toolu_1":"ab12"}}}"#)
            .unwrap();
        assert_eq!(store.load_thread("old").unwrap().ctx.map(|c| c.cleared), Some(3));
    }

    #[test]
    fn anchor_index_keeps_the_first_thread_and_refuses_odd_ids() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        assert_eq!(store.thread_by_anchor("toolu_01abc"), None);
        store.link_anchor("toolu_01abc", "key1");
        store.link_anchor("toolu_01abc", "key2");
        assert_eq!(store.thread_by_anchor("toolu_01abc").as_deref(), Some("key1"));
        for odd in ["../x", "a/b", "", "toolu.1", &"x".repeat(200)] {
            store.link_anchor(odd, "key3");
            assert_eq!(store.thread_by_anchor(odd), None, "{odd:?}");
        }
        assert!(!dir.path().join("threads/x").exists());
        // gc reaches the index too.
        let entry = dir.path().join(ANCHORS).join("toolu_01abc");
        fs::File::options().write(true).open(entry).unwrap().set_modified(SystemTime::UNIX_EPOCH).unwrap();
        assert_eq!(store.gc(Duration::from_secs(3600)).unwrap().0, 1);
        assert_eq!(store.thread_by_anchor("toolu_01abc"), None);
    }
}
