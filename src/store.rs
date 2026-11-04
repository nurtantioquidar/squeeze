//! Local state under `~/.squeeze` (or `$SQUEEZE_HOME`): originals of
//! compressed outputs, a JSONL log of per-request stats, and per-thread
//! prune state. Nothing here ever leaves the machine.

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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreadState {
    pub last_seen: Option<u64>,
    pub boundary: Option<String>,
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
        for sub in ["blobs", "threads", "logs"] {
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

    pub fn save_thread(&self, key: &str, state: &ThreadState) {
        let res =
            serde_json::to_vec(state).map_err(std::io::Error::other).and_then(|b| fs::write(self.thread_path(key), b));
        if let Err(e) = res {
            tracing::warn!("could not save thread state: {e}");
        }
    }

    /// Delete stored originals and thread state older than `max_age`.
    pub fn gc(&self, max_age: Duration) -> Result<(usize, u64)> {
        let cutoff = SystemTime::now() - max_age;
        let (mut files, mut bytes) = (0, 0);
        for sub in ["blobs", "threads"] {
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
        store.save_thread("k", &ThreadState { last_seen: Some(5), boundary: Some("t1".into()) });
        let t = store.load_thread("k").unwrap();
        assert_eq!((t.last_seen, t.boundary.as_deref()), (Some(5), Some("t1")));
    }
}
