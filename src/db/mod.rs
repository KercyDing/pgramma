mod connection;

pub use connection::PgramDb;

use redb::{ReadableDatabase, ReadableTable};

use connection::{CONFIG_TABLE, COUNTERS_TABLE, ENGRAMS_TABLE, EPISODES_TABLE};

use crate::error::{PgrammaError, Result};
use crate::models::{Emotion, Engram, EpisodicEntry, Role};

fn encode<T: bincode::Encode>(val: &T) -> Result<Vec<u8>> {
    bincode::encode_to_vec(val, bincode::config::standard())
        .map_err(|e| PgrammaError::Serialization(e.to_string()))
}

fn decode<T: bincode::Decode<()>>(bytes: &[u8]) -> Result<T> {
    bincode::decode_from_slice(bytes, bincode::config::standard())
        .map(|(val, _)| val)
        .map_err(|e| PgrammaError::Serialization(e.to_string()))
}

// ── internal helpers ─────────────────────────────────────────────

impl PgramDb {
    /// Atomically increment and return the next ID for a table.
    fn next_id(&self, table_name: &str) -> Result<u64> {
        let txn = self.db.begin_write()?;
        let next;
        {
            let mut counters = txn.open_table(COUNTERS_TABLE)?;
            let current = counters
                .get(table_name)?
                .map(|v: redb::AccessGuard<'_, u64>| v.value())
                .unwrap_or(0);
            next = current + 1;
            counters.insert(table_name, next)?;
        }
        txn.commit()?;
        Ok(next)
    }
}

// ── persona_config CRUD ──────────────────────────────────────────

impl PgramDb {
    /// Upsert a key-value pair in persona_config.
    pub fn set_config(&self, key: &str, value: &str) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(CONFIG_TABLE)?;
            table.insert(key, value)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Get a config value by key. Returns `ConfigNotFound` if missing.
    pub fn get_config(&self, key: &str) -> Result<String> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(CONFIG_TABLE)?;
        table
            .get(key)?
            .map(|v: redb::AccessGuard<'_, &str>| v.value().to_owned())
            .ok_or_else(|| PgrammaError::ConfigNotFound(key.to_owned()))
    }
}

// ── episodic_memory CRUD ─────────────────────────────────────────

impl PgramDb {
    /// Append a message to the episodic timeline.
    pub fn append_episode(&self, role: Role, content: &str) -> Result<i64> {
        let id = self.next_id("episodes")?;
        let entry = EpisodicEntry {
            id: id as i64,
            role,
            content: content.to_owned(),
            timestamp: chrono::Utc::now()
                .format("%Y-%m-%dT%H:%M:%S%.3fZ")
                .to_string(),
        };
        let bytes = encode(&entry)?;
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(EPISODES_TABLE)?;
            table.insert(id, bytes.as_slice())?;
        }
        txn.commit()?;
        Ok(id as i64)
    }

    /// Retrieve recent episodes, newest first.
    pub fn get_episodes(&self, limit: i64, offset: i64) -> Result<Vec<EpisodicEntry>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(EPISODES_TABLE)?;
        let mut results = Vec::new();
        let mut skipped = 0i64;

        for item in table.iter()?.rev() {
            let item = item?;
            let val = item.1;
            if skipped < offset {
                skipped += 1;
                continue;
            }
            let entry: EpisodicEntry = decode(val.value())?;
            results.push(entry);
            if results.len() as i64 >= limit {
                break;
            }
        }
        Ok(results)
    }
}

// ── engrams CRUD ─────────────────────────────────────────────────

impl PgramDb {
    /// Insert a new engram. Returns the new row id.
    pub fn insert_engram(
        &self,
        content: &str,
        emotion: Emotion,
        importance: f32,
        embedding: Option<&[f32]>,
    ) -> Result<i64> {
        let id = self.next_id("engrams")?;
        let engram = Engram {
            id: id as i64,
            content: content.to_owned(),
            embedding: embedding.map(|e| e.to_vec()),
            emotion,
            importance,
            access_count: 0,
            last_accessed: None,
            created_at: chrono::Utc::now()
                .format("%Y-%m-%dT%H:%M:%S%.3fZ")
                .to_string(),
        };
        let bytes = encode(&engram)?;
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(ENGRAMS_TABLE)?;
            table.insert(id, bytes.as_slice())?;
        }
        txn.commit()?;
        Ok(id as i64)
    }

    /// Fetch all engrams with importance >= threshold.
    pub fn get_engrams_above(&self, threshold: f32) -> Result<Vec<Engram>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(ENGRAMS_TABLE)?;
        let mut results = Vec::new();
        for item in table.iter()? {
            let item = item?;
            let val = item.1;
            let engram: Engram = decode(val.value())?;
            if engram.importance >= threshold {
                results.push(engram);
            }
        }
        results.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        Ok(results)
    }

    /// Bump access_count and refresh last_accessed timestamp.
    pub fn touch_engram(&self, id: i64) -> Result<()> {
        let key = id as u64;
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(ENGRAMS_TABLE)?;
            // Read and decode first, dropping the AccessGuard before writing
            let updated = {
                let existing = table.get(key)?;
                match existing {
                    Some(guard) => {
                        let mut engram: Engram = decode(guard.value())?;
                        engram.access_count += 1;
                        engram.last_accessed = Some(
                            chrono::Utc::now()
                                .format("%Y-%m-%dT%H:%M:%S%.3fZ")
                                .to_string(),
                        );
                        Some(encode(&engram)?)
                    }
                    None => None,
                }
            };
            if let Some(bytes) = updated {
                table.insert(key, bytes.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Delete an engram by id. Returns true if a row was actually removed.
    pub fn delete_engram(&self, id: i64) -> Result<bool> {
        let key = id as u64;
        let txn = self.db.begin_write()?;
        let removed;
        {
            let mut table = txn.open_table(ENGRAMS_TABLE)?;
            removed = table.remove(key)?.is_some();
        }
        txn.commit()?;
        Ok(removed)
    }

    /// Get all engrams (for batch embedding backfill).
    pub fn get_all_engrams(&self) -> Result<Vec<Engram>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(ENGRAMS_TABLE)?;
        let mut results = Vec::new();
        for item in table.iter()? {
            let item = item?;
            let engram: Engram = decode(item.1.value())?;
            results.push(engram);
        }
        Ok(results)
    }

    /// Update embedding vector for an existing engram.
    pub fn update_engram_embedding(&self, id: i64, embedding: &[f32]) -> Result<()> {
        let key = id as u64;
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(ENGRAMS_TABLE)?;
            let updated = {
                let existing = table.get(key)?;
                match existing {
                    Some(guard) => {
                        let mut engram: Engram = decode(guard.value())?;
                        engram.embedding = Some(embedding.to_vec());
                        Some(encode(&engram)?)
                    }
                    None => None,
                }
            };
            if let Some(bytes) = updated {
                table.insert(key, bytes.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Update importance value for an existing engram.
    pub fn update_engram_importance(&self, id: i64, importance: f32) -> Result<()> {
        let key = id as u64;
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(ENGRAMS_TABLE)?;
            let updated = {
                let existing = table.get(key)?;
                match existing {
                    Some(guard) => {
                        let mut engram: Engram = decode(guard.value())?;
                        engram.importance = importance;
                        Some(encode(&engram)?)
                    }
                    None => None,
                }
            };
            if let Some(bytes) = updated {
                table.insert(key, bytes.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Get the most recently inserted engram, if any.
    pub fn get_latest_engram(&self) -> Result<Option<Engram>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(ENGRAMS_TABLE)?;
        match table.iter()?.next_back() {
            Some(item) => {
                let item = item?;
                let engram: Engram = decode(item.1.value())?;
                Ok(Some(engram))
            }
            None => Ok(None),
        }
    }
}

// ── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn setup() -> PgramDb {
        let file = NamedTempFile::new().expect("tempfile");
        PgramDb::open(file.path()).expect("open db")
    }

    #[test]
    fn config_round_trip() {
        let db = setup();
        db.set_config("system_prompt", "You are helpful.").unwrap();
        assert_eq!(db.get_config("system_prompt").unwrap(), "You are helpful.");

        db.set_config("system_prompt", "Be concise.").unwrap();
        assert_eq!(db.get_config("system_prompt").unwrap(), "Be concise.");
    }

    #[test]
    fn config_not_found() {
        let db = setup();
        let err = db.get_config("nonexistent").unwrap_err();
        assert!(matches!(err, PgrammaError::ConfigNotFound(_)));
    }

    #[test]
    fn episode_append_and_query() {
        let db = setup();
        db.append_episode(Role::System, "You are an assistant.")
            .unwrap();
        db.append_episode(Role::User, "Hello!").unwrap();
        db.append_episode(Role::Assistant, "Hi there!").unwrap();

        let eps = db.get_episodes(10, 0).unwrap();
        assert_eq!(eps.len(), 3);
        assert_eq!(eps[0].role, Role::Assistant);
        assert_eq!(eps[2].role, Role::System);
    }

    #[test]
    fn engram_crud() {
        let db = setup();

        let id = db
            .insert_engram("Rust is great", Emotion::Joy, 0.8, None)
            .unwrap();
        assert_eq!(id, 1);

        let emb = vec![0.1_f32, 0.2, 0.3];
        let id2 = db
            .insert_engram("Memory test", Emotion::Trust, 0.5, Some(&emb))
            .unwrap();
        assert_eq!(id2, 2);

        // query above threshold
        let results = db.get_engrams_above(0.6).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "Rust is great");

        // touch
        db.touch_engram(id).unwrap();
        let updated = db.get_engrams_above(0.6).unwrap();
        assert_eq!(updated[0].access_count, 1);
        assert!(updated[0].last_accessed.is_some());

        // embedding round-trip
        let all = db.get_engrams_above(0.0).unwrap();
        let e2 = all.iter().find(|e| e.id == id2).unwrap();
        assert_eq!(e2.embedding.as_ref().unwrap(), &emb);

        // delete
        assert!(db.delete_engram(id).unwrap());
        assert!(!db.delete_engram(999).unwrap());
        assert_eq!(db.get_engrams_above(0.0).unwrap().len(), 1);

        db.update_engram_importance(id2, 0.9).unwrap();
        let upgraded = db.get_engrams_above(0.8).unwrap();
        assert_eq!(upgraded.len(), 1);
        assert_eq!(upgraded[0].id, id2);
    }
}
