use std::path::Path;

use redb::Database;

use crate::error::Result;

/// Table definitions for the .pgram file.
pub const CONFIG_TABLE: redb::TableDefinition<&str, &str> =
    redb::TableDefinition::new("persona_config");

pub const EPISODES_TABLE: redb::TableDefinition<u64, &[u8]> =
    redb::TableDefinition::new("episodic_memory");

pub const ENGRAMS_TABLE: redb::TableDefinition<u64, &[u8]> = redb::TableDefinition::new("engrams");

pub const COUNTERS_TABLE: redb::TableDefinition<&str, u64> = redb::TableDefinition::new("counters");

/// Handle to a Pgramma redb database.
pub struct PgramDb {
    pub(crate) db: Database,
}

impl PgramDb {
    /// Open (or create) a .pgram file at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let db = Database::create(path)?;
        // Create all tables eagerly so reads never hit "table not found"
        let txn = db.begin_write()?;
        {
            txn.open_table(CONFIG_TABLE)?;
            txn.open_table(EPISODES_TABLE)?;
            txn.open_table(ENGRAMS_TABLE)?;
            txn.open_table(COUNTERS_TABLE)?;
        }
        txn.commit()?;
        Ok(Self { db })
    }
}
