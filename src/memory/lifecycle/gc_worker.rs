use crate::config::LifecycleConfig;
use crate::db::PgramDb;
use crate::error::Result;
use crate::models::Engram;

/// Garbage-collection result summary.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct GcStats {
    pub deleted: usize,
    pub overflow: usize,
}

/// Delete low-importance engrams until `gc_max_engrams` is satisfied.
pub(super) fn collect_garbage(db: &PgramDb, config: &LifecycleConfig) -> Result<GcStats> {
    if config.gc_max_engrams == 0 {
        return Ok(GcStats::default());
    }

    let engrams = db.get_all_engrams()?;
    if engrams.len() <= config.gc_max_engrams {
        return Ok(GcStats::default());
    }

    let mut candidates: Vec<&Engram> = engrams
        .iter()
        .filter(|engram| engram.importance < config.gc_delete_below)
        .collect();
    candidates.sort_by(compare_gc_priority);

    let mut deleted = 0usize;
    let mut remaining = engrams.len();
    for engram in candidates {
        if remaining <= config.gc_max_engrams {
            break;
        }
        if db.delete_engram(engram.id)? {
            deleted += 1;
            remaining -= 1;
        }
    }

    Ok(GcStats {
        deleted,
        overflow: remaining.saturating_sub(config.gc_max_engrams),
    })
}

fn compare_gc_priority(a: &&Engram, b: &&Engram) -> std::cmp::Ordering {
    a.importance
        .partial_cmp(&b.importance)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.created_at.cmp(&b.created_at))
        .then_with(|| a.id.cmp(&b.id))
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use super::*;
    use crate::models::Emotion;

    #[test]
    fn gc_deletes_low_importance_first() {
        let file = NamedTempFile::new().expect("tempfile");
        let db = PgramDb::open(file.path()).expect("open db");

        db.insert_engram("a", Emotion::Neutral, 0.10, None)
            .expect("insert a");
        db.insert_engram("b", Emotion::Neutral, 0.20, None)
            .expect("insert b");
        db.insert_engram("c", Emotion::Neutral, 0.25, None)
            .expect("insert c");
        db.insert_engram("d", Emotion::Neutral, 0.80, None)
            .expect("insert d");
        db.insert_engram("e", Emotion::Neutral, 0.90, None)
            .expect("insert e");

        let config = LifecycleConfig {
            gc_max_engrams: 3,
            gc_delete_below: 0.3,
            ..LifecycleConfig::default()
        };
        let stats = collect_garbage(&db, &config).expect("gc");

        assert_eq!(stats.deleted, 2);
        assert_eq!(stats.overflow, 0);
        assert_eq!(db.get_all_engrams().expect("all").len(), 3);
    }
}
