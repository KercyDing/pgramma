use crate::config::LifecycleConfig;
use crate::db::PgramDb;
use crate::error::Result;

/// Apply Ebbinghaus-style exponential forgetting with access-based retention.
pub(super) fn apply_decay(
    db: &PgramDb,
    config: &LifecycleConfig,
    elapsed_days: f32,
) -> Result<usize> {
    if elapsed_days <= 0.0 || config.half_life_days <= 0.0 {
        return Ok(0);
    }

    let mut updated = 0usize;
    let engrams = db.get_all_engrams()?;
    for engram in engrams {
        if engram.importance <= config.min_importance_floor {
            continue;
        }
        if engram.importance >= config.protect_above_importance {
            continue;
        }

        let next = decay_importance(config, engram.importance, elapsed_days, engram.access_count);
        if (engram.importance - next) < 1e-6 {
            continue;
        }

        db.update_engram_importance(engram.id, next)?;
        updated += 1;
    }

    Ok(updated)
}

fn decay_importance(
    config: &LifecycleConfig,
    importance: f32,
    elapsed_days: f32,
    access_count: i64,
) -> f32 {
    let floor = config.min_importance_floor;
    let base = (importance - floor).max(0.0);
    if base <= 0.0 {
        return importance.clamp(floor, 1.0);
    }

    let half_life = effective_half_life_days(config, access_count).max(1e-6);
    let retain = (-std::f32::consts::LN_2 * elapsed_days / half_life).exp();
    (floor + base * retain).clamp(floor, 1.0)
}

fn effective_half_life_days(config: &LifecycleConfig, access_count: i64) -> f32 {
    let access = access_count.max(0) as f32;
    config.half_life_days * (1.0 + config.access_half_life_gain.max(0.0) * access.ln_1p())
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use super::*;
    use crate::models::Emotion;

    #[test]
    fn decay_reduces_importance() {
        let file = NamedTempFile::new().expect("tempfile");
        let db = PgramDb::open(file.path()).expect("open db");
        let id = db
            .insert_engram("test", Emotion::Neutral, 0.9, None)
            .expect("insert");

        let config = LifecycleConfig {
            half_life_days: 2.0,
            access_half_life_gain: 0.0,
            min_importance_floor: 0.1,
            protect_above_importance: 1.0,
            ..LifecycleConfig::default()
        };

        let changed = apply_decay(&db, &config, 2.0).expect("decay");
        assert_eq!(changed, 1);

        let current = db
            .get_all_engrams()
            .expect("load")
            .into_iter()
            .find(|e| e.id == id)
            .expect("engram");
        assert!((current.importance - 0.5).abs() < 1e-4);
    }

    #[test]
    fn protected_importance_is_not_decayed() {
        let file = NamedTempFile::new().expect("tempfile");
        let db = PgramDb::open(file.path()).expect("open db");
        let id = db
            .insert_engram("anchor", Emotion::Neutral, 0.85, None)
            .expect("insert");

        let config = LifecycleConfig {
            half_life_days: 2.0,
            access_half_life_gain: 0.0,
            min_importance_floor: 0.1,
            protect_above_importance: 0.8,
            ..LifecycleConfig::default()
        };

        let changed = apply_decay(&db, &config, 10.0).expect("decay");
        assert_eq!(changed, 0);

        let current = db
            .get_all_engrams()
            .expect("load")
            .into_iter()
            .find(|e| e.id == id)
            .expect("engram");
        assert!((current.importance - 0.85).abs() < 1e-6);
    }
}
