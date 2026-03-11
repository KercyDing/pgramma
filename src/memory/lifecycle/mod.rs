//! Memory lifecycle maintenance.
//!
//! This module applies periodic decay and garbage collection to engrams.

use chrono::{DateTime, Utc};

use crate::config::LifecycleConfig;
use crate::db::PgramDb;
use crate::error::{PgrammaError, Result};

mod gc_worker;
mod rater;

const LAST_MAINTENANCE_KEY: &str = "lifecycle.last_maintenance_at";

/// Lifecycle execution summary.
#[derive(Debug, Default, Clone, Copy)]
pub struct MaintenanceStats {
    /// Number of engrams whose importance was decayed.
    pub decayed: usize,
    /// Number of engrams physically deleted by GC.
    pub deleted: usize,
    /// Remaining over-limit count after GC.
    pub overflow_after_gc: usize,
    /// True if maintenance was skipped by config/interval.
    pub skipped: bool,
}

/// Run lifecycle maintenance once.
///
/// Applies importance decay and low-importance GC with interval guard.
/// Returns aggregate stats for logging.
pub fn run_maintenance(db: &PgramDb, config: &LifecycleConfig) -> Result<MaintenanceStats> {
    if !config.enabled {
        return Ok(MaintenanceStats {
            skipped: true,
            ..MaintenanceStats::default()
        });
    }

    let now = Utc::now();
    let last_run = read_last_run(db)?;
    if let Some(last) = last_run {
        let elapsed_secs = (now - last).num_seconds();
        if elapsed_secs < config.maintenance_interval_secs {
            return Ok(MaintenanceStats {
                skipped: true,
                ..MaintenanceStats::default()
            });
        }
    }

    let elapsed_days = last_run
        .map(|last| ((now - last).num_seconds().max(0) as f32) / 86_400.0)
        .unwrap_or(0.0);
    let decayed = rater::apply_decay(db, config, elapsed_days)?;
    let gc_stats = gc_worker::collect_garbage(db, config)?;

    db.set_config(
        LAST_MAINTENANCE_KEY,
        &now.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
    )?;

    Ok(MaintenanceStats {
        decayed,
        deleted: gc_stats.deleted,
        overflow_after_gc: gc_stats.overflow,
        skipped: false,
    })
}

fn read_last_run(db: &PgramDb) -> Result<Option<DateTime<Utc>>> {
    match db.get_config(LAST_MAINTENANCE_KEY) {
        Ok(raw) => Ok(Some(parse_utc(&raw)?)),
        Err(PgrammaError::ConfigNotFound(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

fn parse_utc(raw: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| {
            PgrammaError::InvalidData(format!(
                "invalid lifecycle timestamp `{raw}` in persona_config: {e}"
            ))
        })
}
