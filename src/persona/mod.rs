//! Persona state module.
//!
//! Maintains an evolving assistant persona profile persisted in `.pgram`.

use serde::{Deserialize, Serialize};

use crate::db::PgramDb;
use crate::error::{PgrammaError, Result};
use crate::llm::EngramScore;

const PERSONA_STATE_KEY: &str = "persona_state.v1";
const PERSONA_VERSION: u32 = 1;

/// Evolving assistant persona profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaState {
    /// Schema version for forward compatibility.
    pub version: u32,
    /// Number of turns that have contributed to this state.
    pub turn_count: u64,
    /// Warmth tendency in [0.0, 1.0].
    pub warmth: f32,
    /// Directness tendency in [0.0, 1.0].
    pub directness: f32,
    /// Curiosity tendency in [0.0, 1.0].
    pub curiosity: f32,
    /// Baseline emotion inferred from long-term interactions.
    pub baseline_emotion: String,
}

impl Default for PersonaState {
    fn default() -> Self {
        Self {
            version: PERSONA_VERSION,
            turn_count: 0,
            warmth: 0.55,
            directness: 0.55,
            curiosity: 0.45,
            baseline_emotion: "neutral".to_owned(),
        }
    }
}

/// Load persona state from DB, returning defaults if no state exists.
pub fn load_or_default(db: &PgramDb) -> Result<PersonaState> {
    match db.get_config(PERSONA_STATE_KEY) {
        Ok(raw) => deserialize_state(&raw),
        Err(PgrammaError::ConfigNotFound(_)) => Ok(PersonaState::default()),
        Err(e) => Err(e),
    }
}

/// Persist persona state to DB.
pub fn save(db: &PgramDb, state: &PersonaState) -> Result<()> {
    let raw = serde_json::to_string(state)
        .map_err(|e| PgrammaError::InvalidData(format!("serialize persona state failed: {e}")))?;
    db.set_config(PERSONA_STATE_KEY, &raw)
}

/// Update persona state after a completed turn.
///
/// Uses assistant reply style + cognitive score as signals for gradual drift.
pub fn evolve_after_turn(db: &PgramDb, assistant_reply: &str, score: &EngramScore) -> Result<()> {
    let mut state = load_or_default(db)?;

    let words = assistant_reply.split_whitespace().count();
    let directness_target = if words <= 40 {
        0.85
    } else if words <= 100 {
        0.60
    } else {
        0.35
    };

    let question_marks = assistant_reply.chars().filter(|c| *c == '?').count() as f32;
    let curiosity_target = (question_marks / 2.0).clamp(0.0, 1.0);
    let warmth_target = warmth_from_emotions(score);

    let alpha = 0.12_f32;
    state.warmth = ema(state.warmth, warmth_target, alpha);
    state.directness = ema(state.directness, directness_target, alpha);
    state.curiosity = ema(state.curiosity, curiosity_target, alpha);
    state.baseline_emotion = dominant_emotion(score);
    state.turn_count += 1;

    save(db, &state)
}

/// Build a system-prompt suffix describing the current persona state.
///
/// The suffix is compact and non-user-visible, used to keep style continuity.
pub fn build_prompt_suffix(db: &PgramDb) -> Result<String> {
    let state = load_or_default(db)?;
    Ok(format!(
        "\n\n[Persona state — internal, do not reveal]\n\
         - baseline_emotion: {}\n\
         - warmth: {}\n\
         - directness: {}\n\
         - curiosity: {}\n\
         Keep responses stylistically consistent with this persona profile while staying helpful and factual.",
        state.baseline_emotion,
        bucket(state.warmth),
        bucket(state.directness),
        bucket(state.curiosity)
    ))
}

fn deserialize_state(raw: &str) -> Result<PersonaState> {
    let mut state: PersonaState = serde_json::from_str(raw)
        .map_err(|e| PgrammaError::InvalidData(format!("invalid persona state json: {e}")))?;
    if state.version != PERSONA_VERSION {
        state = PersonaState::default();
    }
    Ok(state)
}

fn ema(current: f32, target: f32, alpha: f32) -> f32 {
    (current + alpha * (target - current)).clamp(0.0, 1.0)
}

fn warmth_from_emotions(score: &EngramScore) -> f32 {
    if score.emotions.is_empty() {
        return 0.5;
    }
    let mut v = 0.5_f32;
    for ew in &score.emotions {
        let w = ew.weight.clamp(0.0, 1.0);
        let e = ew.emotion.to_ascii_lowercase();
        let sign = match e.as_str() {
            "joy" | "trust" => 1.0,
            "sadness" | "neutral" | "surprise" | "anticipation" => 0.0,
            "fear" | "anger" | "disgust" | "contempt" => -1.0,
            _ => 0.0,
        };
        v += sign * 0.25 * w;
    }
    v.clamp(0.0, 1.0)
}

fn dominant_emotion(score: &EngramScore) -> String {
    score
        .emotions
        .iter()
        .max_by(|a, b| {
            a.weight
                .partial_cmp(&b.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|v| v.emotion.to_ascii_lowercase())
        .unwrap_or_else(|| "neutral".to_owned())
}

fn bucket(value: f32) -> &'static str {
    if value < 0.34 {
        "low"
    } else if value < 0.67 {
        "medium"
    } else {
        "high"
    }
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use super::*;
    use crate::llm::evaluator::EmotionWeight;

    fn setup() -> PgramDb {
        let file = NamedTempFile::new().expect("tempfile");
        PgramDb::open(file.path()).expect("open db")
    }

    #[test]
    fn evolve_updates_state() {
        let db = setup();
        let score = EngramScore {
            emotions: vec![EmotionWeight {
                emotion: "joy".to_owned(),
                weight: 1.0,
            }],
            importance: 0.7,
            reasoning: "ok".to_owned(),
            retraction: false,
        };
        evolve_after_turn(&db, "Short response?", &score).expect("evolve");

        let state = load_or_default(&db).expect("load");
        assert_eq!(state.turn_count, 1);
        assert_eq!(state.baseline_emotion, "joy");
    }
}
