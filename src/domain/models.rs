use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use super::error::{PgrammaError, Result};

/// Plutchik-based emotion categories, matching the original schema CHECK constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
#[serde(rename_all = "lowercase")]
pub enum Emotion {
    Neutral,
    Joy,
    Sadness,
    Trust,
    Disgust,
    Fear,
    Anger,
    Surprise,
    Anticipation,
    Contempt,
}

impl Emotion {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Neutral => "neutral",
            Self::Joy => "joy",
            Self::Sadness => "sadness",
            Self::Trust => "trust",
            Self::Disgust => "disgust",
            Self::Fear => "fear",
            Self::Anger => "anger",
            Self::Surprise => "surprise",
            Self::Anticipation => "anticipation",
            Self::Contempt => "contempt",
        }
    }

    pub fn from_str_checked(s: &str) -> Result<Self> {
        match s {
            "neutral" => Ok(Self::Neutral),
            "joy" => Ok(Self::Joy),
            "sadness" => Ok(Self::Sadness),
            "trust" => Ok(Self::Trust),
            "disgust" => Ok(Self::Disgust),
            "fear" => Ok(Self::Fear),
            "anger" => Ok(Self::Anger),
            "surprise" => Ok(Self::Surprise),
            "anticipation" => Ok(Self::Anticipation),
            "contempt" => Ok(Self::Contempt),
            other => Err(PgrammaError::InvalidData(format!(
                "unknown emotion: {other}"
            ))),
        }
    }
}

impl std::fmt::Display for Emotion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Conversation role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
        }
    }

    pub fn from_str_checked(s: &str) -> Result<Self> {
        match s {
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            "system" => Ok(Self::System),
            other => Err(PgrammaError::InvalidData(format!("unknown role: {other}"))),
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single entry in the episodic memory timeline.
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct EpisodicEntry {
    pub id: i64,
    pub role: Role,
    pub content: String,
    pub timestamp: String,
}

/// A weighted knowledge unit with optional vector embedding.
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct Engram {
    pub id: i64,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub emotion: Emotion,
    pub importance: f32,
    pub access_count: i64,
    pub last_accessed: Option<String>,
    pub created_at: String,
}
