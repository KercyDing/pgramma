use serde::Deserialize;

/// Top-level application configuration loaded from `config.toml`.
///
/// All fields have sensible defaults — the app works without a config file.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub llm: LlmConfig,
    pub embedding: EmbeddingConfig,
    pub recall: RecallConfig,
    pub chat: ChatConfig,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub model: String,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub model_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct RecallConfig {
    pub top_k: usize,
    pub min_importance: f32,
    pub cosine_weight: f32,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ChatConfig {
    pub context_window: i64,
    pub eval_context_turns: usize,
    pub default_system_prompt: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o".to_owned(),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_id: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_owned(),
        }
    }
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            top_k: 8,
            min_importance: 0.3,
            cosine_weight: 0.7,
        }
    }
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            context_window: 20,
            eval_context_turns: 3,
            default_system_prompt: "You are a thoughtful assistant with emotional awareness."
                .to_owned(),
        }
    }
}

impl AppConfig {
    /// Load config from a TOML file. Falls back to defaults on any error.
    pub fn load(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(content) => toml::from_str(&content).unwrap_or_else(|e| {
                eprintln!("[config] failed to parse {path}: {e}, using defaults");
                Self::default()
            }),
            Err(_) => {
                eprintln!("[config] {path} not found, using defaults");
                Self::default()
            }
        }
    }
}
