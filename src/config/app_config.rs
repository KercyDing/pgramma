use serde::Deserialize;

/// Top-level application configuration loaded from `config.toml`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub llm: LlmConfig,
    pub embedding: EmbeddingConfig,
    pub recall: RecallConfig,
    pub chat: ChatConfig,
}

/// LLM runtime configuration with provider profiles.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// Provider key currently in use.
    ///
    /// Supported values: `openai`, `google`/`gemini`, `grok`/`xai`/`gork`,
    /// `anthropic`, `openrouter`, `custom`.
    pub active_provider: String,
    /// Provider-specific model/API settings.
    pub providers: LlmProvidersConfig,
}

/// Provider profiles for all supported LLM backends.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct LlmProvidersConfig {
    pub openai: LlmProviderProfile,
    pub google: LlmProviderProfile,
    pub grok: LlmProviderProfile,
    pub anthropic: LlmProviderProfile,
    pub openrouter: LlmProviderProfile,
    pub custom: LlmProviderProfile,
}

/// Shared profile fields for each LLM provider.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LlmProviderProfile {
    /// Chat model identifier to use for inference and evaluation.
    pub model: String,
    /// API key for this provider.
    pub api_key: String,
    /// Optional custom endpoint base URL.
    pub base_url: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub model_id: String,
    /// Optional HuggingFace cache root directory.
    ///
    /// If omitted, hf-hub default cache path is used.
    pub cache_dir: Option<String>,
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
            active_provider: "openai".to_owned(),
            providers: LlmProvidersConfig::default(),
        }
    }
}

impl Default for LlmProvidersConfig {
    fn default() -> Self {
        Self {
            openai: LlmProviderProfile {
                model: String::new(),
                api_key: String::new(),
                base_url: Some("https://api.openai.com/v1".to_owned()),
            },
            google: LlmProviderProfile {
                model: String::new(),
                api_key: String::new(),
                base_url: Some("https://generativelanguage.googleapis.com".to_owned()),
            },
            grok: LlmProviderProfile {
                model: String::new(),
                api_key: String::new(),
                base_url: Some("https://api.x.ai".to_owned()),
            },
            anthropic: LlmProviderProfile {
                model: String::new(),
                api_key: String::new(),
                base_url: Some("https://api.anthropic.com".to_owned()),
            },
            openrouter: LlmProviderProfile {
                model: String::new(),
                api_key: String::new(),
                base_url: Some("https://openrouter.ai/api/v1".to_owned()),
            },
            custom: LlmProviderProfile {
                model: String::new(),
                api_key: String::new(),
                base_url: None,
            },
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_id: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_owned(),
            cache_dir: None,
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
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                eprintln!(
                    "[config] {path} not found; copy config.example.toml -> config.toml to customize. using defaults"
                );
                Self::default()
            }
            Err(e) => {
                eprintln!("[config] failed to read {path}: {e}, using defaults");
                Self::default()
            }
        }
    }
}
