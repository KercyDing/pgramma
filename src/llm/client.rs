use rig::providers::{anthropic, gemini, openai, openrouter, xai};

use crate::config::{LlmConfig, LlmProviderProfile};

#[derive(Clone)]
pub(crate) enum ProviderClient {
    OpenAi(openai::Client),
    Google(gemini::Client),
    Grok(xai::Client),
    Anthropic(anthropic::Client),
    OpenRouter(openrouter::Client),
    Custom(openai::Client),
}

/// Thin wrapper over rig provider clients.
///
/// Provider, model, API key, and optional endpoint are configured
/// from `config.toml` via `llm.active_provider` + `llm.providers.*`.
#[derive(Clone)]
pub struct LlmClient {
    pub(crate) client: ProviderClient,
    pub(crate) model: String,
}

impl LlmClient {
    /// Build LLM client from nested config profiles.
    pub fn from_config(config: &LlmConfig) -> Result<Self, String> {
        let provider = config.active_provider.trim().to_ascii_lowercase();
        match provider.as_str() {
            "openai" => Self::build_openai(&config.providers.openai),
            "google" | "gemini" => Self::build_google(&config.providers.google),
            "grok" | "xai" | "gork" => Self::build_grok(&config.providers.grok),
            "anthropic" => Self::build_anthropic(&config.providers.anthropic),
            "openrouter" => Self::build_openrouter(&config.providers.openrouter),
            "custom" => Self::build_custom(&config.providers.custom),
            _ => Err(format!(
                "unsupported llm.active_provider `{}`; expected one of: openai, google/gemini, grok/xai/gork, anthropic, openrouter, custom",
                config.active_provider
            )),
        }
    }

    fn build_openai(profile: &LlmProviderProfile) -> Result<Self, String> {
        let api_key = Self::read_api_key("openai", profile)?;
        let mut builder = openai::Client::builder().api_key(&api_key);
        if let Some(base_url) = Self::normalize_base_url(profile.base_url.as_deref()) {
            builder = builder.base_url(base_url);
        }
        let client = builder
            .build()
            .map_err(|e| format!("failed to build openai client: {e}"))?;
        Ok(Self {
            client: ProviderClient::OpenAi(client),
            model: profile.model.clone(),
        })
    }

    fn build_google(profile: &LlmProviderProfile) -> Result<Self, String> {
        let api_key = Self::read_api_key("google", profile)?;
        let mut builder = gemini::Client::builder().api_key(&api_key);
        if let Some(base_url) = Self::normalize_base_url(profile.base_url.as_deref()) {
            builder = builder.base_url(base_url);
        }
        let client = builder
            .build()
            .map_err(|e| format!("failed to build google(gemini) client: {e}"))?;
        Ok(Self {
            client: ProviderClient::Google(client),
            model: profile.model.clone(),
        })
    }

    fn build_grok(profile: &LlmProviderProfile) -> Result<Self, String> {
        let api_key = Self::read_api_key("grok", profile)?;
        let mut builder = xai::Client::builder().api_key(&api_key);
        if let Some(base_url) = Self::normalize_base_url(profile.base_url.as_deref()) {
            builder = builder.base_url(base_url);
        }
        let client = builder
            .build()
            .map_err(|e| format!("failed to build grok(xai) client: {e}"))?;
        Ok(Self {
            client: ProviderClient::Grok(client),
            model: profile.model.clone(),
        })
    }

    fn build_anthropic(profile: &LlmProviderProfile) -> Result<Self, String> {
        let api_key = Self::read_api_key("anthropic", profile)?;
        let mut builder = anthropic::Client::builder().api_key(&api_key);
        if let Some(base_url) = Self::normalize_base_url(profile.base_url.as_deref()) {
            builder = builder.base_url(base_url);
        }
        let client = builder
            .build()
            .map_err(|e| format!("failed to build anthropic client: {e}"))?;
        Ok(Self {
            client: ProviderClient::Anthropic(client),
            model: profile.model.clone(),
        })
    }

    fn build_openrouter(profile: &LlmProviderProfile) -> Result<Self, String> {
        let api_key = Self::read_api_key("openrouter", profile)?;
        let mut builder = openrouter::Client::builder().api_key(&api_key);
        if let Some(base_url) = Self::normalize_base_url(profile.base_url.as_deref()) {
            builder = builder.base_url(base_url);
        }
        let client = builder
            .build()
            .map_err(|e| format!("failed to build openrouter client: {e}"))?;
        Ok(Self {
            client: ProviderClient::OpenRouter(client),
            model: profile.model.clone(),
        })
    }

    fn build_custom(profile: &LlmProviderProfile) -> Result<Self, String> {
        let api_key = Self::read_api_key("custom", profile)?;
        let base_url = Self::normalize_base_url(profile.base_url.as_deref()).ok_or_else(|| {
            "llm.providers.custom.base_url is required for custom OpenAI-compatible provider"
                .to_owned()
        })?;
        let client = openai::Client::builder()
            .api_key(&api_key)
            .base_url(base_url)
            .build()
            .map_err(|e| format!("failed to build custom(openai-compatible) client: {e}"))?;
        Ok(Self {
            client: ProviderClient::Custom(client),
            model: profile.model.clone(),
        })
    }

    fn read_api_key(provider: &str, profile: &LlmProviderProfile) -> Result<String, String> {
        if profile.model.trim().is_empty() {
            return Err(format!(
                "llm.providers.{provider}.model cannot be empty in config.toml"
            ));
        }
        let api_key = profile.api_key.trim();
        if api_key.is_empty() {
            return Err(format!(
                "llm.providers.{provider}.api_key cannot be empty in config.toml"
            ));
        }
        Ok(api_key.to_owned())
    }

    fn normalize_base_url(raw: Option<&str>) -> Option<&str> {
        match raw {
            Some(value) if !value.trim().is_empty() => Some(value),
            _ => None,
        }
    }
}
