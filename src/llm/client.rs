use rig::client::ProviderClient;
use rig::providers::openai;

/// Thin wrapper over rig's OpenAI client with env-based config.
///
/// Reads environment variables:
/// - `OPENAI_API_KEY` (required) — API key
/// - `OPENAI_BASE_URL` (optional) — custom endpoint for OpenAI-compatible APIs
/// - `LLM_MODEL` (optional, defaults to "gpt-4o") — model name
#[derive(Clone)]
pub struct LlmClient {
    pub(crate) client: openai::Client,
    pub(crate) model: String,
}

impl LlmClient {
    pub fn from_env() -> Result<Self, String> {
        // rig's from_env reads OPENAI_API_KEY and OPENAI_BASE_URL
        let client = openai::Client::from_env();
        let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-4o".to_owned());
        Ok(Self { client, model })
    }
}
