use rig::client::ProviderClient;
use rig::providers::openai;

/// Thin wrapper over rig's OpenAI client.
///
/// API credentials come from environment variables:
/// - `OPENAI_API_KEY` (required) — API key
/// - `OPENAI_BASE_URL` (optional) — custom endpoint for OpenAI-compatible APIs
///
/// The model name is provided by the caller (from config.toml).
#[derive(Clone)]
pub struct LlmClient {
    pub(crate) client: openai::Client,
    pub(crate) model: String,
}

impl LlmClient {
    pub fn from_env(model: &str) -> Result<Self, String> {
        let client = openai::Client::from_env();
        Ok(Self {
            client,
            model: model.to_owned(),
        })
    }
}
