use rig::client::CompletionClient;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::client::{LlmClient, ProviderClient};
use crate::error::{PgrammaError, Result};

// ── scoring criteria (injected via system prompt) ───────────────

const IMPORTANCE_RUBRIC: &str = "\
Output exactly 2 decimal places. \
Avoid multiples of 0.05 (e.g. 0.10, 0.25, 0.80) unless truly warranted. \
Prefer granular values like 0.07, 0.13, 0.42, 0.68, 0.91, 0.97.\n\n\
## Tiers\n\
0.00-0.19: Meaningless small talk, greetings, filler, weather chitchat, \
or verification questions where the user tests recall without providing \
new information (e.g. 'what's my name?', 'do you remember X?').\n\
0.20-0.49: Ordinary factual exchange, general Q&A, trivia.\n\
0.50-0.59: Light preferences — surface-level hobbies, tool/tech preferences, \
casual opinions.\n\
0.60-0.69: Moderate personal sharing — habits, daily concerns, dietary/health \
preferences, emotional venting, language preference.\n\
0.70-0.79: Deep emotional sharing — romantic feelings, past trauma, \
vulnerability, fears, relationship conflicts, significant personal stories.\n\
0.80-1.00: Core memory — information that must be retained permanently. \
Three sub-categories:\n\
  (a) Major life events: death of loved one, birth, marriage/divorce, \
long-term diagnosis, fundamental relationship changes.\n\
  (b) Explicit user directives: 'remember that...', 'from now on...', \
'always/never do X', 'can you stop...' — user is explicitly marking \
this as important. Score 0.80+ regardless of the topic's inherent weight.\n\
  (c) Identity anchors: user's real name, assistant's assigned name/nickname. \
Implicitly permanent — no one shares their name expecting it to be forgotten. \
Score 0.80+ for both user and assistant naming.\n\n\
## Calibration Anchors (do NOT copy these exact values)\n\
0.06 — 'Hi' / 'lol' / 'ok thanks'\n\
0.23 — 'Capital of France?'\n\
0.28 — 'What year did WW2 end?'\n\
0.51 — 'I like using Vim'\n\
0.62 — 'I've been feeling burnt out at work lately'\n\
0.73 — 'I had a crush on someone in high school and never told them'\n\
0.76 — 'My ex and I broke up because he cheated'\n\
0.82 — 'My name is Kira' (identity anchor)\n\
0.82 — 'From now on, keep replies short' (explicit directive)\n\
0.83 — 'Remember: I'm allergic to shellfish, never suggest it' (explicit directive)\n\
0.84 — 'I just got engaged!' (major life event)\n\
0.88 — 'I was diagnosed with bipolar disorder' (major life event)\n\
0.92 — 'My mother passed away last year' (major life event)";

const RETRACTION_RUBRIC: &str = "\
Set retraction=true if this turn supersedes, corrects, retracts, or updates \
information from the immediately preceding exchange. Examples:\n\
- Retraction: 'just kidding', 'I lied', 'actually that's not true'\n\
- Correction: 'wait, my name is actually B', 'no I meant X not Y'\n\
- Preference override: 'actually I prefer dogs over cats'\n\
- Status update: 'I moved to Shanghai now' (overrides prior location)\n\
When true, the previous engram will be invalidated because the CURRENT turn \
carries the up-to-date information.";

const EMOTION_LABELS: &str = "\
Valid emotions (lowercase only): neutral, joy, sadness, trust, disgust, \
fear, anger, surprise, anticipation, contempt. \
Do NOT invent synonyms like happy, anxious, hopeful — use the exact labels above.";

// ── schema types ────────────────────────────────────────────────

/// Single emotion with contribution weight.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EmotionWeight {
    /// Emotion label: neutral, joy, sadness, trust, disgust, fear, anger, surprise, anticipation, contempt.
    pub emotion: String,
    /// Contribution weight of this emotion (0.0-1.0).
    pub weight: f32,
}

/// Cognitive evaluation result for a conversation snippet.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EngramScore {
    /// 1-3 emotions ranked by dominance. Weights must sum to 1.0.
    /// Use 1 emotion for simple cases, 2-3 for mixed/complex cases.
    pub emotions: Vec<EmotionWeight>,
    /// Memory importance weight (0.0-1.0). See system prompt for rubric.
    pub importance: f32,
    /// Brief justification for the weight and emotion assignment, max 20 words.
    pub reasoning: String,
    /// True if this turn supersedes or invalidates the previous exchange.
    /// See system prompt for detailed criteria.
    pub retraction: bool,
}

// ── evaluation logic ────────────────────────────────────────────

/// Build the full system prompt with scoring rubrics.
fn build_system_prompt() -> String {
    format!(
        "You are the cognitive scorer of the Pgramma engine. \
         Objectively evaluate the memory-retention value of the CURRENT conversation turn. \
         You may also receive RECENT CONTEXT (previous turns) — use it to understand \
         who is speaking, what names/nicknames refer to, and to detect retractions \
         or corrections. Score ONLY the CURRENT turn. \
         Respond ONLY with the structured JSON schema provided. \
         Always write the reasoning field in English, regardless of the conversation language.\
         \n\n{IMPORTANCE_RUBRIC}\
         \n\n## Retraction\n{RETRACTION_RUBRIC}\
         \n\n## Emotion Labels\n{EMOTION_LABELS}"
    )
}

/// Score a conversation snippet for emotion and memory importance.
///
/// `recent_turns` contains previous exchanges for context so the evaluator
/// can understand references and detect retractions accurately.
pub async fn evaluate(
    llm: &LlmClient,
    chat_text: &str,
    recent_turns: &[String],
) -> Result<EngramScore> {
    let input = if recent_turns.is_empty() {
        format!("[CURRENT TURN]\n{chat_text}")
    } else {
        let ctx = recent_turns.join("\n---\n");
        format!("[RECENT CONTEXT]\n{ctx}\n\n[CURRENT TURN]\n{chat_text}")
    };

    let system_prompt = build_system_prompt();
    match &llm.client {
        ProviderClient::OpenAi(client) => {
            extract_score(client, &llm.model, &system_prompt, &input).await
        }
        ProviderClient::Google(client) => {
            extract_score(client, &llm.model, &system_prompt, &input).await
        }
        ProviderClient::Grok(client) => {
            extract_score(client, &llm.model, &system_prompt, &input).await
        }
        ProviderClient::Anthropic(client) => {
            extract_score(client, &llm.model, &system_prompt, &input).await
        }
        ProviderClient::OpenRouter(client) => {
            extract_score(client, &llm.model, &system_prompt, &input).await
        }
        ProviderClient::Custom(client) => {
            extract_score(client, &llm.model, &system_prompt, &input).await
        }
    }
}

async fn extract_score<C>(
    client: &C,
    model: &str,
    system_prompt: &str,
    input: &str,
) -> Result<EngramScore>
where
    C: CompletionClient,
{
    let extractor = client
        .extractor::<EngramScore>(model)
        .preamble(system_prompt)
        .build();

    extractor
        .extract(input)
        .await
        .map_err(|e| PgrammaError::InvalidData(format!("evaluation failed: {e}")))
}
