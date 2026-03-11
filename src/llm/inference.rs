use std::io::Write;
use std::sync::Arc;

use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, Text};
use rig::client::CompletionClient;
use rig::streaming::{StreamedAssistantContent, StreamingPrompt};

use super::client::LlmClient;
use crate::db::PgramDb;
use crate::error::{PgrammaError, Result};
use crate::memory::MemoryFragment;
use crate::models::Role;

/// Run a single streaming chat turn.
///
/// Loads recent episodes as context, injects recalled memory fragments
/// into the system prompt, streams LLM reply to stdout,
/// writes user + assistant episodes to db, returns the full reply.
pub async fn chat_stream(
    llm: &LlmClient,
    db: &Arc<PgramDb>,
    user_input: &str,
    system_prompt: &str,
    memories: &[MemoryFragment],
    context_window: i64,
) -> Result<String> {
    // Build context from recent episodes (newest-first → reverse for chronological)
    let episodes = db.get_episodes(context_window, 0)?;
    let mut context = String::new();
    for ep in episodes.iter().rev() {
        context.push_str(&format!("{}: {}\n", ep.role, ep.content));
    }
    context.push_str(&format!("user: {user_input}\n"));

    // Inject memory fragments into system prompt (scores hidden from LLM)
    let full_prompt = if memories.is_empty() {
        system_prompt.to_owned()
    } else {
        let mut p = system_prompt.to_owned();
        p.push_str(
            "\n\n[Memory fragments — let these naturally color your tone, do not recite them]",
        );
        for m in memories {
            p.push_str(&format!("\n- \"{}\"", m.content));
        }
        p
    };

    // Build agent and stream
    let agent = llm.client.agent(&llm.model).preamble(&full_prompt).build();

    let mut stream = agent.stream_prompt(&context).multi_turn(1).await;

    let mut reply = String::new();
    let mut stdout = std::io::stdout();

    loop {
        let Some(chunk) = stream.next().await else {
            break;
        };
        match chunk {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                Text { text },
            ))) => {
                print!("{text}");
                stdout.flush().ok();
                reply.push_str(&text);
            }
            Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
            Err(e) => {
                return Err(PgrammaError::InvalidData(format!("stream error: {e}")));
            }
            _ => continue,
        }
    }
    println!();

    // Persist episodes
    db.append_episode(Role::User, user_input)?;
    db.append_episode(Role::Assistant, &reply)?;

    Ok(reply)
}
