use std::fs::OpenOptions;
use std::io::{self, BufRead, Write};
use std::sync::Arc;

use pgramma::db::PgramDb;
use pgramma::llm::{self, LlmClient};
use pgramma::models::Emotion;

const DEFAULT_SYSTEM_PROMPT: &str = "You are a thoughtful assistant with emotional awareness.";

/// Open (or create) the log file for background eval output.
fn open_log() -> std::fs::File {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open("pgramma.log")
        .expect("failed to open pgramma.log")
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    // Shared log file for background eval tasks
    let log = Arc::new(std::sync::Mutex::new(open_log()));

    // Init DB
    let db = Arc::new(PgramDb::open("pgramma.pgram").expect("failed to open database"));

    // Set default system prompt if not configured
    if db.get_config("system_prompt").is_err() {
        db.set_config("system_prompt", DEFAULT_SYSTEM_PROMPT)
            .expect("failed to set system prompt");
    }
    let system_prompt = db.get_config("system_prompt").unwrap();

    // Init LLM client
    let llm = LlmClient::from_env().expect("failed to init LLM client");

    println!("=== Pgramma ===");
    println!("[system] {system_prompt}\n");

    let stdin = io::stdin();
    /// Recent chat turns for evaluator context (sliding window).
    const EVAL_CONTEXT_TURNS: usize = 3;
    let mut recent_turns: Vec<String> = Vec::new();
    loop {
        print!("> ");
        io::stdout().flush().ok();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).is_err() || input.is_empty() {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/quit" || input == "/exit" {
            break;
        }

        // Streaming chat
        let reply = match llm::chat_stream(&llm, &db, input, &system_prompt).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[error] {e}");
                continue;
            }
        };

        // Background evaluation (output to log file, not terminal)
        let llm2 = llm.clone();
        let db2 = Arc::clone(&db);
        let log2 = Arc::clone(&log);
        let chat_text = format!("User: {input}\nAssistant: {reply}");
        let context_for_eval: Vec<String> = recent_turns.clone();
        // Maintain sliding window
        recent_turns.push(chat_text.clone());
        if recent_turns.len() > EVAL_CONTEXT_TURNS {
            recent_turns.remove(0);
        }
        tokio::spawn(async move {
            let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            match llm::evaluate(&llm2, &chat_text, &context_for_eval).await {
                Ok(score) => {
                    // If retraction detected, delete the previous engram
                    if score.retraction
                        && let Some(prev_engram) = db2.get_latest_engram().ok().flatten()
                    {
                        let _ = db2.delete_engram(prev_engram.id);
                        let _ = writeln!(
                            log2.lock().unwrap(),
                            "[{ts}] retraction: deleted engram #{} — {}",
                            prev_engram.id,
                            score.reasoning
                        );
                    }

                    let emotion = Emotion::from_str_checked(
                        &score
                            .emotions
                            .first()
                            .map(|e| e.emotion.clone())
                            .unwrap_or_else(|| "neutral".to_owned()),
                    )
                    .unwrap_or(Emotion::Neutral);

                    if let Err(e) = db2.insert_engram(&chat_text, emotion, score.importance, None) {
                        let _ = writeln!(log2.lock().unwrap(), "[{ts}] eval error: {e}");
                    } else {
                        let _ = writeln!(
                            log2.lock().unwrap(),
                            "[{ts}] importance={:.2} emotion={} retraction={} — {}",
                            score.importance,
                            emotion,
                            score.retraction,
                            score.reasoning
                        );
                    }
                }
                Err(e) => {
                    let _ = writeln!(log2.lock().unwrap(), "[{ts}] eval error: {e}");
                }
            }
        });

        println!();
    }

    println!("Bye!");
}
