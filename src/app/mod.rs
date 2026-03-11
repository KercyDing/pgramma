//! Application orchestration layer.
//!
//! Handles bootstrap, REPL loop, slash commands, and turn lifecycle wiring.

use std::fs::OpenOptions;
use std::io::{self, BufRead, Write};
use std::sync::Arc;

use crate::config::AppConfig;
use crate::db::PgramDb;
use crate::llm::{self, LlmClient};
use crate::memory::{self, embedder::Embedder};
use crate::models::Emotion;

/// Shared application state threaded through all functions.
struct App {
    llm: LlmClient,
    db: Arc<PgramDb>,
    embedder: Arc<Embedder>,
    log: Arc<std::sync::Mutex<std::fs::File>>,
    system_prompt: String,
    config: AppConfig,
}

fn print_help() {
    println!("  /help         — show this help");
    println!("  /test <path>  — batch seed from file and verify recall");
    println!("  /stats        — show engram statistics");
    println!("  /quit         — exit");
    println!();
}

/// Open (or create) the log file for background eval output.
fn open_log() -> std::io::Result<std::fs::File> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open("pgramma.log")
}

/// Run the Pgramma interactive application.
///
/// Returns a typed startup/runtime error string if initialization fails.
pub async fn run() -> Result<(), String> {
    let config = AppConfig::load("config.toml");

    let db = Arc::new(
        PgramDb::open("pgramma.pgram").map_err(|e| format!("failed to open database: {e}"))?,
    );

    let system_prompt = if let Ok(prompt) = db.get_config("system_prompt") {
        prompt
    } else {
        db.set_config("system_prompt", &config.chat.default_system_prompt)
            .map_err(|e| format!("failed to set system prompt: {e}"))?;
        config.chat.default_system_prompt.clone()
    };

    let llm = LlmClient::from_config(&config.llm)
        .map_err(|e| format!("failed to init LLM client: {e}"))?;

    eprintln!(
        "[init] Loading embedding model: {}...",
        config.embedding.model_id
    );
    let embedder = Arc::new(
        Embedder::load(
            &config.embedding.model_id,
            config.embedding.cache_dir.as_deref(),
        )
        .map_err(|e| format!("failed to load embedding model: {e}"))?,
    );
    eprintln!("[init] Embedding model ready.");

    let app = App {
        llm,
        db,
        embedder,
        log: Arc::new(std::sync::Mutex::new(
            open_log().map_err(|e| format!("failed to open pgramma.log: {e}"))?,
        )),
        system_prompt,
        config,
    };

    println!("=== Pgramma ===");
    println!("[system] {}", app.system_prompt);
    println!("Type /help for commands\n");

    let stdin = io::stdin();
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

        // ── slash commands ───────────────────────────────────────
        if input == "/quit" || input == "/exit" {
            break;
        }
        if input == "/help" {
            print_help();
            continue;
        }
        if input == "/stats" {
            let count = app.db.get_all_engrams().map(|v| v.len()).unwrap_or(0);
            let episodes = app
                .db
                .get_episodes(i64::MAX, 0)
                .map(|v| v.len())
                .unwrap_or(0);
            println!("  engrams: {count}  episodes: {episodes}");
            println!();
            continue;
        }
        if let Some(path) = input.strip_prefix("/test") {
            let path = path.trim();
            if path.is_empty() {
                println!("usage: /test <file_path>");
                println!("  format: one msg per line, # = comment, ? = recall query");
                println!();
                continue;
            }
            run_test(&app, path, &mut recent_turns).await;
            continue;
        }

        // ── normal chat turn ────────────────────────────────────
        chat_turn(&app, &mut recent_turns, input).await;
    }

    println!("Bye!");
    Ok(())
}

/// Execute a single chat turn: recall → stream → background eval.
async fn chat_turn(app: &App, recent_turns: &mut Vec<String>, input: &str) {
    let memories = match memory::recall(
        &app.db,
        &app.embedder,
        input,
        app.config.recall.top_k,
        app.config.recall.min_importance,
        app.config.recall.cosine_weight,
    ) {
        Ok(m) => m,
        Err(e) => {
            let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            let _ = writeln!(app.log.lock().unwrap(), "[{ts}] recall error: {e}");
            Vec::new()
        }
    };

    print!("→ ");
    std::io::stdout().flush().ok();
    let reply = match llm::chat_stream(
        &app.llm,
        &app.db,
        input,
        &app.system_prompt,
        &memories,
        app.config.chat.context_window,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[error] {e}");
            return;
        }
    };

    let llm2 = app.llm.clone();
    let db2 = Arc::clone(&app.db);
    let log2 = Arc::clone(&app.log);
    let embedder2 = Arc::clone(&app.embedder);
    let chat_text = format!("User: {input}\nAssistant: {reply}");
    let user_input_owned = input.to_owned();
    let context_for_eval: Vec<String> = recent_turns.clone();

    let eval_turns = app.config.chat.eval_context_turns;
    recent_turns.push(chat_text.clone());
    if recent_turns.len() > eval_turns {
        recent_turns.remove(0);
    }

    tokio::spawn(async move {
        let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        match llm::evaluate(&llm2, &chat_text, &context_for_eval).await {
            Ok(score) => {
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

                let embedding = embedder2.embed(&user_input_owned).ok();

                match db2.insert_engram(&chat_text, emotion, score.importance, embedding.as_deref())
                {
                    Ok(id) => {
                        let _ = writeln!(
                            log2.lock().unwrap(),
                            "[{ts}] id={id} importance={:.2} emotion={} retraction={} emb={} — {}",
                            score.importance,
                            emotion,
                            score.retraction,
                            embedding.is_some(),
                            score.reasoning
                        );
                    }
                    Err(e) => {
                        let _ = writeln!(log2.lock().unwrap(), "[{ts}] eval error: {e}");
                    }
                }
            }
            Err(e) => {
                let _ = writeln!(log2.lock().unwrap(), "[{ts}] eval error: {e}");
            }
        }
    });

    println!();
}

/// Parse test file, seed chat turns, wait for eval, then verify recall.
///
/// File format: one message per line.
/// - `# ...` → comment (skipped)
/// - `? ...` → recall verification query (run after seeding)
/// - anything else → seed message (sent as chat turn)
async fn run_test(app: &App, path: &str, recent_turns: &mut Vec<String>) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[test] failed to read {path}: {e}");
            return;
        }
    };

    let mut seeds: Vec<&str> = Vec::new();
    let mut queries: Vec<&str> = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(q) = line.strip_prefix('?') {
            queries.push(q.trim());
        } else {
            seeds.push(line);
        }
    }

    // Seed phase
    println!("── seed: {path} ({} messages) ──", seeds.len());
    for (i, msg) in seeds.iter().enumerate() {
        println!("[{}/{}] {msg}", i + 1, seeds.len());
        chat_turn(app, recent_turns, msg).await;
    }

    // Wait for background eval tasks
    println!("── waiting for eval tasks (15s) ──");
    tokio::time::sleep(std::time::Duration::from_secs(15)).await;

    let engram_count = app.db.get_all_engrams().map(|v| v.len()).unwrap_or(0);
    println!("── engrams in DB: {engram_count} ──\n");

    // Recall verification phase
    if !queries.is_empty() {
        println!("── recall verification ({} queries) ──", queries.len());
        for query in &queries {
            println!("  Q: {query}");
            let memories = memory::recall(
                &app.db,
                &app.embedder,
                query,
                3,
                app.config.recall.min_importance,
                app.config.recall.cosine_weight,
            )
            .unwrap_or_default();
            if memories.is_empty() {
                println!("  → (no recall)");
            } else {
                for m in &memories {
                    let preview: String = m.content.chars().take(80).collect();
                    let ellipsis = if m.content.len() > 80 { "…" } else { "" };
                    println!("  → [{:.2}] {preview}{ellipsis}", m.relevance);
                }
            }
            println!();
        }
    }

    println!("── test done ──\n");
}
