//! Application orchestration layer.
//!
//! Handles bootstrap, REPL loop, slash commands, and turn lifecycle wiring.

use std::fs::OpenOptions;
use std::io::{self, BufRead, Write};
use std::sync::Arc;

use tokio::sync::{Semaphore, mpsc, oneshot};

use crate::config::{AppConfig, LifecycleConfig};
use crate::db::PgramDb;
use crate::llm::{self, LlmClient};
use crate::memory::{self, embedder::Embedder};
use crate::models::Emotion;
use crate::persona;

/// Shared application state threaded through all functions.
struct App {
    llm: LlmClient,
    db: Arc<PgramDb>,
    embedder: Arc<Embedder>,
    log: Arc<std::sync::Mutex<std::fs::File>>,
    system_prompt: String,
    config: AppConfig,
}

struct EvalJob {
    llm: LlmClient,
    db: Arc<PgramDb>,
    log: Arc<std::sync::Mutex<std::fs::File>>,
    embedder: Arc<Embedder>,
    chat_text: String,
    engram_text: String,
    assistant_reply: String,
    user_input: String,
    context_for_eval: Vec<String>,
    previous_user_engram_text: Option<String>,
    lifecycle_cfg: LifecycleConfig,
    done_tx: oneshot::Sender<()>,
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

fn extract_user_engram_text(turn: &str) -> Option<String> {
    let first = turn.lines().next()?.trim();
    first
        .strip_prefix("User:")
        .map(|s| format!("User: {}", s.trim()))
}

fn test_min_importance(query: &str, base: f32) -> f32 {
    let q = query.to_ascii_lowercase();
    let personal = q.contains(" my ")
        || q.starts_with("my ")
        || q.contains(" do i ")
        || q.starts_with("do i ")
        || q.contains(" should you ")
        || q.contains("where do i")
        || q.contains("what is my")
        || q.contains("what do i");
    if personal { base.max(0.4) } else { base }
}

async fn wait_eval_tasks(pending_eval_tasks: &mut Vec<oneshot::Receiver<()>>) -> usize {
    let mut completed = 0usize;
    for rx in pending_eval_tasks.drain(..) {
        match rx.await {
            Ok(()) => completed += 1,
            Err(e) => {
                eprintln!("[warn] eval task join error: {e}");
                completed += 1;
            }
        }
    }
    completed
}

async fn process_eval_job(job: EvalJob) {
    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    let EvalJob {
        llm,
        db,
        log,
        embedder,
        chat_text,
        engram_text,
        assistant_reply,
        user_input,
        context_for_eval,
        previous_user_engram_text,
        lifecycle_cfg,
        done_tx,
    } = job;

    match llm::evaluate(&llm, &chat_text, &context_for_eval).await {
        Ok(score) => {
            if score.retraction {
                if let Some(target_content) = previous_user_engram_text.as_deref() {
                    let mut deleted = None;
                    for attempt in 0..3 {
                        match db.delete_latest_engram_by_content(target_content) {
                            Ok(id) if id.is_some() => {
                                deleted = id;
                                break;
                            }
                            Ok(_) if attempt < 2 => {
                                tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                            }
                            Ok(_) => break,
                            Err(e) => {
                                let _ = writeln!(
                                    log.lock().unwrap(),
                                    "[{ts}] retraction delete error: {e}"
                                );
                                break;
                            }
                        }
                    }

                    match deleted {
                        Some(id) => {
                            let _ = writeln!(
                                log.lock().unwrap(),
                                "[{ts}] retraction: deleted engram #{id} ({target_content}) — {}",
                                score.reasoning
                            );
                        }
                        None => {
                            let _ = writeln!(
                                log.lock().unwrap(),
                                "[{ts}] retraction: no matching previous engram for `{target_content}` — {}",
                                score.reasoning
                            );
                        }
                    }
                } else {
                    let _ = writeln!(
                        log.lock().unwrap(),
                        "[{ts}] retraction: skipped (no previous turn context) — {}",
                        score.reasoning
                    );
                }
            }

            let emotion = Emotion::from_str_checked(
                &score
                    .emotions
                    .first()
                    .map(|e| e.emotion.clone())
                    .unwrap_or_else(|| "neutral".to_owned()),
            )
            .unwrap_or(Emotion::Neutral);

            let embedding = embedder.embed(&user_input).ok();

            match db.insert_engram(
                &engram_text,
                emotion,
                score.importance,
                embedding.as_deref(),
            ) {
                Ok(id) => {
                    let _ = writeln!(
                        log.lock().unwrap(),
                        "[{ts}] id={id} importance={:.2} emotion={} retraction={} emb={} — {}",
                        score.importance,
                        emotion,
                        score.retraction,
                        embedding.is_some(),
                        score.reasoning
                    );

                    if let Err(e) = persona::evolve_after_turn(&db, &assistant_reply, &score) {
                        let _ = writeln!(log.lock().unwrap(), "[{ts}] persona evolve error: {e}");
                    }

                    match memory::lifecycle::run_maintenance(&db, &lifecycle_cfg) {
                        Ok(stats)
                            if !stats.skipped
                                && (stats.decayed > 0
                                    || stats.deleted > 0
                                    || stats.overflow_after_gc > 0) =>
                        {
                            let _ = writeln!(
                                log.lock().unwrap(),
                                "[{ts}] lifecycle: decayed={} deleted={} overflow={}",
                                stats.decayed,
                                stats.deleted,
                                stats.overflow_after_gc
                            );
                        }
                        Ok(_) => {}
                        Err(e) => {
                            let _ = writeln!(log.lock().unwrap(), "[{ts}] lifecycle error: {e}");
                        }
                    }
                }
                Err(e) => {
                    let _ = writeln!(log.lock().unwrap(), "[{ts}] eval error: {e}");
                }
            }
        }
        Err(e) => {
            let _ = writeln!(log.lock().unwrap(), "[{ts}] eval error: {e}");
        }
    }

    let _ = done_tx.send(());
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

    let eval_concurrency = app.config.chat.eval_concurrency;
    let (eval_tx, mut eval_rx) = mpsc::unbounded_channel::<EvalJob>();
    let eval_worker = tokio::spawn(async move {
        let sem = Arc::new(Semaphore::new(eval_concurrency));
        while let Some(job) = eval_rx.recv().await {
            let permit = sem.clone().acquire_owned().await.unwrap();
            tokio::spawn(async move {
                process_eval_job(job).await;
                drop(permit);
            });
        }
    });

    println!("=== Pgramma ===");
    println!("[system] {}", app.system_prompt);
    println!("Type /help for commands\n");

    let stdin = io::stdin();
    let mut recent_turns: Vec<String> = Vec::new();
    let mut pending_eval_tasks: Vec<oneshot::Receiver<()>> = Vec::new();

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
            run_test(
                &app,
                &eval_tx,
                path,
                &mut recent_turns,
                &mut pending_eval_tasks,
            )
            .await;
            continue;
        }

        // ── normal chat turn ────────────────────────────────────
        if let Some(done_rx) = chat_turn(&app, &eval_tx, &mut recent_turns, input).await {
            pending_eval_tasks.push(done_rx);
        }
    }

    let _ = wait_eval_tasks(&mut pending_eval_tasks).await;
    drop(eval_tx);
    let _ = eval_worker.await;
    println!("Bye!");
    Ok(())
}

/// Execute a single chat turn: recall → stream → queue background eval.
async fn chat_turn(
    app: &App,
    eval_tx: &mpsc::UnboundedSender<EvalJob>,
    recent_turns: &mut Vec<String>,
    input: &str,
) -> Option<oneshot::Receiver<()>> {
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

    let persona_prompt_suffix = match persona::build_prompt_suffix(&app.db) {
        Ok(v) => Some(v),
        Err(e) => {
            let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            let _ = writeln!(app.log.lock().unwrap(), "[{ts}] persona load error: {e}");
            None
        }
    };

    print!("→ ");
    std::io::stdout().flush().ok();
    let reply = match llm::chat_stream(
        &app.llm,
        &app.db,
        input,
        &app.system_prompt,
        persona_prompt_suffix.as_deref(),
        &memories,
        app.config.chat.context_window,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[error] {e}");
            return None;
        }
    };

    let previous_user_engram_text = recent_turns
        .last()
        .and_then(|t| extract_user_engram_text(t));
    let chat_text = format!("User: {input}\nAssistant: {reply}");
    let engram_text = format!("User: {input}");
    let context_for_eval: Vec<String> = recent_turns.clone();

    let eval_turns = app.config.chat.eval_context_turns;
    recent_turns.push(chat_text.clone());
    if recent_turns.len() > eval_turns {
        recent_turns.remove(0);
    }

    let (done_tx, done_rx) = oneshot::channel();
    let job = EvalJob {
        llm: app.llm.clone(),
        db: Arc::clone(&app.db),
        log: Arc::clone(&app.log),
        embedder: Arc::clone(&app.embedder),
        chat_text,
        engram_text,
        assistant_reply: reply,
        user_input: input.to_owned(),
        context_for_eval,
        previous_user_engram_text,
        lifecycle_cfg: app.config.lifecycle.clone(),
        done_tx,
    };
    if let Err(e) = eval_tx.send(job) {
        eprintln!("[error] failed to queue eval job: {e}");
        return None;
    }

    println!();
    Some(done_rx)
}

/// Parse test file, seed chat turns, wait for eval, then verify recall.
///
/// File format: one message per line.
/// - `# ...` → comment (skipped)
/// - `? ...` → recall verification query (run after seeding)
/// - anything else → seed message (sent as chat turn)
async fn run_test(
    app: &App,
    eval_tx: &mpsc::UnboundedSender<EvalJob>,
    path: &str,
    recent_turns: &mut Vec<String>,
    pending_eval_tasks: &mut Vec<oneshot::Receiver<()>>,
) {
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
        if let Some(done_rx) = chat_turn(app, eval_tx, recent_turns, msg).await {
            pending_eval_tasks.push(done_rx);
        }
    }

    // Wait for background eval tasks
    println!(
        "── waiting for eval tasks ({}) ──",
        pending_eval_tasks.len()
    );
    let completed = wait_eval_tasks(pending_eval_tasks).await;
    println!("── eval tasks done: {completed} ──");

    let engram_count = app.db.get_all_engrams().map(|v| v.len()).unwrap_or(0);
    println!("── engrams in DB: {engram_count} ──\n");

    // Recall verification phase
    if !queries.is_empty() {
        println!("── recall verification ({} queries) ──", queries.len());
        for query in &queries {
            println!("  Q: {query}");
            let min_importance = test_min_importance(query, app.config.recall.min_importance);
            let memories = memory::recall(
                &app.db,
                &app.embedder,
                query,
                3,
                min_importance,
                app.config.recall.cosine_weight,
            )
            .unwrap_or_default();
            if memories.is_empty() {
                println!("  → (no recall)");
            } else {
                for m in &memories {
                    let preview: String = m.content.chars().take(80).collect();
                    let ellipsis = if m.content.len() > 80 { "…" } else { "" };
                    println!(
                        "  → [rank={:.2} cos={:.2} imp={:.2}] {preview}{ellipsis}",
                        m.score, m.relevance, m.importance
                    );
                }
            }
            println!();
        }
    }

    println!("── test done ──\n");
}
