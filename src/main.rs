use pgramma::db::PgramDb;
use pgramma::models::{Emotion, Role};

fn main() {
    let db_path = "demo.pgram";

    println!("=== Pgramma DB Layer Demo ===\n");

    let db = PgramDb::open(db_path).expect("failed to open db");
    println!("[+] Opened {db_path}");

    // persona config
    db.set_config("system_prompt", "You are a thoughtful assistant.")
        .unwrap();
    db.set_config("mood_baseline", "neutral").unwrap();
    println!(
        "[+] system_prompt = {:?}",
        db.get_config("system_prompt").unwrap()
    );

    // episodic memory
    db.append_episode(Role::System, "You are a thoughtful assistant.")
        .unwrap();
    db.append_episode(Role::User, "Tell me about Rust.")
        .unwrap();
    db.append_episode(
        Role::Assistant,
        "Rust is a systems programming language focused on safety and performance.",
    )
    .unwrap();

    let episodes = db.get_episodes(10, 0).unwrap();
    println!(
        "\n[+] Episodic memory ({} entries, newest first):",
        episodes.len()
    );
    for ep in &episodes {
        println!("    #{} [{}] {}", ep.id, ep.role, ep.content);
    }

    // engrams
    let id1 = db
        .insert_engram(
            "Rust ownership model prevents data races",
            Emotion::Trust,
            0.85,
            None,
        )
        .unwrap();
    let _id2 = db
        .insert_engram("User prefers concise answers", Emotion::Neutral, 0.6, None)
        .unwrap();
    let id3 = db
        .insert_engram("Temporary test engram", Emotion::Surprise, 0.2, None)
        .unwrap();

    db.touch_engram(id1).unwrap();

    println!("\n[+] Engrams above 0.5 importance:");
    for eg in db.get_engrams_above(0.5).unwrap() {
        println!(
            "    #{} [{}] imp={:.2} acc={} — {}",
            eg.id, eg.emotion, eg.importance, eg.access_count, eg.content
        );
    }

    let deleted = db.delete_engram(id3).unwrap();
    println!("\n[+] Deleted engram #{id3}: {deleted}");
    println!(
        "[+] Remaining engrams: {}",
        db.get_engrams_above(0.0).unwrap().len()
    );

    drop(db);
    std::fs::remove_file(db_path).ok();
    println!("\n[+] Cleaned up {db_path}");
    println!("=== Demo complete ===");
}
