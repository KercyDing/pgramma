#[tokio::main]
async fn main() {
    if let Err(e) = pgramma::app::run().await {
        eprintln!("\x1b[31m[fatal]\x1b[0m {e}");
        eprintln!("\x1b[34m[hint]\x1b[0m Fix config.toml and run again.");
    }
}
