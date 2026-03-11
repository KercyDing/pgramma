pub mod embedder;
mod gc_worker;
mod rater;

use crate::db::PgramDb;
use crate::error::Result;
use crate::models::Engram;
use embedder::Embedder;

/// A recalled memory fragment ready for chat-context injection.
pub struct MemoryFragment {
    pub content: String,
    pub importance: f32,
    /// Cosine similarity to the query (0.0 when no embedding available).
    pub relevance: f32,
}

/// Retrieve the top-k engrams most relevant to `query`.
///
/// Pipeline: importance filter → embed query → cosine rank → top-k.
/// Each returned engram gets its access_count bumped via `touch`.
pub fn recall(
    db: &PgramDb,
    embedder: &Embedder,
    query: &str,
    top_k: usize,
    min_importance: f32,
    cosine_weight: f32,
) -> Result<Vec<MemoryFragment>> {
    let candidates = db.get_engrams_above(min_importance)?;
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let query_emb = embedder.embed(query)?;
    let imp_weight = 1.0 - cosine_weight;

    let mut scored: Vec<(f32, f32, &Engram)> = candidates
        .iter()
        .map(|eg| {
            let cosine = eg
                .embedding
                .as_ref()
                .map(|v| dot(&query_emb, v))
                .unwrap_or(0.0);
            let score = cosine_weight * cosine + imp_weight * eg.importance;
            (score, cosine, eg)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let results = scored
        .into_iter()
        .take(top_k)
        .map(|(_score, cosine, eg)| {
            let _ = db.touch_engram(eg.id);
            MemoryFragment {
                content: eg.content.clone(),
                importance: eg.importance,
                relevance: cosine,
            }
        })
        .collect();

    Ok(results)
}

/// Dot product of two L2-normalised vectors ≡ cosine similarity.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
