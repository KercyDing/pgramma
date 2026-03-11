use crate::db::PgramDb;
use crate::error::Result;
use crate::models::Engram;

use super::embedder::Embedder;

/// A recalled memory fragment ready for chat-context injection.
pub struct MemoryFragment {
    pub content: String,
    /// Final ranking score after combining cosine/importance/lexical bonuses.
    pub score: f32,
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
    let query_lc = query.to_ascii_lowercase();

    let mut scored: Vec<(f32, f32, &Engram)> = candidates
        .iter()
        .map(|eg| {
            let cosine = eg
                .embedding
                .as_ref()
                .map(|v| dot(&query_emb, v))
                .unwrap_or(0.0);
            let lexical = lexical_bonus(&query_lc, &eg.content);
            let score = cosine_weight * cosine + imp_weight * eg.importance + lexical;
            (score, cosine, eg)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let results = scored
        .into_iter()
        .take(top_k)
        .map(|(score, cosine, eg)| {
            let _ = db.touch_engram(eg.id);
            MemoryFragment {
                content: eg.content.clone(),
                score,
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

fn lexical_bonus(query_lc: &str, content: &str) -> f32 {
    let content_lc = content.to_ascii_lowercase();
    let mut bonus = 0.0_f32;

    let asks_sibling = query_lc.contains("sibling");
    if asks_sibling && (content_lc.contains("sister") || content_lc.contains("brother")) {
        bonus += 0.25;
    }
    if asks_sibling && query_lc.contains("name") && content_lc.contains("named") {
        bonus += 0.10;
    }

    let asks_address = query_lc.contains("where do i live")
        || query_lc.contains("live now")
        || query_lc.contains("address");
    if asks_address
        && (content_lc.contains("live on")
            || content_lc.contains("moved")
            || content_lc.contains("street")
            || content_lc.contains("avenue"))
    {
        bonus += 0.15;
    }

    if query_lc.contains("what should you call me") && content_lc.contains("call me") {
        bonus += 0.20;
    }

    bonus
}

#[cfg(test)]
mod tests {
    use super::lexical_bonus;

    #[test]
    fn sibling_bonus_hits_kinship_terms() {
        let q = "what is my sibling's name";
        let c = "User: I was chatting with my sister Emma yesterday";
        assert!(lexical_bonus(q, c) >= 0.25);
    }

    #[test]
    fn address_bonus_hits_move_updates() {
        let q = "where do i live now";
        let c = "User: Actually I moved again yesterday, now I live on Pine Avenue";
        assert!(lexical_bonus(q, c) >= 0.15);
    }
}
