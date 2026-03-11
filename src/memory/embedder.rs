use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::api::sync::ApiBuilder;
use tokenizers::Tokenizer;

use crate::error::{PgrammaError, Result};

/// Local sentence-embedding encoder backed by candle (CPU).
///
/// Loads a BERT-based sentence-transformer from HuggingFace Hub on first call,
/// then produces L2-normalised vectors (dimension depends on the model).
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

/// Shorthand: convert any Display-able error into PgrammaError::Embedding.
fn emb(e: impl std::fmt::Display) -> PgrammaError {
    PgrammaError::Embedding(e.to_string())
}

impl Embedder {
    /// Download (or use cached) the given sentence-transformer and build on CPU.
    pub fn load(model_id: &str) -> Result<Self> {
        let device = Device::Cpu;
        let cache = hf_hub::Cache::from_env();
        let api = ApiBuilder::from_cache(cache).build().map_err(emb)?;
        let repo = api.model(model_id.to_string());

        let config_path = repo.get("config.json").map_err(emb)?;
        let tokenizer_path = repo.get("tokenizer.json").map_err(emb)?;
        let weights_path = repo.get("model.safetensors").map_err(emb)?;

        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(config_path).map_err(emb)?)
                .map_err(emb)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(emb)?;

        // SAFETY: the safetensors file is downloaded from a trusted source (HF Hub)
        // and is valid by construction.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(emb)?
        };

        let model = BertModel::load(vb, &config).map_err(emb)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Produce a 384-dim L2-normalised embedding for `text`.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text, true).map_err(emb)?;

        let ids = encoding.get_ids();
        let type_ids = encoding.get_type_ids();
        let mask = encoding.get_attention_mask();
        let n = ids.len();

        let input_ids = Tensor::new(ids, &self.device)
            .map_err(emb)?
            .reshape((1, n))
            .map_err(emb)?;

        let token_type_ids = Tensor::new(type_ids, &self.device)
            .map_err(emb)?
            .reshape((1, n))
            .map_err(emb)?;

        let attention_mask = Tensor::new(mask, &self.device)
            .map_err(emb)?
            .reshape((1, n))
            .map_err(emb)?;

        // Forward pass → (1, seq_len, dim); cast to f32 for stable pooling
        let output = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(emb)?
            .to_dtype(DType::F32)
            .map_err(emb)?;

        // Mean pooling weighted by attention mask
        let mask_f = attention_mask
            .to_dtype(DType::F32)
            .map_err(emb)?
            .unsqueeze(2)
            .map_err(emb)?; // (1, seq_len, 1)

        let sum_emb = output
            .broadcast_mul(&mask_f)
            .map_err(emb)?
            .sum(1)
            .map_err(emb)?; // (1, 384)

        let count = mask_f.sum(1).map_err(emb)?; // (1, 1)

        let mean = sum_emb.broadcast_div(&count).map_err(emb)?;

        // L2 normalise
        let norm = mean
            .sqr()
            .map_err(emb)?
            .sum_keepdim(1)
            .map_err(emb)?
            .sqrt()
            .map_err(emb)?; // (1, 1)

        let normalised = mean.broadcast_div(&norm).map_err(emb)?;

        normalised
            .squeeze(0)
            .map_err(emb)? // (384,)
            .to_vec1::<f32>()
            .map_err(emb)
    }

    /// Convenience wrapper — embed each text independently.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}
