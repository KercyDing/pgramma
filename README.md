# Pgramma

[中文文档](README_ZH.md)

A local-first digital persona engine — one file, one identity, irreversible memory.

## Philosophy

A digital persona is not a chatbot with a database. It is a **living state** — an accumulation of every interaction, emotion, and decision that shapes its identity over time.

Pgramma is built on three principles:

- **One file, one persona.** A single `.pgram` (Persona Engram) file encapsulates everything — memory, emotion baseline, personality configuration. Copy the file, and the persona travels with it.
- **Memory is irreversible.** Just like biological memory, engrams are written once and never modified. They may decay, be superseded by newer memories, or fade in relevance — but the original trace is permanent. The episodic timeline is append-only.
- **Cognition is evaluated, not hardcoded.** Every conversation turn passes through a cognitive evaluator that scores importance (0.0–1.0) and classifies emotion. The persona decides what matters, not a rule engine.

## How It Works

```
User input
    │
    ├─→ Recall       importance filter → cosine rank → top-k
    │       │
    │       ▼
    ├─→ Inference    system prompt + recalled engrams + recent episodes → stream
    │       │
    │       ▼
    └─→ Evaluation   importance scoring + emotion tagging + embedding (background)
                │
                ▼
           Engram written to .pgram
```

Every interaction follows this cycle: **recall → respond → evaluate → remember**. The evaluation runs asynchronously — the persona reflects on each exchange after the fact, just as a person might.

## Features

- **Semantic Recall** — retrieves relevant past memories via embedding similarity, not recency
- **Cognitive Evaluation** — LLM-scored importance with a calibrated rubric (identity anchors > emotional depth > preferences > small talk)
- **Local Embeddings** — sentence vectors via [candle](https://github.com/huggingface/candle) (pure Rust, no Python/ONNX)
- **Emotion Spectrum** — Plutchik-based 10-category classification per engram
- **Retraction Detection** — contradictions are flagged, but the original memory persists
- **Streaming Inference** — real-time token output via OpenAI-compatible APIs
- **Single-File Persona** — [redb](https://github.com/cberner/redb) embedded database, portable `.pgram` (Persona Engram) container
- **Lifecycle Evolution** — memory decay, GC, and personality drift (planned)

## Quick Start

### Prerequisites

- Rust 1.80+ (edition 2024)
- An OpenAI-compatible API endpoint

### Setup

```bash
git clone https://github.com/KercyDing/pgramma.git
cd pgramma

cp .env.example .env
# Edit .env: set OPENAI_API_KEY (and optionally OPENAI_BASE_URL, HF_HOME)

cargo run
```

On first launch, the embedding model (~470MB) is downloaded from HuggingFace Hub.

### Usage

```
=== Pgramma ===
Type /help for commands

> My name is Alex
→ Nice to meet you, Alex! ...

  ... 50 messages later ...

> Do you remember my name?
→ Of course — you're Alex.

> /stats
  engrams: 42  episodes: 104

> /quit
Bye!
```

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/test <path>` | Batch seed from file and verify recall |
| `/stats` | Show engram and episode counts |
| `/quit` | Exit |

## Configuration

### `config.toml` — Persona Behavior

```toml
[llm]
model = "gpt-4o"

[embedding]
model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

[recall]
top_k = 8
min_importance = 0.3
cosine_weight = 0.7

[chat]
context_window = 20
eval_context_turns = 3
default_system_prompt = "You are a thoughtful assistant with emotional awareness."
```

All fields have defaults — the engine works without a config file.

### `.env` — Secrets

```bash
OPENAI_API_KEY=sk-...          # Required
OPENAI_BASE_URL=https://...    # Optional
HF_HOME=/path/to/cache         # Optional
```

## Architecture

```
src/
├── main.rs              # REPL + App state
├── config.rs            # TOML config with defaults
├── models.rs            # Engram, Emotion, EpisodicEntry
├── error.rs             # Unified error type
├── db/
│   ├── mod.rs           # Engram / episode / config CRUD
│   └── connection.rs    # redb table definitions
├── llm/
│   ├── client.rs        # OpenAI client wrapper (rig-core)
│   ├── inference.rs     # Streaming chat + memory injection
│   └── evaluator.rs     # Cognitive scorer (importance + emotion)
└── memory/
    ├── mod.rs           # Recall pipeline (filter → rank → top-k)
    └── embedder.rs      # Candle BERT encoder (CPU)

notebooks/               # Python lab — prompt tuning & decay modeling
```

### The `.pgram` Container

The `.pgram` file is a redb database containing three logical layers:

| Layer | Table | Role |
|-------|-------|------|
| **Subconscious** | `persona_config` | System prompt, emotion baseline, counters |
| **Timeline** | `episodic_memory` | Append-only interaction log (immutable) |
| **Engrams** | `engrams` | Weighted memory with embeddings, emotion, importance |

One file. Fully portable. No external state.

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Core | Rust | Performance, safety, single binary |
| LLM | [rig-core](https://github.com/0xPlaygrounds/rig) | Structured extraction, streaming |
| Embeddings | [candle](https://github.com/huggingface/candle) | Pure Rust inference, no FFI |
| Model | paraphrase-multilingual-MiniLM-L12-v2 | Multilingual, 384-dim |
| Storage | [redb](https://github.com/cberner/redb) | Embedded, ACID, single-file |

## Testing

```bash
cargo test                    # Unit tests
cargo run -- <<< '/test seed.txt'   # Recall validation
```

Seed file format:
```
# Comment (skipped)
I have a dog named Biscuit    # Sent as chat message
? What pet do I have          # Recall query (verified after seeding)
```

## Roadmap

- [ ] Memory decay — time-weighted importance degradation
- [ ] GC worker — low-weight pruning + high-weight summarization
- [ ] Personality drift — emotion baseline evolves with interaction history
- [ ] Local inference — GGUF model support via candle (fully offline)
- [ ] GUI shell — frontend for the `.pgram` container

## License

[GPL-3.0](LICENSE)
