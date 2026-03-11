# Pgramma

[中文文档](README_ZH.md)

Pgramma is a local-first digital persona engine: one `.pgram` file stores identity, memory, and conversation history.

## Core Idea

- One file, one persona.
- Memory is append-only and not rewritten.
- Each turn follows: recall -> respond -> evaluate -> store.

## Quick Start

### Prerequisites

- Rust 1.80+ (edition 2024)
- At least one usable LLM provider endpoint

### Setup

```bash
git clone https://github.com/KercyDing/pgramma.git
cd pgramma
cp config.example.toml config.toml
# Edit config.toml: set active_provider, model, api_key
cargo run
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show commands |
| `/test <path>` | Seed from file and verify recall |
| `/stats` | Show engram and episode counts |
| `/quit` | Exit |

## Configuration

Use `config.example.toml` as the template and keep local edits in `config.toml`.

Key points:

- `llm.active_provider` selects provider.
- `llm.providers.<provider>.model` and `api_key` are required for the active provider.
- `embedding.model_id` controls local embedding model.
- `config.toml` is ignored by git.

If required fields are missing, startup prints a fatal message and exits gracefully.

## Architecture

- `src/main.rs`: REPL and app bootstrap
- `src/app/`: bootstrap, REPL loop, and command orchestration
- `src/config/`: TOML config parsing
- `src/domain/`: domain models and error types
- `src/db/`: redb-based storage
- `src/llm/`: multi-provider inference and evaluator
- `src/memory/`: embedding and recall pipeline

## Testing

```bash
cargo test
```

## License

[GPL-3.0](LICENSE)
