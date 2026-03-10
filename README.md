# Pgramma

A local-first digital persona engine with lifecycle evolution, emotion-weighted memory, and irreversible engram management.

## Architecture

- **Rust core** (`src/`) — async inference scheduling, SQLCipher-encrypted `.pgram` state container, memory GC
- **Python lab** (`notebooks/`) — prompt tuning and decay model prototyping via `instructor` + `pydantic`

## Status

Early stage — currently validating cognitive scoring prompts in notebooks before building the production Rust pipeline.
