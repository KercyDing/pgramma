-- Pgramma .pgram internal table schema
-- Backend: SQLite + WAL mode + SQLCipher (AES-256)

-- 1. Persona config (KV store for system-level state)
CREATE TABLE IF NOT EXISTS persona_config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Reserved keys:
--   system_prompt   : initial system prompt
--   mood_baseline   : global mood enum baseline
--   total_tokens    : cumulative token counter (GC trigger threshold)

-- 2. Episodic memory (append-only absolute timeline)
CREATE TABLE IF NOT EXISTS episodic_memory (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    role      TEXT    NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content   TEXT    NOT NULL,
    timestamp TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- 3. Engrams (weighted RAG knowledge base with vector embeddings)
CREATE TABLE IF NOT EXISTS engrams (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    content       TEXT    NOT NULL,
    embedding     BLOB,                -- sqlite-vec vector field
    emotion       TEXT    NOT NULL DEFAULT 'neutral'
                         CHECK (emotion IN (
                             'neutral',
                             'joy', 'sadness', 'trust', 'disgust',
                             'fear', 'anger', 'surprise', 'anticipation',
                             'contempt'
                         )),
    importance    REAL    NOT NULL DEFAULT 0.0 CHECK (importance >= 0.0 AND importance <= 1.0),
    access_count  INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_engrams_importance ON engrams (importance);
CREATE INDEX IF NOT EXISTS idx_engrams_emotion    ON engrams (emotion);
