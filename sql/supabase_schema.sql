-- LLM Learning production schema for Supabase Postgres + pgvector
-- Run once per Supabase project before backend/mcp rollout.

CREATE EXTENSION IF NOT EXISTS vector;

-- =========================
-- Memory / chat state
-- =========================

CREATE TABLE IF NOT EXISTS history (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT,
    query TEXT,
    response TEXT,
    chunk_index INTEGER,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_history_session_timestamp
ON history(session_id, timestamp DESC, id DESC);

CREATE TABLE IF NOT EXISTS file_summaries (
    file_id TEXT PRIMARY KEY,
    summary TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conversation_state (
    session_id TEXT PRIMARY KEY,
    state_json JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    scoped_session_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT,
    title TEXT NOT NULL DEFAULT 'Phiên mới',
    selected_program_id TEXT,
    selected_file_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    archived_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
ON chat_sessions(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    scoped_session_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY(scoped_session_id) REFERENCES chat_sessions(scoped_session_id)
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
ON chat_messages(scoped_session_id, created_at ASC, id ASC);

CREATE TABLE IF NOT EXISTS legacy_chat_migrations (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    migrated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =========================
-- Vector store
-- =========================

CREATE TABLE IF NOT EXISTS vector_documents (
    id BIGSERIAL PRIMARY KEY,
    file_id TEXT,
    file_name TEXT,
    source TEXT,
    page INTEGER,
    chunk_index INTEGER,
    source_line INTEGER,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    page_content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vector_documents_file_id
ON vector_documents(file_id);

CREATE INDEX IF NOT EXISTS idx_vector_documents_file_name
ON vector_documents(file_name);

CREATE INDEX IF NOT EXISTS idx_vector_documents_page
ON vector_documents(page);

CREATE TABLE IF NOT EXISTS vector_embeddings (
    document_id BIGINT PRIMARY KEY REFERENCES vector_documents(id) ON DELETE CASCADE,
    embedding VECTOR(1024) NOT NULL,
    normalized BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vector_embeddings_ann
ON vector_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
