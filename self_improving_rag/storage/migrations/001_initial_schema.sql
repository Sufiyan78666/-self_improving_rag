-- =============================================================
-- Migration 001: Initial Schema for Self-Improving RAG System
-- Compatible with SQLite (dev) and PostgreSQL (prod)
-- =============================================================

-- Stores user query sessions
CREATE TABLE IF NOT EXISTS rag_sessions (
    id          TEXT PRIMARY KEY,          -- UUID
    query       TEXT NOT NULL,             -- Raw user query
    query_embedding BLOB,                  -- Serialized numpy array (float32)
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Stores individual chunk retrievals for each session
CREATE TABLE IF NOT EXISTS rag_retrievals (
    id              TEXT PRIMARY KEY,      -- UUID
    session_id      TEXT NOT NULL,
    chunk_id        TEXT NOT NULL,         -- Unique chunk identifier
    chunk_text      TEXT NOT NULL,         -- Full text of the chunk
    source_doc      TEXT,                  -- Source document name/path
    rank_shown      INTEGER NOT NULL,      -- Rank shown to user (1-indexed)
    vector_score    REAL NOT NULL,         -- Raw FAISS similarity score
    reranker_score  REAL NOT NULL,         -- CrossEncoder reranker score
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES rag_sessions(id)
);

-- Stores individual user feedback signals
CREATE TABLE IF NOT EXISTS feedback_events (
    id          TEXT PRIMARY KEY,          -- UUID
    session_id  TEXT NOT NULL,
    chunk_id    TEXT NOT NULL,             -- Which chunk the feedback is about
    signal_type TEXT NOT NULL,             -- thumbs_up | thumbs_down | citation_click | dwell | re_query
    value       REAL NOT NULL,             -- Raw signal value
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES rag_sessions(id)
);

-- ==============================================================
-- Indexes for query performance
-- ==============================================================
CREATE INDEX IF NOT EXISTS idx_retrievals_session ON rag_retrievals(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_session   ON feedback_events(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_chunk     ON feedback_events(chunk_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created   ON rag_sessions(created_at);

-- ==============================================================
-- VIEW: training_pairs
-- Joins sessions + retrievals + aggregated feedback into
-- (query_text, chunk_text, relevance_score) triples for training.
--
-- Signal weights:
--   thumbs_up      -> +1.0
--   thumbs_down    -> -1.0
--   citation_click -> +0.8
--   dwell          -> +0.3
--   re_query       -> -0.4
-- Relevance score is clamped to [-1, 1] in application code.
-- ==============================================================
CREATE VIEW IF NOT EXISTS training_pairs AS
SELECT
    s.id            AS session_id,
    s.query         AS query_text,
    r.chunk_id      AS chunk_id,
    r.chunk_text    AS chunk_text,
    r.rank_shown    AS rank_shown,
    r.reranker_score AS reranker_score,
    COALESCE(
        SUM(
            CASE f.signal_type
                WHEN 'thumbs_up'      THEN 1.0  * f.value
                WHEN 'thumbs_down'    THEN -1.0 * f.value
                WHEN 'citation_click' THEN 0.8  * f.value
                WHEN 'dwell'          THEN 0.3  * f.value
                WHEN 're_query'       THEN -0.4 * f.value
                ELSE 0.0
            END
        ), 0.0
    ) AS relevance_score_raw
FROM rag_sessions s
JOIN rag_retrievals r ON r.session_id = s.id
LEFT JOIN feedback_events f ON f.session_id = s.id AND f.chunk_id = r.chunk_id
GROUP BY s.id, r.chunk_id;
