-- Migration 002: Experiment Tracking for Training Runs

CREATE TABLE IF NOT EXISTS experiment_runs (
    id              TEXT PRIMARY KEY,      -- UUID
    model_name      TEXT NOT NULL,         -- e.g., MiniLM-L6-v2
    dataset_size    INTEGER NOT NULL,      -- Number of pairs used for training
    epochs          INTEGER NOT NULL,
    ndcg_before     REAL,                  -- NDCG@3 on eval set before training
    ndcg_after      REAL,                  -- NDCG@3 on eval set after training
    mrr_after       REAL,                  -- MRR on eval set after training
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runs_created ON experiment_runs(created_at);
