CREATE EXTENSION IF NOT EXISTS vector;

-- One row per document. The documents of this corpus are 60 to 160 words, so a document is
-- embedded whole and there is no chunk table.
CREATE TABLE IF NOT EXISTS documents (
    id        text PRIMARY KEY,
    kind      text NOT NULL CHECK (kind IN ('incident', 'runbook')),
    service   text NOT NULL,
    title     text NOT NULL,
    body      text NOT NULL,
    embedding vector(384) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || body)) STORED
);

CREATE INDEX IF NOT EXISTS documents_tsv ON documents USING gin (tsv);

-- There is no index on `embedding` on purpose. Without one the nearest-neighbour search is an
-- exact scan, which costs nothing at this size and returns the same ranking every time. An HNSW
-- index would make the search approximate, and the evaluation would then measure the index as
-- well as the retrieval method.
