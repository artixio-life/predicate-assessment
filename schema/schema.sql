-- ============================================================================
-- drug.products — parsed/AI-enriched product layer, downstream of
-- source.drug_predicate_raw_records (owned by the source-predicate crawler
-- repo). Lives in the shared `drug` schema (owned by the regulatory-explorer
-- app) rather than `source`, alongside drug.regulatory_geography and the
-- rest of the drug domain products are matched against. All three repos
-- point at the same Postgres database.
--
-- One flat table. Detail that is only ever displayed (never filtered on)
-- goes in product_data. Full column spec: schema/product_data_spec.md
-- ============================================================================

-- gen_random_uuid() is core in PG13+; pgcrypto is the fallback for older builds.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- The `drug` schema itself is owned/applied by the regulatory-explorer app,
-- and `source.drug_predicate_raw_records` by the source-predicate crawler.
-- Fail loudly and early if either hasn't been applied to this database yet,
-- rather than a cryptic "relation does not exist" FK error partway through.
DO $$
BEGIN
    IF to_regclass('drug.regulatory_geography') IS NULL THEN
        RAISE EXCEPTION 'drug.regulatory_geography not found. drug.products depends on the drug schema — apply the regulatory-explorer schema (database/schemas/drug_domain_schema.sql) to this database first.';
    END IF;
    IF to_regclass('source.drug_predicate_raw_records') IS NULL THEN
        RAISE EXCEPTION 'source.drug_predicate_raw_records not found. drug.products.raw_record_id depends on it — apply the source-predicate schema (schema/schema.sql) to this database first.';
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS drug.products (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- provenance: the raw record this product was promoted from (cross-repo,
    -- cross-schema FK back into source-predicate's crawl-output table).
    -- Makes promotion idempotent (one product row per raw record) and lets
    -- the pipeline re-pull document_url/json_data on a rerun.
    raw_record_id             INTEGER REFERENCES source.drug_predicate_raw_records(id) ON DELETE SET NULL,

    -- naming ----------------------------------------------------------------
    -- product_name: full name as printed, incl. strength and form.
    --               'Pregabalin Wockhardt 225 mg Hard Capsules'
    -- brand_name:   trade-name portion, English/translated. 'Pregabalin Wockhardt'
    -- brand_name_local: same in original script, for CN/BR/etc. '普瑞巴林胶囊'
    --               Never overwrite this with a translation — it is the only
    --               way to re-derive a bad translation without re-crawling.
    product_name              TEXT,
    brand_name                TEXT,
    brand_name_local          TEXT,
    generic_name              TEXT,

    -- geography ---------------------------------------------------------------
    -- Points at drug.regulatory_geography — the curated region/country/agency
    -- reference products are matched against — NOT at source.country, which
    -- stays the crawler-facing geography table for drug_predicate_raw_records
    -- and is unaffected by this. Resolution from a raw record's
    -- source.country row to this FK happens once, at promotion time (see
    -- processing/geography.py).
    country_id                INTEGER REFERENCES drug.regulatory_geography(id) ON DELETE SET NULL,
    regulator                 TEXT,

    -- companies -------------------------------------------------------------
    -- MAH (licence holder) and manufacturer are different roles. Most sources
    -- give you the MAH: SmPC section 7 here is Wockhardt UK Ltd.
    mah_name                  TEXT,
    mah_address               TEXT,
    manufacturer              TEXT,
    manufacturer_address      TEXT,

    -- registration ----------------------------------------------------------
    registration_number       TEXT,
    product_type              TEXT,
    status                    TEXT,          -- regulatory: Active | Withdrawn | Suspended
    registration_date         DATE,
    approval_date             DATE,
    market_authorization_date DATE,
    expiry_date               DATE,
    withdrawal_date           DATE,
    -- SmPC section 10. The label's own version stamp — the only reliable way
    -- to tell a re-crawl from a genuinely changed document.
    label_revision_date       DATE,

    -- classification --------------------------------------------------------
    atc_code                  TEXT,          -- 'N03AX16' — the cross-country match key

    -- clinical --------------------------------------------------------------
    therapeutic_areas         TEXT[],
    indications               TEXT[],
    symptoms                  TEXT[],        -- indication symptoms (what the patient has)
    adverse_reactions         TEXT[],        -- what the drug causes — NOT the same thing
    contraindications         TEXT[],
    active_ingredients        TEXT[],
    strengths                 TEXT[],        -- {'225 mg'}
    dosage_forms              TEXT[],
    routes                    TEXT[],

    is_generic                BOOLEAN,
    reference_product         TEXT,          -- originator, e.g. 'Lyrica'

    -- product_data: the curated, 6-key predicate-assessment contract this
    -- pipeline's AI extraction stage writes into. See product_data_spec.md.
    product_data              JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- json_data: raw structured metadata carried forward from
    -- drug_predicate_raw_records.json_data. Used as the AI extraction input
    -- for products with no document to extract text from (e.g. registry-only
    -- sources), and as extra context alongside document_text otherwise.
    json_data                 JSONB,

    -- document_text: extracted/OCR'd full text of source_url, chunked into
    -- drug.product_chunks for the AI extraction stage.
    document_text             TEXT,

    -- provenance + pipeline -------------------------------------------------
    source_url                TEXT,
    source_language           TEXT,          -- language of the source doc: en, zh, pt

    -- is_active is ROW currency, not regulatory status. A withdrawn product is
    -- status='Withdrawn' AND is_active=true — the row is still current truth.
    -- is_active=false means the row itself is superseded.
    is_active                 BOOLEAN NOT NULL DEFAULT TRUE,
    processing_status         TEXT NOT NULL DEFAULT 'PENDING',

    -- Per-step pipeline tracking, independent of the coarse processing_status
    -- summary above, so a rerun only redoes the step that actually failed.
    -- Each step gets 3 total attempts (see processing/retry.py).
    text_extraction_status    TEXT NOT NULL DEFAULT 'PENDING',
    text_extraction_attempts  INTEGER NOT NULL DEFAULT 0,
    text_extraction_error     TEXT,

    ai_extraction_status      TEXT NOT NULL DEFAULT 'PENDING',
    ai_extraction_attempts    INTEGER NOT NULL DEFAULT 0,
    ai_extraction_error       TEXT,

    created_at                TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at                TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_processing_status CHECK (processing_status IN
        ('PENDING','PARSED','ENRICHED','NEEDS_REVIEW','FAILED','SKIPPED')),
    -- PROCESSING: a worker has atomically claimed this row (see
    -- processing/claim.py) and is actively working it. Never set by hand —
    -- claiming flips PENDING/stale-PROCESSING straight to PROCESSING in one
    -- statement, so two workers can never claim the same row.
    CONSTRAINT chk_text_extraction_status CHECK (text_extraction_status IN
        ('PENDING','PROCESSING','DONE','SKIPPED','FAILED')),
    CONSTRAINT chk_ai_extraction_status CHECK (ai_extraction_status IN
        ('PENDING','PROCESSING','DONE','NEEDS_REVIEW','SKIPPED','FAILED')),

    -- upsert key on re-crawl (NULLs never conflict with each other, so this
    -- doesn't dedupe rows with an unresolved country or no captured
    -- registration number — that's fine, raw_record_id below is the real
    -- promotion idempotency key)
    CONSTRAINT uq_products_country_regnum UNIQUE (country_id, registration_number),

    -- one product row per raw record (Stage A promotion is 1:1 for now —
    -- see README on splitting multi-strength records later)
    CONSTRAINT uq_products_raw_record UNIQUE (raw_record_id)
);

-- Migration for a drug.products created before PROCESSING existed: widen
-- the two CHECK constraints on an already-live table. No-op on a fresh
-- CREATE TABLE above, since it already has the constraint in this shape.
ALTER TABLE drug.products DROP CONSTRAINT IF EXISTS chk_text_extraction_status;
ALTER TABLE drug.products ADD CONSTRAINT chk_text_extraction_status CHECK (text_extraction_status IN
    ('PENDING','PROCESSING','DONE','SKIPPED','FAILED'));
ALTER TABLE drug.products DROP CONSTRAINT IF EXISTS chk_ai_extraction_status;
ALTER TABLE drug.products ADD CONSTRAINT chk_ai_extraction_status CHECK (ai_extraction_status IN
    ('PENDING','PROCESSING','DONE','NEEDS_REVIEW','SKIPPED','FAILED'));

-- raw_record_id already has an index via its UNIQUE constraint above.
CREATE INDEX IF NOT EXISTS idx_products_country     ON drug.products(country_id);
CREATE INDEX IF NOT EXISTS idx_products_generic     ON drug.products(lower(generic_name));
CREATE INDEX IF NOT EXISTS idx_products_atc         ON drug.products(atc_code);
CREATE INDEX IF NOT EXISTS idx_products_regnum      ON drug.products(registration_number);
CREATE INDEX IF NOT EXISTS idx_products_queue       ON drug.products(processing_status)
    WHERE processing_status IN ('PENDING','NEEDS_REVIEW');
CREATE INDEX IF NOT EXISTS idx_products_text_extraction_queue ON drug.products(text_extraction_status)
    WHERE text_extraction_status = 'PENDING';
CREATE INDEX IF NOT EXISTS idx_products_ai_extraction_queue ON drug.products(ai_extraction_status)
    WHERE ai_extraction_status = 'PENDING';
CREATE INDEX IF NOT EXISTS idx_products_ingredients ON drug.products USING GIN(active_ingredients);
CREATE INDEX IF NOT EXISTS idx_products_strengths   ON drug.products USING GIN(strengths);
CREATE INDEX IF NOT EXISTS idx_products_indications ON drug.products USING GIN(indications);
CREATE INDEX IF NOT EXISTS idx_products_data        ON drug.products USING GIN(product_data jsonb_path_ops);


-- ============================================================================
-- drug.product_chunks — text chunks of drug.products.document_text, fed
-- one at a time (in order) to the AI extraction stage, which carries its
-- accumulated JSON result forward from one chunk to the next.
-- ============================================================================

CREATE TABLE IF NOT EXISTS drug.product_chunks (
    id           SERIAL PRIMARY KEY,
    product_id   UUID NOT NULL REFERENCES drug.products(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    chunk_text   TEXT NOT NULL,
    char_count   INTEGER,
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (product_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_product_chunks_product ON drug.product_chunks(product_id);
