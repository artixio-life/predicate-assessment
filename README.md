# Predicate Assessment — AI Extraction Pipeline

Turns crawled regulatory documents into structured, matchable product data.
Downstream of [`source-predicate`](../source-predicate) (the crawler) and
alongside [`regulatory-explorer`](../regulatory-explorer) (owns the shared
`drug` schema) — all three point at the same Postgres database.

```
source.drug_predicate_raw_records   (source-predicate)
              │
              ▼
   Stage A — promote            processing/promote.py
   raw record -> drug.products row, resolves country -> drug.regulatory_geography
              │
              ▼
   Stage B — text extraction    processing/text_extraction.py
   download PDF -> pdf-inspector (text-layer) or Mistral OCR (scanned)
   -> document_text -> chunked into drug.product_chunks
              │
              ▼
   Stage C — AI extraction      processing/ai_extraction.py
   agentic chunk-fold over OpenRouter: each chunk's LLM call is shown the
   JSON accumulated from every prior chunk and returns the updated whole
   -> drug.products.product_data (6-key contract, schema/product_data_spec.md)
```

European Union products are the one exception to "chunk the whole
document": their PDF is Annex I (SmPC) + Annex II (manufacturer/marketing
conditions) + Annex III (Labelling + Package Leaflet), and Annex I alone can
run to hundreds of pages of clinical narrative the AI extraction fold
doesn't need — Annex III is a much shorter, already-summarised version of
the same core facts. `processing/eu_section_extraction.py` slices out Annex
III before chunking (falling back to the full document if that section
isn't found); `document_text` itself always keeps the full extracted text.

Products with no document to OCR (registry-only sources) skip Stage B and
run Stage C directly off their carried-forward `json_data`. Products that
*do* have a document still get their `json_data` folded in alongside the
document's chunks (both are chunked the same way) — a crawler's structured
fields (application number, product table, approval history, TE
cross-references, ...) are often not restated anywhere in the document's
prose, so this isn't only a no-document fallback.

## Status tracking

Every step gets 3 total attempts (`processing/retry.py`) before giving up.
Two fine-grained columns track the two AI-pipeline stages independently, so
a rerun only redoes what actually failed:

- `text_extraction_status` / `_attempts` / `_error`
- `ai_extraction_status` / `_attempts` / `_error`

The coarse `processing_status` summarizes both:
`PENDING → PARSED → ENRICHED` on the happy path, or `NEEDS_REVIEW` /
`FAILED` when a step couldn't fully complete.

## Setup

**1. Apply the schema once, by hand.** `main.py` deliberately does NOT do this
— run `schema/schema.sql` against the target database yourself (every
statement in it is idempotent, so re-running is safe):

```bash
psql "$DATABASE_URL" -f schema/schema.sql
```

It expects the `drug` schema (regulatory-explorer) and
`source.drug_predicate_raw_records` (source-predicate) to already exist, and
checks for both up front, raising a clear error naming whichever is missing
rather than failing on an opaque FK error. If you'd rather apply it from
Python, `db.init_db()` does the same thing.

**2. Run the pipeline.**

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in the same DB/MinIO creds as source-predicate,
                        # plus MISTRAL_API_KEY and OPEN_ROUTER_API_KEY
python main.py         # or: docker compose up
```

`PIPELINE_LIMIT` caps rows per stage, `PIPELINE_COUNTRY` scopes to one
country, `PIPELINE_WORKERS` sets concurrency (default 4).

**Running only some stages.** `PIPELINE_STAGES` selects which of the three
run — useful to stop promoting new raw records and only re-process rows that
are already in `drug.products` (e.g. after changing the extraction prompt):

```bash
PIPELINE_STAGES=text,ai   python main.py   # skip promotion
PIPELINE_STAGES=ai        python main.py   # AI extraction only
```

Unset means all three. Note the stages are still gated by row status, so
`PIPELINE_STAGES=ai` only touches rows sitting at
`processing_status='PARSED'` + `ai_extraction_status='PENDING'` — reset a row
first if you want it re-extracted.

## Out of scope (for now)

- The larger column backlog in source-predicate's `smpc_extraction_gap_analysis.md`
  (ATC code, adverse-reaction/dosing/interaction child tables, etc.).
- Splitting one raw record into multiple `drug.products` rows (e.g. one row
  per strength when a regulator issues separate registration numbers per
  strength) — promotion is 1:1 for now.
- RAG/embeddings-based extraction — this pipeline is a sequential chunk-fold
  instead (see Stage C above).
