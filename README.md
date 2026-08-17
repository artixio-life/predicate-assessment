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

Products with no document to OCR (registry-only sources) skip Stage B and
run Stage C directly off their carried-forward `json_data`.

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

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in the same DB/MinIO creds as source-predicate,
                        # plus MISTRAL_API_KEY and OPEN_ROUTER_API_KEY
python main.py
```

`main.py` applies `schema/schema.sql` (idempotent) and then runs all three
stages once. It expects the `drug` schema (regulatory-explorer) and
`source.drug_predicate_raw_records` (source-predicate) to already exist on
the target database — `schema/schema.sql` checks for both and raises a clear
error naming whichever is missing.

Set `PIPELINE_LIMIT` to cap how many rows each stage processes in one run.

## Out of scope (for now)

- The larger column backlog in source-predicate's `smpc_extraction_gap_analysis.md`
  (ATC code, adverse-reaction/dosing/interaction child tables, etc.).
- Splitting one raw record into multiple `drug.products` rows (e.g. one row
  per strength when a regulator issues separate registration numbers per
  strength) — promotion is 1:1 for now.
- RAG/embeddings-based extraction — this pipeline is a sequential chunk-fold
  instead (see Stage C above).
