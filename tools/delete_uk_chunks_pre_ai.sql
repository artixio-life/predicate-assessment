-- Delete drug.product_chunks for United Kingdom products that have already
-- been text-extracted/chunked, but NOT yet AI-extracted — so the old chunks
-- (from before the SPC+PIL multi-document logic in
-- processing/text_extraction.py) are safe to discard: Stage C never read
-- them, so nothing is lost.
--
-- Deliberately does NOT touch a UK product whose ai_extraction_status is
-- already DONE/NEEDS_REVIEW/PROCESSING/FAILED — those already went through
-- Stage C on the old chunks; re-chunking them is a separate, explicit
-- decision (see the commented block at the bottom), not something this
-- script does implicitly.
--
-- Run the SELECT first to see exactly what would be affected, then the
-- transaction below it.

-- ============================================================================
-- STEP 1 — preview (read-only, run this first)
-- ============================================================================
SELECT p.id, p.product_name, p.text_extraction_status, p.ai_extraction_status,
       (SELECT count(*) FROM drug.product_chunks pc WHERE pc.product_id = p.id) AS chunks_now
FROM drug.products p
LEFT JOIN drug.regulatory_geography g ON g.id = p.country_id
WHERE g.country_name = 'United Kingdom'
  AND p.text_extraction_status = 'DONE'
  AND p.ai_extraction_status = 'PENDING'
ORDER BY p.id;

-- ============================================================================
-- STEP 2 — the actual delete, inside a transaction so you can verify before
-- committing. product_chunks.product_id -> drug.products.id has no other
-- foreign-key dependents, so this is a plain DELETE, not a cascade.
-- ============================================================================
BEGIN;

DELETE FROM drug.product_chunks pc
USING drug.products p
LEFT JOIN drug.regulatory_geography g ON g.id = p.country_id
WHERE pc.product_id = p.id
  AND g.country_name = 'United Kingdom'
  AND p.text_extraction_status = 'DONE'
  AND p.ai_extraction_status = 'PENDING';

-- Verify before committing: 0 remaining chunk rows for the targeted products.
SELECT count(*) AS remaining_chunks_for_targeted_products
FROM drug.product_chunks pc
JOIN drug.products p ON p.id = pc.product_id
LEFT JOIN drug.regulatory_geography g ON g.id = p.country_id
WHERE g.country_name = 'United Kingdom'
  AND p.text_extraction_status = 'DONE'
  AND p.ai_extraction_status = 'PENDING';

-- Sanity check: UK rows that ALREADY went through AI extraction must be
-- untouched — this count should match whatever it was before you ran this.
SELECT p.ai_extraction_status, count(*) AS chunk_rows
FROM drug.product_chunks pc
JOIN drug.products p ON p.id = pc.product_id
LEFT JOIN drug.regulatory_geography g ON g.id = p.country_id
WHERE g.country_name = 'United Kingdom' AND p.ai_extraction_status <> 'PENDING'
GROUP BY 1;

-- If the two SELECTs above look right: COMMIT;
-- If anything looks wrong:                ROLLBACK;
COMMIT;

-- ============================================================================
-- STEP 3 (separate, deliberate) — only the chunk rows are gone after Step 2.
-- text_extraction_status is still 'DONE', so the pipeline's claim query
-- (processing/claim.py) will NOT pick these products up again on its own —
-- they'd sit with zero chunks until this is run too. Uncomment when you're
-- ready to actually regenerate them with the new SPC+PIL chunking:
-- ============================================================================
-- UPDATE drug.products p
-- SET text_extraction_status = 'PENDING'
-- FROM drug.regulatory_geography g
-- WHERE g.id = p.country_id
--   AND g.country_name = 'United Kingdom'
--   AND p.text_extraction_status = 'DONE'
--   AND p.ai_extraction_status = 'PENDING';
