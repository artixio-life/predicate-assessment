-- How many LLM calls processing/llm_field_repair.py would make.
--
-- Mirrors that module's selection rule exactly (one call per qualifying
-- product; multi-component products are skipped without a call):
--   1. raw json_data component has non-empty pack_size text, AND
--   2. EITHER a stored presentation has pack_size null,
--      OR the raw text lists several configs (has a comma) while only one
--         presentation was stored (partially parsed), AND
--   3. the product has at most ONE component (repair_product skips
--      multi-component products rather than mismapping presentations).
--
-- Change the country on the WHERE line to check another one.

WITH candidates AS (
    SELECT
        p.id,
        jsonb_array_length(COALESCE(p.json_data->'components', '[]'::jsonb)) AS n_components,
        length(COALESCE(p.json_data->'components'->0->>'indications', ''))   AS indications_chars,
        length(COALESCE(p.json_data->'components'->0->>'pack_size', ''))     AS pack_size_chars
    FROM drug.products p
    LEFT JOIN drug.regulatory_geography g ON g.id = p.country_id
    WHERE g.country_name = 'Australia'
      AND p.json_data ? 'artg_id'
      -- (1) there is raw pack text to read
      AND EXISTS (
          SELECT 1 FROM jsonb_array_elements(p.json_data->'components') c
          WHERE COALESCE(btrim(c->>'pack_size'), '') <> ''
      )
      -- (2) the stored data looks wrong
      AND (
          EXISTS (
              SELECT 1 FROM jsonb_array_elements(p.product_data->'presentations') pr
              WHERE pr->'pack_size' = 'null'::jsonb OR pr->>'pack_size' IS NULL
          )
          OR (
              jsonb_array_length(COALESCE(p.product_data->'presentations', '[]'::jsonb)) = 1
              AND EXISTS (
                  SELECT 1 FROM jsonb_array_elements(p.json_data->'components') c
                  WHERE c->>'pack_size' LIKE '%,%'
              )
          )
      )
)
SELECT
    count(*)                                          AS rows_fetched,
    count(*) FILTER (WHERE n_components > 1)           AS skipped_multicomponent,
    count(*) FILTER (WHERE n_components <= 1)          AS llm_calls,
    -- Cost drivers. The system prompt is a fixed ~2,600 tokens per call; the
    -- rest of the prompt is the pack_size + indications text, and completion
    -- size tracks the indications text since that is what gets split and
    -- echoed back.
    round(avg(indications_chars) FILTER (WHERE n_components <= 1))  AS avg_indications_chars,
    max(indications_chars) FILTER (WHERE n_components <= 1)         AS max_indications_chars,
    count(*) FILTER (WHERE n_components <= 1 AND indications_chars > 4000) AS calls_with_large_indications,
    round(avg(pack_size_chars) FILTER (WHERE n_components <= 1))    AS avg_pack_size_chars
FROM candidates;
