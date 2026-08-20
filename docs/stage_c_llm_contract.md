# Stage C — what the LLM is actually asked to do

Everything a replacement model (local or hosted) needs in order to be dropped
into `processing/ai_extraction.py` and produce output the rest of the pipeline
accepts. Source of truth for each number is named so it can be re-checked.

Run the benchmark with `tools/local_llm_harness.py` — it imports the real
prompt, chunker and fold loop, so the local model sees byte-for-byte what
OpenRouter sees, and writes nothing back to the database.

```bash
export LLM_SERVICE_BASE_URL=http://localhost:8000/v1
export LLM_SERVICE_MODEL=gemma-4-26b
export LLM_SERVICE_CONTEXT_TOKENS=22000
python tools/local_llm_harness.py --country "United States" --limit 8 --diff-stored
```

Provider routing lives in `processing/llm_service_client.py` (ported from
regulatory-explorer): **self-hosted first, automatic OpenRouter fallback** when
the local endpoint is unreachable or keeps erroring — sticky after
`LLM_SERVICE_FALLBACK_AFTER` consecutive failures, so a dead endpoint does not
cost the full retry/backoff per product. `health_check()` probes both before any
work is spent, and the run summary reports the per-provider call split, so a run
that quietly degraded to the paid API is visible rather than mistaken for a
local result. See §8.

---

## 1. The job in one paragraph

For each product, the pipeline has a pile of text — a regulatory document
(SmPC / label / approval letter, OCR'd to plain text) and/or the crawler's raw
registry JSON. That pile is cut into **excerpts**. The model is shown a batch of
excerpts *plus the JSON it has built so far*, and must return **the complete
updated JSON object** — not a diff. The object returned after the last batch is
the product's extraction. This is a **sequential fold**, not RAG and not
map-reduce: batch *N* cannot start until batch *N-1* returns.

```
excerpts[1..7]   + {}            -> LLM -> J1
excerpts[8..14]  + J1            -> LLM -> J2
excerpts[15..21] + J2            -> LLM -> J3   ... and so on
                                            J_final is what gets stored
```

## 2. Request shape

Exactly two messages, every call. No tools, no function calling, no streaming.

| parameter | value | where |
| --- | --- | --- |
| `messages[0].role` | `system` | `SYSTEM_PROMPT`, `ai_extraction.py:97` |
| `messages[1].role` | `user` | `_build_user_message`, `ai_extraction.py:352` |
| `temperature` | `0` | `_call_llm_merge` |
| `max_tokens` | `$AI_EXTRACTION_MAX_TOKENS` (8192) | sized per call against the window |
| `response_format` | `{"type":"json_object"}` | dropped automatically if the server rejects it |
| primary endpoint | `$LLM_SERVICE_BASE_URL` | self-hosted, `llm_service_client` |
| primary model | `$LLM_SERVICE_MODEL` | e.g. `gemma-4-26b` |
| fallback endpoint | `https://openrouter.ai/api/v1` | only on failure |
| fallback model | `$EXTRACTOR_MODEL` / `$LLM_FALLBACK_MODEL` | paid; logged loudly |

The system prompt is **4,878 tokens / 20,930 chars** and is resent on every
call. It is one long spec of the output contract — worth checking that your
local model's context window and its instruction-following both hold up at that
size, because the whole schema lives in there.

**User message template** (verbatim structure):

```
Product: {product_name}
Registration number: {registration_number}

Excerpts {batch_start}-{batch_end} of {total} (read them in order; they are
consecutive pieces of the same document).

JSON accumulated so far:
{compact JSON of the whole accumulator}

New excerpts:
--- Excerpt {i} of {total} ---
{excerpt text}

--- Excerpt {i+1} of {total} ---
{excerpt text}
...
```

## 3. Required output

One JSON object, **no markdown fences, no prose**, with exactly two top-level
keys. Full field-by-field semantics are in the system prompt itself; the shape:

```json
{
  "columns": {
    "product_name_en": null, "brand_name": null, "brand_name_local": null,
    "generic_name": null, "mah_name": null, "mah_address": null,
    "manufacturer": null, "manufacturer_address": null,
    "registration_number": null, "product_type": null, "status": null,
    "registration_date": null, "approval_date": null,
    "market_authorization_date": null, "expiry_date": null,
    "withdrawal_date": null, "label_revision_date": null, "atc_code": null,
    "is_generic": null, "reference_product": null, "source_language": null,
    "therapeutic_areas": [], "indications": [], "symptoms": [],
    "adverse_reactions": [], "contraindications": [], "active_ingredients": [],
    "strengths": [], "dosage_forms": [], "routes": []
  },
  "product_data": {
    "_schema_version": "1",
    "substance": {"inn": null, "salt_form": null, "modality": null,
                  "target": null, "moa": null},
    "presentations": [{"per_value": null, "per_unit": null, "pack_size": null,
                       "pack_unit": null, "form": null, "route": null,
                       "active_ingredients": [{"substance": "", "strength_value": null,
                                               "strength_unit": null, "salt": null}]}],
    "indications": [{"condition": "", "population": null, "line_of_therapy": null,
                     "biomarker": null, "approval_date": null}],
    "approval": {"pathway": null, "registration_class": null, "conditional": null,
                 "priority_review": null, "reference_product": null},
    "pivotal_evidence": [{"study_id": null, "design": "", "n": null, "endpoint": "",
                          "value": null, "unit": null, "comparator": null,
                          "outcome": null}],
    "key_risks": [],
    "excipients": [{"name": "", "name_local": null, "value": null, "unit": null}]
  }
}
```

Enumerated fields — a local model that invents its own labels here breaks
downstream matching:

- `substance.modality` — `small_molecule` | `biologic` | `adc` | `vaccine` | `peptide`
- `approval.pathway` — `innovator` | `generic` | `biosimilar` | `hybrid` | `similar`
- `pivotal_evidence[].outcome` — `met` | `not_met` | `null`
- all dates — `YYYY-MM-DD`
- all free text — **English**, except `brand_name_local`, `mah_name`,
  `manufacturer`, `registration_number`, `atc_code`, `source_language`

### Acceptance gate

`_extract_one` (`ai_extraction.py:589`) decides the row's fate from the final
object:

| condition | `ai_extraction_status` |
| --- | --- |
| all 7 `product_data` keys present **and** no failed batch | `DONE` |
| any batch failed, or a key is missing | `NEEDS_REVIEW` |
| both `columns` and `product_data` empty | `FAILED` |

The 7 keys: `substance`, `presentations`, `indications`, `approval`,
`pivotal_evidence`, `key_risks`, `excipients`. **Present, not non-empty** — an
empty array counts, a missing key does not. A local model that omits keys it has
nothing for will push every row to `NEEDS_REVIEW`.

### Parsing tolerance

`processing/json_utils.py` runs `json.loads` → sanitize (strip ``` fences,
control chars, prose around the object) → repair (literal newlines in strings,
trailing commas, unclosed braces from truncation). So fenced or slightly
malformed JSON still lands. The harness reports `JSON-REPAIRED` and `TRUNCATED`
counts per run — a model needing frequent repair is a warning sign, not a pass.

## 4. Fold-specific instructions the model must obey

These are the rules that make a *fold* work rather than a single-shot
extraction, and they are where a smaller model is most likely to fail:

1. **Return the whole object, always.** Not a diff. Not only new fields.
2. **Never blank out or downgrade an already-correct field** unless this excerpt
   directly contradicts it.
3. **Arrays accumulate** — keep every existing entry, append genuinely new ones.
   Applies to `columns.*` arrays and `product_data.presentations` /
   `indications` / `pivotal_evidence` / `excipients`.
4. **`key_risks` does *not* accumulate** — it is re-ranked every call down to the
   max 8 most label-driving risks. Dropping a previously-listed lower-severity
   risk is correct behaviour here.
5. **`presentations` identity is (strength set + form + route + pack_size)** and
   nothing else. A multi-ingredient combination is ONE entry. Pack size *is*
   part of identity; container material is *not*. A `null` pack_size must be
   filled in on the existing entry, not added as a second entry.
6. **If an excerpt has nothing relevant, return the JSON unchanged.**
7. **Some excerpts are raw registry JSON, not prose** — extract from them the
   same way.

Rule 2 and rule 3 are the ones to test hardest: a model that regenerates the
object from only the current excerpt will silently erase everything the earlier
excerpts found, and the failure looks like a *quality* problem rather than a
*mechanics* problem.

## 5. How the excerpts are built

Two sources, concatenated as `json_data units + document chunks` — registry JSON
first, because the structured skeleton (application number, product table,
approval dates) is often not restated in the document prose.

**Document text** → `processing/chunking.py`:
- `DEFAULT_TARGET_TOKENS = 1800`, `cl100k_base` via tiktoken (falls back to
  `len/4` if tiktoken is unavailable)
- packs whole paragraphs greedily; splits a paragraph internally on sentence
  boundaries only if it alone exceeds 1800 tokens
- **15% overlap** — each chunk after the first is prefixed with the tail of the
  previous one
- persisted to `drug.product_chunks`, consumed in `chunk_index` order

**Registry JSON** → `_json_data_units` (`ai_extraction.py:444`):
- nested dicts are walked to their leaves; each **list-valued** leaf is packed
  separately into ~1800-token groups by `_pack_list_field`
- all scalar leaves collapse into one leading "application/product summary" unit
- each unit is prefixed `Raw structured registry metadata — {dotted.path} (items
  N-M of T):`

**Batching**: `AI_EXTRACTION_CHUNKS_PER_CALL`, default **5**. One LLM call per
batch of 5 excerpts. It was 7 while this stage ran on OpenRouter's 1M-token
context; 5 is what fits the self-hosted window without triggering splits (§9).

**Retry**: 3 attempts per batch, linear backoff `2s × attempt`
(`processing/retry.py`). A permanently failed batch is skipped, the fold
continues with the accumulator it had, and the row lands `NEEDS_REVIEW`. A
context overflow is exempt (`no_retry_on=(ContextOverflow,)`) — it is split
instead, since retrying an oversized payload fails identically.

## 6. Measured production baseline

8 US products with full SPL labels (JAYPIRCA, KLONOPIN, YONDELIS, PIVYA,
XALATAN, INFED, IMITREX, ZYLOPRIM), `openai/gpt-5.6-luna` via OpenRouter,
read-only re-run:

| metric | value |
| --- | --- |
| products | 8 |
| excerpts | 269 (199 from registry JSON, 70 document chunks) |
| LLM calls | 42 |
| prompt tokens | 575,283 — avg **13,697/call** |
| completion tokens | 66,392 — avg **1,581/call** |
| wall clock | 562 s — **70 s/product**, strictly sequential per product |
| cost | **$0.186** for 8 products (≈ $0.023/product) |
| failures | 0 |

Sizing notes for a local deployment:

- **Context**: 13.7k prompt tokens average, and that is the *average*. Budget
  ~32k to be safe: 3.9k system + accumulator (grows through the fold) + up to 7
  excerpts. One excerpt in this sample was **40,968 tokens** on its own — a
  single oversized `_pack_list_field` group — so a 8k or 16k context model will
  hard-fail on some rows.
- **Output**: ~1.6k tokens/call average. Set `max_tokens` ≥ 8192; the model
  re-emits the entire accumulated object every call, so output grows as the fold
  progresses and truncation gets more likely at the end.
- **Throughput**: 42 sequential calls for 8 products. Concurrency comes from
  `PIPELINE_WORKERS` running *different products* in parallel (default 4), never
  from parallelising one product's fold. Your local server needs to hold
  `PIPELINE_WORKERS` concurrent sessions at ~14k context each.
- **Fold amplification is only 1.5×** — 378,783 tokens of unique excerpt text
  became 575,283 billed prompt tokens. Cheaper than it looks, because the
  accumulator is compact JSON.

### Two inefficiencies worth knowing before you benchmark

Both are in the excerpt builder, not the model, and both inflate any
per-call comparison:

- **44% of excerpts are under 50 tokens.** `_flatten_json_fields` emits a
  separate unit for every list leaf, so `openfda.unii` (one string),
  `openfda.route` (one string) and `openfda.rxcui` (two strings) each become
  their own excerpt. PIVYA gets 30 excerpts / 5 calls where ~6 excerpts of real
  content exist.
- **The 40,968-token excerpt.** `_pack_list_field` packs to ~1800 tokens *per
  item boundary*, so a single list item larger than the target is emitted whole
  and unsplit. A full `spl_labels[]` entry does exactly that.

Neither is a blocker for testing — they affect both models equally — but if the
local model has a smaller context window, the oversized excerpt is the thing
that will break it first, and it is a bug on our side rather than a limitation
on yours.

## 7. Swapping the model in for real

`_client()` and `EXTRACTOR_MODEL` are the only two places the provider is named.
For an OpenAI-compatible local server:

```python
def _client():
    return OpenAI(
        base_url=os.getenv("EXTRACTOR_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )
```

Then `EXTRACTOR_BASE_URL=http://localhost:8000/v1` and
`EXTRACTOR_MODEL=your-model`. Also drop the
`extra_body={"usage": {"include": True}}` in `_call_llm_merge` — that is an
OpenRouter cost-reporting flag and some servers reject unknown body keys — and
add an explicit `max_tokens`. Cost logging (`_record_usage`) degrades gracefully:
token counts still log, `cost` shows `n/a`.

---

## 8. Benchmarking a self-hosted model

`tools/local_llm_harness.py` is read-only: it samples rows that already have a
stored Stage C result, replays the identical excerpt stream against your model,
and diffs the two. The report lands in `local_llm_harness_reports/`.

### Provider routing and fallback

`processing/llm_service_client.py` tries the self-hosted endpoint, then
OpenRouter. Two behaviours are worth knowing before you read any numbers:

- **Degrade, then recover.** A single failure is a blip and the next call still
  prefers the GPU; after `LLM_SERVICE_FALLBACK_AFTER` (default 2) consecutive
  failures the run switches to OpenRouter — but only for
  `LLM_SERVICE_RECOVERY_AFTER` seconds (default 60), after which one caller
  re-probes self-hosted and the run returns to it the moment it answers. A
  failed probe doubles the cooldown up to `LLM_SERVICE_RECOVERY_MAX`, so a dead
  endpoint is checked rarely rather than once per product, and only one probe
  runs at a time so N workers do not all retry a dead endpoint together.
  Self-hosted is always the preferred provider; the fallback is never permanent.
  The summary prints `providers: {'self_hosted': N, 'openrouter': M}` and warns
  explicitly when any call hit the paid API.
- **Context overflow is never failed over.** A payload that overflows one
  model's window overflows the other's too, and unlike a dead endpoint it says
  nothing about the service's health. Such a batch is *split* instead, and the
  failure does not count toward the degradation counter.

Two switches control the routing:

| `LLM_SERVICE_ENABLED` | `LLM_SERVICE_FALLBACK` | behaviour |
| --- | --- | --- |
| `true` (default) | `true` (default) | self-hosted first, temporary fallback on failure |
| `true` | `false` | self-hosted only; failures are fatal (rows land `NEEDS_REVIEW`) |
| `false` | *ignored* | OpenRouter only, no GPU probe |

`LLM_SERVICE_ENABLED=false` is the clean way to run hosted-only: it reports
OpenRouter as the *configured provider* rather than flagging the run as degraded,
so an intentional setup is not confused with a broken GPU in the logs.
`LLM_SERVICE_FALLBACK=false` makes self-hosted failures fatal — use it when you
want the benchmark to measure only the local model.

Note `LLM_SERVICE_FALLBACK` gates falling back *from* self-hosted, so it is
ignored when the self-hosted path is switched off — otherwise a leftover `false`
would disable the only remaining provider.

### Context-aware batch splitting

Before each call the harness estimates the prompt (system + accumulator +
excerpts) and checks it leaves at least `MIN_COMPLETION_TOKENS` (512) of room
under the window. If not, the batch is halved and each half recursed, down to a
single excerpt. Every excerpt still gets processed — at the cost of more, smaller
calls — rather than being silently dropped.

When even one excerpt alone will not fit, the run reports it and the product
lands `NEEDS_REVIEW`, never a false `DONE`. The message names which of the two
causes applies, because the remedies are opposite:

```
the excerpt alone is 41336 tokens and only ~1056 fit; try --sub-chunk-tokens 528
    -> an indivisible oversized excerpt. --sub-chunk-tokens splits it.

the excerpt is only 1336 tokens, but the system prompt + accumulated JSON
already cost 5900; sub-chunking will NOT help, this fold needs a larger
--context-tokens
    -> the fixed cost fills the window. Only a bigger window helps.
```

`--sub-chunk-tokens N` (bare flag = 1800) splits any excerpt above N tokens into
labelled parts. It is off by default because it changes the excerpt stream, so a
run using it is no longer byte-exact against production — but it is the only way
to get the 40,968-token `spl_labels[]` unit through a 22k window.

### Prompt A/B

`--prompt production` sends `ai_extraction.SYSTEM_PROMPT` unchanged (3,888
tokens). `--prompt harness` (default) appends four exhaustive-enumeration rules
— one array entry per distinct pack size / strength / trial, carry the
already-established identity fields onto each new entry, ignore facts belonging
to a different named strength, and always emit all 7 `product_data` keys. That
costs **+560 tokens/call** (4,448 total).

These target the failure mode smaller models show most: collapsing an enumerable
list into one summarising entry. Run both variants on the same rows — if the
extra rules do not change the diff, they are pure cost.

### Batch size: 5 here, 7 in production

The harness defaults to **5 excerpts per call**, not production's 7
(`AI_EXTRACTION_CHUNKS_PER_CALL`). 7 was tuned against a 1M-token hosted context
window; a self-hosted model on ~22k has to fit the system prompt (3.9k–4.4k) plus
an accumulator that grows for the whole fold into that same window, so 7 excerpts
overflow far sooner and trigger the recursive splitting described above. Five
costs more calls but avoids most splits, which is the cheaper trade locally —
a split re-sends the system prompt and accumulator all over again.

Set it with `--chunks-per-call` or `LOCAL_LLM_CHUNKS_PER_CALL`. The run header
and the JSON report both record the value used alongside production's, so
comparisons stay honest. Use `--chunks-per-call 7` when you specifically want to
reproduce production's batching.

### Useful flags

| flag | effect |
| --- | --- |
| `--regnum X` | one product (repeatable) |
| `--status ''` | sample any row, not just `DONE` ones |
| `--diff-stored` | field-level diff + key-coverage table vs the stored result |
| `--context-tokens N` | override the discovered window |
| `--chunks-per-call N` | excerpts per call (harness default **5**; production Stage C uses 7) |
| `--no-json-mode` | drop `response_format=json_object` |
| `--dump-prompts` | print the system prompt |

### Reading the summary

- **`peak prompt` vs `window`** — how close the fold came to overflowing. If
  peak is near the window, expect splits on bigger products.
- **`JSON repairs`** — production's parser fixes fences and trailing commas, so
  these calls still *pass*. A model needing frequent repair is nonetheless a
  warning sign, not a clean result.
- **`truncated (length)`** — the reply hit `max_tokens` mid-object. The fold
  re-emits the whole accumulator every call, so this gets more likely near the
  end of a long product. Raise `--max-tokens`.
- **`dropped excerpts`** — evidence the model never saw. Any non-zero value
  invalidates a quality comparison for that product.
- **`status`** — the production acceptance gate (§3) applied to the local
  output. `NEEDS_REVIEW` from a missing key is a *format* failure, not a
  quality one, and is usually fixed by the harness prompt's "always emit all 7
  keys" rule.

---

## 9. Running on the self-hosted LLM

Stage C's primary provider is the self-hosted model. OpenRouter is the fallback
only. Both live behind `processing/llm_service_client.py`; `_call_llm_merge`
calls `call_json` and never talks to a provider directly.

```bash
LLM_SERVICE_BASE_URL=http://localhost:8000/v1
LLM_SERVICE_MODEL=gemma-4-26b
LLM_SERVICE_CONTEXT_TOKENS=22000     # fallback if GET /models omits max_model_len
EXTRACTOR_MODEL=google/gemini-2.5-flash   # now the FALLBACK model
LLM_SERVICE_FALLBACK=false           # optional: make self-hosted failures fatal
```

`extract_pending` calls `health_check()` before claiming any rows — otherwise a
dead GPU with no fallback would flip the whole queue to `PROCESSING`, fail it,
and burn every row's retry budget before anyone noticed. The end-of-run summary
prints the provider split and warns explicitly when any call hit the paid
fallback, because row statuses look identical either way.

### Context budget

The window has to hold the system prompt, the accumulated JSON, this batch's
excerpts, and the reply. Fixed costs are the prompt (4,878) plus a 300-token
chat-template margin plus a 512-token minimum reply. Measured against the 62
`DONE` rows currently in the database:

| accumulator | excerpt budget in a 22k window |
| --- | --- |
| median (428 tokens) | 15,882 → 8.8 chunks of 1,800 |
| p90 (1,642) | 14,668 → 8.1 chunks |
| max observed (3,560) | 12,750 → 7.1 chunks |

At `CHUNKS_PER_CALL=5` a full batch of 1,800-token chunks costs 9,000 tokens,
leaving **7,310 tokens of headroom** for the accumulator — roughly double the
largest one observed so far. That is the reason for 5 rather than 7: at 7 the
batch alone is 12,600 and only ~3,700 is left, which the biggest accumulators
already exceed.

### Overflow handling

`_fit_completion_tokens` sizes `max_tokens` from what is actually left, and
raises `ContextOverflow` when under 512 remain. `_fold_slice` then halves the
batch and recurses, so every excerpt is still folded in rather than the batch
being abandoned:

```
excerpts 1-5/5 overflow (prompt ~5721 tokens ...) — splitting into 1-2 and 3-5
excerpts 3-5/5 overflow (prompt ~5500 tokens ...) — splitting into 3-3 and 4-5
```

This matters far more here than on OpenRouter: with a 22k window a long
product's accumulator eventually crowds out its own excerpts, and without
splitting the tail of that document would simply never be read.

An excerpt that will not fit even alone cannot be folded. It is recorded in
`failed_chunks`, so the row lands `NEEDS_REVIEW` with the span named — never a
false `DONE` — and the log says what to do (re-chunk with a smaller
`chunking.DEFAULT_TARGET_TOKENS`, or use a larger-context model).

### Truncated replies

A reply that hits `max_tokens` is repaired by `json_utils` and still parses, but
the tail is genuinely lost, so it logs a warning naming
`AI_EXTRACTION_MAX_TOKENS`. Because the fold re-emits the whole accumulated
object every call, this gets more likely near the end of a long product.

### Keeping the standalone prompt in sync

`tools/harness_prompt_standalone.py` is a generated verbatim copy of
`SYSTEM_PROMPT`, for machines without this repo on disk. Nothing imports it in
normal operation, so it would drift silently — `tests/test_standalone_prompt.py`
fails when it does. After any prompt edit:

```bash
python tools/generate_standalone_prompt.py
```
