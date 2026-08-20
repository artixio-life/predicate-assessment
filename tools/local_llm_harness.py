"""
Local-LLM benchmark harness for the drug.products AI-extraction fold.

READ-ONLY: never writes to the database. It picks N already-completed
(ai_extraction_status='DONE') products, rebuilds the exact excerpt stream the
production fold saw, runs the fold against your self-hosted model, and diffs
the result against the product_data/columns already stored (produced by
OpenRouter — see EXTRACTOR_MODEL).

Unlike the cloud draft this replaces, the real pipeline modules ARE imported
here rather than reconstructed, so the excerpt stream is byte-exact on both
halves:

  - document chunks      read verbatim from drug.product_chunks
  - registry-JSON units  built by ai_extraction._json_data_units
  - JSON parse/repair    processing/json_utils.parse_llm_json
  - token counting       processing/chunking.count_tokens (cl100k_base)

The system prompt is production's own, imported — not a variant. Stage C runs on
the self-hosted model now, so the extraction rules live in
ai_extraction.SYSTEM_PROMPT and both paths send identical bytes.

Provider routing is processing/llm_service_client.py: self-hosted first, with
an automatic OpenRouter fallback when the local endpoint is unreachable or
keeps erroring (sticky after LLM_SERVICE_FALLBACK_AFTER consecutive failures).
A context-length rejection is NOT failed over — the batch is split instead,
since a payload that overflows one model's window will overflow the other's
too, and the report tells you which provider served each call.

Usage:
    export LLM_SERVICE_BASE_URL=http://localhost:8000/v1
    export LLM_SERVICE_MODEL=gemma-4-26b
    export LLM_SERVICE_CONTEXT_TOKENS=22000
    python tools/local_llm_harness.py --country "United States" --limit 8 --diff-stored

    # split oversized registry-JSON items instead of emitting them whole
    python tools/local_llm_harness.py --limit 8 --sub-chunk-tokens 1800
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import psycopg2
import psycopg2.extras

import processing.ai_extraction as ai
from processing.chunking import DEFAULT_TARGET_TOKENS, count_tokens
from processing import llm_service_client as llm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("harness")

# Excerpts per fold call. Deliberately 5 rather than production's 7
# (AI_EXTRACTION_CHUNKS_PER_CALL): 7 was tuned for a 1M-token hosted context,
# and a self-hosted model on ~22k has to spend that window on the system prompt
# plus an accumulator that grows all fold long. Smaller batches mean more calls
# but far fewer context-overflow splits, which is the cheaper trade locally.
# Override with LOCAL_LLM_CHUNKS_PER_CALL or --chunks-per-call.
CHUNKS_PER_CALL = int(os.getenv("LOCAL_LLM_CHUNKS_PER_CALL", "5"))
PRODUCTION_CHUNKS_PER_CALL = ai.CHUNKS_PER_CALL
MAX_RETRIES = 3
# Matches ai_extraction.MAX_COMPLETION_TOKENS. Kept modest on purpose: vLLM
# reserves KV cache for the declared max_tokens and defers requests it cannot
# fit, so over-declaring costs concurrency rather than buying safety.
DEFAULT_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "3072"))

# Headroom between our estimate and the server's real count: chat-template
# overhead plus the tokeniser mismatch (cl100k vs the served model's own vocab,
# measured 1.68% low on Gemma). Proportional with a floor — see
# ai_extraction._safety_margin.
CONTEXT_SAFETY_MARGIN = int(os.getenv("LOCAL_LLM_SAFETY_MARGIN", "300"))
CONTEXT_SAFETY_RATIO = float(os.getenv("LOCAL_LLM_SAFETY_RATIO", "0.06"))
# Below this much room for a reply, sending the batch is pointless — the object
# the fold must re-emit will not fit, so split instead.
MIN_COMPLETION_TOKENS = int(os.getenv("LOCAL_LLM_MIN_COMPLETION", "512"))

REQUIRED_PRODUCT_DATA_KEYS = list(ai.PRODUCT_DATA_KEYS)
COLUMN_SCALAR_FIELDS = list(ai.COLUMN_SCALAR_FIELDS)
COLUMN_ARRAY_FIELDS = list(ai.COLUMN_ARRAY_FIELDS)


# --------------------------------------------------------------------------
# Database (read-only)
# --------------------------------------------------------------------------

def get_connection():
    """
    Prefer DATABASE_URL when set (matches the cloud draft and regulatory-
    explorer), otherwise fall back to this repo's POSTGRES_* parts via db.py.
    The session is marked read-only so a bug here cannot mutate the pipeline's
    own tables.
    """
    url = os.getenv("DATABASE_URL")
    if url:
        conn = psycopg2.connect(url)
    else:
        from db import get_db_connection
        conn = get_db_connection()
    conn.set_session(readonly=True)
    return conn


def fetch_products(conn, country_name, limit, regnums=None, status="DONE"):
    conditions, params = [], []
    if country_name:
        conditions.append("rg.country_name = %s")
        params.append(country_name)
    if status:
        conditions.append("p.ai_extraction_status = %s")
        params.append(status)
    if regnums:
        conditions.append("p.registration_number = ANY(%s)")
        params.append(list(regnums))
    where = " AND ".join(conditions) or "TRUE"
    params.append(limit)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT p.*
            FROM drug.products p
            JOIN drug.regulatory_geography rg ON rg.id = p.country_id
            WHERE {where}
            ORDER BY (SELECT count(*) FROM drug.product_chunks pc
                      WHERE pc.product_id = p.id) DESC,
                     p.updated_at DESC
            LIMIT %s
            """,
            params,
        )
        return [dict(row) for row in cur.fetchall()]


# --------------------------------------------------------------------------
# Excerpt stream
# --------------------------------------------------------------------------

def _split_text_by_tokens(text, target_tokens):
    """Slice text into ~target_tokens pieces on token boundaries."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text)
        return [enc.decode(ids[i:i + target_tokens]) for i in range(0, len(ids), target_tokens)]
    except Exception:
        span = target_tokens * 4
        return [text[i:i + span] for i in range(0, len(text), span)]


def split_oversized_units(units, sub_chunk_tokens):
    """
    Production's _pack_list_field packs to ~1800 tokens *per item boundary*, so
    a single list item bigger than the target is emitted whole and unsplit — a
    full spl_labels[] entry does exactly that, and one was measured at 40,968
    tokens. That overflows any local context window on its own, and no amount
    of batch-splitting helps, because a batch cannot be split below one excerpt.

    This splits any such excerpt into sub_chunk_tokens-sized parts, labelled so
    the model knows it is seeing a fragment of one entry rather than a whole
    one. Applied to the WHOLE excerpt stream, not only the registry-JSON half:
    a document chunk is normally ~1800 tokens by construction, but text
    extraction is not guaranteed to have produced chunks at the current target
    (chunk sizes are whatever chunking.py used when the row was parsed), so
    capping both is what actually keeps every excerpt inside the window.

    Off by default: it changes the excerpt stream, so a run using it is no
    longer byte-exact against production.
    """
    if not sub_chunk_tokens:
        return units, 0
    out, split_count = [], 0
    for unit in units:
        if count_tokens(unit) <= sub_chunk_tokens:
            out.append(unit)
            continue
        # Registry-JSON units carry a "Raw structured registry metadata — ..."
        # header line that must be repeated on each part for the fragment to be
        # interpretable; a bare document chunk has no such header.
        if unit.startswith("Raw structured registry metadata"):
            header, _, body = unit.partition("\n")
        else:
            header, body = "", unit
        parts = _split_text_by_tokens(body, sub_chunk_tokens)
        split_count += 1
        for i, part in enumerate(parts, start=1):
            label = f"[oversized excerpt, part {i}/{len(parts)}]"
            out.append(f"{header} {label}\n{part}" if header else f"{label}\n{part}")
    return out, split_count


def build_excerpts(conn, product, sub_chunk_tokens=None):
    """
    The same two-source stream production builds: registry-JSON units first
    (the structured skeleton document prose rarely restates), then document
    chunks in chunk_index order.
    """
    json_units = ai._json_data_units(product)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT chunk_text FROM drug.product_chunks WHERE product_id = %s "
            "ORDER BY chunk_index",
            (product["id"],),
        )
        doc_chunks = [row["chunk_text"] for row in cur.fetchall()]

    units, n_split = split_oversized_units(json_units + doc_chunks, sub_chunk_tokens)
    # Report the post-split counts, since those are the excerpts actually sent.
    n_json = sum(1 for u in units if u.startswith("Raw structured registry metadata"))
    return units, n_json, len(units) - n_json, n_split


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

# The rules that used to be patched on here (exhaustive enumeration, the
# multi-strength sentence handling, the pivotal_evidence one-entry-per-study
# contract, the columns/product_data mirroring requirement) now live in
# ai_extraction.SYSTEM_PROMPT itself, since Stage C runs on the self-hosted
# model in production. There is nothing left to patch: the harness and the
# pipeline send the same prompt by construction, which is the point — a
# benchmark against a prompt the pipeline does not use measures nothing.
SYSTEM_PROMPT = ai.SYSTEM_PROMPT


def build_user_message(product, accumulated, batch, batch_start, batch_end, total):
    """Production's _build_user_message, reused verbatim."""
    return ai._build_user_message(product, accumulated, batch, batch_start, batch_end, total)


# --------------------------------------------------------------------------
# Fold
# --------------------------------------------------------------------------

def _fits(messages, requested_max_tokens, context_window):
    """
    Returns (effective_max_tokens, prompt_estimate) or (None, prompt_estimate)
    when there is not enough room left for a usable reply.
    """
    prompt_estimate = sum(count_tokens(m["content"]) for m in messages) + 16
    available = context_window - prompt_estimate - max(
        CONTEXT_SAFETY_MARGIN, int(prompt_estimate * CONTEXT_SAFETY_RATIO))
    if available < MIN_COMPLETION_TOKENS:
        return None, prompt_estimate
    return max(MIN_COMPLETION_TOKENS, min(requested_max_tokens, available)), prompt_estimate


def _call_batch(system_prompt, user_message, requested_max_tokens, context_window, stats):
    """
    One fold call. Returns (obj_or_None, needs_split).

    needs_split=True means the payload itself is too big for the window — the
    caller must send less, not retry. Distinguished from an ordinary failure so
    a genuinely dead endpoint is not mistaken for an oversized batch.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    effective_max_tokens, prompt_estimate = _fits(messages, requested_max_tokens, context_window)
    if effective_max_tokens is None:
        logger.info(
            f"    [info] estimated prompt {prompt_estimate} tokens leaves no room for a "
            f"{MIN_COMPLETION_TOKENS}-token reply under a {context_window}-token "
            f"window — splitting batch"
        )
        return None, True

    stats["max_prompt_tokens"] = max(stats.get("max_prompt_tokens", 0), prompt_estimate)

    try:
        obj, provider, meta = llm.call_json(
            system_prompt, user_message,
            max_tokens=effective_max_tokens, temperature=0.0,
            json_mode=stats["json_mode"], parse_retries=1,
        )
    except llm.LLMServiceError as e:
        if llm.is_context_error(e):
            logger.warning(f"    [warn] provider reported context overflow: {e} — splitting batch")
            return None, True
        logger.error(f"    [error] batch permanently failed: {e}")
        stats["provider_failures"] += 1
        return None, False

    stats["calls"] += 1
    stats["by_provider"][provider] = stats["by_provider"].get(provider, 0) + 1
    usage = meta.get("usage")
    if usage is not None:
        stats["prompt_tokens"] += getattr(usage, "prompt_tokens", None) or 0
        stats["completion_tokens"] += getattr(usage, "completion_tokens", None) or 0
    else:
        stats["prompt_tokens"] += prompt_estimate
    if meta.get("repaired"):
        stats["json_repaired"] += 1
    if meta.get("finish_reason") == "length":
        stats["truncated"] += 1

    if obj is None:
        logger.error("    [error] reply could not be parsed even after repair")
        stats["unparseable"] += 1
        return None, False
    if not isinstance(obj, dict) or "columns" not in obj or "product_data" not in obj:
        logger.error(
            f"    [error] reply missing columns/product_data: "
            f"{list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__}"
        )
        stats["bad_shape"] += 1
        return None, False
    return obj, False


def _fold_slice(product, accumulated, excerpts, offset, total, system_prompt,
                max_tokens, context_window, stats, depth=0):
    """
    Fold one slice of the excerpt list into the accumulator.

    On a context overflow the slice is halved and each half recursed, so every
    excerpt is still processed — at the cost of more, smaller calls — rather
    than silently dropped. Returns (accumulator, any_unrecoverable).
    """
    if not excerpts:
        return accumulated, False

    batch_start, batch_end = offset + 1, offset + len(excerpts)
    indent = "  " * depth
    logger.info(f"    {indent}batch {batch_start}-{batch_end} of {total} (size {len(excerpts)})")

    user_message = build_user_message(
        product, accumulated, excerpts, batch_start, batch_end, total)
    result, needs_split = _call_batch(
        system_prompt, user_message, max_tokens, context_window, stats)

    if result is not None:
        return result, False

    if needs_split and len(excerpts) > 1:
        stats["splits"] += 1
        mid = len(excerpts) // 2
        logger.info(
            f"    {indent}splitting {batch_start}-{batch_end} into "
            f"{batch_start}-{offset + mid} and {offset + mid + 1}-{batch_end}"
        )
        accumulated, failed_a = _fold_slice(
            product, accumulated, excerpts[:mid], offset, total,
            system_prompt, max_tokens, context_window, stats, depth + 1)
        accumulated, failed_b = _fold_slice(
            product, accumulated, excerpts[mid:], offset + mid, total,
            system_prompt, max_tokens, context_window, stats, depth + 1)
        return accumulated, failed_a or failed_b

    # A single excerpt that still will not fit, or a non-context failure.
    stats["unrecoverable_excerpts"] += len(excerpts)
    hint = ""
    if needs_split:
        # The two causes need opposite remedies, so name the right one. Compare
        # the excerpt itself against the room left once the fixed costs (system
        # prompt + the accumulated JSON the fold must carry) are paid.
        excerpt_tokens = sum(count_tokens(e) for e in excerpts)
        fixed_cost = (count_tokens(system_prompt)
                      + count_tokens(json.dumps(accumulated, default=str)))
        room = context_window - fixed_cost - MIN_COMPLETION_TOKENS - CONTEXT_SAFETY_MARGIN
        if excerpt_tokens > max(0, room):
            hint = (f" — the excerpt alone is {excerpt_tokens} tokens and only ~{max(0, room)} "
                    f"fit; try --sub-chunk-tokens {max(256, room // 2)}")
        else:
            hint = (f" — the excerpt is only {excerpt_tokens} tokens, but the system prompt "
                    f"+ accumulated JSON already cost {fixed_cost}; sub-chunking will NOT "
                    f"help, this fold needs a larger --context-tokens")
    logger.error(
        f"    {indent}[error] excerpt(s) {batch_start}-{batch_end} could not be processed "
        f"even alone — accumulator left unchanged for this slice{hint}"
    )
    return accumulated, True


def compute_status(final_obj, any_batch_failed):
    """Production's acceptance gate — see ai_extraction._extract_one."""
    if not final_obj:
        return "FAILED"
    columns = final_obj.get("columns") or {}
    product_data = final_obj.get("product_data") or {}
    if not columns and not product_data:
        return "FAILED"
    if any_batch_failed:
        return "NEEDS_REVIEW"
    if not all(key in product_data for key in REQUIRED_PRODUCT_DATA_KEYS):
        return "NEEDS_REVIEW"
    return "DONE"


def run_fold(product, excerpts, system_prompt, max_tokens, context_window,
             chunks_per_call, json_mode):
    accumulated = {"columns": {}, "product_data": {}}
    total = len(excerpts)
    any_failed = False
    stats = {
        "calls": 0, "json_repaired": 0, "truncated": 0, "unparseable": 0,
        "bad_shape": 0, "provider_failures": 0, "prompt_tokens": 0,
        "completion_tokens": 0, "splits": 0, "unrecoverable_excerpts": 0,
        "by_provider": {}, "max_prompt_tokens": 0, "json_mode": json_mode,
    }
    started = time.time()
    for start in range(0, total, chunks_per_call):
        accumulated, failed = _fold_slice(
            product, accumulated, excerpts[start:start + chunks_per_call],
            start, total, system_prompt, max_tokens, context_window, stats)
        any_failed = any_failed or failed
    stats["seconds"] = round(time.time() - started, 1)
    stats.pop("json_mode")
    return accumulated, compute_status(accumulated, any_failed), stats


# --------------------------------------------------------------------------
# Diff vs the stored OpenRouter result
# --------------------------------------------------------------------------

def _norm(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


def _hashable_set(items):
    """
    A model can return dicts where a flat string array is expected — confusing
    columns.indications (flat strings) with product_data.indications (objects)
    is the common case. Stringify anything unhashable so the diff still reports
    instead of crashing.
    """
    out = set()
    for item in items or []:
        if isinstance(item, (dict, list)):
            out.add(json.dumps(item, sort_keys=True, default=str))
        else:
            out.add(item)
    return out


def diff_columns(stored_row, local_columns):
    diffs = []
    local_columns = local_columns or {}
    for field in COLUMN_SCALAR_FIELDS:
        stored_value = _norm(stored_row.get(field))
        local_value = _norm(local_columns.get(field))
        if stored_value != local_value:
            diffs.append({"field": field, "stored": stored_value, "local": local_value})
    for field in COLUMN_ARRAY_FIELDS:
        stored_set = _hashable_set(stored_row.get(field))
        local_set = _hashable_set(local_columns.get(field))
        if stored_set != local_set:
            diffs.append({
                "field": field,
                "stored_only": sorted(stored_set - local_set, key=str),
                "local_only": sorted(local_set - stored_set, key=str),
            })
    return diffs


def diff_product_data(stored_pd, local_pd):
    stored_pd, local_pd = stored_pd or {}, local_pd or {}
    summary = {}
    for key in REQUIRED_PRODUCT_DATA_KEYS:
        stored_value, local_value = stored_pd.get(key), local_pd.get(key)
        entry = {
            "present_stored": key in stored_pd,
            "present_local": key in local_pd,
            "equal": stored_value == local_value,
        }
        if isinstance(stored_value, list) or isinstance(local_value, list):
            entry["count_stored"] = len(stored_value) if isinstance(stored_value, list) else None
            entry["count_local"] = len(local_value) if isinstance(local_value, list) else None
        else:
            entry["stored"], entry["local"] = stored_value, local_value
        summary[key] = entry
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--country", default="United States")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--regnum", action="append", dest="regnums",
                        help="restrict to these registration_numbers; repeatable")
    parser.add_argument("--status", default="DONE",
                        help="ai_extraction_status to sample; '' for any (default DONE, "
                             "the rows that have a stored result to diff against)")
    parser.add_argument("--diff-stored", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--chunks-per-call", type=int, default=CHUNKS_PER_CALL,
                        help=f"excerpts per fold call (default {CHUNKS_PER_CALL}; "
                             f"production Stage C uses {PRODUCTION_CHUNKS_PER_CALL})")
    parser.add_argument("--context-tokens", type=int, default=None,
                        help="override the context window (default: discovered from "
                             "GET /models, else LLM_SERVICE_CONTEXT_TOKENS)")
    parser.add_argument("--sub-chunk-tokens", type=int, nargs="?",
                        const=DEFAULT_TARGET_TOKENS, default=None,
                        help="split any single oversized registry-JSON item into "
                             f"~N-token parts (bare flag = {DEFAULT_TARGET_TOKENS})")
    parser.add_argument("--no-json-mode", action="store_true",
                        help="do not send response_format=json_object")
    parser.add_argument("--dump-prompts", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    llm.reset_stats()
    try:
        llm.health_check()
    except llm.LLMServiceError as e:
        # No provider can serve traffic. Fail here with one clear line rather
        # than a traceback, before any database or retrieval work is spent.
        logger.error(f"no LLM provider available: {e}")
        logger.error("set LLM_SERVICE_BASE_URL to a reachable /v1 endpoint, or set "
                     "OPEN_ROUTER_API_KEY and leave LLM_SERVICE_FALLBACK unset/true.")
        return 3
    context_window = args.context_tokens or llm.discover_context_window()
    system_prompt = SYSTEM_PROMPT

    logger.info(f"self-hosted : {llm.LLM_SERVICE_MODEL} @ {llm.LLM_SERVICE_BASE_URL or '(unset)'}")
    logger.info(f"fallback    : {llm.LLM_FALLBACK_MODEL} "
                f"({'enabled' if llm.fallback_available() else 'unavailable'})")
    logger.info(f"context     : {context_window} tokens, max_tokens {args.max_tokens}, "
                f"{args.chunks_per_call} excerpts/call"
                + ("" if args.chunks_per_call == PRODUCTION_CHUNKS_PER_CALL
                   else f" (production Stage C: {PRODUCTION_CHUNKS_PER_CALL})"))
    logger.info(f"prompt      : ai_extraction.SYSTEM_PROMPT "
                f"({count_tokens(system_prompt)} tokens)")

    if args.dump_prompts:
        print("\n=== SYSTEM PROMPT ===\n" + system_prompt)

    conn = get_connection()
    try:
        products = fetch_products(conn, args.country, args.limit, args.regnums,
                                  args.status or None)
        if not products:
            logger.error("no products matched")
            return 2
        logger.info(f"products    : {len(products)}\n")

        report = {
            "self_hosted_model": llm.LLM_SERVICE_MODEL,
            "self_hosted_base_url": llm.LLM_SERVICE_BASE_URL,
            "fallback_model": llm.LLM_FALLBACK_MODEL,
            "context_window": context_window,
            "chunks_per_call": args.chunks_per_call,
            "production_chunks_per_call": PRODUCTION_CHUNKS_PER_CALL,
            "sub_chunk_tokens": args.sub_chunk_tokens,
            "products": [],
        }
        totals = {
            "calls": 0, "json_repaired": 0, "truncated": 0, "unparseable": 0,
            "bad_shape": 0, "provider_failures": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "splits": 0, "unrecoverable_excerpts": 0,
            "seconds": 0.0,
        }
        peak_prompt = 0
        status_counts = {}

        for index, product in enumerate(products, start=1):
            logger.info(f"[{index}/{len(products)}] {product['product_name']} "
                        f"({product['registration_number']})")
            excerpts, n_json, n_doc, n_split = build_excerpts(
                conn, product, args.sub_chunk_tokens)
            sizes = [count_tokens(e) for e in excerpts]
            logger.info(
                f"  excerpts: {len(excerpts)} ({n_json} registry-JSON, {n_doc} document)"
                f"  max {max(sizes) if sizes else 0} tokens"
                + (f", {n_split} oversized item(s) sub-chunked" if n_split else "")
            )
            if not excerpts:
                logger.info("  SKIPPED (no chunks, no json_data)")
                continue

            result, status, stats = run_fold(
                product, excerpts, system_prompt, args.max_tokens, context_window,
                args.chunks_per_call, json_mode=not args.no_json_mode)

            for key in totals:
                totals[key] += stats.get(key, 0)
            peak_prompt = max(peak_prompt, stats.get("max_prompt_tokens", 0))
            status_counts[status] = status_counts.get(status, 0) + 1

            flags = []
            if stats["json_repaired"]:
                flags.append(f"JSON-REPAIRED×{stats['json_repaired']}")
            if stats["truncated"]:
                flags.append(f"TRUNCATED×{stats['truncated']}")
            if stats["splits"]:
                flags.append(f"SPLIT×{stats['splits']}")
            if stats["unrecoverable_excerpts"]:
                flags.append(f"DROPPED×{stats['unrecoverable_excerpts']}")
            product_data = result.get("product_data") or {}
            logger.info(
                f"  {stats['calls']} calls, {stats['seconds']}s, "
                f"keys {sum(1 for k in REQUIRED_PRODUCT_DATA_KEYS if k in product_data)}/7, "
                f"status={status} {' '.join(flags)}"
            )

            entry = {
                "product_id": str(product["id"]),
                "product_name": product["product_name"],
                "registration_number": product["registration_number"],
                "excerpt_count": len(excerpts),
                "registry_json_units": n_json,
                "document_chunks": n_doc,
                "max_excerpt_tokens": max(sizes) if sizes else 0,
                "stored_ai_extraction_status": product["ai_extraction_status"],
                "local_status": status,
                "stats": stats,
                "local_result": result,
            }
            if args.diff_stored:
                entry["column_diffs"] = diff_columns(product, result.get("columns"))
                entry["product_data_diff"] = diff_product_data(
                    product.get("product_data"), result.get("product_data"))
                logger.info(f"  diff vs stored: {len(entry['column_diffs'])} column field(s) differ")
            report["products"].append(entry)

        n = len(report["products"]) or 1
        report["aggregate"] = {
            **totals,
            "status_counts": status_counts,
            "providers": llm.provider_stats(),
            "seconds_per_product": round(totals["seconds"] / n, 1),
            "peak_prompt_tokens": peak_prompt,
        }

        logger.info("\n=== aggregate ===")
        logger.info(f"calls              : {totals['calls']}")
        logger.info(f"prompt tokens      : {totals['prompt_tokens']:,} "
                    f"(avg {totals['prompt_tokens'] // max(1, totals['calls']):,}/call)")
        logger.info(f"completion tokens  : {totals['completion_tokens']:,} "
                    f"(avg {totals['completion_tokens'] // max(1, totals['calls']):,}/call)")
        logger.info(f"wall clock         : {totals['seconds']:.1f}s "
                    f"({report['aggregate']['seconds_per_product']}s/product)")
        logger.info(f"peak prompt        : {peak_prompt:,} tokens "
                    f"(window {context_window:,})")
        logger.info(f"JSON repairs       : {totals['json_repaired']} / {totals['calls']}")
        logger.info(f"truncated (length) : {totals['truncated']} / {totals['calls']}")
        logger.info(f"batch splits       : {totals['splits']}")
        logger.info(f"dropped excerpts   : {totals['unrecoverable_excerpts']}")
        logger.info(f"unparseable/shape  : {totals['unparseable']} / {totals['bad_shape']}")
        logger.info(f"status             : {status_counts}")
        logger.info(f"providers          : {llm.provider_stats()['calls']}")
        if llm.provider_stats()["degraded"]:
            logger.warning("NOTE: the run degraded to the OpenRouter fallback — "
                           "the numbers above are NOT purely self-hosted.")

        if args.diff_stored:
            logger.info("\n=== product_data key coverage: LOCAL vs STORED ===")
            logger.info(f"{'key':20} {'local':>8} {'stored':>8}")
            for key in REQUIRED_PRODUCT_DATA_KEYS:
                local_n = sum(
                    1 for p in report["products"]
                    if (p["local_result"].get("product_data") or {}).get(key)
                )
                stored_n = sum(
                    1 for p in report["products"]
                    if p["product_data_diff"][key]["present_stored"]
                    and (p["product_data_diff"][key].get("count_stored")
                         or p["product_data_diff"][key].get("stored"))
                )
                logger.info(f"{key:20} {local_n:>6}/{len(report['products'])} "
                            f"{stored_n:>6}/{len(report['products'])}")

        out_path = args.out or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..",
            "local_llm_harness_reports", f"run_{int(time.time())}.json")
        out_path = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info(f"\nreport: {out_path}")

        return 1 if (totals["provider_failures"] or totals["unrecoverable_excerpts"]) else 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
