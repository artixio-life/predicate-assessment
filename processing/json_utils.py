"""
LLM JSON response parsing: sanitize -> parse -> repair cascade.

Ported from regulatory-explorer's processing/matchers/unified_matcher.py
(_sanitize_json_response / _repair_json) — proven against the same kind of
model output (markdown-fenced JSON, trailing commas, truncated responses)
this pipeline's AI extraction stage will see.
"""
import json
import re
from typing import Optional


class JsonParseError(ValueError):
    pass


def sanitize_json_response(content: str) -> str:
    """Strip markdown fences, control characters, and surrounding prose."""
    if not content:
        return ""

    content = content.strip()

    # Remove control characters that break json.loads (except \n, \r, \t)
    content = "".join(c for c in content if ord(c) >= 32 or c in '\n\r\t')

    # Remove markdown formatting
    content = re.sub(r'^```json\s*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'^```\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    content = content.strip()

    # Extract only the JSON object part
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1 and end > start:
        return content[start:end + 1]
    return content


def repair_json(content: str) -> Optional[str]:
    """Attempt to fix common LLM JSON errors: literal newlines in strings,
    trailing commas, and truncated/unclosed responses. Returns repaired JSON
    text on success, or None if it couldn't be fixed."""
    if not content:
        return None

    content = content.strip()

    # Fix literal newlines inside strings (common in LLM output)
    fixed_newlines = []
    in_string = False
    escaped = False
    for char in content:
        if char == '"' and not escaped:
            in_string = not in_string
        if char == '\n' and in_string:
            fixed_newlines.append('\\n')
        else:
            fixed_newlines.append(char)
        escaped = (char == '\\' and not escaped)
    content = "".join(fixed_newlines)

    # Fix trailing commas
    repaired = re.sub(r',\s*([\]}])', r'\1', content)
    try:
        json.loads(repaired)
        return repaired
    except Exception:
        pass

    # Handle truncation (response ended prematurely). Re-apply the
    # trailing-comma fix here too: a response can be both truncated AND
    # have trailing commas earlier in the array/object (e.g. a chunk-fold
    # response cut off by max_tokens mid-array), and the brace-closing
    # logic below produces invalid JSON if an inner trailing comma survives.
    content = re.sub(r',\s*([\]}])', r'\1', content)
    content = re.sub(r':\s*$', ': null', content.strip())
    content = re.sub(r',\s*$', '', content.strip())

    actual_quotes = len(re.findall(r'(?<!\\)"', content))
    if actual_quotes % 2 != 0:
        content += '"'

    open_braces = content.count('{')
    close_braces = content.count('}')
    open_brackets = content.count('[')
    close_brackets = content.count(']')

    if open_braces > close_braces or open_brackets > close_brackets:
        test_content = content
        if open_brackets > close_brackets:
            test_content += "]" * (open_brackets - close_brackets)
        if open_braces > close_braces:
            test_content += "}" * (open_braces - close_braces)
        try:
            json.loads(test_content)
            return test_content
        except Exception:
            pass

    return None


def parse_llm_json(content: str) -> dict:
    """
    Parse an LLM's JSON response, trying progressively more aggressive
    recovery: as-is -> sanitized -> repaired. Raises JsonParseError with the
    original decode error if every attempt fails, so the caller's retry loop
    (see processing/retry.py) treats it as a failed attempt.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    cleaned = sanitize_json_response(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as original_error:
        repaired = repair_json(cleaned)
        if repaired is not None:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        raise JsonParseError(f"Could not parse LLM JSON response: {original_error}") from original_error
