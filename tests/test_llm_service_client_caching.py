"""
Tests for processing/llm_service_client.py's OpenRouter prompt-caching support.

_with_cache_control adds the cache_control breakpoint OpenRouter needs to
discount a repeated system prompt (Anthropic/Gemini require it explicitly;
OpenAI ignores it and caches automatically anyway) — but must never touch
the messages a self-hosted vLLM server would receive, since it only
understands plain string content.

_cached_tokens reads back whatever the provider reported was served from
cache, so a run's logs can show whether the discount actually landed instead
of only inferring it after the fact from cost math.

Run: python -m unittest discover -s tests
"""
import unittest
from types import SimpleNamespace

from processing.llm_service_client import _with_cache_control, _cached_tokens


class TestWithCacheControl(unittest.TestCase):
    def test_adds_breakpoint_to_system_message(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ]
        result = _with_cache_control(messages)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(
            result[0]["content"],
            [{"type": "text", "text": "You are a helpful assistant.",
              "cache_control": {"type": "ephemeral"}}],
        )
        self.assertEqual(result[1], {"role": "user", "content": "hello"})

    def test_does_not_mutate_the_original_list_or_dict(self):
        # The same `messages` object is reused for the self-hosted attempt
        # (see llm_service_client.call_chat) — a self-hosted vLLM server
        # only understands plain string content, so this must be a copy.
        original = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hi"},
        ]
        _with_cache_control(original)
        self.assertEqual(original[0]["content"], "system prompt")
        self.assertIsInstance(original[0]["content"], str)

    def test_no_system_message_is_left_unchanged(self):
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(_with_cache_control(messages), messages)

    def test_already_structured_system_content_is_left_unchanged(self):
        # Idempotent: calling this twice (or being passed already-cached
        # content) must not double-wrap or error.
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
            ]},
            {"role": "user", "content": "hi"},
        ]
        self.assertEqual(_with_cache_control(messages), messages)

    def test_empty_messages_is_left_unchanged(self):
        self.assertEqual(_with_cache_control([]), [])


class TestCachedTokens(unittest.TestCase):
    def test_reads_cached_tokens_when_present(self):
        usage = SimpleNamespace(
            prompt_tokens=100,
            prompt_tokens_details=SimpleNamespace(cached_tokens=80),
        )
        self.assertEqual(_cached_tokens(usage), 80)

    def test_none_when_details_absent(self):
        usage = SimpleNamespace(prompt_tokens=100)
        self.assertIsNone(_cached_tokens(usage))

    def test_none_when_details_present_but_no_cached_field(self):
        usage = SimpleNamespace(prompt_tokens=100, prompt_tokens_details=SimpleNamespace())
        self.assertIsNone(_cached_tokens(usage))


if __name__ == "__main__":
    unittest.main()
