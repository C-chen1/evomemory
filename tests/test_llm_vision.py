"""Vision plumbing and the reasoning-model empty-response guard.

Both are regressions: the empty-content guard exists because a reasoning model
spent its whole completion budget thinking and returned nothing, which the old
code recorded as a refusal; the image path exists because P3 needs real pixels.
"""

import base64
import io
import json

import pytest

from evomemory.llm import (
    BudgetedLLM, DeepSeekClient, LLMError, ScriptedLLM, _content_parts,
)


# -- content assembly ------------------------------------------------------


def test_text_only_content_is_a_plain_string():
    assert _content_parts("hello", None) == "hello"
    assert _content_parts("hello", []) == "hello"


def test_images_become_data_url_parts():
    parts = _content_parts("look", [b"\x89PNG-fake"])
    assert isinstance(parts, list) and len(parts) == 2
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["type"] == "image_url"
    url = parts[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNG-fake"


def test_multiple_images_preserve_order():
    parts = _content_parts("two", [b"a", b"b"])
    urls = [p["image_url"]["url"] for p in parts[1:]]
    assert [base64.b64decode(u.split(",", 1)[1]) for u in urls] == [b"a", b"b"]


# -- the client attaches images to the last user turn ----------------------


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_post(monkeypatch, payload):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return captured


def test_images_are_attached_to_the_last_user_message(monkeypatch):
    payload = {"choices": [{"message": {"content": "ok", "role": "assistant"},
                            "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    captured = _capture_post(monkeypatch, payload)
    client = DeepSeekClient("sk-test", "m", base_url="https://example.test")
    client.complete([{"role": "system", "content": "s"},
                     {"role": "user", "content": "look"}],
                    purpose="act", images=[b"img"])
    msgs = captured["body"]["messages"]
    assert msgs[0]["content"] == "s"
    assert isinstance(msgs[1]["content"], list)
    assert msgs[1]["content"][1]["type"] == "image_url"


def test_empty_content_with_reasoning_tokens_raises(monkeypatch):
    """The regression: an all-reasoning reply used to look like a refusal."""
    payload = {"choices": [{"message": {"content": "", "role": "assistant"},
                            "finish_reason": "length"}],
               "usage": {"prompt_tokens": 30, "completion_tokens": 1200,
                         "completion_tokens_details": {"reasoning_tokens": 1200}}}
    _capture_post(monkeypatch, payload)
    client = DeepSeekClient("sk-test", "m", base_url="https://example.test")
    with pytest.raises(LLMError) as exc:
        client.complete([{"role": "user", "content": "hi"}], purpose="act")
    message = str(exc.value)
    assert "empty content" in message
    assert "reasoning_tokens=1200" in message
    assert "max_tokens" in message


def test_budget_forwards_images_and_counts_the_call():
    seen = {}

    class _Client:
        model = "fake"

        def complete(self, messages, **kw):
            seen.update(kw)
            from evomemory.llm import LLMResponse
            return LLMResponse(text="ok", input_tokens=3, output_tokens=1)

    llm = BudgetedLLM(_Client(), 5, 1000)
    llm.complete([{"role": "user", "content": "x"}], purpose="act", images=[b"i"])
    assert seen["images"] == [b"i"]
    assert llm.calls == 1


def test_scripted_client_records_that_it_saw_images():
    llm = ScriptedLLM(lambda m: "{}")
    llm.complete([{"role": "user", "content": "x"}], purpose="act", images=[b"a", b"b"])
    assert llm.images_seen == 2
    assert llm.calls[0]["images"] == 2
