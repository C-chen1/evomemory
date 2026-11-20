"""LLM access with a hard budget and a deterministic offline client.

Two clients, one interface:

* ``ScriptedLLM`` — a pure function of the messages. No network, no cost, fully
  reproducible. Every test and every offline benchmark run uses this.
* ``DeepSeekClient`` — an OpenAI-compatible ``/chat/completions`` call. Only
  constructed when a key *and* a model id are configured; the model id is never
  guessed.

Both sit behind ``BudgetedLLM``, which refuses the call that would cross the cap
rather than discovering it afterwards.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from .db import Database
from .models import now_iso


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    """Rough token estimate, used only where the provider gives us nothing.

    Deliberately conservative (4 chars/token). Every place this is used is
    labelled as an estimate in the results; real runs use provider-reported
    usage.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    model: str = ""
    estimated: bool = True
    raw: Dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    model: str

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        purpose: str,
        task_id: str = "",
        run_id: str = "",
        prompt_version: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------
# offline client
# --------------------------------------------------------------------------


class ScriptedLLM:
    """Deterministic client driven by a responder function.

    ``responder(messages) -> str`` is called for every request. Because the
    benchmark's agent policy is expressed as a responder, an offline run is a
    reproducible experiment about the *harness*, not about model quality —
    results are labelled accordingly and are never presented as evidence that
    memory "improves" anything.
    """

    def __init__(self, responder: Callable[[List[Dict[str, str]]], str],
                 model: str = "scripted-offline"):
        self.responder = responder
        self.model = model
        self.calls: List[Dict[str, Any]] = []

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        purpose: str,
        task_id: str = "",
        run_id: str = "",
        prompt_version: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        started = time.perf_counter()
        text = self.responder(messages)
        latency = int((time.perf_counter() - started) * 1000)
        input_tokens = estimate_tokens("\n".join(m.get("content", "") for m in messages))
        output_tokens = estimate_tokens(text)
        self.calls.append({
            "purpose": purpose, "task_id": task_id,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
        })
        return LLMResponse(
            text=text, input_tokens=input_tokens, output_tokens=output_tokens,
            latency_ms=latency, model=self.model, estimated=True,
        )


class StaticLLM(ScriptedLLM):
    """Always returns the same text. Handy for plumbing tests."""

    def __init__(self, text: str, model: str = "static-offline"):
        super().__init__(lambda messages: text, model=model)


# --------------------------------------------------------------------------
# real client
# --------------------------------------------------------------------------


class DeepSeekClient:
    """OpenAI-compatible chat completion over stdlib urllib."""

    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://api.deepseek.com",
                 timeout: int = 60):
        if not api_key:
            raise ValueError("DeepSeekClient requires an api key")
        if not model:
            raise ValueError("DeepSeekClient requires an explicit model id")
        self.api_key = api_key
        self.model = model
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            self.chat_url = f"{base}/chat/completions"
            self.models_url = f"{base}/models"
        else:
            self.chat_url = f"{base}/v1/chat/completions"
            self.models_url = f"{base}/v1/models"
        self.timeout = timeout

    # -- helpers --------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def list_models(self) -> List[str]:
        req = urllib.request.Request(self.models_url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            raise LLMError(f"model listing failed: {exc}") from exc
        return [m.get("id", "") for m in payload.get("data", [])]

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        purpose: str,
        task_id: str = "",
        run_id: str = "",
        prompt_version: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(self.chat_url, data=body, headers=self._headers(),
                                     method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"HTTP {exc.code} from provider: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"request failed: {exc}") from exc
        latency = int((time.perf_counter() - started) * 1000)

        try:
            text = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(f"malformed response: {json.dumps(payload)[:300]}") from exc
        usage = payload.get("usage") or {}
        finish_reason = payload["choices"][0].get("finish_reason")
        reasoning_tokens = ((usage.get("completion_tokens_details") or {})
                            .get("reasoning_tokens", 0))

        # Reasoning models spend the completion budget on hidden tokens first. If
        # max_tokens is too small the visible content comes back empty with
        # finish_reason="length" — which used to look like a model that refused to
        # answer, and silently burned a step. Say what actually happened.
        if not text.strip() and int(usage.get("completion_tokens", 0)) > 0:
            raise LLMError(
                f"empty content (finish_reason={finish_reason!r}, "
                f"reasoning_tokens={reasoning_tokens}); the token budget was likely "
                f"consumed by reasoning. Increase max_tokens."
            )

        return LLMResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", estimate_tokens(
                "\n".join(m.get("content", "") for m in messages)))),
            output_tokens=int(usage.get("completion_tokens", estimate_tokens(text))),
            latency_ms=latency,
            model=payload.get("model", self.model),
            estimated=False,
            raw={"id": payload.get("id", ""), "finish_reason": finish_reason,
                 "reasoning_tokens": reasoning_tokens},
        )


# --------------------------------------------------------------------------
# budget
# --------------------------------------------------------------------------


class BudgetedLLM:
    """Enforces the call/token cap and writes the ledger row for every call.

    The check happens before the request: crossing the cap raises, it does not
    silently truncate an experiment half way through a task.
    """

    def __init__(self, client: LLMClient, budget_calls: int, budget_tokens: int,
                 db: Optional[Database] = None):
        self.client = client
        self.max_calls = budget_calls
        self.max_tokens = budget_tokens
        self.db = db
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def model(self) -> str:
        return self.client.model

    def remaining(self) -> Dict[str, int]:
        return {
            "calls": self.max_calls - self.calls,
            "tokens": self.max_tokens - (self.input_tokens + self.output_tokens),
        }

    def complete(self, messages: List[Dict[str, str]], *, purpose: str,
                 task_id: str = "", run_id: str = "", prompt_version: str = "",
                 temperature: float = 0.0, max_tokens: int = 1024) -> LLMResponse:
        if self.calls + 1 > self.max_calls:
            raise BudgetExceeded(
                f"call budget exhausted ({self.calls}/{self.max_calls})")
        spent = self.input_tokens + self.output_tokens
        if spent >= self.max_tokens:
            raise BudgetExceeded(
                f"token budget exhausted ({spent}/{self.max_tokens})")

        ok, error = True, ""
        try:
            resp = self.client.complete(
                messages, purpose=purpose, task_id=task_id, run_id=run_id,
                prompt_version=prompt_version, temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - recorded then re-raised
            ok, error = False, f"{type(exc).__name__}: {exc}"
            self.calls += 1
            self._log(run_id, task_id, purpose, prompt_version, 0, 0, 0, ok, error)
            raise

        self.calls += 1
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self._log(run_id, task_id, purpose, prompt_version,
                  resp.input_tokens, resp.output_tokens, resp.latency_ms, ok, error)
        if self.input_tokens + self.output_tokens > self.max_tokens:
            raise BudgetExceeded(
                f"token budget exceeded after the call "
                f"({self.input_tokens + self.output_tokens}/{self.max_tokens}); "
                "stopping further requests")
        return resp

    def _log(self, run_id: str, task_id: str, purpose: str, prompt_version: str,
             in_tok: int, out_tok: int, latency: int, ok: bool, error: str) -> None:
        if self.db is None:
            return
        self.db.insert("llm_calls", {
            "run_id": run_id, "task_id": task_id, "purpose": purpose,
            "prompt_version": prompt_version, "model": self.client.model,
            "input_tokens": in_tok, "output_tokens": out_tok,
            "latency_ms": latency, "ok": 1 if ok else 0, "error": error,
            "created_at": now_iso(),
        })

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self.client.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "max_calls": self.max_calls,
            "max_tokens": self.max_tokens,
        }
