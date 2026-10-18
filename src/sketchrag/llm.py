"""Ollama chat backend shared by every system (same model and decoding for all).

Talks to the Ollama HTTP API (/api/chat) with only the standard library, so the same code
runs against a local CPU server during development and against an Ollama server inside a
Kaggle GPU kernel for the bulk experiments (set OLLAMA_HOST).

Decoding is greedy (temperature 0, fixed seed), so runs are repeatable. Usage is
recorded from Ollama's own counters (prompt_eval_count / eval_count).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_MODEL = os.environ.get("SKETCHRAG_MODEL", "llama3.2:3b")
THINKING_MODELS = ("qwen3", "deepseek-r1", "gpt-oss", "magistral")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    llm_s: float = 0.0  # wall-clock seconds spent inside the model server

    def add(self, resp: dict, seconds: float) -> None:
        self.input_tokens += resp.get("prompt_eval_count", 0) or 0
        self.output_tokens += resp.get("eval_count", 0) or 0
        self.calls += 1
        self.llm_s += seconds

    @property
    def context_tokens(self) -> int:
        return self.input_tokens

    def cost(self, model: str) -> float:
        return 0.0  # local open-weight model: reported as tokens and seconds instead of dollars


@dataclass
class Completion:
    text: str
    stop_reason: str
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    refused: bool = False


@dataclass
class ChatResult:
    message: dict          # assistant message, ready to append to the conversation
    text: str
    tool_calls: list[dict]  # [{"name": str, "arguments": dict}]
    done_reason: str
    raw: dict


class LLM:
    def __init__(self, model: str = DEFAULT_MODEL, host: str | None = None, num_ctx: int = 8192,
                 temperature: float = 0.0, seed: int = 0, timeout_s: int = 1800, retries: int = 3):
        self.model = model
        host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.host = host if host.startswith("http") else f"http://{host}"
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.seed = seed
        self.timeout_s = timeout_s
        self.retries = retries

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        last = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(f"{self.host}/api/chat", data, {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    return json.load(r)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Ollama request failed after {self.retries + 1} attempts: {last}")

    def chat(self, messages: list[dict], usage: Usage, tools: list[dict] | None = None,
             fmt: dict | None = None, max_tokens: int = 512, num_ctx: int | None = None) -> ChatResult:
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"num_ctx": num_ctx or self.num_ctx, "num_predict": max_tokens,
                            "temperature": self.temperature, "seed": self.seed}}
        if self.model.startswith(THINKING_MODELS):
            body["think"] = False  # answer directly; thinking would multiply latency on CPU
        if tools:
            body["tools"] = tools
        if fmt:
            body["format"] = fmt
        t0 = time.perf_counter()
        resp = self._post(body)
        usage.add(resp, time.perf_counter() - t0)
        msg = resp.get("message", {})
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"name": fn.get("name", ""), "arguments": args})
        return ChatResult(message={"role": "assistant", "content": msg.get("content", ""),
                                   **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {})},
                          text=(msg.get("content") or "").strip(), tool_calls=calls,
                          done_reason=resp.get("done_reason", "stop"), raw=resp)

    def complete(self, system: str, user: str, max_tokens: int = 512, num_ctx: int | None = None) -> Completion:
        usage = Usage()
        t0 = time.perf_counter()
        r = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], usage,
                      max_tokens=max_tokens, num_ctx=num_ctx)
        return Completion(r.text, r.done_reason, usage, time.perf_counter() - t0)
