"""P10: mini flash_next serve smoke.

The mini ``qwen3_8_flash_next`` stub is served through the SAME Engine/serve
route as the 27B target (no new HTTP stack): launch ``build/apps/ninfer-serve.exe``
on the mini fixture and check the OpenAI envelope. Adapted from the 27B full
contract in ``tools/smoke/serve_contract.py`` to the mini subset that this
stub supports -- /health + /v1/models + /v1/chat/completions (nonstream).
Vision and thinking are NOT exercised (the mini stub has neither).

The emitted content is deterministic-but-garbage (the ~1e21 MoE logit scale,
flagged for P12); the smoke asserts the ENVELOPE (one choice, string content,
valid finish_reason, consistent usage), not semantic correctness -- exactly as
the 27B smoke does.

Labels: flash_next;flash_next_P10.
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

from tools.smoke.serve_contract import (
    ContractError,
    json_response,
    openai_nonstream,
    request,
)

REPO = Path(__file__).resolve().parents[3]  # ninfer-win root
SERVE_EXE = REPO / "build" / "apps" / "ninfer-serve.exe"
FIXTURE = REPO / "out" / "flash_next_dev" / "qwen3_8_flash_next_mini.ninfer"
MODEL = "qwen3.8-flash-next"
OUT_DIR = REPO / "out" / "flash_next_dev"

pytestmark = pytest.mark.skipif(
    not (SERVE_EXE.exists() and FIXTURE.exists()),
    reason=f"missing serve exe ({SERVE_EXE.name}) or mini fixture ({FIXTURE.name})",
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_healthy(base_url: str, timeout: float) -> None:
    """Robust health poll: the live serve's /health is a rich dict, so match on
    the status field (serve_contract.wait_for_health's ``== {"status":"ok"}``
    does not match it)."""
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            health = json_response(base_url, "GET", "/health")
            if health.get("status") == "ok":
                return
        except (ContractError, OSError) as error:
            last = error
        time.sleep(0.25)
    raise ContractError(f"mini serve did not become healthy in {timeout:g}s: {last}")


def test_mini_serve_smoke() -> None:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    out_log = OUT_DIR / "p10_serve_smoke_out.log"
    err_log = OUT_DIR / "p10_serve_smoke_err.log"

    env = dict(os.environ)
    with out_log.open("wb") as out, err_log.open("wb") as err:
        proc = subprocess.Popen(
            [
                str(SERVE_EXE),
                str(FIXTURE),
                "--host", "127.0.0.1",
                "--port", str(port),
                "--max-context", "64",
                "--kv-capacity", "64",
                "--max-concurrency", "1",
                "--vision", "off",
                "--greedy",
                "--request-log-jsonl", "off",
            ],
            cwd=str(SERVE_EXE.parent),  # DLLs resolve next to the exe
            env=env,
            stdout=out,
            stderr=err,
        )
        try:
            _wait_healthy(base_url, timeout=120)

            # /health: identity + status.
            health = json_response(base_url, "GET", "/health")
            assert health.get("model_id") == MODEL
            assert health.get("context", {}).get("max_context") == 64

            # /v1/models: object list, id + owner (mirrors the 27B smoke).
            models = json_response(base_url, "GET", "/v1/models")
            assert models.get("object") == "list"
            data = models.get("data")
            assert isinstance(data, list) and len(data) == 1
            assert data[0].get("id") == MODEL
            assert data[0].get("owned_by") == "ninfer"

            # /v1/chat/completions: the canonical OpenAI envelope (one choice,
            # string content, valid finish_reason, usage.total == prompt+completion).
            resp = openai_nonstream(
                base_url, MODEL, [{"role": "user", "content": "hi"}], max_tokens=8
            )
            assert resp.get("model") == MODEL
            choice = resp["choices"][0]
            assert len(choice["message"]["content"]) >= 1  # stub emits to budget
            usage = resp["usage"]
            assert usage["prompt_tokens"] == 2  # "hi" -> 2 byte tokens

            # Content must be valid UTF-8 (the response was JSON-decoded above,
            # which only succeeds for valid UTF-8).
            choice["message"]["content"].encode("utf-8")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=20)
