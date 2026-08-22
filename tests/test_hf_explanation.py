"""
Tests for the Hugging Face explanation layer (backend/app/services/hf_explanation_service.py).

NOT VERIFIED IN THIS SANDBOX -- the container this was authored in has no
network egress and no installed dependencies (fastapi/httpx/pytest are not
importable here), so these tests could not actually be executed as part of
this change. They are written to run under the project's normal
`pytest` environment (see requirements.txt: httpx>=0.27, pytest>=8.0).
Run them for real with: `pytest tests/test_hf_explanation.py -v`.
"""

from __future__ import annotations

import os

import pytest

from backend.app.services import hf_explanation_service as hf


AVAILABLE_EXPLANATION = {
    "available": True,
    "success": True,
    "steps": 42,
    "reward": 12.5,
    "penalty": 3.0,
    "rescued": 4,
    "unmet": 0,
    "sections": {"MISSION SUMMARY": "MISSION COMPLETE — 4 rescued in 42 steps."},
}

UNAVAILABLE_EXPLANATION = {
    "available": False,
    "message": "No recorded episode to explain.",
}


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    # Ensure each test starts from a clean, disabled-by-default HF config,
    # matching the real Settings() defaults (HF_ENABLED unset -> false).
    for key in ("HF_ENABLED", "HF_API_TOKEN", "HF_MODEL", "HF_TIMEOUT_SECONDS", "HF_MAX_NEW_TOKENS"):
        monkeypatch.delenv(key, raising=False)
    yield


@pytest.mark.asyncio
async def test_falls_back_when_hf_disabled():
    result = await hf.explain_with_hf(AVAILABLE_EXPLANATION)
    assert result["source"] == hf.SOURCE_FALLBACK
    assert result["narrative"] is None
    # The original rule-based content must still be present, unmodified.
    assert result["sections"] == AVAILABLE_EXPLANATION["sections"]


@pytest.mark.asyncio
async def test_falls_back_when_no_recorded_episode():
    result = await hf.explain_with_hf(UNAVAILABLE_EXPLANATION)
    assert result["source"] == hf.SOURCE_FALLBACK
    assert result["narrative"] is None


@pytest.mark.asyncio
async def test_falls_back_when_token_missing_even_if_enabled(monkeypatch):
    monkeypatch.setenv("HF_ENABLED", "true")
    monkeypatch.setenv("HF_API_TOKEN", "")
    result = await hf.explain_with_hf(AVAILABLE_EXPLANATION)
    assert result["source"] == hf.SOURCE_FALLBACK
    assert result["narrative"] is None


@pytest.mark.asyncio
async def test_uses_hf_and_reports_source_when_call_succeeds(monkeypatch):
    monkeypatch.setenv("HF_ENABLED", "true")
    monkeypatch.setenv("HF_API_TOKEN", "fake-token-for-test")

    async def fake_call_hf(context):
        assert context == AVAILABLE_EXPLANATION  # only real data reaches the "model"
        return "MISSION SUMMARY: 4 rescued in 42 steps."

    monkeypatch.setattr(hf, "_call_hf", fake_call_hf)
    result = await hf.explain_with_hf(AVAILABLE_EXPLANATION)
    assert result["source"] == hf.SOURCE_HF
    assert result["narrative"] == "MISSION SUMMARY: 4 rescued in 42 steps."


@pytest.mark.asyncio
async def test_falls_back_on_hf_call_failure_without_fabricating(monkeypatch):
    monkeypatch.setenv("HF_ENABLED", "true")
    monkeypatch.setenv("HF_API_TOKEN", "fake-token-for-test")

    async def failing_call_hf(context):
        raise TimeoutError("simulated HF timeout")

    monkeypatch.setattr(hf, "_call_hf", failing_call_hf)
    result = await hf.explain_with_hf(AVAILABLE_EXPLANATION)
    assert result["source"] == hf.SOURCE_FALLBACK
    assert result["narrative"] is None
    assert "hf_error" in result
