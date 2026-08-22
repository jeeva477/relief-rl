"""
Hugging Face explanation layer for DisasterMind AI (Section 18).

Architecture (one-way, explanation only):

    SIMULATION -> PPO / QR-DQN -> REAL DECISION -> REAL METRICS
        -> STRUCTURED CONTEXT -> HUGGING FACE -> EXPLANATION

Hard rules enforced by this module:
  * This service NEVER controls PPO, QR-DQN, routes, vehicles, the
    environment, or rewards. It only reads already-computed session/frame
    data and asks an LLM to narrate it in natural language.
  * The LLM is given ONLY real, already-recorded simulation data (built by
    `sim_service.explain_episode`, itself computed from real frames). It is
    explicitly instructed not to invent incidents, vehicles, routes,
    metrics, rewards, locations, or actions, and this module never fabricates
    a response on its own behalf either.
  * If HF is disabled, unconfigured, or the call fails/times out, this
    module returns the existing rule-based explanation unchanged, tagged
    SOURCE: RULE-BASED FALLBACK -- it never claims Hugging Face inference
    succeeded when it did not.
  * The API token lives only on the backend (read from environment via
    Settings) and is never sent to or exposed through the frontend.
  * Calls are made with httpx.AsyncClient so a slow/unavailable HF endpoint
    cannot block the asyncio event loop running the simulation/WebSocket
    loop; a hard timeout (HF_TIMEOUT_SECONDS) enforces this further.
  * Only called for meaningful events (mission completion/failure/stop,
    high-risk or anomalous steps, route changes) -- callers decide when to
    invoke `explain_with_hf`, this module does not gate that itself.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from backend.app.config import get_settings

logger = logging.getLogger(__name__)

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

SOURCE_HF = "HUGGING FACE"
SOURCE_FALLBACK = "RULE-BASED FALLBACK"

_SYSTEM_PROMPT = (
    "You are an explanation layer for a disaster-response reinforcement-learning "
    "simulator. You do not control the simulation, the AI policy, routes, vehicles, "
    "or rewards -- you only explain a mission that has ALREADY happened, using the "
    "structured data given to you. Rules:\n"
    "1. Use ONLY the numbers, events, and facts provided in the data below.\n"
    "2. Do NOT invent incidents, vehicles, routes, metrics, rewards, locations, or "
    "AI actions that are not present in the data.\n"
    "3. If the data is insufficient to answer something, say so plainly instead of "
    "guessing.\n"
    "4. Be concise, operational, and specific -- write for an emergency-management "
    "audience reviewing an after-action report.\n"
    "Respond with plain text organized under these headings: MISSION SUMMARY, "
    "WHY THE AI DECIDED THIS, WHAT WENT WELL, WHAT WENT WRONG, RECOMMENDATION."
)


def _build_user_prompt(structured_context: dict[str, Any]) -> str:
    """Serialize only the real, already-computed episode data. No values are
    added, inferred, or embellished here -- this is a straight JSON dump of
    what `sim_service.explain_episode` already computed from real frames."""
    return (
        "Real simulation data for one completed mission (JSON). Explain this "
        "mission using only what is below:\n\n"
        f"{json.dumps(structured_context, indent=2, default=str)}"
    )


async def _call_hf(structured_context: dict[str, Any]) -> str:
    """Raises on any failure (timeout, HTTP error, malformed response) so the
    caller can fall back cleanly -- this function never itself invents a
    response to paper over a failed call."""
    settings = get_settings()
    payload = {
        "model": settings.hf_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(structured_context)},
        ],
        "max_tokens": settings.hf_max_new_tokens,
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=settings.hf_timeout_seconds) as client:
        resp = await client.post(HF_ROUTER_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    text = data["choices"][0]["message"]["content"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Hugging Face returned an empty response")
    return text.strip()


async def explain_with_hf(rule_based_explanation: dict[str, Any]) -> dict[str, Any]:
    """Wrap the existing rule-based explanation with an optional Hugging Face
    narrative layer.

    `rule_based_explanation` is the dict already produced by
    `sim_service.explain_episode` (or `SimSession.explanation()`) -- this is
    the ONLY simulation data the LLM ever sees. The rule-based explanation
    itself is always included in the response unchanged, so a caller/UI can
    show it verbatim regardless of whether HF succeeded.

    Returns the rule-based dict plus:
      - "source": "HUGGING FACE" | "RULE-BASED FALLBACK"
      - "narrative": the LLM's text (HF) or None (fallback)
      - "hf_error": present only when HF was enabled but failed
    """
    settings = get_settings()
    result = dict(rule_based_explanation)

    if not rule_based_explanation.get("available", False):
        result["source"] = SOURCE_FALLBACK
        result["narrative"] = None
        return result

    if not settings.hf_enabled or not settings.hf_api_token:
        result["source"] = SOURCE_FALLBACK
        result["narrative"] = None
        return result

    try:
        narrative = await _call_hf(rule_based_explanation)
        result["source"] = SOURCE_HF
        result["narrative"] = narrative
        result["hf_model"] = settings.hf_model
        return result
    except Exception as exc:  # noqa: BLE001 - any HF failure falls back, never fabricates
        logger.warning("Hugging Face explanation call failed, using rule-based fallback: %s", exc)
        result["source"] = SOURCE_FALLBACK
        result["narrative"] = None
        result["hf_error"] = str(exc)
        return result
