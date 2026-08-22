from __future__ import annotations

from fastapi import APIRouter

from rl.envs.scenarios import Difficulty

router = APIRouter()


@router.post("/api/alerts/test")
def test_alert():
    """Fires a synthetic high-severity alert payload for mobile UI testing."""
    return {
        "nearby": True,
        "severity": "HIGH",
        "distance_m": 420,
        "hazard_id": "HZ-TEST-001",
        "recommended_action": "EVACUATE",
        "note": "This is a synthetic test alert, not a real hazard detection.",
    }


@router.get("/api/scenarios")
def list_scenarios():
    return {"difficulties": [d.value for d in Difficulty]}
