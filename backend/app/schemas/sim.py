from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SimStartRequest(BaseModel):
    policy: str = Field(default="ppo", description="ppo | qrdqn | random | heuristic | shortest")
    difficulty: str = Field(default="MEDIUM", description="EASY | MEDIUM | HARD | EXTREME")
    disaster: str = Field(default="any", description="any, flood, wildfire, earthquake, ...")
    seed: int = Field(default=7, ge=0)
    grid_size: int = Field(default=10, ge=4, le=20)
    max_steps: int = Field(default=100, ge=10, le=500)
    speed: float = Field(default=1.0, ge=0.25, le=16.0)


class HazardView(BaseModel):
    id: str
    x: float
    y: float
    radius: float
    severity: float
    type: str
    hard: bool
    velocity: list[float]


class IncidentView(BaseModel):
    x: float
    y: float
    victims: int


class RewardBreakdown(BaseModel):
    progress: float = 0.0
    distance_cost: float = 0.0
    time_cost: float = 0.0
    risk_cost: float = 0.0
    traffic_cost: float = 0.0
    blocked_penalty: float = 0.0
    hard_violation_penalty: float = 0.0
    safe_zone_bonus: float = 0.0
    success_bonus: float = 0.0
    rescue_bonus: float = 0.0
    efficiency_bonus: float = 0.0
    reroute_cost: float = 0.0
    unnecessary_move: float = 0.0
    resource_waste: float = 0.0
    dispatch_bonus: float = 0.0
    failed_rescue: float = 0.0


class SimFrame(BaseModel):
    session_id: str
    step: int
    status: str  # running | paused | done
    policy: str
    model_label: str
    grid_size: int
    disaster: str
    agent: dict[str, float]
    goal: dict[str, float]
    action: dict[str, Any]
    action_valid: bool
    valid_mask: list[int]
    reward: float
    reward_breakdown: RewardBreakdown
    penalty: float
    cumulative_reward: float
    cumulative_penalty: float
    score: float
    victims_total: int
    victims_rescued: int
    unmet: int
    resources: int
    vehicles: int
    priority: float
    dispatch_steps_left: int
    weather: str
    severity: float
    traffic_level: float
    time_frac: float
    distance_to_goal: float
    route_distance: int
    hard_violations: int
    blocked_attempts: int
    wasted_actions: int
    blocked_cells: list[list[int]]
    hazards: list[HazardView]
    incidents: list[IncidentView]
    terminated: bool
    truncated: bool
    success: bool
    timed_out: bool
    explanation: str
    response_time_s: float | None = None
    inference_ms: float | None = None
    env_step_ms: float | None = None
    episode_metrics: dict[str, Any] | None = None
    risk_quantiles: list[float] | None = None
    route_risk: str = "LOW"  # LOW | MEDIUM | HIGH -- deterministic, from real frame state
    route_risk_score: float = 0.0
    anomaly_status: str = "NORMAL"  # NORMAL | WARNING | ANOMALY -- lightweight statistical detection
    anomaly_reasons: list[str] = Field(default_factory=list)


class SimState(BaseModel):
    session_id: str
    status: str
    policy: str
    step: int
    difficulty: str
    disaster: str
    seed: int
    frames: int


class SimEvent(BaseModel):
    event_id: str
    step: int
    simulation_time: float
    type: str
    severity: str
    message: str
    text: str  # kept for older frontend clients that read `text`
    location: dict[str, float] | None = None
    vehicle: str | None = None
    incident: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplayResponse(BaseModel):
    session_id: str
    policy: str
    frames: list[SimFrame]
    events: list[SimEvent] = Field(default_factory=list)


class TrainStartRequest(BaseModel):
    episodes: int = Field(default=200, ge=10, le=2000)
    difficulty: str = "MEDIUM"
    disaster: str = "any"
    seed: int = 42
    algo: str = Field(default="ppo", description="ppo | qrdqn")


class TrainStatus(BaseModel):
    running: bool
    run_id: str | None = None
    episode: int | None = None
    total_episodes: int | None = None
    latest: dict[str, Any] | None = None
    checkpoint_dir: str | None = None
    message: str | None = None


class ModelStatus(BaseModel):
    available: bool
    model_name: str | None = None
    model_version: str | None = None
    algo: str | None = None
    obs_dim: int | None = None
    n_actions: int | None = None
    hidden_dim: int | None = None
    episode: int | None = None
    mean_reward: float | None = None
    path: str | None = None
    compatible: bool
    incompatible_reason: str | None = None
    fallback_policy: str | None = None


class EvalRunRequest(BaseModel):
    episodes: int = Field(default=20, ge=1, le=200)
    seeds: int = Field(default=1, ge=1, le=10)
    difficulty: str = "MEDIUM"
    disaster: str = "any"
    grid_size: int = 10
    max_steps: int = 100
    seed: int = 123


class EvalStatus(BaseModel):
    running: bool
    kind: str | None = None
    message: str | None = None
    result: dict[str, Any] | None = None


class LearningTrend(BaseModel):
    available: bool
    episodes: list[dict[str, Any]]
    checkpoint_dir: str | None = None
    source: str | None = None