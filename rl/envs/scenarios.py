"""
Scenario generation for Relief-RL training and evaluation.

Difficulty levels:
    EASY    - few static hazards, low traffic, large safe area
    MEDIUM  - multiple hazards, some moving, moderate traffic
    HARD    - many hazards, road closures, high traffic, multiple safe zones
    EXTREME - dynamic/expanding hazards, high traffic, blocked routes,
              limited safe zones

DISASTER SCENARIOS
------------------
The environment is generalised across disaster types. Every scenario is
generated from a :class:`ScenarioConfig`, which describes:

    disaster_type   one of FLOOD / WILDFIRE / EARTHQUAKE / CYCLONE / TSUNAMI /
                    LANDSLIDE / HEAVY_RAIN / ROAD_BLOCKAGE / TRAFFIC_JAM / COMBINED
    severity        overall disaster severity in [0, 1]
    traffic         road traffic level in [0, 1]
    victims         number of affected people waiting at the goal zone
    weather         CLEAR / RAIN / HEAVY_RAIN / STORM
    blocked_fraction  fraction of roads randomly closed
    resources       number of rescue vehicles / resource units available

Each disaster type maps to a different hazard mix:

    FLOOD          - large slow-growing water zones (high radius, low growth)
    WILDFIRE       - small fast-growing fire zones (hard constraint, growth)
    EARTHQUAKE     - many moderate zones, tremors (road closures, moderate growth)
    CYCLONE        - wide storm zones, high traffic, heavy rain
    TSUNAMI        - very large slow water zones sweeping across the map
    LANDSLIDE      - many closed roads plus rockfall zones on a "mountain" line
    HEAVY_RAIN     - low visibility: high traffic cost, weather effect
    ROAD_BLOCKAGE  - the most closed roads, few hazards
    TRAFFIC_JAM    - very high traffic, few hazards
    COMBINED       - a mixture of two or three of the above

Scenarios are randomized per episode (subject to a seed) so the agent
is never trained or evaluated on a single fixed map.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

from rl.envs.hazard import Hazard, HazardType


class Difficulty(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    EXTREME = "EXTREME"


class DisasterType(str, Enum):
    FLOOD = "flood"
    WILDFIRE = "wildfire"
    EARTHQUAKE = "earthquake"
    CYCLONE = "cyclone"
    TSUNAMI = "tsunami"
    LANDSLIDE = "landslide"
    HEAVY_RAIN = "heavy_rain"
    ROAD_BLOCKAGE = "road_blockage"
    TRAFFIC_JAM = "traffic_jam"
    COMBINED = "combined"


class WeatherType(str, Enum):
    CLEAR = "clear"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    STORM = "storm"


@dataclass
class ScenarioConfig:
    """Complete, configurable description of a disaster scenario."""

    disaster_type: str = "combined"
    severity: float = 0.5           # overall severity in [0, 1]
    traffic: float = 0.4            # traffic level in [0, 1]
    victims: int = 40               # affected population at the goal zone
    weather: str = "clear"
    blocked_fraction: float = 0.1   # fraction of roads randomly closed
    resources: int = 4              # rescue vehicles / resource units
    difficulty: str = "MEDIUM"
    grid_size: int = 10
    max_steps: int = 100
    seed: int | None = None


@dataclass
class Scenario:
    difficulty: Difficulty
    disaster_type: DisasterType
    grid_size: int
    start: tuple[int, int]
    goal: tuple[int, int]
    hazards: list[Hazard]
    blocked_cells: set[tuple[int, int]]
    traffic_level: float  # 0.0 (free flow) .. 1.0 (gridlock)
    victims: int = 0                 # affected population at the goal
    weather: WeatherType = WeatherType.CLEAR
    resources: int = 4               # rescue resource units available
    severity: float = 0.5            # overall disaster severity in [0, 1]
    priority: float = 0.5            # emergency priority of the goal zone in [0, 1]
    config: ScenarioConfig | None = None


_DIFFICULTY_PARAMS = {
    Difficulty.EASY: dict(n_hazards=(1, 2), moving_frac=0.0, blocked=(0, 1),
                           traffic=(0.0, 0.2), severity=(0.2, 0.4), hard_frac=0.0),
    Difficulty.MEDIUM: dict(n_hazards=(2, 4), moving_frac=0.3, blocked=(0, 2),
                             traffic=(0.2, 0.5), severity=(0.3, 0.6), hard_frac=0.2),
    Difficulty.HARD: dict(n_hazards=(4, 6), moving_frac=0.5, blocked=(2, 5),
                           traffic=(0.4, 0.7), severity=(0.4, 0.8), hard_frac=0.35),
    Difficulty.EXTREME: dict(n_hazards=(6, 9), moving_frac=0.7, blocked=(4, 8),
                              traffic=(0.6, 0.9), severity=(0.5, 1.0), hard_frac=0.5),
}

class _DisasterParams(TypedDict):
    n_hazards: tuple[int, int]
    radius: tuple[float, float]
    growth: tuple[float, float]
    hard_frac: float
    traffic_mult: float
    blocked_mult: float
    weather: WeatherType


def _p(
    n_hazards: tuple[int, int],
    radius: tuple[float, float],
    growth: tuple[float, float],
    hard_frac: float,
    traffic_mult: float,
    blocked_mult: float,
    weather: WeatherType,
) -> _DisasterParams:
    return {
        "n_hazards": n_hazards,
        "radius": radius,
        "growth": growth,
        "hard_frac": hard_frac,
        "traffic_mult": traffic_mult,
        "blocked_mult": blocked_mult,
        "weather": weather,
    }


# Per-disaster tuning of the hazard mix.
#   n_hazards    (min, max) number of hazard zones
#   radius       (min, max) zone radius
#   growth       (min, max) radius growth per step
#   hard_frac    fraction of hard-constraint zones
#   traffic_mult multiplier applied to the configured traffic level
#   blocked_mult multiplier applied to the configured blocked fraction
#   weather      suggested weather
_DISASTER_PARAMS: dict[DisasterType, _DisasterParams] = {
    DisasterType.FLOOD: _p((3, 6), (0.08, 0.18), (0.001, 0.004), 0.30, 1.2, 1.0, WeatherType.RAIN),
    DisasterType.WILDFIRE: _p((2, 4), (0.05, 0.12), (0.004, 0.010), 0.50, 0.8, 0.6, WeatherType.CLEAR),
    DisasterType.EARTHQUAKE: _p((3, 5), (0.06, 0.14), (0.002, 0.005), 0.35, 1.3, 1.5, WeatherType.CLEAR),
    DisasterType.CYCLONE: _p((2, 4), (0.10, 0.20), (0.003, 0.008), 0.25, 2.0, 1.3, WeatherType.STORM),
    DisasterType.TSUNAMI: _p((1, 2), (0.15, 0.25), (0.005, 0.012), 0.30, 1.4, 1.2, WeatherType.STORM),
    DisasterType.LANDSLIDE: _p((2, 4), (0.06, 0.12), (0.0, 0.001), 0.40, 1.0, 1.8, WeatherType.RAIN),
    DisasterType.HEAVY_RAIN: _p((1, 3), (0.06, 0.14), (0.001, 0.002), 0.10, 1.8, 1.2, WeatherType.HEAVY_RAIN),
    DisasterType.ROAD_BLOCKAGE: _p((0, 2), (0.05, 0.10), (0.0, 0.0), 0.20, 1.2, 2.2, WeatherType.CLEAR),
    DisasterType.TRAFFIC_JAM: _p((0, 2), (0.05, 0.10), (0.0, 0.0), 0.10, 2.4, 0.4, WeatherType.CLEAR),
    DisasterType.COMBINED: _p((3, 6), (0.05, 0.18), (0.001, 0.008), 0.35, 1.4, 1.4, WeatherType.RAIN),
}

_DISASTER_HAZARD_TYPES = {
    DisasterType.FLOOD: [HazardType.FLOOD],
    DisasterType.WILDFIRE: [HazardType.WILDFIRE],
    DisasterType.EARTHQUAKE: [HazardType.EARTHQUAKE],
    DisasterType.CYCLONE: [HazardType.STORM, HazardType.FLOOD],
    DisasterType.TSUNAMI: [HazardType.FLOOD],
    DisasterType.LANDSLIDE: [HazardType.LANDSLIDE],
    DisasterType.HEAVY_RAIN: [HazardType.STORM, HazardType.FLOOD],
    DisasterType.ROAD_BLOCKAGE: [HazardType.GENERIC],
    DisasterType.TRAFFIC_JAM: [HazardType.GENERIC],
    DisasterType.COMBINED: [HazardType.FLOOD, HazardType.WILDFIRE, HazardType.EARTHQUAKE,
                             HazardType.STORM, HazardType.LANDSLIDE, HazardType.GENERIC],
}


def _as_disaster(value: str | DisasterType) -> DisasterType:
    return value if isinstance(value, DisasterType) else DisasterType(value)


def _as_weather(value: str | WeatherType) -> WeatherType:
    return value if isinstance(value, WeatherType) else WeatherType(value)


def _as_difficulty(value: str | Difficulty) -> Difficulty:
    return value if isinstance(value, Difficulty) else Difficulty(value)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _difficulty_multipliers(difficulty: Difficulty) -> tuple[float, float, float, float]:
    """(hard_mult, growth_mult, radius_mult, n_hazards_mult) per difficulty.

    This restores the difficulty progression: EASY has almost no hard
    hazards and slow growth; EXTREME has many hard, fast-growing zones.
    """
    return {
        Difficulty.EASY: (0.15, 0.5, 0.8, 0.6),
        Difficulty.MEDIUM: (0.5, 1.0, 1.0, 1.0),
        Difficulty.HARD: (0.85, 1.3, 1.15, 1.3),
        Difficulty.EXTREME: (1.0, 1.6, 1.3, 1.5),
    }[difficulty]


def _spawn_hazards(
    disaster: DisasterType,
    cfg: ScenarioConfig,
    params: _DisasterParams,
    rng: random.Random,
) -> list[Hazard]:
    """Generate the hazard zones for a disaster config."""
    hard_mult, growth_mult, radius_mult, n_mult = _difficulty_multipliers(
        _as_difficulty(cfg.difficulty)
    )
    hazards: list[Hazard] = []
    n_hazards = int(rng.randint(*params["n_hazards"]) * n_mult)
    n_hazards = max(1, n_hazards)
    for i in range(n_hazards):
        hx, hy = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)
        radius = rng.uniform(*params["radius"]) * (0.7 + 0.6 * cfg.severity) * radius_mult
        growth = rng.uniform(*params["growth"]) * (0.5 + cfg.severity) * growth_mult
        severity = _clamp01(rng.uniform(0.3, 1.0) * (0.5 + 0.5 * cfg.severity))
        hard = rng.random() < params["hard_frac"] * hard_mult
        is_moving = rng.random() < 0.3
        velocity = (rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01)) if is_moving else (0.0, 0.0)
        htype = rng.choice(_DISASTER_HAZARD_TYPES[disaster])
        hazards.append(
            Hazard(
                id=f"{disaster.value.upper()}-{i:03d}",
                x=hx,
                y=hy,
                radius=radius,
                severity=severity,
                hazard_type=htype,
                velocity=velocity,
                growth_rate=growth,
                hard_constraint=hard,
            )
        )
    return hazards


def _pick_start_goal(grid_size: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    start = (rng.randint(0, grid_size // 3), rng.randint(0, grid_size // 3))
    goal = (
        rng.randint(2 * grid_size // 3, grid_size - 1),
        rng.randint(2 * grid_size // 3, grid_size - 1),
    )
    return start, goal


def _sample_blocked(
    blocked_fraction: float,
    grid_size: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    rng: random.Random,
) -> set[tuple[int, int]]:
    """Randomly close `blocked_fraction` of the roads (excluding start/goal)."""
    cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    cells = [cell for cell in cells if cell not in (start, goal)]
    rng.shuffle(cells)
    n_blocked = int(len(cells) * blocked_fraction)
    return set(cells[:n_blocked])


def generate_scenario(
    difficulty: str | Difficulty,
    grid_size: int = 10,
    rng: random.Random | None = None,
) -> Scenario:
    """
    Backward-compatible generator: builds a random ScenarioConfig from the
    difficulty level and delegates to :func:`generate_scenario_from_config`.
    The disaster type is chosen at random so the training distribution
    covers every disaster family.
    """
    difficulty = _as_difficulty(difficulty)
    rng = rng or random.Random()
    params = _DIFFICULTY_PARAMS[difficulty]

    config = ScenarioConfig(
        disaster_type=rng.choice(list(DisasterType)).value,
        severity=float(rng.uniform(*params["severity"])),
        traffic=float(rng.uniform(*params["traffic"])),
        victims=int(rng.randint(20, 80)),
        weather=rng.choice(list(WeatherType)).value,
        blocked_fraction=float(rng.uniform(*params["blocked"]) / max(grid_size * grid_size, 1)),
        resources=int(rng.randint(2, 5)),
        difficulty=difficulty.value,
        grid_size=grid_size,
        max_steps=max(50, int(grid_size * 10)),
    )
    return generate_scenario_from_config(config, rng=rng)


def generate_scenario_from_config(
    config: ScenarioConfig,
    rng: random.Random | None = None,
) -> Scenario:
    """
    Generate a scenario from an explicit :class:`ScenarioConfig`. This is
    the entry point used by the live simulation API, so that every
    disaster type, severity, traffic, victim count, weather and road
    blockage chosen by the user maps to a concrete environment.
    """
    rng = rng or random.Random(config.seed)
    cfg = config
    grid_size = cfg.grid_size

    disaster = _as_disaster(cfg.disaster_type)
    weather = _as_weather(cfg.weather)
    difficulty = _as_difficulty(cfg.difficulty)

    params = _DISASTER_PARAMS[disaster]
    traffic = _clamp01(cfg.traffic * params["traffic_mult"])
    blocked_fraction = _clamp01(cfg.blocked_fraction * params["blocked_mult"])

    start, goal = _pick_start_goal(grid_size, rng)
    hazards = _spawn_hazards(disaster, cfg, params, rng)
    blocked_cells = _sample_blocked(blocked_fraction, grid_size, start, goal, rng)

    # Heavy weather lifts traffic further.
    if weather in (WeatherType.HEAVY_RAIN, WeatherType.STORM):
        traffic = _clamp01(traffic + 0.15)

    return Scenario(
        difficulty=difficulty,
        disaster_type=disaster,
        grid_size=grid_size,
        start=start,
        goal=goal,
        hazards=hazards,
        blocked_cells=blocked_cells,
        traffic_level=traffic,
        victims=cfg.victims,
        weather=weather,
        resources=cfg.resources,
        severity=_clamp01(cfg.severity),
        priority=0.5 + 0.5 * _clamp01(cfg.severity),  # worse disaster -> higher priority
        config=cfg,
    )
