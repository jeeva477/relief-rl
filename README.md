# DisasterMind AI — 3D AI Disaster-Response Simulator

*(codename: Relief-RL — the underlying repo/package names are unchanged to
avoid breaking imports and tooling; "DisasterMind AI" is the user-facing
product name shown in the UI.)*

A playable 3D disaster-response simulator driven by a real Proximal Policy
Optimization (PPO) agent. The agent evacuates civilians, fights hazards,
dispatches emergency vehicles and learns from penalties — all inside a
React Three Fiber world that mirrors the actual RL environment.

```
┌───────────────────────────┐      ┌───────────────────────────┐
│  React + Vite + R3F       │      │  FastAPI + WebSocket      │
│  SimulatorView (3D)       │◄────►│  /api/sim/*  +  /ws/sim   │
│  HUD / panels / minimap   │      │  SimManager (sessions)    │
│  replay / charts          │      └────────────┬──────────────┘
└───────────────────────────┘                   │
                                                ▼
                          ┌───────────────────────────┐
                          │  Gymnasium RL environment │
                          │  EvacuationEnv            │
                          │  scenarios / dynamics     │
                          └──────┬────────────┬───────┘
                                 ▼            ▼
                     PPO trainer       evaluation suite
                     (rl/algorithms)   (rl/training, metrics)
```

---

## Project purpose

The simulator lets anyone start a disaster mission (flood, wildfire,
earthquake, cyclone, tsunami, …) and watch an RL agent act autonomously:
the 3D world reacts, vehicles move, incidents evolve, penalties appear,
the mission ends, and the system explains what happened using the episode's
actual metrics — never fabricated values.

## Architecture

| Layer | Technology | Location |
|---|---|---|
| Frontend | React 19, Vite, TypeScript, React Three Fiber, Three.js | `frontend/` |
| Backend | FastAPI, WebSocket, uvicorn | `backend/app/` |
| RL env | Gymnasium (`EvacuationEnv`) | `rl/envs/` |
| RL algorithm | PPO (clipped surrogate + GAE) | `rl/algorithms/ppo.py` |
| Evaluation | baselines + PPO on identical seeds | `rl/training/evaluate.py` |
| Tests | pytest | `tests/` |

## RL environment

`EvacuationEnv` is the single source of truth. The 3D scene, HUD and
explanations are visualizations of this environment — there is no separate
simulation.

- Grid world (default 10×10), the agent must reach the goal while a
  disaster evolves (hazard zones, hard-hazard no-entry zones, blocked roads,
  traffic) and while responding to incidents (civilian rescue).
- **Action space (8):** STAY, N, E, S, W, REROUTE, DISPATCH, PRIORITIZE.
- **Observation (37 dims):** agent/goal coordinates, distance-to-goal,
  blocked-cell mask, hazard proximity/severity, incidents, vehicles,
  priority, resources, weather, severity, traffic level.
- **Action masking:** blocked roads are masked out of the action
  distribution during both rollout and training (`-1e8` logit fill, which
  is numerically identical to `-inf` but cannot produce `0 * -inf` NaN in
  the entropy term).
- **Rewards:** progress, distance/time/risk/traffic costs, blocked- and
  hard-violation penalties, reroute cost, resource-waste penalty,
  dispatch bonus, rescue reward, success bonus, route efficiency. Weights
  in `rl/envs/evacuation_env.py::RewardWeights`. All penalties flow into
  GAE advantages and shrink the probability of the bad actions (verified
  by tests in `tests/test_ppo.py`).
- **Disasters:** flood, wildfire, earthquake, cyclone, tsunami, landslide,
  heavy_rain, road_blockage, traffic_jam, combined.
- **Difficulties:** EASY / MEDIUM / HARD / EXTREME scale hazard hardness,
  hazard growth, zone radius and hazard count (`rl/envs/scenarios.py`).

## PPO

`rl/algorithms/ppo.py` is a genuine PPO (Schulman et al., 2017), not A2C
relabeled:

- clipped surrogate objective `L = min(r·A, clip(r, 1±ε)·A)`
- Generalized Advantage Estimation (`λ = 0.95, γ = 0.99`)
- multi-epoch minibatch updates over the rollout
- per-minibatch advantage normalization, clipped value loss, entropy bonus,
  gradient clipping
- NaN-safety: non-finite minibatches are skipped with a warning

The trainer in `rl/training/train.py` collects `rollout_episodes` (8)
per update, clears the buffer, logs every episode to
`rl/checkpoints/metrics.csv` + `training_log.json`, and saves
`latest_model.pt` / `best_model.pt` (best = highest window mean reward —
never the final episode).

## Training

```
python scripts/train.py --episodes 800 --difficulty MEDIUM --seed 42 \
    --checkpoint-dir rl/checkpoints --eval-after-episodes 50
```

Options: `--algo ppo|a2c`, `--disaster any|flood|wildfire|...`,
`--learning-rate`, `--gamma`, `--gae-lambda`, `--clip-epsilon`,
`--entropy-coef`, `--n-epochs`, `--batch-size`, `--rollout-episodes`,
`--max-steps`, `--grid-size`, `--checkpoint-frequency`.

Training logs:

| Episode | Reward | Penalty | Net | Success | Rescued |
|---|---|---|---|---|---|
| … | 64.26 | −12.4 | 51.9 | 1.00 | 442 |
| … | 31.19 | −9.8 | 21.4 | 0.90 | 431 |

(`metrics.csv` — real values from the training loop.)

## Evaluation

`rl/training/evaluate.py` runs any policy (Random, ShortestSafe,
RuleHeuristic, PPO) on identical seeded scenarios:

```
python scripts/compare_agents.py --episodes 25 --seed 7 --difficulty EASY
```

- `evaluate_policy(...)` → per-episode metrics: reward, penalty, success,
  rescued, unmet, response time, route efficiency, resource usage,
  violations, failed actions.
- `compare_agents(...)` → before/after table + CSV export.
- `--seed-offset` evaluates on a hold-out seed range (**unseen** scenarios,
  never used for checkpoint selection). `--eval-after-episodes N` in the
  trainer runs the same hold-out evaluation after training and writes it
  to `run_config.json`.

**Unseen evaluation (40 episodes, hold-out seeds, current checkpoint):**
success ≈ 0.38–0.48 depending on difficulty; mean response time ≈ 32–34 s;
zero hard violations. These are measured results, not targets.

## Simulation API

REST (`backend/app/api/sim.py`):

| Endpoint | Purpose |
|---|---|
| `POST /api/sim/start` | create a session (difficulty, disaster, policy) |
| `GET  /api/sim/state` | current session state |
| `POST /api/sim/step` | advance one step, return frame + events |
| `POST /api/sim/reset` | reset session |
| `GET  /api/sim/history` | recorded frames of a session |
| `GET  /api/sim/learning` | training history (real `metrics.csv`) |
| `GET  /api/sim/model-status` | checkpoint metadata + compatibility |
| `POST /api/sim/train` / `GET /api/sim/train/status` / `POST /api/sim/train/stop` | background training |
| `POST /api/sim/evaluate` / `GET /api/sim/evaluate` | evaluation (known + unseen seeds) |
| `POST /api/sim/evaluate/before-after` | policy comparison on identical seeds |
| `POST /api/sim/explain` | rule-based explanation from actual episode data |

## WebSocket

`WS /ws/sim` streams the live simulation. The client sends commands
(`start`, `pause`, `resume`, `step`, `reset`, `set_speed`, `ping`); the
server pushes compact frames (step, timestamp, vehicle/incident/road/hazard
updates, AI decision, reward, penalty, events, done) plus lifecycle events:

```
connected → simulation_started → frame → event … → mission_complete | mission_failed
```

Frames carry deltas, not the whole world, and are throttled to the
requested playback speed. Replay uses recorded frames — no re-simulation.

## 3D simulator

`frontend/src/components/SimulatorView.tsx` (React Three Fiber):

- island terrain, roads (blocked cells become rubble), buildings,
  instanced forest, water (rises for floods/tsunami), emergency facilities
- emergency vehicles **synchronized with RL state**, interpolated movement,
  rotated to heading; vehicle type auto-selects from the disaster
  (flood/tsunami/heavy_rain → rescue boat, otherwise ambulance)
- disaster effects: rain, fire+smoke, dust, hazard domes/rings
- camera presets (OVERVIEW / FOLLOW / INCIDENT / DISASTER / FREE), minimap,
  HUD, incident/vehicle/disaster/AI panels, event timeline, replay with
  scrub, post-game report, learning charts, PPO-vs-DQN panel (DQN shows
  INSUFFICIENT DATA — no fabricated comparison), research mode, perf HUD (D)
- performance practices: instancing (forest, traffic), shared materials,
  object pooling for effects, throttled HUD state (no React renders per
  animation frame)

## Post-game explanation

`POST /api/sim/explain` produces sections from the episode's real numbers:
MISSION SUMMARY, WHY DID THE AI MAKE THESE DECISIONS?, WHAT CAUSED
PENALTIES?, WHAT WENT WELL?, WHAT WENT WRONG?, WHAT DID THE AI LEARN?,
RECOMMENDATION. If data is missing, a safe fallback is returned — the
system never invents an explanation (no LLM is used).

## Installation

Backend (Python 3.12):

```
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Frontend (Node 20+):

```
cd frontend && npm install
```

## Running

```
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
cd frontend && npm run dev        # http://localhost:5173 (proxies /api and /ws to :8000)
```

The model checkpoint ships in `rl/checkpoints/best_model.pt`
(`MODEL_PATH` env var overrides). Start a mission → the AI acts → watch,
pause, step, replay, retrain in Research mode.

## Testing

```
python -m pytest tests -q            # 76 tests
cd frontend && npm run build         # type-check + production build
```

Coverage includes: env (gymnasium compliance, masks, determinism),
reward/penalty components, PPO (GAE, clipped objective, masking,
NaN-entropy regression, reward/penalty propagation, checkpoint
save/load + reload), simulation API, WebSocket lifecycle, and a live
smoke path for end-to-end runs.

## Performance

Measured (not claimed) on the dev machine:

| Metric | Value |
|---|---|
| RL inference | ~5 ms / decision |
| Env step | < 1 ms |
| WS frame latency | localhost |
| Frontend | 3D scene instanced; draw calls dominated by terrain + instanced meshes |

Run the live smoke script or open the perf HUD (D) for current numbers.

## Asset sources

All 3D assets are procedural primitives generated in-code (no external
GLB/GLTF downloads): terrain is a flat layered island (cylinder + ring
beach, no elevation/heightmap yet), buildings/vehicles from boxes/cylinders
with shared materials, forest via instancing, water via an animated plane.

## Deployment

See [DEPLOY.md](./DEPLOY.md) for going live with Docker Compose (quick
HTTP test, or a real domain with automatic HTTPS via Caddy).