import type { Dispatch, SetStateAction } from "react";

// ---------------------------------------------------------------------------
// Types mirroring backend/app/schemas/sim.py
// ---------------------------------------------------------------------------

export interface SimStartRequest {
  policy: string;
  difficulty: string;
  disaster: string;
  seed: number;
  grid_size: number;
  max_steps: number;
  speed: number;
}

export interface HazardView {
  id: string;
  x: number;
  y: number;
  radius: number;
  severity: number;
  type: string;
  hard: boolean;
  velocity: [number, number];
}

export interface IncidentView {
  x: number;
  y: number;
  victims: number;
}

export interface RewardBreakdown {
  progress: number;
  distance_cost: number;
  time_cost: number;
  risk_cost: number;
  traffic_cost: number;
  blocked_penalty: number;
  hard_violation_penalty: number;
  safe_zone_bonus: number;
  success_bonus: number;
  rescue_bonus: number;
  efficiency_bonus: number;
  reroute_cost: number;
  unnecessary_move: number;
  resource_waste: number;
  dispatch_bonus: number;
  failed_rescue: number;
}

export interface SimFrame {
  session_id: string;
  step: number;
  status: string;
  policy: string;
  model_label: string;
  grid_size: number;
  disaster: string;
  agent: { x: number; y: number };
  goal: { x: number; y: number };
  action: { id: number; name: string; valid: boolean };
  action_valid: boolean;
  valid_mask: number[];
  reward: number;
  reward_breakdown: RewardBreakdown;
  penalty: number;
  cumulative_reward: number;
  cumulative_penalty: number;
  score: number;
  victims_total: number;
  victims_rescued: number;
  unmet: number;
  resources: number;
  vehicles: number;
  priority: number;
  dispatch_steps_left: number;
  weather: string;
  severity: number;
  traffic_level: number;
  time_frac: number;
  distance_to_goal: number;
  route_distance: number;
  hard_violations: number;
  blocked_attempts: number;
  wasted_actions: number;
  blocked_cells: number[][];
  hazards: HazardView[];
  incidents: IncidentView[];
  terminated: boolean;
  truncated: boolean;
  success: boolean;
  timed_out: boolean;
  explanation: string;
  response_time_s: number | null;
  inference_ms: number | null;
  env_step_ms: number | null;
  episode_metrics: Record<string, unknown> | null;
  risk_quantiles: number[] | null;
  route_risk: "LOW" | "MEDIUM" | "HIGH";
  route_risk_score: number;
  anomaly_status: "NORMAL" | "WARNING" | "ANOMALY";
  anomaly_reasons: string[];
}

export interface SimState {
  session_id: string;
  status: string;
  policy: string;
  step: number;
  difficulty: string;
  disaster: string;
  seed: number;
  frames: number;
}

export interface SimEvent {
  event_id?: string;
  step?: number;
  simulation_time?: number;
  type: string;
  severity: string;
  message?: string;
  text: string;
  location?: { x: number; y: number } | null;
  vehicle?: string | null;
  incident?: string | null;
  metadata?: Record<string, unknown>;
}

export interface AiPerformance {
  model: string;
  ai_decisions: number;
  valid_decisions: number;
  masked_actions: number;
  route_changes: number;
  dispatches: number;
}

export interface VehicleSummary {
  active_vehicles: number;
  dispatches: number;
  note: string;
}

export interface TrainStatus {
  running: boolean;
  run_id: string | null;
  episode: number | null;
  total_episodes: number | null;
  latest: Record<string, string> | null;
  checkpoint_dir: string | null;
  message: string | null;
}

export interface ModelStatus {
  available: boolean;
  model_name: string | null;
  model_version: string | null;
  algo: string | null;
  obs_dim: number | null;
  n_actions: number | null;
  hidden_dim: number | null;
  episode: number | null;
  mean_reward: number | null;
  path: string | null;
  compatible: boolean;
  incompatible_reason: string | null;
  fallback_policy: string | null;
}

export interface TrendPoint {
  episode: number;
  reward: number | null;
  penalty: number | null;
  net_reward: number | null;
  success: boolean | null;
  success_rate: number | null;
  response_time_s: number | null;
  rescued: number | null;
  route_efficiency: number | null;
  hard_violations: number | null;
  wasted_actions: number | null;
  blocked_attempts: number | null;
  policy_loss: number | null;
  value_loss: number | null;
  entropy: number | null;
}

export interface LearningTrend {
  available: boolean;
  episodes: TrendPoint[];
  checkpoint_dir: string | null;
  source: string | null;
}

export interface ExplanationSection {
  available: boolean;
  generated_from: string | null;
  success: boolean | null;
  end_reason: "completed" | "failed" | "timeout" | "user_stopped" | null;
  reward: number | null;
  penalty: number | null;
  rescued: number | null;
  unmet: number | null;
  victims: number | null;
  steps: number | null;
  response_time_s: number | null;
  route_efficiency: number | null;
  route_risk_summary?: {
    mean_score: number;
    peak_level: "LOW" | "MEDIUM" | "HIGH";
    steps_low: number;
    steps_medium: number;
    steps_high: number;
    route_changes: number;
  };
  anomaly_summary?: { event_count: number; status: "NORMAL" | "WARNING" | "ANOMALY" };
  sections: Record<string, string>;
  ai_performance?: AiPerformance;
  reward_components?: Record<string, number>;
  penalty_components?: Record<string, number>;
  vehicle_summary?: VehicleSummary;
  final_state?: SimFrame | null;
  policy?: string;
  model_label?: string;
  difficulty?: string;
  disaster?: string;
  seed?: number;
  // Hugging Face explanation layer (Section 18) -- "HUGGING FACE" only when
  // a real HF call succeeded this request; otherwise "RULE-BASED FALLBACK".
  // `narrative` is the LLM's text when source is HUGGING FACE, else null --
  // never fabricated client-side.
  source?: "HUGGING FACE" | "RULE-BASED FALLBACK";
  narrative?: string | null;
  hf_model?: string;
  hf_error?: string;
}

// Mirrors rl/evaluation/metrics.py::aggregate_to_dict output.
export interface AgentAggregateMetrics {
  n_episodes: number;
  success_rate: number;
  mean_reward: number;
  std_reward: number;
  mean_steps: number;
  mean_response_time_s: number | null;
  mean_rescues: number;
  mean_unmet: number;
  mean_resource_usage: number;
  mean_failed_actions: number;
  mean_distance: number;
  route_efficiency: number | null;
  violation_rate: number;
  timeout_rate: number;
  mean_penalty?: number;
  std_penalty?: number;
}

// Mirrors backend/app/services/sim_service.py::ppo_vs_qrdqn_comparison output
// (and rl/checkpoints/ppo_vs_qrdqn.json written by scripts/compare_ppo_qrdqn.py).
export interface PpoVsQrdqnComparison {
  episodes: number;
  seed: number;
  difficulty: string;
  disaster: string;
  agents: Record<string, AgentAggregateMetrics>;
  note?: string;
  source?: string;
}

export interface WsMessage {
  type: string;
  payload?: Record<string, unknown>;
}

export type WsHandler = (msg: WsMessage) => void;

// ---------------------------------------------------------------------------
// REST client
// ---------------------------------------------------------------------------

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api/sim";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail ?? body.message ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export function startSimulation(req: SimStartRequest): Promise<SimState> {
  return http("/start", { method: "POST", body: JSON.stringify(req) });
}

export function stepSimulation(sessionId: string) {
  return http<{ frame: SimFrame; events: SimEvent[]; summary: Record<string, unknown> }>(
    `/step?session_id=${sessionId}`,
    { method: "POST" }
  );
}

export function resetSimulation(sessionId: string, seed?: number): Promise<SimState> {
  return http(`/reset?session_id=${sessionId}${seed != null ? `&seed=${seed}` : ""}`, { method: "POST" });
}

export interface StopSummary {
  session_id: string;
  status: string;
  end_reason: "user_stopped" | "completed" | "failed" | "timeout" | null;
  reward: number;
  penalty: number;
  rescued: number;
  unmet: number;
  victims: number;
  steps: number;
  success: boolean;
}

export function stopSimulation(sessionId: string): Promise<StopSummary> {
  return http(`/stop?session_id=${sessionId}`, { method: "POST" });
}

export function pauseSimulation(sessionId: string): Promise<SimState> {
  return http(`/pause?session_id=${sessionId}`, { method: "POST" });
}

export function resumeSimulation(sessionId: string): Promise<SimState> {
  return http(`/resume?session_id=${sessionId}`, { method: "POST" });
}

export function getSimulationState(sessionId: string): Promise<SimState> {
  return http(`/state?session_id=${sessionId}`);
}

export function getHistory(sessionId: string): Promise<{ session_id: string; policy: string; frames: SimFrame[]; events: SimEvent[] }> {
  return http(`/history?session_id=${sessionId}`);
}

export function explainSession(sessionId: string): Promise<ExplanationSection> {
  return http(`/explain?session_id=${sessionId}`, { method: "POST" });
}

export function getLearningTrend(): Promise<LearningTrend> {
  return http("/learning");
}

export function getModelStatus(algo: string = "ppo"): Promise<ModelStatus> {
  return http(`/model-status?algo=${encodeURIComponent(algo)}`);
}

export function startTraining(body: { episodes: number; difficulty: string; disaster?: string; seed?: number }): Promise<TrainStatus> {
  return http("/train", { method: "POST", body: JSON.stringify(body) });
}

export function getTrainingStatus(): Promise<TrainStatus> {
  return http("/train/status");
}

export function stopTraining(): Promise<TrainStatus> {
  return http("/train/stop", { method: "POST" });
}

export function runEvaluation(body: Record<string, unknown>) {
  return http("/evaluate", { method: "POST", body: JSON.stringify(body) });
}

export function getEvaluation() {
  return http<Record<string, unknown>>("/evaluate");
}

export function runBeforeAfter(params: { episodes?: number; seed?: number; difficulty?: string; disaster?: string }) {
  const q = new URLSearchParams(
    Object.entries(params).map(([k, v]) => [k, String(v)])
  ).toString();
  return http<Record<string, unknown>>(`/evaluate/before-after?${q}`, { method: "POST" });
}

export function getPpoVsQrdqn(params?: {
  live?: boolean;
  episodes?: number;
  seed?: number;
  difficulty?: string;
  disaster?: string;
}): Promise<PpoVsQrdqnComparison> {
  const q = new URLSearchParams(
    Object.entries(params ?? {}).map(([k, v]) => [k, String(v)])
  ).toString();
  return http(`/compare/ppo-vs-qrdqn${q ? `?${q}` : ""}`);
}

// ---------------------------------------------------------------------------
// WebSocket client (single connection, reconnect, cleanup)
// ---------------------------------------------------------------------------

const WS_BASE = import.meta.env.VITE_WS_BASE ?? "";

export class SimSocket {
  private ws: WebSocket | null = null;
  private handlers = new Set<WsHandler>();
  private statusHandlers = new Set<Dispatch<SetStateAction<string>>>();
  private reconnectAttempts = 0;
  private shouldRun = false;
  private pending: Record<string, unknown>[] = [];
  private url: string;

  constructor(sessionId?: string) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const q = sessionId ? `?session_id=${sessionId}` : "";
    this.url = `${proto}://${location.host}${WS_BASE}/ws/sim${q}`;
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    this.shouldRun = true;
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.emitStatus("connected");
      while (this.pending.length) {
        const m = this.pending.shift()!;
        ws.send(JSON.stringify(m));
      }
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data as string) as WsMessage;
        this.handlers.forEach((h) => h(msg));
      } catch {
        /* ignore malformed frames */
      }
    };
    ws.onclose = () => {
      this.emitStatus("closed");
      if (this.shouldRun) this.scheduleReconnect();
    };
    ws.onerror = () => {
      this.emitStatus("error");
      ws.close();
    };
  }

  private scheduleReconnect() {
    const delay = Math.min(1000 * 2 ** this.reconnectAttempts, 10000);
    this.reconnectAttempts += 1;
    window.setTimeout(() => {
      if (this.shouldRun) this.connect();
    }, delay);
  }

  disconnect() {
    this.shouldRun = false;
    this.handlers.clear();
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
    this.emitStatus("closed");
  }

  send(obj: Record<string, unknown>) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
    } else if (this.shouldRun) {
      this.pending.push(obj);
    }
  }

  subscribe(handler: WsHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  onStatus(handler: Dispatch<SetStateAction<string>>): () => void {
    this.statusHandlers.add(handler);
    return () => this.statusHandlers.delete(handler);
  }

  private emitStatus(status: string) {
    this.statusHandlers.forEach((h) => h(status));
  }

  start(req: SimStartRequest) {
    this.send({ type: "start", payload: { ...req } });
  }

  pause() {
    this.send({ type: "pause" });
  }

  resume() {
    this.send({ type: "resume" });
  }

  stepOnce() {
    this.send({ type: "step" });
  }

  reset(seed?: number) {
    this.send({ type: "reset", payload: { seed } });
  }

  /** User-initiated stop: freezes the session in its real current state
   * server-side (end_reason='user_stopped') and triggers a "mission_stopped"
   * message back -- never just closes the socket or discards data. */
  stop() {
    this.send({ type: "stop" });
  }

  setSpeed(speed: number) {
    this.send({ type: "set_speed", payload: { speed } });
  }

  ping() {
    this.send({ type: "ping", payload: { t: Date.now() } });
  }
}