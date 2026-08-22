import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import * as THREE from "three";
import {
  Activity, AlertTriangle, Camera, Crosshair, Gauge, Map as MapIcon,
  Play, Pause, RotateCcw, Repeat, StepForward, Zap, Siren, Cpu, Radio, Square,
} from "lucide-react";
import {
  SimSocket, startSimulation, stepSimulation, getHistory,
  getLearningTrend, getModelStatus, startTraining, getTrainingStatus,
  runEvaluation, runBeforeAfter, explainSession, stopSimulation,
  type SimFrame, type SimStartRequest, type SimEvent, type LearningTrend,
  type ModelStatus, type TrainStatus, type ExplanationSection,
} from "../services/sim";
import { Terrain, Roads, GoalMarker } from "./Simulator/Terrain";
import { Buildings } from "./Simulator/Buildings";
import { Forest } from "./Simulator/Forest";
import { Facilities } from "./Simulator/Facilities";
import { Vehicles, CivilianTraffic, Civilians } from "./Simulator/Vehicles";
import { Incidents } from "./Simulator/Incidents";
import { HazardZones, FireAndSmoke, Dust, Rain } from "./Simulator/DisasterEffects";
import { CameraController, PerformanceMeter } from "./Simulator/CameraController";
import { MiniMap } from "./HUD/MiniMap";
import { EventTimeline } from "./HUD/EventTimeline";
import { StartOverlay } from "./HUD/StartOverlay";
import { PostGameReport } from "./HUD/PostGameReport";
import { LearningCharts } from "./Analytics/LearningCharts";
import { PpoVsDqn } from "./Analytics/PpoVsDqn";
import { cellToWorld, normToWorld, type CamMode } from "./Simulator/world";

type RunStatus = "idle" | "running" | "paused" | "done";

function useThrottled<T>(value: T, ms: number): T {
  const [state, setState] = useState(value);
  const ref = useRef(value);
  useEffect(() => {
    ref.current = value;
    const id = window.setTimeout(() => setState(ref.current), ms);
    return () => window.clearTimeout(id);
  }, [value, ms]);
  return state;
}

function fmtMissionTime(step: number): string {
  const totalSec = Math.max(0, step * 2.0);
  const m = Math.floor(totalSec / 60);
  const s = Math.floor(totalSec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function SimulatorView() {
  const frameRef = useRef<SimFrame | null>(null);
  const startRef = useRef<[number, number]>([0, 0]);
  const focusTarget = useRef<THREE.Vector3 | null>(null);
  const hazardRefs = useRef(new Map<string, THREE.Vector3>());
  const blockedSetRef = useRef(new Set<string>());
  const [blockedVersion, setBlockedVersion] = useState(0);
  const socketRef = useRef<SimSocket | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [wsStatus, setWsStatus] = useState("disconnected");
  const [frame, setFrame] = useState<SimFrame | null>(null);
  const [events, setEvents] = useState<SimEvent[]>([]);
  const [speed, setSpeed] = useState(1);
  const [policy, setPolicy] = useState("ppo");
  const [difficulty, setDifficulty] = useState("EASY");
  const [disaster, setDisaster] = useState("any");
  const [seed, setSeed] = useState(7);
  const [camMode, setCamMode] = useState<CamMode>("overview");
  const [trend, setTrend] = useState<LearningTrend>({ available: false, episodes: [], checkpoint_dir: null, source: null });
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const [report, setReport] = useState<ExplanationSection | null>(null);
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const [replay, setReplay] = useState<{ frames: SimFrame[]; idx: number; playing: boolean } | null>(null);
  const [research, setResearch] = useState(false);
  const [showPerf, setShowPerf] = useState(false);
  const [perf, setPerf] = useState({ fps: 0, frameMs: 0, draws: 0, tris: 0 });
  const [aiLatency, setAiLatency] = useState<number | null>(null);
  const [apiLatency, setApiLatency] = useState<number | null>(null);
  const [replayPos, setReplayPos] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [beforeAfter, setBeforeAfter] = useState<Record<string, unknown> | null>(null);
  const [toasts, setToasts] = useState<{ id: number; kind: "reward" | "penalty" | "rescue" | "alert"; text: string }[]>([]);
  const [quality, setQuality] = useState(0);
  const routeSetRef = useRef(new Set<string>());
  const [routeVersion, setRouteVersion] = useState(0);
  const toastId = useRef(0);
  const lastToastStep = useRef(0);
  const lastRescued = useRef(0);
  const lastWentDown = useRef(false);

  useEffect(() => {
    if (wsStatus === "connected" && lastWentDown.current) {
      const id = window.setTimeout(() => { lastWentDown.current = false; }, 4000);
      return () => window.clearTimeout(id);
    }
    if (wsStatus === "error" || wsStatus === "closed") lastWentDown.current = true;
  }, [wsStatus]);

  const pushToast = useCallback((kind: "reward" | "penalty" | "rescue" | "alert", text: string) => {
    toastId.current += 1;
    const id = toastId.current;
    setToasts((prev) => [...prev.slice(-3), { id, kind, text }]);
    window.setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 2800);
  }, []);

  const frameSnapshot = useThrottled(frame, 200);

  const startSim = useCallback((opts?: Partial<SimStartRequest>) => {
    const req: SimStartRequest = {
      policy, difficulty, disaster, seed, grid_size: 10, max_steps: 100,
      speed, ...opts,
    };
    setError(null);
    setReport(null);
    setReplay(null);
    setEvents([]);
    setToasts([]);
    lastRescued.current = 0;
    lastToastStep.current = 0;
    routeSetRef.current.clear();
    setRouteVersion((v) => v + 1);
    setStatus("running");
    const socket = socketRef.current;
    if (socket) {
      socket.start(req);
    } else {
      startSimulation(req)
        .then((state) => {
          setSessionId(state.session_id);
          setStatus(state.status === "running" ? "running" : "idle");
        })
        .catch((err) => { setError(err.message); setStatus("idle"); });
    }
  }, [policy, difficulty, disaster, seed, speed]);

  useEffect(() => {
    const socket = new SimSocket(sessionId ?? undefined);
    socketRef.current = socket;
    socket.connect();
    socket.onStatus(setWsStatus);
    const unsub = socket.subscribe((msg) => {
      switch (msg.type) {
        case "simulation_started": {
          const sid = (msg.payload?.session_id as string) ?? "";
          if (sid) setSessionId(sid);
          setStatus("running");
          break;
        }
        case "frame": {
          const f = msg.payload as unknown as SimFrame;
          frameRef.current = f;
          if (f) {
            setFrame(f);
            if (f.step === 1) startRef.current = [f.agent.x, f.agent.y];
            setAiLatency(f.inference_ms);
            if (f.hazards[0]) {
              const [wx, wz] = normToWorld(f.hazards[0].x, f.hazards[0].y, f.grid_size);
              focusTarget.current = new THREE.Vector3(wx, 1, wz);
            }
          }
          break;
        }
        case "event": {
          const ev = msg.payload as unknown as SimEvent;
          setEvents((prev) => [ev, ...prev].slice(0, 40));
          if (ev.severity === "critical") pushToast("alert", ev.text);
          else if (ev.severity === "warning") pushToast("alert", ev.text);
          break;
        }
        case "mission_complete":
        case "mission_failed":
        case "mission_stopped": {
          setStatus("done");
          const sid = sessionIdRef.current;
          if (sid) {
            explainSession(sid).then(setReport).catch(() => setReport(null));
          }
          break;
        }
        case "state":
          if (msg.payload?.status === "paused") setStatus("paused");
          if (msg.payload?.status === "running") setStatus("running");
          break;
        case "error":
          setError(String(msg.payload?.message ?? "server error"));
          break;
        default:
          break;
      }
    });
    const ping = window.setInterval(() => socket.ping(), 5000);
    return () => {
      unsub();
      window.clearInterval(ping);
      socket.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync the blocked-road set used by the instanced road mesh.
  useEffect(() => {
    const cells = frameSnapshot?.blocked_cells ?? [];
    const next = new Set(cells.map(([x, y]) => `${x},${y}`));
    let changed = next.size !== blockedSetRef.current.size;
    if (!changed) {
      for (const k of next) { if (!blockedSetRef.current.has(k)) { changed = true; break; } }
    }
    blockedSetRef.current = next;
    if (changed) setBlockedVersion((v) => v + 1);
  }, [frameSnapshot]);

  // Reward / penalty / rescue toasts driven by REAL frame values (deduped per step).
  useEffect(() => {
    const f = frameSnapshot;
    if (!f || f.step === lastToastStep.current) return;
    lastToastStep.current = f.step;
    if (f.penalty > 0.01) {
      pushToast("penalty", `−${f.penalty.toFixed(1)} PENALTY`);
    } else if (f.reward > 0.01) {
      pushToast("reward", `+${f.reward.toFixed(1)} REWARD`);
    }
    if (f.victims_rescued > lastRescued.current) {
      const delta = f.victims_rescued - lastRescued.current;
      lastRescued.current = f.victims_rescued;
      pushToast("rescue", `RESCUE +${delta} VICTIMS`);
    } else {
      lastRescued.current = f.victims_rescued;
    }
  }, [frameSnapshot, pushToast]);

  // Active AI route trail — cells actually visited by the agent (real positions).
  useEffect(() => {
    const f = frameSnapshot;
    if (!f) return;
    const key = `${f.agent.x},${f.agent.y}`;
    if (routeSetRef.current.has(key)) return;
    routeSetRef.current.add(key);
    setRouteVersion((v) => v + 1);
  }, [frameSnapshot]);

  // Load learning trend + model + training status on mount / research toggle
  // / whenever the selected policy switches between the two trainable algos.
  const statusAlgo = policy === "qrdqn" ? "qrdqn" : "ppo";
  useEffect(() => {
    getLearningTrend().then(setTrend).catch(() => {});
    getModelStatus(statusAlgo).then(setModelStatus).catch(() => {});
    getTrainingStatus().then(setTrainStatus).catch(() => {});
    const id = window.setInterval(() => {
      getLearningTrend().then(setTrend).catch(() => {});
      if (research) getTrainingStatus().then(setTrainStatus).catch(() => {});
    }, 8000);
    return () => window.clearInterval(id);
  }, [research, statusAlgo]);

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "d") setShowPerf((v) => !v);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Adaptive quality: drop settings while FPS is low, restore gradually when healthy.
  useEffect(() => {
    if (perf.fps <= 0) return;
    const id = window.setTimeout(() => {
      if (perf.fps < 30 && quality < 2) setQuality((q) => Math.min(2, q + 1));
      else if (perf.fps > 50 && quality > 0) setQuality((q) => Math.max(0, q - 1));
    }, 1500);
    return () => window.clearTimeout(id);
  }, [perf.fps, quality]);

  const pause = useCallback(() => socketRef.current?.pause(), []);
  const resume = useCallback(() => socketRef.current?.resume(), []);
  const stepOnce = useCallback(() => socketRef.current?.stepOnce(), []);

  // STOP MISSION: freezes the real session state server-side and marks it
  // end_reason='user_stopped' -- distinct from a natural completion/failure.
  // Falls back to the REST endpoint if the WebSocket isn't connected, so the
  // mission can always actually be stopped rather than silently no-op'ing.
  const stopMission = useCallback(async () => {
    setShowStopConfirm(false);
    const sid = sessionIdRef.current;
    if (socketRef.current && wsStatus === "connected") {
      socketRef.current.stop();
      return;
    }
    if (sid) {
      try {
        await stopSimulation(sid);
        setStatus("done");
        explainSession(sid).then(setReport).catch(() => setReport(null));
      } catch (err) {
        setError(err instanceof Error ? err.message : "stop failed");
      }
    }
  }, [wsStatus]);

  const playAgain = useCallback(() => {
    setReport(null);
    setReplay(null);
    setEvents([]);
    socketRef.current?.reset(seed);
    socketRef.current?.resume();
  }, [seed]);

  const startReplay = useCallback(async () => {
    if (!sessionId) return;
    pause();
    try {
      const { frames } = await getHistory(sessionId);
      setReplay({ frames, idx: 0, playing: false });
      setReplayPos(0);
      setStatus("paused");
    } catch (err) {
      setError(err instanceof Error ? err.message : "replay failed");
    }
  }, [sessionId, pause]);

  // Replay playback loop (uses recorded frames, never runs a new sim).
  useEffect(() => {
    if (!replay || !replay.playing) return;
    const id = window.setInterval(() => {
      setReplay((r) => {
        if (!r) return r;
        const idx = Math.min(r.idx + 1, r.frames.length - 1);
        frameRef.current = r.frames[idx];
        setFrame(r.frames[idx]);
        setReplayPos(idx);
        if (idx >= r.frames.length - 1) return { ...r, playing: false };
        return { ...r, idx };
      });
    }, Math.max(30, (2.0 / speed) * 1000));
    return () => window.clearInterval(id);
  }, [replay?.playing, speed, replay]);

  // Which frame the 3D scene reads: live frame (unless replaying).
  useEffect(() => {
    if (replay && replay.playing && replay.frames[replay.idx]) {
      frameRef.current = replay.frames[replay.idx];
    }
  }, [replay]);

  const stepREST = useCallback(async () => {
    if (!sessionId) return;
    const t0 = performance.now();
    try {
      const res = await stepSimulation(sessionId);
      setApiLatency(performance.now() - t0);
      frameRef.current = res.frame;
      setFrame(res.frame);
      setStatus(res.frame.status === "done" ? "done" : status);
      setEvents((prev) => [...res.events, ...prev].slice(0, 40));
    } catch (err) {
      setError(err instanceof Error ? err.message : "step failed");
    }
  }, [sessionId, status]);

  const train = useCallback(async () => {
    if (!trainStatus?.running) {
      await startTraining({ episodes: 150, difficulty }).catch((err) => setError(err.message));
    }
  }, [difficulty, trainStatus]);

  const evalRun = useCallback(async () => {
    const t0 = performance.now();
    try {
      const res = await runBeforeAfter({ episodes: 10, seed: 7, difficulty, disaster });
      setApiLatency(performance.now() - t0);
      setBeforeAfter(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "eval failed");
    }
  }, [difficulty, disaster]);

  const focusedIncident = useMemo(() => {
    if (frame && frame.incidents[0]) {
      const [wx, wz] = normToWorld(frame.incidents[0].x, frame.incidents[0].y, frame.grid_size);
      return new THREE.Vector3(wx, 1, wz);
    }
    return null;
  }, [frame]);

  const hazardTypes = useMemo(() => frame?.hazards.map((h) => h.type) ?? [], [frame?.hazards]);

  return (
    <div className="simulator" aria-label="DisasterMind AI 3D disaster response simulator">
      <div className="sim-canvas">
        {status !== "idle" && (
        <Canvas dpr={quality === 0 ? [1, 1.5] : quality === 1 ? [0.85, 1.15] : [0.6, 1]} camera={{ position: [30, 26, 30], fov: 45, near: 0.1, far: 300 }} gl={{ antialias: quality === 0, powerPreference: "high-performance" }}>
          <color attach="background" args={["#081420"]} />
          <fog attach="fog" args={["#0a1b2b", 70, 140]} />
          <ambientLight intensity={0.55} />
          <directionalLight position={[40, 60, 20]} intensity={1.2} color="#cfe8ff" castShadow />
          <directionalLight position={[-30, 20, -40]} intensity={0.25} color="#4db8ff" />
          <hemisphereLight args={["#9fc8ff", "#2f6a3f", 0.5]} />
          {quality === 0 && <Stars radius={120} depth={40} count={900} factor={3} fade speed={0.5} />}
          <Terrain grid={10} hazardTypes={hazardTypes} />
          <Roads grid={10} blockedSetRef={blockedSetRef} blockedVersion={blockedVersion} routeSetRef={routeSetRef} routeVersion={routeVersion} />
          <Buildings grid={10} />
          <Forest grid={10} density={quality === 0 ? 1 : quality === 1 ? 0.5 : 0.25} />
          <Facilities grid={10} />
          {quality < 2 && <CivilianTraffic grid={10} frameRef={frameRef} />}
          {quality === 0 && <Civilians grid={10} frameRef={frameRef} />}
          <Vehicles frameRef={frameRef} startRef={startRef} />
          <Incidents frameRef={frameRef} />
          <HazardZones frameRef={frameRef} />
          <FireAndSmoke frameRef={frameRef} hazardRefs={hazardRefs} maxFlames={quality === 0 ? 60 : quality === 1 ? 28 : 0} />
          <Dust frameRef={frameRef} />
          <Rain frameRef={frameRef} maxParticles={quality === 0 ? 500 : quality === 1 ? 200 : 0} />
          <GoalMarker frameRef={frameRef} />
          <CameraController frameRef={frameRef} mode={camMode} focusTarget={focusTarget} />
          <PerformanceMeter onStats={setPerf} />
          <OrbitControls makeDefault enablePan maxPolarAngle={Math.PI / 2.15} minDistance={4} maxDistance={90} />
        </Canvas>
        )}

        {/* TOP HUD */}
        <div className="hud-top">
          <div className="hud-brand">
            <div className="brand-mark"><Siren size={18} /></div>
            <div><strong>DISASTERMIND AI</strong><span>AI DISASTER RESPONSE</span></div>
          </div>
          <div className={`hud-ai ${status === "running" ? "active" : ""}`}>
            <span className="dot" /> AI {status === "running" ? "ACTIVE" : "STANDBY"}
            <span className="hud-chip-model">{frame?.model_label ?? "PPO"}</span>
          </div>
          {/* Top-center: live disaster chip (real frame data) */}
          <div className="hud-disaster">
            <span className="hud-disaster-icon">
              {frameSnapshot && ["flood", "tsunami", "heavy_rain"].includes(frameSnapshot.disaster) ? "🌊" : "🔥"}
            </span>
            <b>{frameSnapshot?.disaster?.toUpperCase() ?? "NO ACTIVE DISASTER"}</b>
            <span className="hud-sev">
              {frameSnapshot && frameSnapshot.severity >= 0.7 ? "HIGH" : frameSnapshot && frameSnapshot.severity >= 0.4 ? "MODERATE" : frameSnapshot ? "LOW" : "—"}
            </span>
            <span className="hud-traf">TRAFFIC {frameSnapshot ? (frameSnapshot.traffic_level >= 0.6 ? "↑" : frameSnapshot.traffic_level >= 0.3 ? "→" : "↓") : "—"}</span>
            {frameSnapshot && (
              <span className={`hud-risk risk-${frameSnapshot.route_risk.toLowerCase()}`} title={`Route risk score ${frameSnapshot.route_risk_score.toFixed(2)} (deterministic: traffic, hazard severity, blocked roads, hard-zone proximity)`}>
                ROUTE RISK {frameSnapshot.route_risk}
              </span>
            )}
            {frameSnapshot && frameSnapshot.anomaly_status !== "NORMAL" && (
              <span className={`hud-anomaly anomaly-${frameSnapshot.anomaly_status.toLowerCase()}`} title={frameSnapshot.anomaly_reasons.join("; ")}>
                <AlertTriangle size={11} /> {frameSnapshot.anomaly_status}
              </span>
            )}
          </div>
          <div className="hud-speed">
            {[0.5, 1, 2, 4, 8, 16].map((s) => (
              <button key={s} className={`speed-btn ${speed === s ? "active" : ""}`} onClick={() => { setSpeed(s); socketRef.current?.setSpeed(s); }}>
                {s}x
              </button>
            ))}
          </div>
        </div>

        {/* CONNECTION / ERROR BANNERS (actual connection state only) */}
        {status !== "idle" && (wsStatus === "error" || wsStatus === "closed") && (
          <div className="conn-banner bad">⚠ AI CONNECTION LOST <span>↻ RECONNECTING...</span></div>
        )}
        {status !== "idle" && wsStatus === "connected" && lastWentDown.current && (
          <div className="conn-banner ok">● AI CONNECTION RESTORED</div>
        )}
        {status === "idle" && (wsStatus === "error" || wsStatus === "closed") && (
          <div className="conn-banner bad">⚠ SERVER UNAVAILABLE — simulation backend not reachable</div>
        )}
        {modelStatus && !(modelStatus.available && modelStatus.compatible) && (
          <div className="conn-banner warn">⚠ AI FALLBACK — no compatible trained checkpoint, heuristic policy active</div>
        )}

        {/* METRICS BAR */}
        <div className="hud-metrics">
          <div className="metric"><small>Reward</small><strong className={frameSnapshot && frameSnapshot.cumulative_reward >= 0 ? "pos" : "neg"}>{frameSnapshot?.cumulative_reward?.toFixed(1) ?? "0.0"}</strong></div>
          <div className="metric"><small>Penalty</small><strong className="neg">{frameSnapshot?.cumulative_penalty?.toFixed(1) ?? "0.0"}</strong></div>
          <div className="metric"><small>Score</small><strong className="ai">{frameSnapshot?.score?.toFixed(1) ?? "0.0"}</strong></div>
          <div className="metric"><small>Rescued</small><strong>{frameSnapshot?.victims_rescued ?? 0} / {frameSnapshot?.victims_total ?? 0}</strong></div>
          <div className="metric"><small>Unmet</small><strong className={frameSnapshot && frameSnapshot.unmet > 0 ? "neg" : "pos"}>{frameSnapshot?.unmet ?? 0}</strong></div>
          <div className="metric"><small>Mission time</small><strong className="time">{fmtMissionTime(frameSnapshot?.step ?? 0)}</strong></div>
        </div>

        {/* AI CONTROL PANEL */}
        <div className="panel ai-panel">
          <div className="panel-head"><Cpu size={14} /> AI CONTROL <span className={`ai-state ${status === "running" ? "deciding" : status === "done" ? "done" : ""}`}>
            {status === "running" && !frameSnapshot ? "● DECIDING" : status === "running" ? "● ACTIVE" : status === "done" ? "● COMPLETED" : "○ STANDBY"}
          </span></div>
          <div className="ai-action">
            <span className="ai-action-name">{frameSnapshot?.action?.name ?? "—"}</span>
            {frameSnapshot && !frameSnapshot.action_valid && <span className="chip warn">MASKED</span>}
          </div>
          <div className="ai-detail"><small>REASON</small><p>
            {status === "running" && !frameSnapshot
              ? "AI DECISION PENDING..."
              : (frameSnapshot?.explanation ?? "Start a mission to see the AI decision context.")}
          </p></div>
          <div className="ai-detail"><small>RESULT</small><p>
            {frameSnapshot ? (frameSnapshot.reward >= 0 ? `+${frameSnapshot.reward.toFixed(2)} reward` : `${frameSnapshot.reward.toFixed(2)} (penalty ${frameSnapshot.penalty.toFixed(2)})`) : "—"}
          </p></div>
        </div>

        {/* INCIDENT / VEHICLE / DISASTER PANELS */}
        <div className="panel side-panel top-left">
          <div className="panel-head"><Activity size={14} /> INCIDENTS</div>
          {frameSnapshot?.incidents.length ? frameSnapshot.incidents.slice(0, 4).map((inc, i) => (
            <div className="row" key={i}>
              <span className="row-tag danger" />
              <div><b>Cluster {i + 1}</b><small>victims: {inc.victims}</small></div>
            </div>
          )) : <div className="muted">No incidents yet.</div>}
        </div>
        <div className="panel side-panel bottom-left">
          <div className="panel-head"><Siren size={14} /> VEHICLES</div>
          <div className="row"><span className="row-tag ai" /><div><b>{frameSnapshot && ["flood", "tsunami", "heavy_rain"].includes(frameSnapshot.disaster) ? "Rescue boat" : "Ambulance"}</b><small>{frameSnapshot?.model_label ?? "—"}</small></div></div>
          <div className="row"><span className="row-tag warn" /><div><b>Dispatched units</b><small>{Math.max(0, (frameSnapshot?.vehicles ?? 1) - 1)}</small></div></div>
          <div className="row"><span className="row-tag" /><div><b>Resources</b><small>{frameSnapshot?.resources ?? 0} · priority {frameSnapshot?.priority ?? 0}</small></div></div>
        </div>
        <div className="panel side-panel right-top">
          <div className="panel-head"><AlertTriangle size={14} /> DISASTER</div>
          <div className="row"><span className="row-tag danger" /><div><b>{frameSnapshot?.weather?.toUpperCase() ?? "CLEAR"}</b><small>severity {frameSnapshot?.severity ?? 0} · traffic {frameSnapshot?.traffic_level ?? 0}</small></div></div>
          {frameSnapshot?.hazards.slice(0, 3).map((h) => (
            <div className="row" key={h.id}><span className={`row-tag ${h.hard ? "danger" : "warn"}`} /><div><b>{h.type.toUpperCase()}</b><small>{Math.round(h.severity * 100)}%{h.hard ? " · hard" : ""}</small></div></div>
          ))}
        </div>

        {/* MINIMAP */}
        <div className="minimap-wrap">
          <div className="minimap-head"><MapIcon size={13} /> MINIMAP</div>
          <MiniMap
            frameRef={frameRef}
            mode={camMode}
            routeSetRef={routeSetRef}
            onFocus={(x, y) => {
              const g = frameSnapshot?.grid_size ?? 10;
              const [wx, wz] = cellToWorld(x, y, g);
              focusTarget.current = new THREE.Vector3(wx, 1, wz);
              setCamMode("incident");
            }}
          />
        </div>

        {/* REWARD / PENALTY / EVENT TOASTS (real backend values) */}
        <div className="toast-stack">
          {toasts.map((t) => (
            <div key={t.id} className={`toast toast-${t.kind}`}>
              {t.text}
            </div>
          ))}
        </div>

        {/* EVENT TIMELINE */}
        <div className="panel timeline-panel">
          <div className="panel-head"><Activity size={14} /> EVENT TIMELINE</div>
          <EventTimeline events={events} />
        </div>

        {/* CAMERA CONTROLS */}
        <div className="cam-controls">
          <button className={camMode === "overview" ? "active" : ""} onClick={() => setCamMode("overview")} title="Overview"><Camera size={14} /> Overview</button>
          <button className={camMode === "follow" ? "active" : ""} onClick={() => setCamMode("follow")} title="Follow vehicle"><Crosshair size={14} /> Follow</button>
          <button className={camMode === "incident" ? "active" : ""} onClick={() => { if (focusedIncident) focusTarget.current = focusedIncident; setCamMode("incident"); }} title="Focus incident"><AlertTriangle size={14} /> Incident</button>
          <button className={camMode === "disaster" ? "active" : ""} onClick={() => setCamMode("disaster")} title="Focus disaster"><Zap size={14} /> Disaster</button>
        </div>

        {/* GAME CONTROLS */}
        <div className="game-controls">
          {(status === "idle" || status === "done") && (
            <button className="primary big" onClick={() => startSim()}><Play size={17} /> {status === "done" ? "NEW MISSION" : "START MISSION"}</button>
          )}
          {status === "running" && <button className="secondary" onClick={pause}><Pause size={16} /> Pause</button>}
          {status === "paused" && (
            <>
              <button className="primary" onClick={resume}><Play size={16} /> Resume</button>
              <button className="secondary" onClick={stepOnce}><StepForward size={16} /> Step</button>
              <button className="secondary" onClick={startReplay}><Repeat size={16} /> Replay</button>
            </>
          )}
          {(status === "running" || status === "paused") && (
            <button className="stop-mission" onClick={() => setShowStopConfirm(true)}>
              <Square size={14} /> Stop Mission
            </button>
          )}
          {status === "done" && report && (
            <button className="primary" onClick={playAgain}><RotateCcw size={16} /> Play Again</button>
          )}
          <button className="ghost-btn" onClick={() => setResearch((v) => !v)}><Radio size={15} /> {research ? "Exit Research" : "Research"}</button>
          <button className="ghost-btn" onClick={() => setShowPerf((v) => !v)}><Gauge size={15} /> Debug</button>
        </div>

        {/* REPLAY BAR */}
        {replay && (
          <div className="replay-bar">
            <span>REPLAY {replayPos + 1}/{replay.frames.length}</span>
            <button className="secondary" onClick={() => setReplay((r) => r && { ...r, playing: !r.playing })}>
              {replay.playing ? <Pause size={14} /> : <Play size={14} />}
            </button>
            <input
              type="range" min={0} max={Math.max(replay.frames.length - 1, 1)}
              value={replayPos}
              onChange={(e) => {
                const idx = Number(e.target.value);
                setReplay((r) => r && { ...r, idx, playing: false });
                setReplayPos(idx);
                frameRef.current = replay.frames[idx];
                setFrame(replay.frames[idx]);
              }}
            />
          </div>
        )}

        {/* PERFORMANCE HUD */}
        {showPerf && (
          <div className="perf-hud">
            <h4>PERFORMANCE</h4>
            <div><span>FPS</span><b>{perf.fps.toFixed(0)}</b></div>
            <div><span>Frame time</span><b>{perf.frameMs.toFixed(1)} ms</b></div>
            <div><span>DPR</span><b>{quality === 0 ? "1.5" : quality === 1 ? "1.15" : "1.0"}</b></div>
            <div><span>Quality</span><b>{quality === 0 ? "HIGH" : quality === 1 ? "MEDIUM" : "LOW"} (adaptive)</b></div>
            <div><span>Particles</span><b>{quality === 0 ? 700 : quality === 1 ? 228 : 0}</b></div>
            <div><span>Draw calls</span><b>{perf.draws}</b></div>
            <div><span>Triangles</span><b>{perf.tris}</b></div>
            <div><span>AI latency</span><b>{aiLatency != null ? `${aiLatency.toFixed(1)} ms` : "—"}</b></div>
            <div><span>API latency</span><b>{apiLatency != null ? `${apiLatency.toFixed(1)} ms` : "—"}</b></div>
            <div><span>WebSocket</span><b className={wsStatus === "connected" ? "ok" : "bad"}>{wsStatus}</b></div>
            <div><span>Step latency</span><b>{frame?.env_step_ms != null ? `${frame.env_step_ms.toFixed(1)} ms` : "—"}</b></div>
          </div>
        )}

        {/* ERROR */}
        {error && <div className="error-toast">{error} <button onClick={() => setError(null)}>✕</button></div>}

        {/* START GAME BOOT OVERLAY (only before the first mission) */}
        {status === "idle" && !report && !replay && (
          <StartOverlay
            difficulty={difficulty}
            disaster={disaster}
            seed={seed}
            policy={policy}
            modelReady={!!(modelStatus?.available && modelStatus.compatible)}
            wsReady={wsStatus === "connected"}
            onStart={() => startSim()}
          />
        )}

        {/* POST-GAME REPORT */}
        {report && <PostGameReport report={report} trend={trend} sessionId={sessionId} onReplay={startReplay} onAgain={playAgain} onClose={() => setReport(null)} />}
        {showStopConfirm && (
          <div className="confirm-overlay">
            <div className="confirm-card">
              <h4>STOP CURRENT MISSION?</h4>
              <p>The current simulation will end. A final report will be generated from the real mission state.</p>
              <div className="confirm-actions">
                <button className="secondary" onClick={() => setShowStopConfirm(false)}>Cancel</button>
                <button className="danger" onClick={stopMission}><Square size={14} /> Stop Mission</button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* SIDE INFO PANEL */}
      <div className="sim-sidebar">
        <div className="sidebar-tabs">
          <button className="active">MISSION</button>
          <button onClick={() => { setResearch(true); }}>LEARNING</button>
          <button onClick={() => { setResearch(true); }}>RESEARCH</button>
        </div>

        <div className="sidebar-section">
          <h3>Mission setup</h3>
          <div className="field">
            <label>Difficulty</label>
            <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
              <option>EASY</option><option>MEDIUM</option><option>HARD</option><option>EXTREME</option>
            </select>
          </div>
          <div className="field">
            <label>Disaster</label>
            <select value={disaster} onChange={(e) => setDisaster(e.target.value)}>
              <option value="any">Any</option>
              <option value="flood">Flood</option>
              <option value="wildfire">Wildfire</option>
              <option value="earthquake">Earthquake</option>
              <option value="cyclone">Cyclone</option>
              <option value="tsunami">Tsunami</option>
              <option value="landslide">Landslide</option>
              <option value="heavy_rain">Heavy Rain</option>
              <option value="road_blockage">Road Blockage</option>
              <option value="traffic_jam">Traffic Jam</option>
              <option value="combined">Combined</option>
            </select>
          </div>
          <div className="field">
            <label>Policy</label>
            <select value={policy} onChange={(e) => setPolicy(e.target.value)} disabled={status === "running"}>
              <option value="ppo">PPO</option>
              <option value="qrdqn">QR-DQN</option>
              <option value="heuristic">Rule heuristic</option>
              <option value="shortest">Shortest safe path</option>
              <option value="random">Random (untrained)</option>
            </select>
            {(policy === "ppo" || policy === "qrdqn") && !modelStatus?.available && (
              <small className="muted">
                <Cpu size={14} /> Trained {policy.toUpperCase()} checkpoint missing/incompatible — heuristic fallback will run.
              </small>
            )}
          </div>
          <div className="field">
            <label>Seed</label>
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </div>
        </div>

        <div className="sidebar-section">
          <h3>Episode</h3>
          <div className="epi-stats">
            <div><small>Step</small><b>{frameSnapshot?.step ?? 0}</b></div>
            <div><small>Distance</small><b>{frameSnapshot?.distance_to_goal?.toFixed(2) ?? "—"}</b></div>
            <div><small>Route len</small><b>{frameSnapshot?.route_distance ?? 0}</b></div>
            <div><small>Vehicles</small><b>{frameSnapshot?.vehicles ?? 1}</b></div>
            <div><small>Blocked</small><b>{frameSnapshot?.blocked_cells?.length ?? 0}</b></div>
            <div><small>Hard viol.</small><b>{frameSnapshot?.hard_violations ?? 0}</b></div>
          </div>
        </div>

        <LearningCharts trend={trend} />

        {research && (
          <div className="sidebar-section">
            <h3>Research mode</h3>
            {modelStatus && (
              <div className="model-box">
                <div><small>Algorithm</small><b>{modelStatus.algo ?? "—"}</b></div>
                <div><small>Model</small><b>{modelStatus.model_name ?? "—"} v{modelStatus.model_version ?? "?"}</b></div>
                <div><small>Obs / Actions</small><b>{modelStatus.obs_dim ?? "?"} / {modelStatus.n_actions ?? "?"}</b></div>
                <div><small>Hidden</small><b>{modelStatus.hidden_dim ?? "?"}</b></div>
                <div><small>Trained</small><b>ep {modelStatus.episode ?? "—"}</b></div>
                <div><small>Mean reward</small><b>{modelStatus.mean_reward ?? "—"}</b></div>
                <div><small>Checkpoint</small><b className="muted">{modelStatus.path ?? "none"}</b></div>
                {!modelStatus.compatible && <div className="muted warn-text">{modelStatus.incompatible_reason}</div>}
                {modelStatus.available && !modelStatus.compatible && <div className="muted">Fallback: {modelStatus.fallback_policy}</div>}
              </div>
            )}
            <div className="action-row">
              <button className="secondary" disabled={trainStatus?.running} onClick={train}>
                {trainStatus?.running ? "Training…" : "Train PPO (150 ep)"}
              </button>
              <button className="secondary" onClick={evalRun}><Gauge size={14} /> Before/After</button>
            </div>
            {trainStatus?.running && (
              <div className="train-progress">
                <div>Episode {trainStatus.episode ?? 0}/{trainStatus.total_episodes ?? "?"}</div>
                <div className="progress"><div style={{ width: `${((trainStatus.episode ?? 0) / Math.max(trainStatus.total_episodes ?? 1, 1)) * 100}%` }} /></div>
              </div>
            )}
            {trainStatus?.message && <p className="muted">{trainStatus.message}</p>}
            {beforeAfter && (
              <div className="before-after">
                <h4>Before vs After (identical seeds)</h4>
                <table>
                  <thead><tr><th>Metric</th><th>Untrained</th><th>PPO</th></tr></thead>
                  <tbody>
                    {(Object.entries((beforeAfter.policies as Record<string, Record<string, number>>) ?? {})[0]?.[1] ?? {}).length > 0 &&
                      ["mean_reward", "success_rate", "mean_rescues", "mean_unmet", "mean_response_time_s", "mean_failed_actions"].map((k) => (
                        <tr key={k}>
                          <td>{k}</td>
                          {(Object.values(beforeAfter.policies as Record<string, Record<string, number>>)).map((m, i) => (
                            <td key={i}>{k === "success_rate" ? `${(m[k] * 100).toFixed(0)}%` : (m[k]?.toFixed?.(1) ?? "—")}</td>
                          ))}
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            )}
            <PpoVsDqn trend={trend} />
            <div className="model-box">
              <h4>Evaluation</h4>
              <button className="secondary" onClick={() => { runEvaluation({ episodes: 15, seeds: 1, difficulty, disaster }).catch((e) => setError(e.message)); }}>Run unseen-seed eval</button>
              <p className="muted">Evaluation uses a seed range never used during training.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}