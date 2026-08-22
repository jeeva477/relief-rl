import { useEffect, useState } from "react";
import { Play, Siren } from "lucide-react";

const BOOT_STEPS = [
  "Terrain",
  "Buildings",
  "Roads",
  "Vehicles",
  "Disaster System",
  "AI Connection",
] as const;

type StepState = "pending" | "loading" | "ready" | "failed";

export function StartOverlay({ difficulty, disaster, seed, policy, modelReady, wsReady, onStart }: {
  difficulty: string;
  disaster: string;
  seed: number;
  policy: string;
  modelReady: boolean;
  wsReady: boolean;
  onStart: () => void;
}) {
  const [stage, setStage] = useState<"ready" | "booting" | "ready2">("ready");
  const [steps, setSteps] = useState<StepState[]>(() => BOOT_STEPS.map(() => "pending"));
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    if (stage !== "booting") return;
    if (current >= BOOT_STEPS.length) {
      const id = window.setTimeout(() => setStage("ready2"), 300);
      return () => window.clearTimeout(id);
    }
    const step = BOOT_STEPS[current];
    if (step === "AI Connection") {
      setSteps((s) => s.map((v, i) => (i === current ? "loading" : v)));
      if (wsReady) {
        const id = window.setTimeout(() => {
          setSteps((s) => s.map((v, i) => (i === current ? "ready" : v)));
          setCurrent((c) => c + 1);
        }, 350);
        return () => window.clearTimeout(id);
      }
      const fail = window.setTimeout(() => {
        setSteps((s) => s.map((v, i) => (i === current ? "failed" : v)));
        setCurrent((c) => c + 1);
      }, 2600);
      return () => window.clearTimeout(fail);
    }
    setSteps((s) => s.map((v, i) => (i === current ? "loading" : v)));
    const id = window.setTimeout(() => {
      setSteps((s) => s.map((v, i) => (i === current ? "ready" : v)));
      setCurrent((c) => c + 1);
    }, 260);
    return () => window.clearTimeout(id);
  }, [stage, current, wsReady]);

  return (
    <div className="start-overlay">
      <div className="start-card">
        <div className="start-brand">
          <div className="brand-mark"><Siren size={22} /></div>
          <div>
            <strong>RELIEFRL</strong>
            <span>AI DISASTER SIMULATOR</span>
          </div>
        </div>

        <div className="start-chips">
          <span className={`chip ${modelReady ? "ok" : "warn"}`}>
            {modelReady ? "● PPO ACTIVE" : "⚠ AI FALLBACK · HEURISTIC"}
          </span>
          <span className={`chip ${wsReady ? "ok" : "warn"}`}>
            {wsReady ? "● SERVER ONLINE" : "⚠ SERVER OFFLINE"}
          </span>
        </div>

        <div className="start-meta">
          <div><small>Difficulty</small><b>{difficulty}</b></div>
          <div><small>Disaster</small><b>{disaster === "any" ? "Any" : disaster.toUpperCase()}</b></div>
          <div><small>Seed</small><b>{seed}</b></div>
          <div><small>Policy</small><b>{policy.toUpperCase()}</b></div>
        </div>

        {stage === "ready" ? (
          <button className="primary big start-btn" onClick={() => setStage("booting")}>
            <Play size={18} /> START MISSION
          </button>
        ) : (
          <div className="boot-list">
            {stage === "booting" && current < BOOT_STEPS.length && (
              <div className="boot-init">INITIALIZING ENVIRONMENT</div>
            )}
            {BOOT_STEPS.map((step, i) => {
              const st = steps[i];
              return (
                <div key={step} className={`boot-step ${st === "ready" ? "done" : ""} ${st === "failed" ? "bad" : ""} ${st === "loading" ? "loading" : ""}`}>
                  <span className="boot-check">
                    {st === "ready" ? "✓" : st === "failed" ? "⚠" : st === "loading" ? "◌" : "·"}
                  </span>
                  {step}
                  {st === "loading" && <em>LOADING</em>}
                  {st === "failed" && <em>FAILED</em>}
                </div>
              );
            })}
            {stage === "ready2" && (
              <>
                <div className="boot-ready">SYSTEM READY</div>
                <button className="primary big start-btn" onClick={onStart}>
                  <Play size={18} /> START MISSION
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}