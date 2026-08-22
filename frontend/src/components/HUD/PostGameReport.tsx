import { Fragment, useEffect, useState } from "react";
import { Repeat, RotateCcw, Download, Printer } from "lucide-react";
import { getHistory, type ExplanationSection, type LearningTrend, type SimFrame, type SimEvent } from "../../services/sim";
import { LearningCharts } from "../Analytics/LearningCharts";
import { PpoVsDqn } from "../Analytics/PpoVsDqn";
import { EventTimeline } from "./EventTimeline";

function downloadBlob(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** Every value here comes straight from the /api/sim/explain + /api/sim/history
 * responses -- this component renders what the backend actually measured and
 * never invents a number. Anything the backend didn't compute is shown as
 * "N/A" or omitted rather than guessed. */
export function PostGameReport({ report, trend, sessionId, onReplay, onAgain, onClose }: {
  report: ExplanationSection;
  trend: LearningTrend;
  sessionId: string | null;
  onReplay: () => void;
  onAgain: () => void;
  onClose: () => void;
}) {
  const [frames, setFrames] = useState<SimFrame[]>([]);
  const [fullEvents, setFullEvents] = useState<SimEvent[]>([]);

  // Full mission record (every frame + the permanent event log), fetched
  // once when the report opens -- backs Event Timeline / Final State /
  // JSON / CSV export. Independent of the live HUD's capped 40-event view.
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    getHistory(sessionId)
      .then((h) => {
        if (cancelled) return;
        setFrames(h.frames ?? []);
        setFullEvents(h.events ?? []);
      })
      .catch(() => {
        if (!cancelled) {
          setFrames([]);
          setFullEvents([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const success = report.success;
  const stopped = report.end_reason === "user_stopped";
  const timedOut = report.end_reason === "timeout";
  const title = stopped
    ? "MISSION STOPPED BY USER"
    : success
    ? "MISSION COMPLETE"
    : timedOut
    ? "MISSION TIMEOUT"
    : "MISSION FAILED";
  const titleClass = stopped ? "stopped" : success ? "ok" : "bad";
  const allSections = Object.entries(report.sections ?? {});
  const sections = allSections.filter(([t]) => t !== "ROUTE ANALYSIS" && t !== "ANOMALY ANALYSIS");
  const routeProse = report.sections?.["ROUTE ANALYSIS"];
  const anomalyProse = report.sections?.["ANOMALY ANALYSIS"];
  const rr = report.route_risk_summary;
  const an = report.anomaly_summary;
  const ai = report.ai_performance;
  const veh = report.vehicle_summary;
  const rewardParts = Object.entries(report.reward_components ?? {});
  const penaltyParts = Object.entries(report.penalty_components ?? {});
  const finalState = report.final_state;

  const missionId = sessionId ?? "—";

  function handleDownloadJson() {
    const payload = {
      mission_id: missionId,
      overview: {
        scenario: report.disaster, difficulty: report.difficulty, seed: report.seed,
        policy: report.policy, model: report.model_label, result: report.end_reason,
        success: report.success, steps: report.steps, response_time_s: report.response_time_s,
      },
      metrics: {
        reward: report.reward, penalty: report.penalty, rescued: report.rescued,
        unmet: report.unmet, victims: report.victims, route_efficiency: report.route_efficiency,
      },
      ai_performance: ai ?? null,
      reward_components: report.reward_components ?? {},
      penalty_components: report.penalty_components ?? {},
      vehicle_summary: veh ?? null,
      route_risk_summary: rr ?? null,
      anomaly_summary: an ?? null,
      explanation_sections: report.sections ?? {},
      final_state: finalState ?? null,
      events: fullEvents,
      frames,
    };
    downloadBlob(`mission-${missionId}-report.json`, JSON.stringify(payload, null, 2), "application/json");
  }

  function handleDownloadCsv() {
    if (frames.length === 0) {
      downloadBlob(`mission-${missionId}-frames.csv`, "step\n(no frames recorded)\n", "text/csv");
      return;
    }
    const cols: (keyof SimFrame)[] = [
      "step", "status", "action", "reward", "penalty", "cumulative_reward", "cumulative_penalty",
      "victims_total", "victims_rescued", "unmet", "route_risk", "route_risk_score",
      "anomaly_status", "weather", "severity", "traffic_level",
    ];
    const header = cols.join(",");
    const rows = frames.map((fr) =>
      cols
        .map((c) => {
          const v = c === "action" ? fr.action?.name : (fr as unknown as Record<string, unknown>)[c];
          const s = v == null ? "" : String(v);
          return s.includes(",") ? `"${s}"` : s;
        })
        .join(",")
    );
    downloadBlob(`mission-${missionId}-frames.csv`, [header, ...rows].join("\n"), "text/csv");
  }

  function handlePrint() {
    window.print();
  }

  return (
    <div className="report-overlay">
      <div className="report-card report-scroll">
        <div className="report-head">
          <div className={`report-title ${titleClass}`}>
            {title}
          </div>
          <button className="ghost-btn" onClick={onClose}>✕</button>
        </div>

        <div className="report-section">
          <h5>MISSION OVERVIEW</h5>
          <div className="report-kv">
            <span>Mission ID</span><b>{missionId}</b>
            <span>Scenario</span><b>{report.disaster ?? "—"}</b>
            <span>Difficulty</span><b>{report.difficulty ?? "—"}</b>
            <span>Seed</span><b>{report.seed ?? "—"}</b>
            <span>Policy</span><b>{report.policy ?? "—"}</b>
            <span>Model</span><b>{report.model_label ?? "—"}</b>
            <span>Result</span><b>{report.end_reason ?? "—"}</b>
          </div>
        </div>

        <div className="report-metrics">
          <div><small>Reward</small><strong>{report.reward?.toFixed(1) ?? "—"}</strong></div>
          <div><small>Penalty</small><strong>{report.penalty?.toFixed(1) ?? "—"}</strong></div>
          <div><small>Rescued</small><strong>{report.rescued} / {report.victims}</strong></div>
          <div><small>Unmet</small><strong>{report.unmet}</strong></div>
          <div><small>Response time</small><strong>{report.response_time_s != null ? `${report.response_time_s}s` : "—"}</strong></div>
          <div><small>Steps</small><strong>{report.steps}</strong></div>
        </div>

        <div className="report-actions">
          <button className="primary" onClick={onReplay}><Repeat size={15} /> Replay</button>
          <button className="secondary" onClick={onAgain}><RotateCcw size={15} /> Play again</button>
          <button className="secondary" onClick={handleDownloadJson}><Download size={15} /> JSON</button>
          <button className="secondary" onClick={handleDownloadCsv}><Download size={15} /> CSV</button>
          <button className="secondary" onClick={handlePrint}><Printer size={15} /> Print</button>
        </div>

        {ai && (
          <div className="report-section">
            <h5>AI PERFORMANCE</h5>
            <div className="report-kv">
              <span>Model</span><b>{ai.model}</b>
              <span>AI decisions</span><b>{ai.ai_decisions}</b>
              <span>Valid decisions</span><b>{ai.valid_decisions}</b>
              <span>Masked actions</span><b>{ai.masked_actions}</b>
              <span>Route changes</span><b>{ai.route_changes}</b>
              <span>Dispatches</span><b>{ai.dispatches}</b>
            </div>
          </div>
        )}

        {veh && (
          <div className="report-section">
            <h5>VEHICLE PERFORMANCE</h5>
            <div className="report-kv">
              <span>Active vehicles</span><b>{veh.active_vehicles}</b>
              <span>Dispatches</span><b>{veh.dispatches}</b>
            </div>
            <p className="muted">{veh.note}</p>
          </div>
        )}

        <div className="report-section">
          <h5>INCIDENT ANALYSIS</h5>
          <div className="report-kv">
            <span>Total victims</span><b>{report.victims}</b>
            <span>Rescued</span><b>{report.rescued}</b>
            <span>Unmet</span><b>{report.unmet}</b>
            <span>Response time</span><b>{report.response_time_s != null ? `${report.response_time_s}s` : "N/A"}</b>
          </div>
        </div>

        {(rr || an) && (
          <div className="report-riskrow">
            {rr && (
              <div className="risk-block">
                <h5>ROUTE ANALYSIS</h5>
                <div className="risk-chips">
                  <span className={`hud-risk risk-${rr.peak_level.toLowerCase()}`}>PEAK {rr.peak_level}</span>
                  <span className="risk-chip-neutral">avg score {rr.mean_score.toFixed(2)}</span>
                  <span className="risk-chip-neutral">{rr.route_changes} reroutes</span>
                </div>
                {routeProse && <p>{routeProse}</p>}
              </div>
            )}
            {an && (
              <div className="risk-block">
                <h5>ANOMALY ANALYSIS</h5>
                <div className="risk-chips">
                  <span className={`hud-anomaly anomaly-${an.status.toLowerCase()}`}>{an.status}</span>
                  <span className="risk-chip-neutral">{an.event_count} signal(s)</span>
                </div>
                {anomalyProse && <p>{anomalyProse}</p>}
              </div>
            )}
          </div>
        )}

        {(rewardParts.length > 0 || penaltyParts.length > 0) && (
          <div className="report-riskrow">
            {rewardParts.length > 0 && (
              <div className="risk-block">
                <h5>REWARD ANALYSIS</h5>
                <div className="report-kv">
                  {rewardParts.map(([k, v]) => (
                    <Fragment key={k}>
                      <span>{k}</span>
                      <b>+{v.toFixed(2)}</b>
                    </Fragment>
                  ))}
                </div>
              </div>
            )}
            {penaltyParts.length > 0 && (
              <div className="risk-block">
                <h5>PENALTY ANALYSIS</h5>
                <div className="report-kv">
                  {penaltyParts.map(([k, v]) => (
                    <Fragment key={k}>
                      <span>{k}</span>
                      <b>-{v.toFixed(2)}</b>
                    </Fragment>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <div className="report-analysis">
          <h4>LEARNING ANALYSIS</h4>
          <LearningCharts trend={trend} />
          <PpoVsDqn trend={trend} />
        </div>

        <div className="report-sections">
          <h4>AI EXPLANATION</h4>
          <p className="muted">
            SOURCE: {report.source ?? "RULE-BASED FALLBACK"}
            {report.source === "HUGGING FACE" && report.hf_model ? ` (${report.hf_model})` : ""}
            {report.source !== "HUGGING FACE" ? " — rule-based analysis of actual simulation metrics." : ""}
          </p>
          {report.source === "HUGGING FACE" && report.narrative && (
            <div className="report-section">
              <h5>AI NARRATIVE</h5>
              <p style={{ whiteSpace: "pre-wrap" }}>{report.narrative}</p>
            </div>
          )}
          {sections.map(([title, body]) => (
            <div key={title} className="report-section">
              <h5>{title}</h5>
              <p>{body}</p>
            </div>
          ))}
        </div>

        <div className="report-section">
          <h4>EVENT TIMELINE</h4>
          {fullEvents.length > 0 ? (
            <EventTimeline events={fullEvents} />
          ) : (
            <p className="muted">
              {sessionId ? "Loading full event log…" : "No session id available for this report."}
            </p>
          )}
        </div>

        {finalState && (
          <div className="report-section">
            <h4>FINAL STATE</h4>
            <div className="report-kv">
              <span>Step</span><b>{finalState.step}</b>
              <span>Status</span><b>{finalState.status}</b>
              <span>Agent position</span><b>({finalState.agent.x.toFixed(2)}, {finalState.agent.y.toFixed(2)})</b>
              <span>Goal position</span><b>({finalState.goal.x.toFixed(2)}, {finalState.goal.y.toFixed(2)})</b>
              <span>Weather</span><b>{finalState.weather}</b>
              <span>Severity</span><b>{finalState.severity.toFixed(2)}</b>
              <span>Traffic</span><b>{finalState.traffic_level.toFixed(2)}</b>
            </div>
          </div>
        )}

        <div className="report-foot muted">
          Every figure above is measured from the recorded episode (frames + event log) or the training
          log -- nothing on this report is fabricated. Fields the backend did not track are shown as N/A.
        </div>
      </div>
    </div>
  );
}
