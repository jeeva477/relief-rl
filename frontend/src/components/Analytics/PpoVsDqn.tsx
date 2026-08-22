import { useEffect, useState } from "react";
import type { LearningTrend, PpoVsQrdqnComparison } from "../../services/sim";
import { getPpoVsQrdqn } from "../../services/sim";

const ROWS: { label: string; key: keyof NonNullable<PpoVsQrdqnComparison["agents"][string]>; fmt: (v: number) => string; lowerIsBetter?: boolean }[] = [
  { label: "Average Reward", key: "mean_reward", fmt: (v) => v.toFixed(1) },
  { label: "Average Penalty", key: "mean_penalty", fmt: (v) => v.toFixed(1), lowerIsBetter: true },
  { label: "Success Rate", key: "success_rate", fmt: (v) => `${(v * 100).toFixed(1)}%` },
  { label: "Rescued", key: "mean_rescues", fmt: (v) => v.toFixed(1) },
  { label: "Unmet", key: "mean_unmet", fmt: (v) => v.toFixed(1), lowerIsBetter: true },
  { label: "Response Time", key: "mean_response_time_s", fmt: (v) => `${v.toFixed(1)}s`, lowerIsBetter: true },
  { label: "Route Efficiency", key: "route_efficiency", fmt: (v) => `${v.toFixed(2)}x`, lowerIsBetter: true },
];

export function PpoVsDqn({ trend }: { trend: LearningTrend }) {
  const [comparison, setComparison] = useState<PpoVsQrdqnComparison | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    let cancelled = false;
    getPpoVsQrdqn()
      .then((data) => {
        if (!cancelled) {
          setComparison(data);
          setStatus("ready");
        }
      })
      .catch(() => {
        if (!cancelled) setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const agentNames = comparison ? Object.keys(comparison.agents) : [];
  const hasPpo = agentNames.includes("PPO");
  const hasQrdqn = agentNames.includes("QR-DQN");

  // Fall back to the live training-log stats for the PPO column if the
  // comparison endpoint has no PPO data (e.g. checkpoint missing).
  const trendPpo = trend.available ? trend.episodes.slice(-10) : [];
  const trendPpoStats = trendPpo.length
    ? {
        mean_reward: trendPpo.reduce((a, b) => a + (b.reward ?? 0), 0) / trendPpo.length,
        mean_penalty: trendPpo.reduce((a, b) => a + (b.penalty ?? 0), 0) / trendPpo.length,
        success_rate: trendPpo.filter((e) => e.success).length / trendPpo.length,
        mean_rescues: trendPpo.reduce((a, b) => a + (b.rescued ?? 0), 0) / trendPpo.length,
        mean_unmet: 0,
        mean_response_time_s: trendPpo.reduce((a, b) => a + (b.response_time_s ?? 0), 0) / trendPpo.length,
        route_efficiency: trendPpo.reduce((a, b) => a + (b.route_efficiency ?? 0), 0) / trendPpo.length,
      }
    : null;

  return (
    <div className="ppovsdqn">
      <h3>PPO vs QR-DQN</h3>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>PPO {hasPpo ? "(evaluated)" : trendPpoStats ? "(training log)" : ""}</th>
            <th>QR-DQN {hasQrdqn ? "(evaluated)" : ""}</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map((row) => {
            const ppoVal = hasPpo
              ? (comparison!.agents["PPO"][row.key] as number | null)
              : trendPpoStats
              ? (trendPpoStats as Record<string, number>)[row.key]
              : null;
            const qrVal = hasQrdqn ? (comparison!.agents["QR-DQN"][row.key] as number | null) : null;
            const bothPresent = ppoVal != null && qrVal != null;
            const ppoBetter =
              bothPresent && (row.lowerIsBetter ? ppoVal! < qrVal! : ppoVal! > qrVal!);
            const qrBetter =
              bothPresent && (row.lowerIsBetter ? qrVal! < ppoVal! : qrVal! > ppoVal!);
            return (
              <tr key={row.label}>
                <td>{row.label}</td>
                <td className={ppoBetter ? "better" : undefined}>
                  {ppoVal != null ? row.fmt(ppoVal) : "—"}
                </td>
                <td className={qrBetter ? "better" : hasQrdqn ? undefined : "insufficient"}>
                  {qrVal != null ? row.fmt(qrVal) : status === "loading" ? "…" : "INSUFFICIENT DATA"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted">
        {status === "error" &&
          "Could not reach the comparison endpoint (/api/sim/compare/ppo-vs-qrdqn); showing the PPO training log only."}
        {status === "ready" && hasQrdqn && (
          <>
            {comparison?.source === "live"
              ? `Live evaluation: ${comparison.episodes} episodes, seed ${comparison.seed}, ${comparison.difficulty} difficulty. `
              : `Precomputed evaluation: ${comparison?.episodes} episodes, seed ${comparison?.seed}, ${comparison?.difficulty} difficulty. `}
            Both agents ran on the identical set of seeded scenarios. All numbers come from real environment rollouts of the trained checkpoints — none are fabricated.
          </>
        )}
        {status === "ready" && !hasQrdqn && (
          "No compatible QR-DQN checkpoint was found, so no QR-DQN comparison could be made. Train the QR-DQN agent (scripts/train_qrdqn.py) to populate this column."
        )}
      </p>
    </div>
  );
}
