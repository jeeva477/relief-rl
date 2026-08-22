import type { LearningTrend, TrendPoint } from "../../services/sim";

function MiniChart({ title, points, color, fmt, yLabel }: {
  title: string;
  points: (number | null)[];
  color: string;
  fmt?: (v: number) => string;
  yLabel?: string;
}) {
  const W = 260, H = 84;
  const vals = points.filter((p): p is number => p != null);
  if (vals.length < 2) {
    return (
      <div className="chart">
        <div className="chart-title">{title}</div>
        <div className="chart-empty">INSUFFICIENT DATA</div>
      </div>
    );
  }
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = max - min || 1;
  const pts = points.map((p, i) => {
    const x = (i / Math.max(points.length - 1, 1)) * W;
    const y = H - 6 - ((p == null ? min : (p - min) / range) * (H - 14));
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return (
    <div className="chart">
      <div className="chart-title">{title}</div>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="chart-svg">
        <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth="1.6" />
        <text x="4" y={H - 4} fill="#7d8ea3" fontSize="8">{yLabel ?? ""}</text>
        <text x={W - 4} y={H - 4} fill="#7d8ea3" fontSize="8" textAnchor="end">
          {fmt ? `${fmt(max)} / ${fmt(min)}` : `${max.toFixed(1)} / ${min.toFixed(1)}`}
        </text>
      </svg>
    </div>
  );
}

export function LearningCharts({ trend }: { trend: LearningTrend }) {
  if (!trend.available) {
    return (
      <div className="charts-block">
        <h3>Learning trends</h3>
        <div className="chart-empty big">INSUFFICIENT DATA</div>
        <p className="muted">No training history found. Start a training run or train the agent to populate these charts.</p>
      </div>
    );
  }
  const eps = trend.episodes;
  const R = (k: keyof TrendPoint) => eps.map((e) => (typeof e[k] === "number" ? (e[k] as number) : null));
  const window = (arr: (number | null)[], n: number) => arr.map((_, i) => {
    const seg = arr.slice(Math.max(0, i - n + 1), i + 1).filter((v): v is number => v != null);
    return seg.length ? seg.reduce((a, b) => a + b, 0) / seg.length : null;
  });
  const smooth = (arr: (number | null)[]) => window(arr, 10);
  return (
    <div className="charts-block">
      <h3>Learning trends <span className="muted">· from real training log</span></h3>
      <div className="charts-grid">
        <MiniChart title="Reward vs Episode" points={smooth(R("reward"))} color="#4ade80" />
        <MiniChart title="Penalty vs Episode" points={smooth(R("penalty"))} color="#f87171" />
        <MiniChart title="Net Reward vs Episode" points={smooth(R("net_reward"))} color="#38bdf8" />
        <MiniChart title="Success Rate (10-ep window)" points={smooth(R("success_rate"))} color="#a78bfa" fmt={(v) => `${Math.round(v * 100)}%`} />
        <MiniChart title="Response Time (s)" points={smooth(R("response_time_s"))} color="#fbbf24" fmt={(v) => `${Math.round(v)}s`} />
        <MiniChart title="Route Efficiency (x shortest)" points={smooth(R("route_efficiency"))} color="#2dd4bf" fmt={(v) => `${v.toFixed(1)}x`} />
      </div>
    </div>
  );
}