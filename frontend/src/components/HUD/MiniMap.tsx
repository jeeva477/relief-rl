import { useEffect, useRef, useState } from "react";
import type { SimFrame } from "../../services/sim";
import type { CamMode } from "../Simulator/world";

export function MiniMap({ frameRef, mode, onFocus, routeSetRef }: {
  frameRef: React.MutableRefObject<SimFrame | null>;
  mode: CamMode;
  onFocus?: (cellX: number, cellY: number) => void;
  /** Real cells the agent has actually visited this episode (from
   * SimulatorView's routeSetRef) -- drawn as the "AI ROUTE" trail. Never a
   * fabricated/planned path, since PPO/QR-DQN don't pre-plan one. */
  routeSetRef?: React.MutableRefObject<Set<string>>;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastKey = useRef("");
  const [grid, setGrid] = useState(10);
  useEffect(() => {
    const id = window.setInterval(() => setGrid(frameRef.current?.grid_size ?? 10), 500);
    return () => window.clearInterval(id);
  }, [frameRef]);
  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !onFocus) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * grid);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * grid);
    onFocus(Math.max(0, Math.min(grid - 1, x)), Math.max(0, Math.min(grid - 1, y)));
  };
  useEffect(() => {
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const f = frameRef.current;
      const canvas = canvasRef.current;
      if (!f || !canvas) return;
      const key = `${f.grid_size}|${f.agent.x},${f.agent.y}|${f.blocked_cells.length}|${f.hazards.length}|${f.success}`;
      if (key === lastKey.current && f.status !== "running") return;
      lastKey.current = key;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const size = canvas.width = canvas.height = 128;
      ctx.clearRect(0, 0, size, size);
      const s = size / grid;
      ctx.fillStyle = "#31455c";
      for (let x = 0; x < grid; x++) for (let y = 0; y < grid; y++) ctx.fillRect(x * s, y * s, s, s);
      ctx.fillStyle = "#7a4a3a";
      f.blocked_cells.forEach(([x, y]) => ctx.fillRect(x * s, y * s, s, s));
      f.hazards.forEach((h) => {
        const hx = h.x * size, hy = h.y * size;
        const hr = Math.max(4, h.radius * size * 0.25);
        ctx.beginPath();
        ctx.arc(hx, hy, hr, 0, Math.PI * 2);
        ctx.fillStyle = h.hard ? "#ffdd45" : "#e5484d";
        ctx.globalAlpha = 0.7;
        ctx.fill();
        ctx.globalAlpha = 1;
      });
      f.incidents.forEach((inc) => {
        ctx.fillStyle = "#ff6b70";
        ctx.beginPath();
        ctx.arc(inc.x * size, inc.y * size, 2.5, 0, Math.PI * 2);
        ctx.fill();
      });
      // AI ROUTE: the agent's real visited-cell trail this episode (never a
      // fabricated planned path -- PPO/QR-DQN choose reactively, step by
      // step, so the only honest "route" is where it has actually been).
      if (routeSetRef?.current && routeSetRef.current.size > 1) {
        ctx.fillStyle = "#22d3ee";
        ctx.globalAlpha = 0.55;
        routeSetRef.current.forEach((key) => {
          const [rx, ry] = key.split(",").map(Number);
          ctx.fillRect(rx * s + s * 0.3, ry * s + s * 0.3, s * 0.4, s * 0.4);
        });
        ctx.globalAlpha = 1;
      }
      ctx.fillStyle = "#22c55e";
      ctx.fillRect(f.goal.x * s, f.goal.y * s, s, s);
      ctx.fillStyle = "#4db8ff";
      ctx.beginPath();
      ctx.arc(f.agent.x * s + s / 2, f.agent.y * s + s / 2, 4, 0, Math.PI * 2);
      ctx.fill();
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [grid]);
  return <canvas ref={canvasRef} className="minimap" onClick={handleClick} title="Click to focus camera" />;
}