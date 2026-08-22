import * as THREE from "three";

export const CELL = 3;
export const ROAD_Y = 0.06;
export const WATER_LEVEL = 0.22;
export const MAX_HAZARDS = 8;
export const MAX_INCIDENTS = 8;

export type CamMode = "overview" | "follow" | "incident" | "disaster";

export function cellToWorld(x: number, y: number, grid: number): [number, number] {
  const half = (grid - 1) / 2;
  return [(x - half) * CELL, (y - half) * CELL];
}

export function normToWorld(nx: number, ny: number, grid: number): [number, number] {
  return [(nx - 0.5) * grid * CELL, (ny - 0.5) * grid * CELL];
}

export function seeded(seed: number) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

export function lerpVec(out: THREE.Vector3, a: THREE.Vector3, b: THREE.Vector3, t: number) {
  out.lerpVectors(a, b, t);
}

/** Shortest-path angle interpolation (handles the -pi/+pi wraparound), used
 * for real vehicle/NPC turning instead of an instant heading snap. */
export function lerpAngle(current: number, target: number, t: number): number {
  let diff = (target - current) % (Math.PI * 2);
  if (diff > Math.PI) diff -= Math.PI * 2;
  if (diff < -Math.PI) diff += Math.PI * 2;
  return current + diff * Math.min(1, Math.max(0, t));
}