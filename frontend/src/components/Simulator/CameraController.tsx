import { useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { cellToWorld, lerpVec, type CamMode } from "./world";
import type { SimFrame } from "../../services/sim";

export function CameraController({ frameRef, mode, focusTarget }: {
  frameRef: React.MutableRefObject<SimFrame | null>;
  mode: CamMode;
  focusTarget: React.MutableRefObject<THREE.Vector3 | null>;
}) {
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null;
  const { camera } = useThree();
  const currentTarget = useRef(new THREE.Vector3(0, 0, 0));
  // Earthquake "shake" is applied as a bounded, self-cancelling offset on
  // top of the real camera/target position each frame (undo-then-reapply),
  // rather than a permanent mutation -- otherwise the camera would drift
  // away in a random walk for the rest of the episode.
  const shakeOffset = useRef(new THREE.Vector3());
  const shakeTargetOffset = useRef(new THREE.Vector3());
  useFrame((_, dt) => {
    if (!controls) return;
    const frame = frameRef.current;
    let desired: THREE.Vector3 | null = null;
    if (mode === "follow" && frame) {
      const [wx, wz] = cellToWorld(frame.agent.x, frame.agent.y, frame.grid_size);
      desired = new THREE.Vector3(wx, 1, wz);
    } else if (mode === "incident") {
      desired = focusTarget.current;
    } else if (mode === "disaster") {
      desired = focusTarget.current;
    }
    if (desired) {
      lerpVec(currentTarget.current, currentTarget.current, desired, Math.min(1, dt * 3));
      controls.target.copy(currentTarget.current);
    }
    if (mode === "follow" && frame) {
      const [wx, wz] = cellToWorld(frame.agent.x, frame.agent.y, frame.grid_size);
      const camPos = new THREE.Vector3(wx + 9, 7.5, wz + 9);
      camera.position.lerp(camPos, Math.min(1, dt * 2.4));
    }

    // Undo last frame's shake before computing this frame's shake, so the
    // camera always oscillates around its "true" position/target instead
    // of accumulating a permanent offset.
    camera.position.sub(shakeOffset.current);
    controls.target.sub(shakeTargetOffset.current);

    let shake = 0;
    frame?.hazards.forEach((h) => {
      if (h.type === "earthquake") shake = Math.max(shake, h.severity);
    });
    if (shake > 0 && mode !== "disaster") {
      const a = shake * 0.5;
      shakeOffset.current.set(
        (Math.random() - 0.5) * a,
        (Math.random() - 0.5) * a * 0.7,
        (Math.random() - 0.5) * a
      );
      shakeTargetOffset.current.set((Math.random() - 0.5) * a * 0.4, 0, (Math.random() - 0.5) * a * 0.4);
    } else {
      shakeOffset.current.set(0, 0, 0);
      shakeTargetOffset.current.set(0, 0, 0);
    }
    camera.position.add(shakeOffset.current);
    controls.target.add(shakeTargetOffset.current);

    controls.update();
  });
  return null;
}

export function PerformanceMeter({ onStats }: { onStats: (s: { fps: number; frameMs: number; draws: number; tris: number }) => void }) {
  const { gl } = useThree();
  const frames = useRef(0);
  const last = useRef(performance.now());
  useFrame(() => {
    frames.current += 1;
    const now = performance.now();
    if (now - last.current >= 500) {
      const fps = (frames.current * 1000) / (now - last.current);
      onStats({
        fps,
        frameMs: 1000 / Math.max(fps, 0.1),
        draws: gl.info.render.calls,
        tris: gl.info.render.triangles,
      });
      frames.current = 0;
      last.current = now;
    }
  });
  return null;
}