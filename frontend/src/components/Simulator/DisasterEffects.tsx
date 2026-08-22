import { useEffect, useMemo, useRef, useState } from "react";
import { Html } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { CELL, MAX_HAZARDS, normToWorld } from "./world";
import type { SimFrame } from "../../services/sim";

export function Rain({ frameRef, maxParticles = 500 }: { frameRef: React.MutableRefObject<SimFrame | null>; maxParticles?: number }) {
  const pointsRef = useRef<THREE.Points>(null);
  const positions = useMemo(() => new Float32Array(maxParticles * 3), [maxParticles]);
  const velocity = useMemo(() => new Float32Array(maxParticles), [maxParticles]);
  const mat = useMemo(() => new THREE.PointsMaterial({
    color: "#9fc8ff", size: 0.09, transparent: true, opacity: 0.7, depthWrite: false,
  }), []);
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return g;
  }, [positions]);
  const [enabled, setEnabled] = useState(false);
  useEffect(() => {
    const id = window.setInterval(() => {
      const f = frameRef.current;
      const raining = maxParticles > 0 && !!f && ["rain", "heavy_rain", "storm"].includes(f.weather) &&
        f.hazards.some((h) => ["flood", "heavy_rain", "storm", "tsunami", "cyclone"].includes(h.type));
      setEnabled(raining);
    }, 500);
    return () => window.clearInterval(id);
  }, [frameRef, maxParticles]);

  useFrame((state) => {
    if (!enabled) return;
    const pts = pointsRef.current;
    if (!pts) return;
    const grid = frameRef.current?.grid_size ?? 10;
    const span = grid * CELL * 1.1;
    const attr = geo.attributes.position as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    const dt = state.clock.getDelta();
    for (let i = 0; i < maxParticles; i++) {
      let y = arr[i * 3 + 1] - dt * 6;
      if (y < 0.2) y = 8 + Math.random() * 2;
      arr[i * 3] = (Math.random() - 0.5) * span;
      arr[i * 3 + 1] = y;
      arr[i * 3 + 2] = (Math.random() - 0.5) * span;
    }
    attr.needsUpdate = true;
  });
  return (
    <group visible={enabled}>
      <points ref={pointsRef} geometry={geo} material={mat} />
    </group>
  );
}

export function FireAndSmoke({ frameRef, hazardRefs, maxFlames = 60 }: {
  frameRef: React.MutableRefObject<SimFrame | null>;
  hazardRefs: React.MutableRefObject<Map<string, THREE.Vector3>>;
  maxFlames?: number;
}) {
  const flamesRef = useRef<THREE.InstancedMesh>(null);
  const smokeRef = useRef<THREE.Points>(null);
  const smokePos = useMemo(() => new Float32Array(200 * 3), []);
  const smokeGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(smokePos, 3));
    return g;
  }, [smokePos]);
  const smokeMat = useMemo(() => new THREE.PointsMaterial({
    color: "#3a3a3a", size: 0.7, transparent: true, opacity: 0.35, depthWrite: false,
  }), []);
  const flameMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: "#ff8c1a", emissive: "#ff5a00", emissiveIntensity: 1.6, transparent: true, opacity: 0.9,
  }), []);
  const tmp = useMemo(() => new THREE.Object3D(), []);
  const MAX_FLAMES = Math.max(maxFlames, 0);

  useEffect(() => {
    const arr = new Float32Array(200 * 3);
    for (let i = 0; i < 200; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 30;
      arr[i * 3 + 1] = Math.random() * 6;
      arr[i * 3 + 2] = (Math.random() - 0.5) * 30;
    }
    smokePos.set(arr);
  }, [smokePos]);

  useFrame((state) => {
    const mesh = flamesRef.current;
    const fires = new Map<string, THREE.Vector3>();
    frameRef.current?.hazards.forEach((h) => {
      if (["wildfire", "fire"].includes(h.type)) {
        const [wx, wz] = normToWorld(h.x, h.y, frameRef.current?.grid_size ?? 10);
        fires.set(h.id, new THREE.Vector3(wx, 0, wz));
      }
    });
    hazardRefs.current = fires;
    if (mesh && MAX_FLAMES > 0) {
      let idx = 0;
      fires.forEach((pos) => {
        const n = Math.min(6, MAX_FLAMES - idx);
        for (let k = 0; k < n && idx < MAX_FLAMES; k++, idx++) {
          const t = state.clock.elapsedTime;
          tmp.position.set(pos.x + (Math.random() - 0.5) * 1.6, 0.3 + (Math.random() - 0.5) * 0.2, pos.z + (Math.random() - 0.5) * 1.6);
          tmp.scale.setScalar(0.4 + Math.sin(t * 6 + idx) * 0.12);
          tmp.updateMatrix();
          mesh.setMatrixAt(idx, tmp.matrix);
        }
      });
      for (; idx < MAX_FLAMES; idx++) {
        tmp.position.set(0, -10, 0);
        tmp.scale.setScalar(0.01);
        tmp.updateMatrix();
        mesh.setMatrixAt(idx, tmp.matrix);
      }
      mesh.count = idx;
      mesh.instanceMatrix.needsUpdate = true;
    }
    const pts = smokeRef.current;
    if (pts && fires.size > 0 && MAX_FLAMES > 0) {
      const attr = smokeGeo.attributes.position as THREE.BufferAttribute;
      const arr = attr.array as Float32Array;
      const centers = Array.from(fires.values());
      const dt = state.clock.getDelta();
      for (let i = 0; i < 200; i++) {
        let y = arr[i * 3 + 1] + dt * 1.1;
        if (y > 6) y = 0.4;
        const c = centers[i % centers.length];
        arr[i * 3] += (Math.random() - 0.5) * dt * 0.8;
        arr[i * 3 + 1] = y;
        arr[i * 3 + 2] += (Math.random() - 0.5) * dt * 0.8;
        if (c) { arr[i * 3] = c.x + (arr[i * 3] - c.x) * 0.5; arr[i * 3 + 2] = c.z + (arr[i * 3 + 2] - c.z) * 0.5; }
      }
      attr.needsUpdate = true;
    }
  });

  return (
    <group>
      <instancedMesh ref={flamesRef} args={[undefined, undefined, MAX_FLAMES]}>
        <coneGeometry args={[0.28, 0.9, 6]} />
        <primitive object={flameMat} attach="material" />
      </instancedMesh>
      <points ref={smokeRef} geometry={smokeGeo} material={smokeMat} visible={false} />
    </group>
  );
}

export function Dust({ frameRef }: { frameRef: React.MutableRefObject<SimFrame | null> }) {
  const pointsRef = useRef<THREE.Points>(null);
  const pos = useMemo(() => new Float32Array(120 * 3), []);
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    return g;
  }, [pos]);
  const mat = useMemo(() => new THREE.PointsMaterial({
    color: "#9c8a6a", size: 0.25, transparent: true, opacity: 0.5, depthWrite: false,
  }), []);
  useFrame((state) => {
    const f = frameRef.current;
    if (!f) return;
    const dusty = f.hazards.some((h) => ["earthquake", "landslide"].includes(h.type));
    if (!dusty) return;
    const attr = geo.attributes.position as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    const grid = f?.grid_size ?? 10;
    const dt = state.clock.getDelta();
    for (let i = 0; i < 120; i++) {
      let y = arr[i * 3 + 1] + dt * 0.5;
      if (y > 2.5) y = 0.05;
      arr[i * 3] += (Math.random() - 0.5) * dt * 0.6;
      arr[i * 3 + 1] = y;
      arr[i * 3 + 2] += (Math.random() - 0.5) * dt * 0.6;
      if (i % 10 === 0) {
        const h = f.hazards.find((hh) => ["earthquake", "landslide"].includes(hh.type));
        if (h) {
          const [wx, wz] = normToWorld(h.x, h.y, grid);
          arr[i * 3] = wx + (Math.random() - 0.5) * h.radius * grid * CELL;
          arr[i * 3 + 2] = wz + (Math.random() - 0.5) * h.radius * grid * CELL;
        }
      }
    }
    attr.needsUpdate = true;
  });
  return (
    <points ref={pointsRef} geometry={geo} material={mat} />
  );
}

export function HazardZones({ frameRef }: { frameRef: React.MutableRefObject<SimFrame | null> }) {
  const [hazards, setHazards] = useState<SimFrame["hazards"]>([]);
  useEffect(() => {
    const id = window.setInterval(() => {
      const f = frameRef.current;
      setHazards((prev) => (prev === f?.hazards ? prev : (f?.hazards ?? [])));
    }, 400);
    return () => window.clearInterval(id);
  }, [frameRef]);
  const [grid, setGrid] = useState(10);
  useEffect(() => {
    const id = window.setInterval(() => setGrid(frameRef.current?.grid_size ?? 10), 600);
    return () => window.clearInterval(id);
  }, [frameRef]);
  const domeMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: "#e5484d", transparent: true, opacity: 0.22, emissive: "#ff6b6b",
    emissiveIntensity: 0.5, roughness: 0.4, side: THREE.DoubleSide, depthWrite: false,
  }), []);
  const ringMat = useMemo(() => new THREE.MeshBasicMaterial({
    color: "#ff5d67", transparent: true, opacity: 0.4, side: THREE.DoubleSide, depthWrite: false,
  }), []);
  const pulse = useRef<{ m: THREE.Mesh; r: THREE.Mesh }[]>([]);
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    pulse.current.forEach((p, i) => {
      if (!p.m || !p.r) return;
      const s = 1 + Math.sin(t * 2.4 + i) * 0.12;
      p.r.scale.set(s, s, s);
      (p.r.material as THREE.MeshBasicMaterial).opacity = 0.35 + Math.sin(t * 3 + i) * 0.15;
      (p.m.material as THREE.MeshStandardMaterial).opacity = 0.2 + Math.sin(t * 2 + i) * 0.06;
    });
  });
  return (
    <group>
      {hazards.slice(0, MAX_HAZARDS).map((h, i) => {
        const [wx, wz] = normToWorld(h.x, h.y, grid);
        const radius = Math.min(Math.max(h.radius * grid * CELL * 0.5, 1.2), grid * CELL * 0.7);
        const fire = h.type === "wildfire";
        const flood = ["flood", "tsunami", "heavy_rain"].includes(h.type);
        return (
          <group key={h.id} position={[wx, 0, wz]}>
            <mesh
              ref={(el) => { if (el) { pulse.current[i] = { m: el, r: pulse.current[i]?.r as THREE.Mesh }; } }}
              position={[0, radius * 0.4, 0]}
            >
              <sphereGeometry args={[radius, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2]} />
              <primitive object={domeMat} attach="material" />
            </mesh>
            <mesh
              ref={(el) => { if (el) { pulse.current[i] = { m: pulse.current[i]?.m as THREE.Mesh, r: el }; } }}
              rotation-x={-Math.PI / 2}
              position={[0, 0.15, 0]}
            >
              <ringGeometry args={[radius * 0.82, radius, 32]} />
              <primitive object={ringMat} attach="material" />
            </mesh>
            {h.hard && (
              <mesh position={[0, 0.18, 0]}>
                <ringGeometry args={[radius * 0.62, radius * 0.7, 32]} />
                <meshBasicMaterial color="#ffdd45" transparent opacity={0.9} side={THREE.DoubleSide} depthWrite={false} />
              </mesh>
            )}
            {flood && (
              <mesh position={[0, 0.3, 0]}>
                <cylinderGeometry args={[radius, radius, 0.6, 20]} />
                <meshStandardMaterial color="#1b6fb5" transparent opacity={0.6} roughness={0.2} />
              </mesh>
            )}
            {fire && (
              <pointLight position={[0, 4, 0]} color="#ff6a00" intensity={2} distance={radius * 4} decay={2} />
            )}
            <Html position={[0, radius * 0.7 + 1.2, 0]} center distanceFactor={24} zIndexRange={[10, 0]}>
              <div className={`hud-label ${h.hard ? "danger" : "warn"}`}>
                {h.type.toUpperCase()} · {Math.round(h.severity * 100)}%{h.hard ? " · HARD" : ""}
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
}