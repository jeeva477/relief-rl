import { useEffect, useMemo, useRef, useState } from "react";
import { Html } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { CELL, ROAD_Y, cellToWorld } from "./world";
import type { SimFrame } from "../../services/sim";

export function Terrain({ grid, hazardTypes }: { grid: number; hazardTypes: string[] }) {
  const islandRadius = grid * CELL * 0.56;
  const isAquatic = hazardTypes.some((t) => ["flood", "tsunami", "heavy_rain"].includes(t));
  const waterLevel = isAquatic ? 0.32 : 0.05;
  return (
    <group>
      <mesh position={[0, -0.4, 0]}>
        <circleGeometry args={[grid * CELL * 2.4, 48]} />
        <meshStandardMaterial color="#0a3350" roughness={0.35} metalness={0.15} />
      </mesh>
      <mesh position={[0, -0.2, 0]} receiveShadow>
        <cylinderGeometry args={[islandRadius, islandRadius * 1.12, 0.9, 40]} />
        <meshStandardMaterial color="#8a6a45" roughness={1} />
      </mesh>
      <mesh position={[0, 0, 0]} receiveShadow>
        <cylinderGeometry args={[islandRadius * 0.97, islandRadius * 0.97, 0.14, 40]} />
        <meshStandardMaterial color="#2f7a43" roughness={0.95} />
      </mesh>
      <mesh position={[0, -0.02, 0]}>
        <ringGeometry args={[islandRadius * 0.94, islandRadius, 40]} />
        <meshStandardMaterial color="#c8b078" roughness={1} />
      </mesh>
      <mesh position={[0, waterLevel - 0.15, 0]}>
        <cylinderGeometry args={[islandRadius * 0.9, islandRadius * 0.9, 0.3, 32]} />
        <meshStandardMaterial color="#1b6fb5" transparent opacity={0.55} roughness={0.3} />
      </mesh>
    </group>
  );
}

export function Roads({ grid, blockedSetRef, blockedVersion, routeSetRef, routeVersion }: {
  grid: number;
  blockedSetRef: React.MutableRefObject<Set<string>>;
  blockedVersion: number;
  routeSetRef?: React.MutableRefObject<Set<string>>;
  routeVersion?: number;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const cells = useMemo(() => {
    const list: [number, number][] = [];
    for (let x = 0; x < grid; x++) for (let y = 0; y < grid; y++) list.push([x, y]);
    return list;
  }, [grid]);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const m = new THREE.Matrix4();
    const color = new THREE.Color();
    cells.forEach(([x, y], i) => {
      const [wx, wz] = cellToWorld(x, y, grid);
      m.setPosition(wx, ROAD_Y, wz);
      m.makeScale(1, 1, 1);
      mesh.setMatrixAt(i, m);
      const key = `${x},${y}`;
      const blocked = blockedSetRef.current.has(key);
      const route = routeSetRef?.current.has(key) ?? false;
      mesh.setColorAt(i, color.set(blocked ? "#8a4030" : route ? "#155e7d" : "#3d3a38"));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [cells, grid, blockedSetRef, blockedVersion, routeSetRef, routeVersion]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, cells.length]} castShadow receiveShadow>
      <boxGeometry args={[CELL * 0.82, 0.1, CELL * 0.82]} />
      <meshStandardMaterial roughness={0.85} />
    </instancedMesh>
  );
}

export function GoalMarker({ frameRef }: { frameRef: React.MutableRefObject<SimFrame | null> }) {
  const [pos, setPos] = useState<[number, number] | null>(null);
  const [grid, setGrid] = useState(10);
  useEffect(() => {
    const id = window.setInterval(() => {
      const f = frameRef.current;
      if (!f) return;
      setGrid(f.grid_size);
      const [wx, wz] = cellToWorld(f.goal.x, f.goal.y, f.grid_size);
      setPos([wx, wz]);
    }, 400);
    return () => window.clearInterval(id);
  }, [frameRef]);
  const ringRef = useRef<THREE.Mesh>(null);
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (ringRef.current) {
      const s = 1 + Math.sin(t * 2.6) * 0.15;
      ringRef.current.scale.set(s, s, s);
      (ringRef.current.material as THREE.MeshBasicMaterial).opacity = 0.5 + Math.sin(t * 3.2) * 0.2;
    }
  });
  if (!pos) return null;
  return (
    <group position={[pos[0], 0, pos[1]]}>
      <mesh position={[0, 0.22, 0]}>
        <cylinderGeometry args={[0.55, 0.62, 0.42, 6]} />
        <meshStandardMaterial color="#22c55e" emissive="#16a34a" emissiveIntensity={0.7} />
      </mesh>
      <mesh ref={ringRef} rotation-x={-Math.PI / 2} position={[0, 0.08, 0]}>
        <ringGeometry args={[0.7, 1.0, 24]} />
        <meshBasicMaterial color="#4ade80" transparent opacity={0.5} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <Html position={[0, 1.1, 0]} center distanceFactor={26} zIndexRange={[10, 0]}>
        <div className="hud-label ok">SAFE ZONE</div>
      </Html>
    </group>
  );
}