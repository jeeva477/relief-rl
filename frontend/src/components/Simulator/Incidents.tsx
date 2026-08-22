import { useEffect, useRef, useState } from "react";
import { Html } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { MAX_INCIDENTS, normToWorld } from "./world";
import type { SimFrame } from "../../services/sim";

export function Incidents({ frameRef }: { frameRef: React.MutableRefObject<SimFrame | null> }) {
  const [incidents, setIncidents] = useState<SimFrame["incidents"]>([]);
  useEffect(() => {
    const id = window.setInterval(() => {
      const f = frameRef.current;
      setIncidents((prev) => (prev === f?.incidents ? prev : (f?.incidents ?? [])));
    }, 300);
    return () => window.clearInterval(id);
  }, [frameRef]);
  const pulseRefs = useRef<THREE.Mesh[]>([]);
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    pulseRefs.current.forEach((m, i) => {
      if (!m) return;
      const s = 1 + Math.sin(t * 3 + i) * 0.18;
      m.scale.set(s, s, s);
      (m.material as THREE.MeshBasicMaterial).opacity = 0.5 + Math.sin(t * 4 + i) * 0.2;
    });
  });
  return (
    <group>
      {incidents.slice(0, MAX_INCIDENTS).map((inc, i) => {
        const [wx, wz] = normToWorld(inc.x, inc.y, frameRef.current?.grid_size ?? 10);
        return (
          <group key={`${inc.x.toFixed(2)}-${inc.y.toFixed(2)}-${i}`} position={[wx, 0, wz]}>
            <mesh position={[0, 0.9, 0]}>
              <cylinderGeometry args={[0.35, 0.5, 1.8, 10]} />
              <meshStandardMaterial color="#e5484d" emissive="#e5484d" emissiveIntensity={0.8} />
            </mesh>
            <mesh
              ref={(el) => { if (el) pulseRefs.current[i] = el; }}
              rotation-x={-Math.PI / 2}
              position={[0, 0.12, 0]}
            >
              <ringGeometry args={[0.5, 0.8, 20]} />
              <meshBasicMaterial color="#ff6b70" transparent opacity={0.6} side={THREE.DoubleSide} depthWrite={false} />
            </mesh>
            <Html position={[0, 2.0, 0]} center distanceFactor={26} zIndexRange={[10, 0]}>
              <div className="hud-label danger">INCIDENT · {inc.victims} VICTIMS</div>
            </Html>
          </group>
        );
      })}
    </group>
  );
}