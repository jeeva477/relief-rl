import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { cellToWorld, seeded } from "./world";

export function Buildings({ grid }: { grid: number }) {
  const rand = useMemo(() => seeded(grid * 97 + 11), [grid]);
  const boxes = useMemo(() => {
    const list: { pos: [number, number, number]; size: [number, number, number]; c: string }[] = [];
    const count = Math.min(grid * grid, 180);
    for (let i = 0; i < count; i++) {
      const x = Math.floor(rand() * grid);
      const y = Math.floor(rand() * grid);
      const [wx, wz] = cellToWorld(x, y, grid);
      if (rand() < 0.22) continue;
      const w = 0.7 + rand() * 1.1;
      const h = 0.9 + rand() * 2.6;
      list.push({
        pos: [wx, h / 2, wz],
        size: [w, h, w],
        c: rand() > 0.6 ? "#b9c4cd" : "#8d9aa5",
      });
    }
    return list;
  }, [grid]);
  const meshRef = useRef<THREE.InstancedMesh>(null);
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const m = new THREE.Matrix4();
    const color = new THREE.Color();
    boxes.forEach((b, i) => {
      m.compose(new THREE.Vector3(...b.pos), new THREE.Quaternion(), new THREE.Vector3(...b.size));
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(i, color.set(b.c));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [boxes]);
  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, boxes.length]} castShadow receiveShadow>
      <boxGeometry />
      <meshStandardMaterial roughness={0.9} />
    </instancedMesh>
  );
}