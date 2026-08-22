import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { cellToWorld, seeded } from "./world";

export function Forest({ grid, density = 1 }: { grid: number; density?: number }) {
  const rand = useMemo(() => seeded(grid * 31 + 5), [grid]);
  const trees = useMemo(() => {
    const list: { pos: [number, number, number]; s: number }[] = [];
    const count = Math.min(grid * grid * 2, Math.round(400 * density));
    for (let i = 0; i < count; i++) {
      const x = Math.floor(rand() * grid);
      const y = Math.floor(rand() * grid);
      const [wx, wz] = cellToWorld(x, y, grid);
      const s = 0.4 + rand() * 0.8;
      list.push({ pos: [wx + (rand() - 0.5), 0, wz + (rand() - 0.5)], s });
    }
    return list;
  }, [grid, density]);
  const trunkRef = useRef<THREE.InstancedMesh>(null);
  const leafRef = useRef<THREE.InstancedMesh>(null);
  useEffect(() => {
    const m = new THREE.Matrix4();
    const color = new THREE.Color();
    trees.forEach((t, i) => {
      m.compose(new THREE.Vector3(...t.pos), new THREE.Quaternion(), new THREE.Vector3(t.s, t.s, t.s));
      if (trunkRef.current) trunkRef.current.setMatrixAt(i, m);
      if (leafRef.current) {
        leafRef.current.setMatrixAt(i, m);
        leafRef.current.setColorAt(i, color.set(`hsl(${100 + ((i * 13) % 30)},45%,${32 + ((i * 7) % 16)}%)`));
      }
    });
    if (trunkRef.current) trunkRef.current.instanceMatrix.needsUpdate = true;
    if (leafRef.current) { leafRef.current.instanceMatrix.needsUpdate = true; if (leafRef.current.instanceColor) leafRef.current.instanceColor.needsUpdate = true; }
  }, [trees]);
  return (
    <group>
      <instancedMesh ref={trunkRef} args={[undefined, undefined, trees.length]} castShadow>
        <cylinderGeometry args={[0.12, 0.2, 0.8, 5]} />
        <meshStandardMaterial color="#5a4026" roughness={1} />
      </instancedMesh>
      <instancedMesh ref={leafRef} args={[undefined, undefined, trees.length]} castShadow>
        <coneGeometry args={[0.7, 1.6, 6]} />
        <meshStandardMaterial roughness={0.9} />
      </instancedMesh>
    </group>
  );
}