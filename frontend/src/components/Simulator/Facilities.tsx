import { Html } from "@react-three/drei";
import { CELL } from "./world";

function Facility({ pos, color, label, icon }: { pos: [number, number, number]; color: string; label: string; icon: string }) {
  return (
    <group position={pos}>
      <mesh position={[0, 1.1, 0]} castShadow>
        <boxGeometry args={[2.2, 2.2, 2.2]} />
        <meshStandardMaterial color={color} roughness={0.6} />
      </mesh>
      <mesh position={[0, 2.4, 0]}>
        <boxGeometry args={[1.9, 0.25, 1.9]} />
        <meshStandardMaterial color="#f5f5f5" />
      </mesh>
      <Html position={[0, 3.2, 0]} center distanceFactor={30} zIndexRange={[10, 0]}>
        <div className="facility-label"><span>{icon}</span>{label}</div>
      </Html>
    </group>
  );
}

export function Facilities({ grid }: { grid: number }) {
  const r = (grid - 1) * 0.5 * CELL;
  const positions: [number, number, number][] = [
    [-r * 0.72, 0, -r * 0.72],
    [r * 0.72, 0, -r * 0.6],
    [-r * 0.6, 0, r * 0.72],
  ];
  return (
    <group>
      <Facility pos={positions[0]} color="#b3261e" label="HOSPITAL" icon="+" />
      <Facility pos={positions[1]} color="#d35400" label="FIRE STATION" icon="▲" />
      <Facility pos={positions[2]} color="#1f6fb0" label="RESCUE CENTER" icon="✚" />
    </group>
  );
}