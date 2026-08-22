import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { WATER_LEVEL, ROAD_Y, cellToWorld, lerpVec, lerpAngle, seeded } from "./world";
import type { SimFrame } from "../../services/sim";

export type VehiclePose = { pos: THREE.Vector3; yaw: number };
export type ArrivalState = { arrived: boolean; since: number };

function Ambulance({ poseRef, flashRef, arrivedRef }: {
  poseRef: React.MutableRefObject<VehiclePose>;
  flashRef: React.MutableRefObject<{ phase: number }>;
  arrivedRef?: React.MutableRefObject<ArrivalState>;
}) {
  const group = useRef<THREE.Group>(null);
  const beaconMat = useRef<THREE.MeshStandardMaterial>(null);
  const beacon2Mat = useRef<THREE.MeshStandardMaterial>(null);
  const sirenMat = useRef<THREE.MeshStandardMaterial>(null);
  useFrame((state, dt) => {
    const g = group.current;
    if (!g) return;
    g.position.copy(poseRef.current.pos);
    g.rotation.y = poseRef.current.yaw;
    const t = state.clock.elapsedTime;
    flashRef.current.phase += state.clock.getDelta();
    const red = Math.sin(flashRef.current.phase * 6) > 0;
    const blue = Math.sin(flashRef.current.phase * 6 + Math.PI) > 0;

    // ARRIVAL: once the mission is a real, confirmed success (frame.success
    // from the environment, not a guess), the urgent siren flash winds down
    // to a calm steady glow over ~1.5s and the vehicle does a brief victory
    // bounce -- a visible payoff for a real terminal event instead of the
    // vehicle just quietly stopping.
    const arrival = arrivedRef?.current;
    const woundDown = arrival?.arrived ? Math.min(1, arrival.since / 1.5) : 0;
    const beaconUrgency = 1 - woundDown; // 1 = full emergency flash, 0 = calm
    if (beaconMat.current) beaconMat.current.emissiveIntensity = (red ? 2.6 : 0.2) * beaconUrgency + 0.5 * woundDown;
    if (beacon2Mat.current) beacon2Mat.current.emissiveIntensity = (blue ? 2.6 : 0.2) * beaconUrgency + 0.5 * woundDown;
    if (sirenMat.current) sirenMat.current.emissiveIntensity = (1.4 + Math.sin(t * 5) * 0.7) * beaconUrgency;

    if (arrival?.arrived && arrival.since < 1.0) {
      const bounce = Math.sin((arrival.since / 1.0) * Math.PI);
      g.scale.setScalar(1 + bounce * 0.12);
    } else {
      g.scale.setScalar(1);
    }
  });
  return (
    <group ref={group}>
      <mesh position={[0, 0.42, 0]} castShadow>
        <boxGeometry args={[1.15, 0.42, 2.3]} />
        <meshStandardMaterial color="#f2f2f2" roughness={0.4} />
      </mesh>
      <mesh position={[0, 0.95, -0.1]} castShadow>
        <boxGeometry args={[1.05, 0.78, 1.7]} />
        <meshStandardMaterial color="#f8f8f8" roughness={0.35} />
      </mesh>
      <mesh position={[0, 0.95, -0.1]}>
        <boxGeometry args={[0.86, 0.5, 0.04]} />
        <meshStandardMaterial color="#b3261e" />
      </mesh>
      <mesh position={[0, 0.95, -0.1]}>
        <boxGeometry args={[0.5, 0.86, 0.05]} />
        <meshStandardMaterial color="#b3261e" />
      </mesh>
      <mesh position={[0, 0.78, 1.02]}>
        <boxGeometry args={[0.9, 0.42, 0.03]} />
        <meshStandardMaterial color="#17313f" metalness={0.6} roughness={0.2} />
      </mesh>
      <mesh position={[0, 1.42, -0.1]}>
        <boxGeometry args={[0.8, 0.18, 0.5]} />
        <meshStandardMaterial color="#222" />
      </mesh>
      <mesh position={[-0.2, 1.53, -0.1]}>
        <boxGeometry args={[0.32, 0.14, 0.4]} />
        <meshStandardMaterial ref={beaconMat} color="#ff3b45" emissive="#ff3b45" />
      </mesh>
      <mesh position={[0.2, 1.53, -0.1]}>
        <boxGeometry args={[0.32, 0.14, 0.4]} />
        <meshStandardMaterial ref={beacon2Mat} color="#2f7bff" emissive="#2f7bff" />
      </mesh>
      <mesh ref={sirenMat as never} position={[0, 0.78, 1.25]}>
        <sphereGeometry args={[0.16, 10, 10]} />
        <meshStandardMaterial color="#8ab8ff" emissive="#4d9bff" transparent opacity={0.9} />
      </mesh>
      {[[-0.62, 0.22, 0.75], [0.62, 0.22, 0.75], [-0.62, 0.22, -0.75], [0.62, 0.22, -0.75]].map((p, i) => (
        <mesh key={i} position={p as [number, number, number]} rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.26, 0.26, 0.2, 12]} />
          <meshStandardMaterial color="#111" roughness={0.9} />
        </mesh>
      ))}
      <pointLight position={[0, 1.6, 0]} intensity={0} distance={0} />
    </group>
  );
}

function DispatchVan({ pose }: { pose: VehiclePose }) {
  const group = useRef<THREE.Group>(null);
  useFrame(() => {
    if (group.current) group.current.position.copy(pose.pos);
  });
  return (
    <group ref={group}>
      <mesh position={[0, 0.4, 0]} castShadow>
        <boxGeometry args={[1.1, 0.4, 2.1]} />
        <meshStandardMaterial color="#e74c3c" roughness={0.4} />
      </mesh>
      <mesh position={[0, 0.9, 0]} castShadow>
        <boxGeometry args={[1.0, 0.66, 1.6]} />
        <meshStandardMaterial color="#f1c40f" roughness={0.4} />
      </mesh>
      <mesh position={[0, 1.32, 0.1]}>
        <boxGeometry args={[0.9, 0.16, 0.4]} />
        <meshStandardMaterial color="#e67e22" emissive="#ff8c00" />
      </mesh>
    </group>
  );
}

function RescueBoat({ poseRef, flashRef, arrivedRef }: {
  poseRef: React.MutableRefObject<VehiclePose>;
  flashRef: React.MutableRefObject<{ phase: number }>;
  arrivedRef?: React.MutableRefObject<ArrivalState>;
}) {
  const group = useRef<THREE.Group>(null);
  const beaconMat = useRef<THREE.MeshStandardMaterial>(null);
  const beacon2Mat = useRef<THREE.MeshStandardMaterial>(null);
  useFrame((state) => {
    const g = group.current;
    if (!g) return;
    g.position.copy(poseRef.current.pos);
    g.rotation.y = poseRef.current.yaw;
    const t = state.clock.elapsedTime;
    flashRef.current.phase += state.clock.getDelta();
    const red = Math.sin(flashRef.current.phase * 6) > 0;
    const blue = Math.sin(flashRef.current.phase * 6 + Math.PI) > 0;
    const arrival = arrivedRef?.current;
    const woundDown = arrival?.arrived ? Math.min(1, arrival.since / 1.5) : 0;
    const beaconUrgency = 1 - woundDown;
    if (beaconMat.current) beaconMat.current.emissiveIntensity = (red ? 2.6 : 0.2) * beaconUrgency + 0.5 * woundDown;
    if (beacon2Mat.current) beacon2Mat.current.emissiveIntensity = (blue ? 2.6 : 0.2) * beaconUrgency + 0.5 * woundDown;
    const bounce = arrival?.arrived && arrival.since < 1.0 ? Math.sin((arrival.since / 1.0) * Math.PI) : 0;
    g.scale.setScalar(1 + bounce * 0.12);
    g.position.y = WATER_LEVEL + Math.sin(t * 2 + poseRef.current.pos.x) * 0.06;
    g.rotation.z = Math.sin(t * 1.6 + poseRef.current.pos.z) * 0.03;
    g.rotation.x = Math.cos(t * 1.9 + poseRef.current.pos.x) * 0.03;
  });
  return (
    <group ref={group}>
      <mesh position={[0, 0, 0]} castShadow>
        <boxGeometry args={[1.3, 0.4, 2.6]} />
        <meshStandardMaterial color="#e74c3c" roughness={0.5} />
      </mesh>
      <mesh position={[0, 0.12, 0]}>
        <boxGeometry args={[1.05, 0.28, 2.45]} />
        <meshStandardMaterial color="#f8f8f8" roughness={0.4} />
      </mesh>
      <mesh position={[0, 0.3, 0.2]}>
        <boxGeometry args={[0.75, 0.04, 0.45]} />
        <meshStandardMaterial color="#b3261e" />
      </mesh>
      <mesh position={[0, 0.3, 0.2]}>
        <boxGeometry args={[0.45, 0.04, 0.75]} />
        <meshStandardMaterial color="#b3261e" />
      </mesh>
      <mesh position={[0, 0.18, 1.25]} rotation-x={-0.25} castShadow>
        <boxGeometry args={[1.1, 0.3, 0.35]} />
        <meshStandardMaterial color="#c0392b" roughness={0.5} />
      </mesh>
      <mesh position={[0, 0.72, -0.35]} castShadow>
        <boxGeometry args={[0.9, 0.7, 1.1]} />
        <meshStandardMaterial color="#dfe6ea" roughness={0.35} />
      </mesh>
      <mesh position={[0, 0.72, 0.18]}>
        <boxGeometry args={[0.8, 0.5, 0.03]} />
        <meshStandardMaterial color="#17313f" metalness={0.5} roughness={0.2} />
      </mesh>
      <mesh position={[0, 1.12, -0.35]}>
        <boxGeometry args={[0.7, 0.16, 0.4]} />
        <meshStandardMaterial color="#222" />
      </mesh>
      <mesh position={[-0.18, 1.22, -0.35]}>
        <boxGeometry args={[0.28, 0.12, 0.32]} />
        <meshStandardMaterial ref={beaconMat} color="#ff3b45" emissive="#ff3b45" />
      </mesh>
      <mesh position={[0.18, 1.22, -0.35]}>
        <boxGeometry args={[0.28, 0.12, 0.32]} />
        <meshStandardMaterial ref={beacon2Mat} color="#2f7bff" emissive="#2f7bff" />
      </mesh>
      <mesh position={[0.7, 0.42, 0.9]} rotation-z={Math.PI / 2}>
        <torusGeometry args={[0.22, 0.07, 8, 14]} />
        <meshStandardMaterial color="#ff8c00" emissive="#ff8c00" emissiveIntensity={0.4} />
      </mesh>
      <mesh position={[0, 0.02, -1.5]} rotation-x={-Math.PI / 2}>
        <planeGeometry args={[1.6, 0.9]} />
        <meshBasicMaterial color="#9fd4ff" transparent opacity={0.35} depthWrite={false} />
      </mesh>
    </group>
  );
}

function DispatchBoat({ pose }: { pose: VehiclePose }) {
  const group = useRef<THREE.Group>(null);
  useFrame((state) => {
    if (group.current) {
      group.current.position.copy(pose.pos);
      group.current.position.y = WATER_LEVEL + Math.sin(state.clock.elapsedTime * 2 + pose.pos.x) * 0.05;
    }
  });
  return (
    <group ref={group}>
      <mesh position={[0, 0, 0]} castShadow>
        <boxGeometry args={[1.0, 0.32, 2.0]} />
        <meshStandardMaterial color="#2f7bff" roughness={0.5} />
      </mesh>
      <mesh position={[0, 0.55, -0.1]} castShadow>
        <boxGeometry args={[0.7, 0.5, 0.9]} />
        <meshStandardMaterial color="#f1c40f" roughness={0.4} />
      </mesh>
    </group>
  );
}

function VehicleKind({ disaster }: { disaster: string | null }) {
  if (!disaster) return "ambulance";
  if (["flood", "tsunami", "heavy_rain"].includes(disaster)) return "boat";
  return "ambulance";
}

export function Vehicles({ frameRef, startRef }: {
  frameRef: React.MutableRefObject<SimFrame | null>;
  startRef: React.MutableRefObject<[number, number]>;
}) {
  const poseRef = useRef<VehiclePose>({ pos: new THREE.Vector3(0, 0, 0), yaw: 0 });
  const flashRef = useRef({ phase: 0 });
  const vansRef = useRef<VehiclePose[]>([]);
  const [vanCount, setVanCount] = useState(0);
  const [kind, setKind] = useState<"ambulance" | "boat">("ambulance");
  const speedRef = useRef(0); // current speed, world units/sec -- real accel/brake state
  const arrivedRef = useRef<ArrivalState>({ arrived: false, since: 0 });

  useEffect(() => {
    const id = window.setInterval(() => {
      const k = VehicleKind({ disaster: frameRef.current?.disaster ?? null });
      setKind((prev) => (prev === k ? prev : k));
    }, 400);
    return () => window.clearInterval(id);
  }, [frameRef]);

  useFrame((_, dt) => {
    const frame = frameRef.current;
    const [sx, sy] = startRef.current;
    if (!frame) {
      const [wx, wz] = cellToWorld(sx, sy, 10);
      poseRef.current.pos.set(wx, kind === "boat" ? WATER_LEVEL : 0, wz);
      return;
    }
    // ARRIVAL tracking: real, sticky, driven only by the environment's own
    // `success` flag -- reset when a fresh episode starts running again.
    if (frame.status === "running" && arrivedRef.current.arrived) {
      arrivedRef.current.arrived = false;
      arrivedRef.current.since = 0;
    }
    if (frame.success) {
      arrivedRef.current.arrived = true;
    }
    if (arrivedRef.current.arrived) {
      arrivedRef.current.since += dt;
    }

    const [ax, ay] = [frame.agent.x, frame.agent.y];
    const target = new THREE.Vector3(...cellToWorld(ax, ay, frame.grid_size));
    const current = poseRef.current.pos;
    target.y = kind === "boat" ? WATER_LEVEL : 0;

    // Real acceleration/braking instead of a flat lerp: the vehicle ramps
    // up to a max speed, and brakes (decelerating faster than it
    // accelerates, like a real vehicle) as it closes in on the target cell
    // so it eases to a stop there rather than snapping to it.
    const toTarget = target.clone().sub(current);
    toTarget.y = 0;
    const dist = toTarget.length();
    const MAX_SPEED = 8.5;    // world units / sec
    const ACCEL = 16;         // units / sec^2
    const BRAKE = 26;         // units / sec^2 (brakes harder than it accelerates)
    const BRAKING_DISTANCE = 2.4;
    const desiredSpeed = dist < BRAKING_DISTANCE ? MAX_SPEED * (dist / BRAKING_DISTANCE) : MAX_SPEED;
    if (speedRef.current < desiredSpeed) {
      speedRef.current = Math.min(desiredSpeed, speedRef.current + ACCEL * dt);
    } else {
      speedRef.current = Math.max(desiredSpeed, speedRef.current - BRAKE * dt);
    }
    if (dist > 1e-4) {
      const dir = toTarget.multiplyScalar(1 / dist);
      const step = Math.min(dist, speedRef.current * dt);
      current.addScaledVector(dir, step);
    }
    const dx = target.x - current.x;
    const dz = target.z - current.z;
    if (dx * dx + dz * dz > 1e-6) {
      // Real steering: turn toward the new heading at a bounded rate
      // instead of snapping instantly (which looked like teleport-rotation
      // whenever the agent reversed direction between grid cells).
      const desiredYaw = Math.atan2(dx, dz);
      poseRef.current.yaw = lerpAngle(poseRef.current.yaw, desiredYaw, dt * 6);
    }
    const count = Math.max(0, (frame.vehicles || 1) - 1);
    while (vansRef.current.length < Math.min(count, 6)) {
      vansRef.current.push({ pos: current.clone(), yaw: poseRef.current.yaw });
    }
    vansRef.current = vansRef.current.slice(0, Math.min(count, 6));
    // Each dispatched escort vehicle trails the lead ambulance/boat in its
    // own formation slot (fanned out behind it) rather than all chasing the
    // exact same point -- otherwise every extra DISPATCH would be invisible,
    // perfectly overlapping the lead vehicle and each other.
    const ringRadius = 1.7;
    vansRef.current.forEach((v, i) => {
      const slotAngle = poseRef.current.yaw + Math.PI + ((i - (vansRef.current.length - 1) / 2) * 0.55);
      const slotTarget = new THREE.Vector3(
        current.x + Math.sin(slotAngle) * ringRadius,
        current.y,
        current.z + Math.cos(slotAngle) * ringRadius
      );
      const t = Math.min(1, dt * 2.2);
      lerpVec(v.pos, v.pos, slotTarget, t);
      const dvx = slotTarget.x - v.pos.x;
      const dvz = slotTarget.z - v.pos.z;
      if (dvx * dvx + dvz * dvz > 1e-6) {
        v.yaw = lerpAngle(v.yaw, Math.atan2(dvx, dvz), dt * 6);
      }
    });
    if (vansRef.current.length !== vanCount) setVanCount(vansRef.current.length);
  });

  return (
    <group>
      {kind === "boat" ? (
        <RescueBoat poseRef={poseRef} flashRef={flashRef} arrivedRef={arrivedRef} />
      ) : (
        <Ambulance poseRef={poseRef} flashRef={flashRef} arrivedRef={arrivedRef} />
      )}
      {/* Render every dispatched escort vehicle (up to 6), not just one --
         DISPATCH is a real backend action that can add several vehicles. */}
      {vansRef.current.map((pose, i) => (
        kind === "boat"
          ? <DispatchBoat key={i} pose={pose} />
          : <DispatchVan key={i} pose={pose} />
      ))}
    </group>
  );
}

// Civilian traffic (cars) and pedestrian NPCs. Both are ambient/visual --
// no rescue outcomes are ever attributed to them -- but their *behavior* is
// driven by real per-step simulation state (active hazards, blocked
// cells, disaster severity) read live off frameRef, not fabricated. During
// a disaster, real hazards near a car/pedestrian make it swerve/flee away
// and severity scales how fast/panicked the crowd moves.
const NPC_HAZARD_AVOID_RADIUS = 1.35; // grid cells

function hazardAvoidance(frame: SimFrame | null, gx: number, gy: number, grid: number): { ax: number; ay: number; panic: number } {
  if (!frame) return { ax: 0, ay: 0, panic: 0 };
  let ax = 0, ay = 0, panic = 0;
  for (const h of frame.hazards) {
    const hx = h.x * grid, hy = h.y * grid;
    const dx = gx - hx, dy = gy - hy;
    const d = Math.sqrt(dx * dx + dy * dy) || 1e-3;
    const influence = h.radius * grid * NPC_HAZARD_AVOID_RADIUS;
    if (d < influence) {
      const strength = (1 - d / influence) * (0.5 + h.severity);
      ax += (dx / d) * strength;
      ay += (dy / d) * strength;
      panic = Math.max(panic, strength);
    }
  }
  return { ax, ay, panic: Math.min(1, panic) };
}

export function CivilianTraffic({ grid, frameRef }: { grid: number; frameRef?: React.MutableRefObject<SimFrame | null> }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const cars = useMemo(() => {
    const rand = seeded(grid * 71 + 3);
    return Array.from({ length: Math.min(grid * 4, 30) }, () => {
      const x = Math.floor(rand() * grid);
      const y = Math.floor(rand() * grid);
      const dir = Math.floor(rand() * 4);
      return { x, y, dir, speed: 0.8 + rand() * 0.6 };
    });
  }, [grid]);
  const tmp = useMemo(() => new THREE.Object3D(), []);
  useFrame((state) => {
    const mesh = ref.current;
    if (!mesh) return;
    const t = state.clock.elapsedTime;
    const frame = frameRef?.current ?? null;
    const blocked = frame ? new Set(frame.blocked_cells.map(([bx, by]) => `${bx},${by}`)) : null;
    const severity = frame?.severity ?? 0;
    cars.forEach((car, i) => {
      let { x, y } = car;
      const { ax, ay, panic } = hazardAvoidance(frame, x, y, grid);
      const speedMul = 1 + panic * 1.8 + severity * 0.5; // panic + general disaster urgency
      const step = t * car.speed * 0.05 * speedMul;
      switch (car.dir) {
        case 0: x += step; break;
        case 1: x -= step; break;
        case 2: y += step; break;
        default: y -= step; break;
      }
      // Real hazard proximity steers the car off its lane a little (visual
      // swerve), and a car heading straight into a currently-blocked cell
      // re-routes to a random new lane instead of driving through it.
      x += ax * 0.02;
      y += ay * 0.02;
      const nextCellKey = `${Math.round(x)},${Math.round(y)}`;
      const hitBlocked = blocked?.has(nextCellKey) ?? false;
      const g = grid;
      if (x < 0 || x >= g || y < 0 || y >= g || hitBlocked) {
        car.x = Math.floor(Math.random() * g);
        car.y = Math.floor(Math.random() * g);
        car.dir = Math.floor(Math.random() * 4);
      } else {
        car.x = x; car.y = y;
      }
      const [wx, wz] = cellToWorld(car.x, car.y, grid);
      tmp.position.set(wx, ROAD_Y + 0.16, wz);
      tmp.rotation.set(0, (car.dir * Math.PI) / 2 + ax * 0.15, 0);
      tmp.scale.setScalar(0.55);
      tmp.updateMatrix();
      mesh.setMatrixAt(i, tmp.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  });
  return (
    <instancedMesh ref={ref} args={[undefined, undefined, cars.length]}>
      <boxGeometry args={[0.9, 0.35, 1.6]} />
      <meshStandardMaterial color="#6b7a8f" roughness={0.6} />
    </instancedMesh>
  );
}

/** Low-cost pedestrian NPCs: instanced capsule figures that wander near
 * buildings and visibly flee active hazards. Visual population only --
 * never a source of rescue/casualty numbers, which come solely from the
 * environment's victims_total/victims_rescued/unmet fields. */
export function Civilians({ grid, frameRef }: { grid: number; frameRef?: React.MutableRefObject<SimFrame | null> }) {
  const bodyRef = useRef<THREE.InstancedMesh>(null);
  const headRef = useRef<THREE.InstancedMesh>(null);
  const people = useMemo(() => {
    const rand = seeded(grid * 53 + 29);
    const count = Math.min(grid * 3, 24);
    return Array.from({ length: count }, () => ({
      x: rand() * grid,
      y: rand() * grid,
      homeX: 0,
      homeY: 0,
      heading: rand() * Math.PI * 2,
      speed: 0.15 + rand() * 0.12,
      wanderPhase: rand() * Math.PI * 2,
    })).map((p) => ({ ...p, homeX: p.x, homeY: p.y }));
  }, [grid]);
  const tmp = useMemo(() => new THREE.Object3D(), []);
  useFrame((state, dt) => {
    const bodies = bodyRef.current;
    const heads = headRef.current;
    if (!bodies || !heads) return;
    const t = state.clock.elapsedTime;
    const frame = frameRef?.current ?? null;
    people.forEach((p, i) => {
      const { ax, ay, panic } = hazardAvoidance(frame, p.x, p.y, grid);
      if (panic > 0.05) {
        // Flee: turn to face directly away from the nearest hazard, but
        // steer into it rather than snapping (a panicked pedestrian still
        // has to physically turn, not teleport-rotate).
        const fleeHeading = Math.atan2(ax, ay);
        p.heading = lerpAngle(p.heading, fleeHeading, dt * 5);
      } else {
        // Idle wander: gentle drift back toward home spot, small meander.
        p.wanderPhase += dt * 0.4;
        const toHomeX = p.homeX - p.x, toHomeY = p.homeY - p.y;
        const homeDist = Math.sqrt(toHomeX * toHomeX + toHomeY * toHomeY);
        const wanderAngle = Math.sin(p.wanderPhase) * 0.6;
        p.heading = homeDist > 1.5 ? Math.atan2(toHomeX, toHomeY) : p.heading + wanderAngle * dt;
      }
      const speedMul = 1 + panic * 2.2;
      const dx = Math.sin(p.heading) * p.speed * speedMul * dt;
      const dy = Math.cos(p.heading) * p.speed * speedMul * dt;
      p.x = Math.min(grid - 0.2, Math.max(0.2, p.x + dx));
      p.y = Math.min(grid - 0.2, Math.max(0.2, p.y + dy));

      const [wx, wz] = cellToWorld(p.x, p.y, grid);
      const bob = Math.sin(t * 6 + i) * 0.03 * (0.3 + speedMul * 0.5);

      tmp.position.set(wx, 0.28 + bob, wz);
      tmp.rotation.set(0, p.heading, 0);
      tmp.scale.setScalar(1);
      tmp.updateMatrix();
      bodies.setMatrixAt(i, tmp.matrix);

      tmp.position.set(wx, 0.58 + bob, wz);
      tmp.updateMatrix();
      heads.setMatrixAt(i, tmp.matrix);
    });
    bodies.instanceMatrix.needsUpdate = true;
    heads.instanceMatrix.needsUpdate = true;
  });
  return (
    <group>
      <instancedMesh ref={bodyRef} args={[undefined, undefined, people.length]} castShadow>
        <capsuleGeometry args={[0.09, 0.32, 3, 6]} />
        <meshStandardMaterial color="#d9c39a" roughness={0.85} />
      </instancedMesh>
      <instancedMesh ref={headRef} args={[undefined, undefined, people.length]} castShadow>
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshStandardMaterial color="#e7b98f" roughness={0.8} />
      </instancedMesh>
    </group>
  );
}