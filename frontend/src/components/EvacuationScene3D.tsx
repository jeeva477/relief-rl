import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Cloud, Grid, Html, OrbitControls, Sky } from "@react-three/drei";
import * as THREE from "three";
import type { LatLng, RLDecision } from "../services/api";
import type { Hazard } from "../services/hazards";

const EARTH_R = 6371000;
const SCALE = 25;
const ROUTE_Y = 0.9;

type Vec = { x: number; z: number };

function project(p: LatLng, origin: LatLng): Vec {
  const x = (((p.longitude - origin.longitude) * Math.PI) / 180) * EARTH_R * Math.cos((origin.latitude * Math.PI) / 180);
  const z = (((p.latitude - origin.latitude) * Math.PI) / 180) * EARTH_R;
  return { x: x / SCALE, z: z / SCALE };
}

function seeded(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function Ground() {
  return (
    <mesh rotation-x={-Math.PI / 2} position={[0, -0.05, 0]}>
      <planeGeometry args={[1200, 1200]} />
      <meshStandardMaterial color="#0a1d2e" roughness={0.95} metalness={0.05} />
    </mesh>
  );
}

function Buildings({ hazards, origin }: { hazards: Hazard[]; origin: LatLng }) {
  const rand = seeded(1337);
  const hazardsLocal = hazards.map((h) => ({ ...project(h, origin), r: Math.max(h.radius_m / SCALE, 2) }));
  const boxes = useMemo(() => {
    const list: Array<{ x: number; z: number; w: number; d: number; h: number; c: string }> = [];
    for (let i = 0; i < 90; i++) {
      const angle = rand() * Math.PI * 2;
      const radius = 45 + rand() * 170;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const tooClose = hazardsLocal.some((h) => Math.hypot(x - h.x, z - h.z) < h.r + 12);
      if (tooClose) continue;
      list.push({
        x, z,
        w: 2.5 + rand() * 4,
        d: 2.5 + rand() * 4,
        h: 2.5 + rand() * 9,
        c: rand() > 0.6 ? "#16344d" : "#12283c",
      });
    }
    return list;
  }, [hazardsLocal]);

  return (
    <group>
      {boxes.map((b, i) => (
        <mesh key={i} position={[b.x, b.h / 2, b.z]}>
          <boxGeometry args={[b.w, b.h, b.d]} />
          <meshStandardMaterial color={b.c} roughness={0.85} metalness={0.1} />
        </mesh>
      ))}
    </group>
  );
}

function Trees() {
  const rand = seeded(4242);
  const trees = useMemo(
    () => Array.from({ length: 50 }, () => {
      const angle = rand() * Math.PI * 2;
      const radius = 40 + rand() * 190;
      return { x: Math.cos(angle) * radius, z: Math.sin(angle) * radius, s: 0.7 + rand() * 0.9 };
    }),
    []
  );
  return (
    <group>
      {trees.map((t, i) => (
        <group key={i} position={[t.x, 0, t.z]} scale={t.s}>
          <mesh position={[0, 0.7, 0]}>
            <cylinderGeometry args={[0.18, 0.26, 1.4, 6]} />
            <meshStandardMaterial color="#5a4026" roughness={1} />
          </mesh>
          <mesh position={[0, 2.1, 0]}>
            <coneGeometry args={[1.1, 2.2, 7]} />
            <meshStandardMaterial color="#1d5c3f" roughness={0.9} />
          </mesh>
        </group>
      ))}
    </group>
  );
}

function HazardZone({ hazard, origin }: { hazard: Hazard; origin: LatLng }) {
  const pos = project(hazard, origin);
  const radius = Math.min(Math.max(hazard.radius_m / SCALE, 2), 26);
  const dome = useRef<THREE.Mesh>(null);
  const ring = useRef<THREE.Mesh>(null);
  const light = useRef<THREE.PointLight>(null);
  const group = useRef<THREE.Group>(null);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (group.current) group.current.position.y = Math.abs(Math.sin(t * 1.8 + pos.x)) * 0.5;
    if (ring.current) {
      const s = 1 + Math.sin(t * 2.4 + pos.x) * 0.14;
      ring.current.scale.set(s, s, s);
      (ring.current.material as THREE.MeshBasicMaterial).opacity = 0.35 + Math.sin(t * 3) * 0.18;
    }
    if (dome.current) (dome.current.material as THREE.MeshStandardMaterial).opacity = 0.22 + Math.sin(t * 2.2 + pos.z) * 0.09;
    if (light.current) light.current.intensity = 2.6 + Math.sin(t * 3.2) * 1.4;
  });

  return (
    <group ref={group} position={[pos.x, 0, pos.z]}>
      <mesh ref={dome} position={[0, radius * 0.55, 0]}>
        <sphereGeometry args={[radius, 24, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#e5484d" transparent opacity={0.24} emissive="#e5484d" emissiveIntensity={0.6} roughness={0.4} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <mesh ref={ring} rotation-x={-Math.PI / 2} position={[0, 0.15, 0]}>
        <ringGeometry args={[radius * 0.85, radius, 40]} />
        <meshBasicMaterial color="#ff5d67" transparent opacity={0.4} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <mesh position={[0, 3.2, 0]}>
        <cylinderGeometry args={[0.16, 0.16, 3, 8]} />
        <meshStandardMaterial color="#c3373f" emissive="#ff3b45" emissiveIntensity={1.4} />
      </mesh>
      <pointLight ref={light} position={[0, 6, 0]} color="#ff4b55" distance={radius * 3.2} decay={2} intensity={3} />
      <Html position={[0, radius + 3.4, 0]} center distanceFactor={22} zIndexRange={[10, 0]}>
        <div className="hud-label danger">
          {hazard.hazard_type.replaceAll("_", " ")} · {Math.round(hazard.severity * 100)}%
        </div>
      </Html>
    </group>
  );
}

function UserBeacon({ position }: { position: Vec }) {
  const ring = useRef<THREE.Mesh>(null);
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (ring.current) {
      ring.current.rotation.z = t * 0.9;
      const s = 1.25 + Math.sin(t * 2.6) * 0.15;
      ring.current.scale.set(s, s, 1);
    }
  });
  return (
    <group position={[position.x, 0, position.z]}>
      <mesh position={[0, 1.1, 0]}>
        <cylinderGeometry args={[0.55, 0.75, 2.2, 18]} />
        <meshStandardMaterial color="#1f8fdb" emissive="#1f8fdb" emissiveIntensity={0.55} metalness={0.5} roughness={0.3} />
      </mesh>
      <mesh ref={ring} rotation-x={-Math.PI / 2} position={[0, 0.12, 0]}>
        <ringGeometry args={[1.15, 1.55, 32]} />
        <meshBasicMaterial color="#4db8ff" transparent opacity={0.85} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <pointLight position={[0, 2.5, 0]} color="#4db8ff" intensity={2.4} distance={16} decay={2} />
      <Html position={[0, 3.4, 0]} center distanceFactor={22} zIndexRange={[10, 0]}>
        <div className="hud-label user">YOU</div>
      </Html>
    </group>
  );
}

function DestinationBeacon({ position }: { position: Vec }) {
  const beam = useRef<THREE.Mesh>(null);
  const ring = useRef<THREE.Mesh>(null);
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (beam.current) (beam.current.material as THREE.MeshBasicMaterial).opacity = 0.16 + Math.abs(Math.sin(t * 1.4)) * 0.2;
    if (ring.current) {
      ring.current.rotation.z = -t * 1.1;
      const s = 1 + Math.sin(t * 2.2) * 0.12;
      ring.current.scale.set(s, s, 1);
    }
  });
  return (
    <group position={[position.x, 0, position.z]}>
      <mesh position={[0, 1.4, 0]}>
        <cylinderGeometry args={[0.4, 0.7, 2.8, 18]} />
        <meshStandardMaterial color="#22c55e" emissive="#22c55e" emissiveIntensity={0.5} metalness={0.4} roughness={0.3} />
      </mesh>
      <mesh ref={beam} position={[0, 14, 0]}>
        <cylinderGeometry args={[0.1, 1.6, 26, 16, 1, true]} />
        <meshBasicMaterial color="#34f58a" transparent opacity={0.2} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <mesh ref={ring} rotation-x={-Math.PI / 2} position={[0, 0.12, 0]}>
        <ringGeometry args={[1.3, 1.8, 32]} />
        <meshBasicMaterial color="#34f58a" transparent opacity={0.8} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <pointLight position={[0, 3, 0]} color="#34f58a" intensity={2.6} distance={18} decay={2} />
      <Html position={[0, 3.8, 0]} center distanceFactor={22} zIndexRange={[10, 0]}>
        <div className="hud-label safe">DESTINATION</div>
      </Html>
    </group>
  );
}

const dashVertex = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const dashFragment = `
uniform float uTime;
uniform vec3 uColor;
varying vec2 vUv;
void main() {
  float seg = fract(vUv.x * 20.0 - uTime * 1.5);
  float dash = step(0.45, seg);
  float glow = pow(1.0 - abs(seg - 0.25) * 3.0, 2.0);
  float alpha = dash * (0.5 + glow * 0.5);
  vec3 col = uColor * (0.65 + glow * 1.35);
  gl_FragColor = vec4(col, alpha);
}
`;

function FlowTube({ curve }: { curve: THREE.CatmullRomCurve3 }) {
  const tube = useMemo(() => new THREE.TubeGeometry(curve, 240, 0.55, 8, false), [curve]);
  const dashMat = useRef<THREE.ShaderMaterial>(null);
  useFrame((state) => {
    if (dashMat.current) dashMat.current.uniforms.uTime.value = state.clock.elapsedTime;
  });
  return (
    <group>
      <mesh geometry={tube}>
        <meshStandardMaterial color="#0e8f7c" emissive="#0e8f7c" emissiveIntensity={0.5} transparent opacity={0.5} roughness={0.5} metalness={0.3} depthWrite={false} />
      </mesh>
      <mesh geometry={tube}>
        <shaderMaterial
          ref={dashMat}
          transparent
          depthWrite={false}
          uniforms={{ uTime: { value: 0 }, uColor: { value: new THREE.Color("#4ed5b2") } }}
          vertexShader={dashVertex}
          fragmentShader={dashFragment}
        />
      </mesh>
    </group>
  );
}

function RouteVehicle({ curve }: { curve: THREE.CatmullRomCurve3 }) {
  const group = useRef<THREE.Group>(null);
  const progress = useRef(Math.random());
  useFrame((_, dt) => {
    const g = group.current;
    if (!g) return;
    progress.current = (progress.current + dt * 0.16) % 1;
    const p = curve.getPointAt(progress.current);
    const t = curve.getTangentAt(progress.current);
    g.position.set(p.x, p.y, p.z);
    const look = new THREE.Vector3(p.x + t.x, p.y, p.z + t.z);
    g.lookAt(look);
    g.rotation.z = 0;
  });
  const wheels: Array<[number, number, number]> = [
    [-0.6, 0.22, 0.75], [0.6, 0.22, 0.75], [-0.6, 0.22, -0.75], [0.6, 0.22, -0.75],
  ];
  return (
    <group ref={group} scale={0.95}>
      <mesh position={[0, 0.42, 0]} castShadow>
        <boxGeometry args={[1.2, 0.42, 2.4]} />
        <meshStandardMaterial color="#2fd6c0" metalness={0.55} roughness={0.3} />
      </mesh>
      <mesh position={[0, 0.85, -0.15]} castShadow>
        <boxGeometry args={[1.0, 0.48, 1.35]} />
        <meshStandardMaterial color="#0f2f42" metalness={0.2} roughness={0.5} />
      </mesh>
      <mesh position={[0, 1.05, 0.75]}>
        <cylinderGeometry args={[0.12, 0.12, 0.9, 12]} />
        <meshStandardMaterial color="#ffd166" emissive="#ffb020" emissiveIntensity={1.8} />
      </mesh>
      {wheels.map((pos, i) => (
        <mesh key={i} position={pos} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.24, 0.24, 0.2, 14]} />
          <meshStandardMaterial color="#0a1520" roughness={0.9} />
        </mesh>
      ))}
      <pointLight position={[0, 1.2, 1.6]} intensity={2.2} distance={9} color="#7df5dc" />
    </group>
  );
}

function Clouds() {
  return (
    <group>
      <Cloud position={[40, 34, -60]} speed={0.5} opacity={0.4} segments={20} />
      <Cloud position={[-70, 40, -30]} speed={0.35} opacity={0.32} segments={16} />
      <Cloud position={[80, 46, 20]} speed={0.45} opacity={0.28} segments={22} />
      <Cloud position={[-40, 52, 60]} speed={0.4} opacity={0.3} segments={18} />
    </group>
  );
}

type Props = {
  current: LatLng | null;
  destination: LatLng | null;
  decision: RLDecision | null;
  hazards: Hazard[];
};

export function EvacuationScene3D({ current, destination, decision, hazards }: Props) {
  const { origin, curve } = useMemo(() => {
    const pts = [
      ...(current ? [current] : []),
      ...(destination ? [destination] : []),
      ...(decision?.route?.flatMap((s) => s.coordinates || [s.start, s.end]) || []),
    ];
    const all = pts.length ? pts : [{ latitude: 11.0168, longitude: 76.9558 }];
    const origin = {
      latitude: all.reduce((a, p) => a + p.latitude, 0) / all.length,
      longitude: all.reduce((a, p) => a + p.longitude, 0) / all.length,
    };
    const routePts = decision?.route?.flatMap((s) => s.coordinates || [s.start, s.end]) || [];
    const curve =
      routePts.length > 2
        ? new THREE.CatmullRomCurve3(routePts.map((p) => { const v = project(p, origin); return new THREE.Vector3(v.x, ROUTE_Y, v.z); }))
        : null;
    return { origin, curve };
  }, [current, destination, decision]);

  const userPos = current ? project(current, origin) : null;
  const destPos = destination ? project(destination, origin) : null;

  return (
    <div className="scene3d" aria-label="DisasterMind AI 3D evacuation simulation">
      <Canvas dpr={[1, 2]} camera={{ position: [70, 65, 95], fov: 45, near: 0.1, far: 2200 }} gl={{ antialias: true }}>
        <color attach="background" args={["#07111f"]} />
        <fog attach="fog" args={["#0a1b2b", 160, 520]} />
        <ambientLight intensity={0.55} />
        <directionalLight position={[90, 130, 50]} intensity={1.15} color="#cfe8ff" />
        <directionalLight position={[-60, 40, -80]} intensity={0.25} color="#4db8ff" />
        <Sky distance={450000} sunPosition={[120, 60, -80]} turbidity={8} rayleigh={0.6} mieCoefficient={0.005} mieDirectionalG={0.8} />
        <Ground />
        <Grid
          position={[0, 0.02, 0]}
          args={[10, 10]}
          cellSize={1.6}
          cellThickness={0.6}
          cellColor="#1d3a52"
          sectionSize={8}
          sectionThickness={1.1}
          sectionColor="#2a4d6b"
          fadeDistance={430}
          fadeStrength={2}
          infiniteGrid
        />
        <Buildings hazards={hazards} origin={origin} />
        <Trees />
        <Clouds />
        {hazards.map((h) => <HazardZone key={h.id} hazard={h} origin={origin} />)}
        {userPos && <UserBeacon position={userPos} />}
        {destPos && <DestinationBeacon position={destPos} />}
        {curve && <FlowTube curve={curve} />}
        {curve && <RouteVehicle curve={curve} />}
        <OrbitControls
          makeDefault
          enablePan
          maxPolarAngle={Math.PI / 2.1}
          minDistance={10}
          maxDistance={430}
          autoRotate={!decision}
          autoRotateSpeed={0.5}
        />
      </Canvas>
      <div className="scene-hud">
        <span className="hud-chip">3D SIMULATION</span>
        <span className="hud-chip">{hazards.length} HAZARD ZONES</span>
        <span className="hud-chip">{decision ? "RL ROUTE ACTIVE" : "DRAG TO ORBIT · SCROLL TO ZOOM"}</span>
      </div>
    </div>
  );
}