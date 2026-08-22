import { useEffect, useMemo, useState } from "react";
import { Activity, AlertTriangle, Box, Crosshair, MapPinned, Navigation, Radio, ShieldCheck, Siren, Settings2, Gamepad2 } from "lucide-react";
import { getRLDecision, type LatLng, type RLDecision } from "./services/api";
import { useGeolocation } from "./hooks/useGeolocation";
import { GoogleMap } from "./components/GoogleMap";
import { PlaceSearch } from "./components/PlaceSearch";
import { getActiveHazards, type Hazard } from "./services/hazards";
import { EmergencyMode } from "./components/EmergencyMode";
import { AdminHazardDashboard } from "./components/AdminHazardDashboard";
import { AdminLogin } from "./components/AdminLogin";
import { isAdminAuthenticated } from "./services/auth";
import { EvacuationScene3D } from "./components/EvacuationScene3D";
import { SimulatorView } from "./components/SimulatorView";

type AppView = "dashboard" | "simulator";

function App() {
  const { location, error: locationError } = useGeolocation();
  const [destinationText, setDestinationText] = useState("");
  const [destination, setDestination] = useState<LatLng | null>(null);
  const [decision, setDecision] = useState<RLDecision | null>(null);
  const [hazards, setHazards] = useState<Hazard[]>([]);
  const [loading, setLoading] = useState(false);
  const [emergencyOpen, setEmergencyOpen] = useState(false);
  const [adminOpen, setAdminOpen] = useState(false);
  const [loginOpen, setLoginOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [adminAuth, setAdminAuth] = useState(isAdminAuthenticated());
  const [view, setView] = useState<AppView>("dashboard");
  const [view3d, setView3d] = useState<"map" | "3d">("map");

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const data = await getActiveHazards();
        if (active) setHazards(data);
      } catch {}
    };
    load();
    const t = window.setInterval(load, 15000);
    return () => {
      active = false;
      window.clearInterval(t);
    };
  }, [adminOpen]);

  const stats = useMemo(
    () => [
      ["Safety score", decision?.safety_score != null ? `${Math.round(decision.safety_score * 100)}%` : "—", decision?.risk_level || "Awaiting route"],
      ["Remaining", decision?.remaining_distance_m != null ? `${(decision.remaining_distance_m / 1000).toFixed(1)} km` : "—", decision?.estimated_time_s != null ? `${Math.round(decision.estimated_time_s / 60)} min` : "—"],
      ["Hazard level", decision?.risk_level || "—", decision?.hazard_level != null ? `${Math.round(decision.hazard_level * 100)}% exposure` : `${hazards.length} active zones`],
    ],
    [decision, hazards.length]
  );

  async function findRoute() {
    setError(null);
    if (!location) return setError("Allow browser location access first.");
    if (!destination) return setError("Select a destination from the location suggestions.");
    setLoading(true);
    try {
      setDecision(await getRLDecision(location, destination));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Route request failed");
      setDecision(null);
    } finally {
      setLoading(false);
    }
  }

  function openAdmin() {
    if (adminAuth) setAdminOpen(true);
    else setLoginOpen(true);
  }

  function logout() {
    setAdminAuth(false);
    setAdminOpen(false);
  }

  return (
    <main className={`app-shell ${view === "simulator" ? "sim-mode" : ""}`}>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><ShieldCheck size={22} /></div>
          <div><strong>DisasterMind AI</strong><span>Emergency Route Intelligence</span></div>
        </div>
        <div className="live"><span className="pulse" /> SYSTEM OPERATIONAL</div>
        <nav className="app-nav" aria-label="Application views">
          <button className={view === "dashboard" ? "active" : ""} onClick={() => setView("dashboard")}><MapPinned size={15} /> Dashboard</button>
          <button className={view === "simulator" ? "active" : ""} onClick={() => setView("simulator")}><Gamepad2 size={15} /> Simulator</button>
        </nav>
        <div className="top-actions">
          <button className="admin-trigger" onClick={openAdmin}><Settings2 size={17} /> Admin</button>
          <button className="emergency" onClick={() => setEmergencyOpen(true)}><Siren size={18} /> Emergency mode</button>
        </div>
      </header>

      {view === "simulator" ? (
        <SimulatorView />
      ) : (
        <>
      <section className="hero">
        <div>
          <p className="eyebrow">REAL-TIME EVACUATION CONTROL</p>
          <h1>Find the safest way out.</h1>
          <p className="hero-copy">DisasterMind AI combines live location data, hazard intelligence and reinforcement learning to evaluate evacuation routes.</p>
        </div>
        <div className="location-pill">
          <Crosshair size={17} /> {location ? "GPS active" : "GPS waiting"} <b>{location ? `${location.latitude.toFixed(4)}, ${location.longitude.toFixed(4)}` : "—"}</b>
        </div>
      </section>

      <section className="workspace">
        <div className="map-card">
          <div className="map-toolbar">
            <span><MapPinned size={17} /> Live evacuation map</span>
            <div className="toolbar-right">
              <span className="map-status"><Radio size={14} /> {decision ? "RL route evaluated" : `${hazards.length} hazard zones monitored`}</span>
              <div className="view-toggle" role="tablist" aria-label="View mode">
                <button className={view3d === "map" ? "active" : ""} onClick={() => setView3d("map")}><MapPinned size={14} /> 2D</button>
                <button className={view3d === "3d" ? "active" : ""} onClick={() => setView3d("3d")}><Box size={14} /> 3D</button>
              </div>
            </div>
          </div>
          {view3d === "3d" ? (
            <EvacuationScene3D current={location} destination={destination} decision={decision} hazards={hazards} />
          ) : (
            <GoogleMap current={location} destination={destination} decision={decision} hazards={hazards} />
          )}
        </div>

        <aside className="control-panel">
          <div className="panel-title">
            <div><span className="eyebrow">ROUTE PLANNER</span><h2>Safe evacuation</h2></div>
            <Navigation size={20} />
          </div>
          <label>Destination</label>
          <div className="input">
            <MapPinned size={17} />
            <PlaceSearch
              value={destinationText}
              onChange={(v) => { setDestinationText(v); setDestination(null); }}
              onSelect={(p, l) => { setDestination(p); setDestinationText(l); }}
            />
          </div>
          <button className="primary" onClick={findRoute} disabled={loading}>
            <Navigation size={17} /> {loading ? "Evaluating route…" : "Find safest route"}
          </button>
          {locationError && <div className="notice"><AlertTriangle size={16} /> {locationError}</div>}
          {error && <div className="notice"><AlertTriangle size={16} /> {error}</div>}
          <div className="hazard-summary"><span><Siren size={15} /> Active hazards</span><strong>{hazards.length}</strong></div>
          {decision && (
            <div className={`decision ${decision.status === "NO_SAFE_ROUTE" ? "danger" : ""}`}>
              <div className="decision-icon"><ShieldCheck size={20} /></div>
              <div>
                <small>RL DECISION</small>
                <strong>{decision.status === "SAFE_ROUTE" ? "SAFE ROUTE AVAILABLE" : "NO SAFE ROUTE"}</strong>
                <span>{decision.message || `${decision.decision_source || "RL"} evaluated the route.`}</span>
              </div>
            </div>
          )}
          <div className="stats">
            {stats.map(([label, value, sub]) => (
              <div className="stat" key={label}>
                <small>{label}</small>
                <strong>{value}</strong>
                <span>{sub}</span>
              </div>
            ))}
          </div>
          <div className="log">
            <div><Activity size={15} /> Decision log</div>
            {decision ? (
              <>
                <p><b>Now</b> Route candidate evaluated</p>
                <p><b>Now</b> Hazard exposure checked</p>
                <p><b>Now</b> Safety validator result: {decision.status}</p>
              </>
            ) : (
              <p>Waiting for a route evaluation.</p>
            )}
          </div>
        </aside>
      </section>
        </>
      )}

      <EmergencyMode open={emergencyOpen} location={location} hazards={hazards} onClose={() => setEmergencyOpen(false)} />
      <AdminLogin
        open={loginOpen}
        onSuccess={() => { setAdminAuth(true); setLoginOpen(false); setAdminOpen(true); }}
        onClose={() => setLoginOpen(false)}
      />
      <AdminHazardDashboard open={adminOpen} onClose={() => setAdminOpen(false)} onLogout={logout} />
    </main>
  );
}

export default App;