import { AlertTriangle, MapPin, ShieldAlert, X } from "lucide-react";
import type { Hazard } from "../services/hazards";
import type { LatLng } from "../services/api";

type Props = { open: boolean; location: LatLng | null; hazards: Hazard[]; onClose: () => void };

function haversineKm(a: LatLng, b: LatLng) {
  const r = 6371; const dLat = (b.latitude - a.latitude) * Math.PI / 180; const dLng = (b.longitude - a.longitude) * Math.PI / 180;
  const x = Math.sin(dLat / 2) ** 2 + Math.cos(a.latitude * Math.PI / 180) * Math.cos(b.latitude * Math.PI / 180) * Math.sin(dLng / 2) ** 2;
  return r * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
}

export function EmergencyMode({ open, location, hazards, onClose }: Props) {
  if (!open) return null;
  const nearby = location ? hazards.map((h) => ({ ...h, distanceKm: haversineKm(location, { latitude: h.latitude, longitude: h.longitude }) })).sort((a, b) => a.distanceKm - b.distanceKm) : [];
  const closest = nearby[0];
  const critical = closest && (closest.distanceKm * 1000 <= closest.radius_m || closest.severity >= 0.9);

  return <div className="emergency-overlay" role="dialog" aria-modal="true" aria-label="Emergency mode">
    <div className="emergency-modal">
      <button className="close" onClick={onClose} aria-label="Close"><X /></button>
      <div className="emergency-icon"><ShieldAlert size={34} /></div>
      <span className="eyebrow">EMERGENCY MODE</span>
      <h2>{critical ? "IMMEDIATE HAZARD WARNING" : "EMERGENCY MONITORING ACTIVE"}</h2>
      <p className="emergency-copy">DisasterMind AI is monitoring your location against active disaster zones. Do not enter a marked hazard area.</p>
      {closest ? <div className={`closest-hazard ${critical ? "critical" : ""}`}><AlertTriangle size={20}/><div><strong>{closest.hazard_type.replaceAll("_", " ")}</strong><span>{closest.distanceKm.toFixed(2)} km from your location · severity {Math.round(closest.severity * 100)}%</span></div></div> : <div className="closest-hazard"><MapPin size={20}/><div><strong>No active hazard detected nearby</strong><span>Continue monitoring the live map for changes.</span></div></div>}
      <div className="emergency-actions"><button className="primary" onClick={onClose}>Return to route planner</button></div>
      <small className="disclaimer">DisasterMind AI is a decision-support system. Follow official emergency instructions and local authorities.</small>
    </div>
  </div>;
}
