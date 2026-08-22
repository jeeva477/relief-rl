import type { LatLng, RLDecision } from "../services/api";

type Props = {
  current: LatLng | null;
  destination: LatLng | null;
  decision: RLDecision | null;
};

function project(point: LatLng, bounds: { minLat: number; maxLat: number; minLng: number; maxLng: number }) {
  const x = ((point.longitude - bounds.minLng) / Math.max(bounds.maxLng - bounds.minLng, 0.00001)) * 100;
  const y = 100 - ((point.latitude - bounds.minLat) / Math.max(bounds.maxLat - bounds.minLat, 0.00001)) * 100;
  return `${Math.min(96, Math.max(4, x))}% ${Math.min(96, Math.max(4, y))}%`;
}

export function RouteMap({ current, destination, decision }: Props) {
  const routePoints = decision?.route?.flatMap((segment) => segment.coordinates || [segment.start, segment.end]) || [];
  const points = [current, destination, ...routePoints].filter(Boolean) as LatLng[];
  const bounds = points.length
    ? {
        minLat: Math.min(...points.map((p) => p.latitude)),
        maxLat: Math.max(...points.map((p) => p.latitude)),
        minLng: Math.min(...points.map((p) => p.longitude)),
        maxLng: Math.max(...points.map((p) => p.longitude)),
      }
    : { minLat: 11.0, maxLat: 11.1, minLng: 77.0, maxLng: 77.1 };

  const polyline = routePoints.map((p) => project(p, bounds)).join(", ");

  return (
    <div className="route-map" aria-label="DisasterMind AI route map">
      <div className="map-grid" />
      <div className="map-label">LIVE EVACUATION VIEW</div>
      <div className="map-legend">
        <span><i className="dot user" /> You</span>
        <span><i className="dot route" /> RL Route</span>
        <span><i className="dot hazard" /> Hazard</span>
      </div>
      {polyline && <svg className="route-svg" viewBox="0 0 100 100" preserveAspectRatio="none"><polyline points={polyline.replaceAll("%", "").replaceAll(" ", ",")} /></svg>}
      {current && <span className="map-marker user-marker" style={{ left: project(current, bounds).split(" ")[0], top: project(current, bounds).split(" ")[1] }} />}
      {destination && <span className="map-marker destination-marker" style={{ left: project(destination, bounds).split(" ")[0], top: project(destination, bounds).split(" ")[1] }} />}
      {!decision && <div className="map-empty">Enable location and select a destination to calculate an evacuation route.</div>}
    </div>
  );
}
