import { useEffect, useRef } from "react";
import type { LatLng, RLDecision } from "../services/api";
import type { Hazard } from "../services/hazards";

declare global { interface Window { L?: any; } }

type Props = { current: LatLng | null; destination: LatLng | null; decision: RLDecision | null; hazards?: Hazard[] };

function loadLeaflet(): Promise<any> {
  return new Promise<any>((resolve, reject) => {
    if (window.L?.map) return resolve(window.L);
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
    document.head.appendChild(link);
    const script = document.createElement("script");
    script.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
    script.onload = () => resolve(window.L);
    script.onerror = () => reject(new Error("Leaflet failed to load."));
    document.head.appendChild(script);
  });
}

function toLL(p: LatLng) { return { lat: p.latitude, lng: p.longitude }; }

export function GoogleMap({ current, destination, decision, hazards = [] }: Props) {
  const mapRef = useRef<HTMLDivElement>(null);
  const map = useRef<any>(null);
  const overlays = useRef<any[]>([]);
  const routeLine = useRef<any>(null);

  const clearOverlays = () => {
    overlays.current.forEach((o) => { try { o.setMap ? o.setMap(null) : o.remove(); } catch {} });
    overlays.current = [];
    if (routeLine.current) { try { routeLine.current.setMap ? routeLine.current.setMap(null) : routeLine.current.remove(); } catch {} routeLine.current = null; }
  };

  const draw = () => {
    const m = map.current;
    const L = window.L;
    if (!m || !L) return;
    clearOverlays();

    if (current) overlays.current.push(L.marker(toLL(current)).addTo(m).bindPopup("Current location"));
    if (destination) overlays.current.push(L.marker(toLL(destination), { icon: L.divIcon({ className: "destination-marker", html: "D" }) }).addTo(m).bindPopup("Destination"));

    hazards.forEach((hazard) => {
      const center = { lat: hazard.latitude, lng: hazard.longitude };
      const circle = L.circle(center, { radius: hazard.radius_m, color: "#e5484d", fillColor: "#e5484d", fillOpacity: hazard.hard_constraint ? 0.32 : 0.18, weight: 2 }).addTo(m);
      circle.bindPopup(`${hazard.hazard_type.replaceAll("_", " ")} — severity ${Math.round(hazard.severity * 100)}%`);
      overlays.current.push(circle);
    });

    const points = decision?.route?.flatMap((segment) => segment.coordinates || [segment.start, segment.end]) || [];
    if (points.length > 1) {
      const latlngs = points.map(toLL);
      routeLine.current = L.polyline(latlngs, { color: "#4ed5b2", weight: 5, opacity: 0.9 }).addTo(m);
      const bounds = L.latLngBounds(latlngs); m.fitBounds(bounds.pad(0.2));
    }
  };

  useEffect(() => {
    let cancelled = false;
    const center = current || { latitude: 11.0168, longitude: 76.9558 };
    loadLeaflet().then((L) => {
      if (cancelled || !mapRef.current || map.current) return;
      map.current = L.map(mapRef.current, { zoomControl: true }).setView([center.latitude, center.longitude], 13);
      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors (ODbL)" }).addTo(map.current);
      setTimeout(() => { try { map.current?.invalidateSize(); } catch {} }, 150);
      draw();
    }).catch(() => {});
    return () => { cancelled = true; clearOverlays(); try { map.current?.remove?.(); } catch {} map.current = null; };
  }, []);

  useEffect(() => { draw(); }, [current, destination, decision, hazards]);

  return <div ref={mapRef} className="google-map" aria-label="DisasterMind AI evacuation map" />;
}