import { useEffect, useRef, useState } from "react";
import type { LatLng } from "../services/api";

declare global { interface Window { google?: any; } }

type Props = { value: string; onChange: (value: string) => void; onSelect: (location: LatLng, label: string) => void };
type Suggestion = { label: string; lat: number; lon: number };

const API_KEY = import.meta.env.VITE_GOOGLE_MAPS_BROWSER_KEY || "";

export function PlaceSearch({ value, onChange, onSelect }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const autocomplete = useRef<any>(null);
  const [ready, setReady] = useState(false);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [open, setOpen] = useState(false);
  const timer = useRef<any>(null);

  useEffect(() => {
    if (!API_KEY) { setReady(true); return; }
    const iv = window.setInterval(() => {
      if (window.google?.maps?.places?.Autocomplete && inputRef.current && !autocomplete.current) {
        autocomplete.current = new window.google.maps.places.Autocomplete(inputRef.current, {
          fields: ["geometry", "formatted_address", "name"],
          types: ["geocode", "establishment"],
        });
        autocomplete.current.addListener("place_changed", () => {
          const place = autocomplete.current.getPlace();
          const point = place.geometry?.location;
          if (!point) return;
          const label = place.formatted_address || place.name || "Selected destination";
          onChange(label);
          onSelect({ latitude: point.lat(), longitude: point.lng() }, label);
        });
        setReady(true);
        window.clearInterval(iv);
      }
    }, 250);
    return () => window.clearInterval(iv);
  }, [onChange, onSelect]);

  const search = (q: string) => {
    if (timer.current) window.clearTimeout(timer.current);
    if (!q.trim() || q.trim().length < 3) { setSuggestions([]); setOpen(false); return; }
    timer.current = window.setTimeout(async () => {
      try {
        const res = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&limit=5&q=${encodeURIComponent(q)}`,
          { headers: { Accept: "application/json" } }
        );
        if (!res.ok) return;
        const data = await res.json();
        setSuggestions((data || []).map((d: any) => ({
          label: d.display_name || d.name,
          lat: parseFloat(d.lat),
          lon: parseFloat(d.lon),
        })));
        setOpen(true);
      } catch { /* offline: keep typed value */ }
    }, 400);
  };

  const pick = (s: Suggestion) => {
    setOpen(false);
    onChange(s.label);
    onSelect({ latitude: s.lat, longitude: s.lon }, s.label);
  };

  return (
    <div className="place-search">
      <input ref={inputRef} value={value} onChange={(e) => { onChange(e.target.value); if (!API_KEY) search(e.target.value); }}
        onFocus={() => suggestions.length && setOpen(true)}
        onBlur={() => window.setTimeout(() => setOpen(false), 200)}
        placeholder={ready ? "Search a destination" : "Loading location search…"} aria-label="Destination" />
      {open && suggestions.length > 0 && (
        <ul className="place-suggestions">
          {suggestions.map((s, i) => (
            <li key={i} onMouseDown={(e) => { e.preventDefault(); pick(s); }}>{s.label}</li>
          ))}
        </ul>
      )}
    </div>
  );
}