const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

export type LatLng = { latitude: number; longitude: number };

export type RLDecision = {
  status: "SAFE_ROUTE" | "NO_SAFE_ROUTE";
  action?: string;
  safety_score?: number;
  risk_level?: string;
  remaining_distance_m?: number;
  estimated_time_s?: number;
  hazard_level?: number;
  decision_source?: string;
  model_name?: string;
  model_version?: string;
  message?: string;
  route?: Array<{
    start: LatLng;
    end: LatLng;
    distance_m: number;
    duration_s: number;
    traffic_factor: number;
    coordinates?: LatLng[];
    risk: number;
    blocked: boolean;
  }>;
};

export async function getHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) throw new Error(`Backend health check failed (${response.status})`);
  return response.json();
}

export async function getRLDecision(currentLocation: LatLng, destination: LatLng): Promise<RLDecision> {
  const response = await fetch(`${API_BASE_URL}/api/rl/decision`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ current_location: currentLocation, destination }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || "Route request failed");
  return body;
}

export { API_BASE_URL };
