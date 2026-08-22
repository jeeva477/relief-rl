import { API_BASE_URL } from "./api";

export type Hazard = {
  id: string;
  latitude: number;
  longitude: number;
  radius_m: number;
  severity: number;
  hazard_type: string;
  hard_constraint: boolean;
  source: string;
  active: boolean;
};

export async function getActiveHazards(): Promise<Hazard[]> {
  const response = await fetch(`${API_BASE_URL}/api/hazards`);
  if (!response.ok) throw new Error(`Hazard service failed (${response.status})`);
  return response.json();
}
