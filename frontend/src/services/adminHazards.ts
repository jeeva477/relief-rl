import { API_BASE_URL } from "./api";
import type { Hazard } from "./hazards";
import { adminHeaders } from "./auth";

export type HazardPayload = Omit<Hazard, "active">;
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { headers: { "Content-Type": "application/json", ...adminHeaders(), ...(options?.headers || {}) }, ...options });
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.detail || `Request failed (${response.status})`);
  return body as T;
}
export const listAllHazards = () => request<Hazard[]>("/api/admin/hazards");
export const createHazard = (hazard: HazardPayload) => request<Hazard>("/api/admin/hazards", { method: "POST", body: JSON.stringify(hazard) });
export const updateHazard = (id: string, hazard: HazardPayload) => request<Hazard>(`/api/admin/hazards/${encodeURIComponent(id)}`, { method: "PUT", body: JSON.stringify(hazard) });
export const deactivateHazard = (id: string) => request<{ id: string; active: boolean }>(`/api/admin/hazards/${encodeURIComponent(id)}/deactivate`, { method: "POST" });
export const deleteHazard = (id: string) => request<{ id: string; deleted: boolean }>(`/api/admin/hazards/${encodeURIComponent(id)}`, { method: "DELETE" });
