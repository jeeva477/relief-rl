import { API_BASE_URL } from "./api";

const TOKEN_KEY = "relief_rl_admin_token";
export async function adminLogin(email: string, password: string) {
  const r = await fetch(`${API_BASE_URL}/api/auth/admin/login`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }) });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "Admin login failed");
  const data = await r.json(); localStorage.setItem(TOKEN_KEY, data.access_token); return data;
}
export function getAdminToken() { return localStorage.getItem(TOKEN_KEY); }
export function adminLogout() { localStorage.removeItem(TOKEN_KEY); }
export function isAdminAuthenticated() { return Boolean(getAdminToken()); }
export function adminHeaders(): HeadersInit { const token = getAdminToken(); return token ? { Authorization: `Bearer ${token}` } : {}; }
