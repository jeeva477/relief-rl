import { useState } from "react";
import { LockKeyhole, ShieldCheck } from "lucide-react";
import { adminLogin } from "../services/auth";

type Props = { open: boolean; onSuccess: () => void; onClose: () => void };
export function AdminLogin({ open, onSuccess, onClose }: Props) {
  if (!open) return null;
  const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [error, setError] = useState(""); const [loading, setLoading] = useState(false);
  async function submit(e: React.FormEvent) { e.preventDefault(); setError(""); setLoading(true); try { await adminLogin(email, password); onSuccess(); } catch (err) { setError(err instanceof Error ? err.message : "Login failed"); } finally { setLoading(false); } }
  return <div className="admin-overlay"><section className="login-modal"><button className="close" onClick={onClose}>×</button><div className="login-icon"><ShieldCheck size={30}/></div><span className="eyebrow">DISASTERMIND AI ADMIN</span><h2>Administrator sign in</h2><p>Authorized personnel only.</p><form onSubmit={submit}><label>Email</label><input type="email" value={email} onChange={e=>setEmail(e.target.value)} placeholder="admin@example.com" required/><label>Password</label><div className="password-input"><LockKeyhole size={16}/><input type="password" value={password} onChange={e=>setPassword(e.target.value)} placeholder="Password" required/></div>{error&&<div className="notice">{error}</div>}<button className="primary" disabled={loading}>{loading?"Signing in…":"Sign in"}</button></form></section></div>;
}
