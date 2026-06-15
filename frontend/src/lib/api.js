// Thin wrapper around the FastAPI backend.
//
// VITE_API_BASE points the frontend at the backend's origin. Leave it empty for
// same-origin (local dev via the Vite proxy, or when FastAPI serves the build);
// set it to the deployed backend URL (e.g. https://punjabi-ocr-api.onrender.com)
// when the frontend is hosted separately on Cloudflare Pages.
const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export function apiUrl(path) {
  return `${API_BASE}${path}`;
}

// Admin token (only needed when the backend sets ADMIN_TOKEN). Kept in
// sessionStorage so it is entered once per browser session.
const TOKEN_KEY = "punjabi_ocr_admin_token";
export const getAdminToken = () => sessionStorage.getItem(TOKEN_KEY) || "";
export const setAdminToken = (t) =>
  t ? sessionStorage.setItem(TOKEN_KEY, t) : sessionStorage.removeItem(TOKEN_KEY);

function adminHeaders(extra = {}) {
  const token = getAdminToken();
  return token ? { ...extra, "X-Admin-Token": token } : extra;
}

async function asError(res) {
  let detail;
  try {
    const body = await res.json();
    detail = body.detail;
  } catch {
    /* not JSON */
  }
  const err = new Error(detail || `Request failed (${res.status})`);
  err.status = res.status;
  return err;
}

export async function uploadImage(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(apiUrl("/api/upload"), { method: "POST", body: fd });
  if (!res.ok) throw await asError(res);
  const data = await res.json(); // { id, image_url }
  // image_url is backend-relative — make it absolute so the <img> loads cross-origin
  return { ...data, image_url: apiUrl(data.image_url) };
}

export async function runOcr(id) {
  const res = await fetch(apiUrl(`/api/ocr/${id}`), { method: "POST" });
  if (!res.ok) throw await asError(res);
  return res.json(); // { width, height, words[], full_text }
}

export async function refine(id) {
  const res = await fetch(apiUrl(`/api/refine/${id}`), { method: "POST" });
  if (!res.ok) throw await asError(res);
  return res.json(); // { refined_text }
}

export function downloadUrl(id, refined) {
  return apiUrl(`/api/results/${id}/download${refined ? "?refined=true" : ""}`);
}

export async function getSettings() {
  const res = await fetch(apiUrl("/api/admin/settings"), { headers: adminHeaders() });
  if (!res.ok) throw await asError(res);
  return res.json();
}

export async function saveSettings(payload) {
  const res = await fetch(apiUrl("/api/admin/settings"), {
    method: "POST",
    headers: adminHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await asError(res);
  return res.json();
}
