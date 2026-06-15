// Thin wrapper around the FastAPI backend. All paths are relative so the same
// build works behind the Vite dev proxy and when served by FastAPI directly.

async function asError(res) {
  try {
    const body = await res.json();
    if (body.detail) return new Error(body.detail);
  } catch {
    /* not JSON */
  }
  return new Error(`Request failed (${res.status})`);
}

export async function uploadImage(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: fd });
  if (!res.ok) throw await asError(res);
  return res.json(); // { id, image_url }
}

export async function runOcr(id) {
  const res = await fetch(`/api/ocr/${id}`, { method: "POST" });
  if (!res.ok) throw await asError(res);
  return res.json(); // { width, height, words[], full_text }
}

export async function refine(id) {
  const res = await fetch(`/api/refine/${id}`, { method: "POST" });
  if (!res.ok) throw await asError(res);
  return res.json(); // { refined_text }
}

export function downloadUrl(id, refined) {
  return `/api/results/${id}/download${refined ? "?refined=true" : ""}`;
}

export async function getSettings() {
  const res = await fetch("/api/admin/settings");
  if (!res.ok) throw await asError(res);
  return res.json();
}

export async function saveSettings(payload) {
  const res = await fetch("/api/admin/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await asError(res);
  return res.json();
}
