/* Admin portal — load/save API keys and AI provider settings. */

const $ = (sel) => document.querySelector(sel);

const SECRET_FIELDS = ["google_api_key", "azure_vision_key", "openai_api_key", "azure_api_key"];
const PLAIN_FIELDS = [
  "azure_vision_endpoint",
  "openai_model",
  "azure_endpoint",
  "azure_deployment",
  "azure_api_version",
];
const BADGES = {
  google_api_key: "badge-google",
  azure_vision_key: "badge-azure-vision",
  openai_api_key: "badge-openai",
  azure_api_key: "badge-azure",
};

let toastTimer;
function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  requestAnimationFrame(() => el.classList.add("show"));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.remove("show");
    toastTimer = setTimeout(() => (el.hidden = true), 300);
  }, 3500);
}

function toggleGroups() {
  const fd = new FormData($("#settings-form"));
  const ai = fd.get("ai_provider") || "";
  $("#group-openai").classList.toggle("off", ai !== "openai");
  $("#group-azure").classList.toggle("off", ai !== "azure");
  const ocr = fd.get("ocr_provider") || "google";
  $("#group-ocr-google").classList.toggle("off", ocr !== "google");
  $("#group-ocr-azure").classList.toggle("off", ocr !== "azure");
}

async function loadSettings() {
  const res = await fetch("/api/admin/settings");
  if (!res.ok) {
    toast("Could not load settings.");
    return;
  }
  const s = await res.json();

  for (const key of SECRET_FIELDS) {
    const input = document.getElementById(key);
    const badge = document.getElementById(BADGES[key]);
    if (s[key] && s[key].set) {
      input.placeholder = `${s[key].hint || "saved"} — type to replace`;
      badge.classList.add("on");
    }
  }
  for (const key of PLAIN_FIELDS) {
    document.getElementById(key).value = s[key] || "";
  }
  const aiProvider = s.ai_provider || "";
  const aiRadio = document.querySelector(`input[name="ai_provider"][value="${aiProvider}"]`);
  if (aiRadio) aiRadio.checked = true;
  const ocrProvider = s.ocr_provider || "google";
  const ocrRadio = document.querySelector(`input[name="ocr_provider"][value="${ocrProvider}"]`);
  if (ocrRadio) ocrRadio.checked = true;
  toggleGroups();
}

async function saveSettings(event) {
  event.preventDefault();
  const fd = new FormData($("#settings-form"));
  const payload = {
    ai_provider: fd.get("ai_provider") || "",
    ocr_provider: fd.get("ocr_provider") || "google",
  };
  for (const key of SECRET_FIELDS) {
    const value = document.getElementById(key).value.trim();
    if (value) payload[key] = value; // empty = leave the stored key unchanged
  }
  for (const key of PLAIN_FIELDS) {
    payload[key] = document.getElementById(key).value.trim();
  }

  const btn = $("#settings-form button[type=submit]");
  btn.disabled = true;
  try {
    const res = await fetch("/api/admin/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Save failed (${res.status})`);
    }
    for (const key of SECRET_FIELDS) document.getElementById(key).value = "";
    toast("Settings saved ✓");
    await loadSettings();
  } catch (err) {
    toast(err.message);
  }
  btn.disabled = false;
}

$("#settings-form").addEventListener("submit", saveSettings);
$("#settings-form").addEventListener("change", toggleGroups);
loadSettings();
