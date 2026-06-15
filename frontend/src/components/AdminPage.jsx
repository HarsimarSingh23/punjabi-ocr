import { useEffect, useState } from "react";
import { motion } from "framer-motion";

import AuroraBackground from "./AuroraBackground.jsx";
import TopBar from "./TopBar.jsx";
import { getSettings, saveSettings, setAdminToken } from "../lib/api.js";
import { ToastProvider, useToast } from "../lib/useToast.jsx";

const SECRET_FIELDS = [
  "google_api_key",
  "azure_vision_key",
  "nvidia_api_key",
  "openai_api_key",
  "azure_api_key",
];
const PLAIN_FIELDS = [
  "azure_vision_endpoint",
  "nvidia_model",
  "openai_model",
  "azure_endpoint",
  "azure_deployment",
  "azure_api_version",
];

const EMPTY = {
  ocr_provider: "google",
  ai_provider: "",
  ...Object.fromEntries(PLAIN_FIELDS.map((k) => [k, ""])),
  ...Object.fromEntries(SECRET_FIELDS.map((k) => [k, ""])),
};

export default function AdminPage() {
  return (
    <ToastProvider>
      <AuroraBackground />
      <div className="app-shell">
        <TopBar />
        <AdminForm />
      </div>
    </ToastProvider>
  );
}

function AdminForm() {
  const [values, setValues] = useState(EMPTY);
  const [saved, setSaved] = useState({}); // which secrets already have a stored value
  const [busy, setBusy] = useState(false);
  const [locked, setLocked] = useState(false); // backend wants an admin token
  const toast = useToast();

  function load() {
    return getSettings()
      .then((s) => {
        const next = { ...EMPTY };
        next.ocr_provider = s.ocr_provider || "google";
        next.ai_provider = s.ai_provider || "";
        for (const k of PLAIN_FIELDS) next[k] = s[k] || "";
        const savedFlags = {};
        for (const k of SECRET_FIELDS) savedFlags[k] = s[k] && s[k].set ? s[k].hint || "saved" : "";
        setValues(next);
        setSaved(savedFlags);
        setLocked(false);
        return true;
      })
      .catch((e) => {
        if (e.status === 401) setLocked(true);
        else toast(e.message);
        return false;
      });
  }

  useEffect(() => {
    load();
  }, []);

  if (locked) {
    return <TokenGate onUnlock={load} toast={toast} />;
  }

  const set = (k) => (e) => setValues((v) => ({ ...v, [k]: e.target.value }));

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    const payload = { ocr_provider: values.ocr_provider, ai_provider: values.ai_provider };
    for (const k of PLAIN_FIELDS) payload[k] = values[k];
    for (const k of SECRET_FIELDS) if (values[k].trim()) payload[k] = values[k].trim();
    try {
      await saveSettings(payload);
      setValues((v) => ({ ...v, ...Object.fromEntries(SECRET_FIELDS.map((k) => [k, ""])) }));
      const s = await getSettings();
      const flags = {};
      for (const k of SECRET_FIELDS) flags[k] = s[k] && s[k].set ? s[k].hint || "saved" : "";
      setSaved(flags);
      toast("Settings saved ✓");
    } catch (err) {
      if (err.status === 401) setLocked(true);
      toast(err.message);
    }
    setBusy(false);
  }

  const ocr = values.ocr_provider;
  const ai = values.ai_provider;

  return (
    <main className="admin-main">
      <motion.form
        className="admin-card"
        onSubmit={submit}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
      >
        <h1>API keys &amp; providers</h1>
        <p>Keys are stored in the app's local database and never shown back in full.</p>

        <fieldset>
          <legend>OCR engine</legend>
          <div className="provider-row">
            <Radio name="ocr_provider" value="google" current={ocr} onChange={set("ocr_provider")}>
              Google Cloud Vision
            </Radio>
            <Radio name="ocr_provider" value="azure" current={ocr} onChange={set("ocr_provider")}>
              Azure AI Vision
            </Radio>
            <Radio name="ocr_provider" value="nvidia" current={ocr} onChange={set("ocr_provider")}>
              NVIDIA vision
            </Radio>
          </div>

          {ocr === "google" && (
            <Secret
              id="google_api_key"
              label="Google API key"
              placeholder="AIza…"
              value={values.google_api_key}
              saved={saved.google_api_key}
              onChange={set("google_api_key")}
              hint={
                <>
                  Create one in the{" "}
                  <a href="https://console.cloud.google.com/apis/credentials" target="_blank" rel="noopener noreferrer">
                    Google Cloud console
                  </a>{" "}
                  with the Cloud Vision API enabled.
                </>
              }
            />
          )}

          {ocr === "azure" && (
            <>
              <Text id="azure_vision_endpoint" label="Endpoint" type="url"
                placeholder="https://my-resource.cognitiveservices.azure.com"
                value={values.azure_vision_endpoint} onChange={set("azure_vision_endpoint")} />
              <Secret id="azure_vision_key" label="Azure Vision key"
                value={values.azure_vision_key} saved={saved.azure_vision_key}
                onChange={set("azure_vision_key")}
                hint="Azure AI Services / Computer Vision resource (Image Analysis 4.0 Read). Keys are under Resource Management → Keys and Endpoint." />
            </>
          )}

          {ocr === "nvidia" && (
            <>
              <Secret id="nvidia_api_key" label="NVIDIA API key" placeholder="nvapi-…"
                value={values.nvidia_api_key} saved={saved.nvidia_api_key} onChange={set("nvidia_api_key")} />
              <Text id="nvidia_model" label="Vision model"
                placeholder="meta/llama-3.2-11b-vision-instruct"
                value={values.nvidia_model} onChange={set("nvidia_model")}
                hint="A vision-capable model from integrate.api.nvidia.com. This engine returns plain text (no word boxes), so words animate from the image rather than from exact boxes." />
            </>
          )}
        </fieldset>

        <fieldset>
          <legend>AI text refinement</legend>
          <div className="provider-row">
            <Radio name="ai_provider" value="" current={ai} onChange={set("ai_provider")}>None</Radio>
            <Radio name="ai_provider" value="openai" current={ai} onChange={set("ai_provider")}>OpenAI</Radio>
            <Radio name="ai_provider" value="azure" current={ai} onChange={set("ai_provider")}>Azure OpenAI</Radio>
          </div>

          {ai === "openai" && (
            <>
              <Secret id="openai_api_key" label="OpenAI API key" placeholder="sk-…"
                value={values.openai_api_key} saved={saved.openai_api_key} onChange={set("openai_api_key")} />
              <Text id="openai_model" label="Model" placeholder="gpt-4o-mini"
                value={values.openai_model} onChange={set("openai_model")} />
            </>
          )}

          {ai === "azure" && (
            <>
              <Text id="azure_endpoint" label="Endpoint" type="url"
                placeholder="https://my-resource.openai.azure.com"
                value={values.azure_endpoint} onChange={set("azure_endpoint")} />
              <Secret id="azure_api_key" label="Azure API key"
                value={values.azure_api_key} saved={saved.azure_api_key} onChange={set("azure_api_key")} />
              <Text id="azure_deployment" label="Deployment name" placeholder="gpt-4o-mini"
                value={values.azure_deployment} onChange={set("azure_deployment")} />
              <Text id="azure_api_version" label="API version"
                placeholder="leave blank for the v1 API (recommended)"
                value={values.azure_api_version} onChange={set("azure_api_version")}
                hint="Blank uses the modern /openai/v1 route. Set a version like 2024-06-01 only for older resources." />
            </>
          )}
        </fieldset>

        <div className="admin-actions">
          <button type="submit" className="cta small" disabled={busy}>
            {busy ? "Saving…" : "Save settings"}
          </button>
        </div>
      </motion.form>
    </main>
  );
}

function TokenGate({ onUnlock, toast }) {
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);

  async function unlock(e) {
    e.preventDefault();
    if (!token.trim()) return;
    setBusy(true);
    setAdminToken(token.trim());
    const ok = await onUnlock();
    if (!ok) {
      setAdminToken("");
      toast("Invalid admin token.");
    }
    setBusy(false);
  }

  return (
    <main className="admin-main">
      <motion.form
        className="admin-card token-gate"
        onSubmit={unlock}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <h1>Admin locked 🔒</h1>
        <p>This backend requires an admin token to view or change provider keys.</p>
        <div className="field">
          <label htmlFor="admin_token">Admin token</label>
          <input
            id="admin_token"
            type="password"
            autoComplete="off"
            autoFocus
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="value of the backend's ADMIN_TOKEN"
          />
        </div>
        <div className="admin-actions">
          <button type="submit" className="cta small" disabled={busy || !token.trim()}>
            {busy ? "Unlocking…" : "Unlock"}
          </button>
        </div>
      </motion.form>
    </main>
  );
}

function Radio({ name, value, current, onChange, children }) {
  return (
    <label className={current === value ? "selected" : ""}>
      <input type="radio" name={name} value={value} checked={current === value} onChange={onChange} />
      {children}
    </label>
  );
}

function Text({ id, label, type = "text", placeholder, value, onChange, hint }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input id={id} type={type} placeholder={placeholder} value={value} onChange={onChange} autoComplete="off" />
      {hint && <div className="hintline">{hint}</div>}
    </div>
  );
}

function Secret({ id, label, placeholder, value, saved, onChange, hint }) {
  return (
    <div className="field">
      <label htmlFor={id}>
        {label} {saved && <span className="badge on">saved ✓</span>}
      </label>
      <input
        id={id}
        type="password"
        placeholder={saved ? `${saved} — type to replace` : placeholder}
        value={value}
        onChange={onChange}
        autoComplete="off"
      />
      {hint && <div className="hintline">{hint}</div>}
    </div>
  );
}
