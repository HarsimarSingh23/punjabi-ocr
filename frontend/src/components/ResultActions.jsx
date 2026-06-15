import { useState } from "react";
import { motion } from "framer-motion";

import { downloadUrl, refine } from "../lib/api.js";
import { useToast } from "../lib/useToast.jsx";

export default function ResultActions({ id, currentText, refined, onRefined, onStatus }) {
  const [refining, setRefining] = useState(false);
  const toast = useToast();

  async function handleRefine() {
    setRefining(true);
    onStatus("Asking the AI to clean up the text…");
    try {
      const { refined_text } = await refine(id);
      onRefined(refined_text);
    } catch (err) {
      toast(err.message);
      onStatus("Saved to backend ✓");
    }
    setRefining(false);
  }

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(currentText);
      toast("Copied to clipboard ✓");
    } catch {
      toast("Could not access the clipboard.");
    }
  }

  return (
    <motion.div
      className="pane-foot result-actions"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <a className="cta small" href={downloadUrl(id, refined)}>
        ⬇ Download .txt
      </a>
      <button className="ghost" onClick={handleCopy}>
        ⧉ Copy
      </button>
      <button className="ghost" onClick={handleRefine} disabled={refining}>
        {refining ? "Refining…" : "🪄 Refine with AI"}
      </button>
    </motion.div>
  );
}
