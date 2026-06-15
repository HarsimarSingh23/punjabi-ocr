import { useRef, useState } from "react";
import { motion } from "framer-motion";

import { uploadImage } from "../lib/api.js";
import { useToast } from "../lib/useToast.jsx";

export default function UploadPane({ onUploaded }) {
  const inputRef = useRef();
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  async function handleFile(file) {
    if (!file) return;
    if (!/^image\/(png|jpe?g|webp)$/.test(file.type)) {
      toast("Please choose a JPG, PNG or WebP image.");
      return;
    }
    setBusy(true);
    try {
      const { id, image_url } = await uploadImage(file);
      // preload so the workspace shows the image immediately
      await new Promise((ok, bad) => {
        const img = new Image();
        img.onload = ok;
        img.onerror = () => bad(new Error("Could not display the image."));
        img.src = image_url;
      });
      onUploaded({ id, url: image_url });
    } catch (err) {
      toast(err.message);
      setBusy(false);
    }
  }

  return (
    <motion.section
      className="upload-pane"
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{ duration: 0.35 }}
    >
      <motion.div
        className={`dropzone${over ? " over" : ""}${busy ? " busy" : ""}`}
        role="button"
        tabIndex={0}
        whileHover={{ y: -3 }}
        onClick={() => inputRef.current.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          handleFile(e.dataTransfer.files[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          hidden
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <div className="dz-icon">{busy ? "⏳" : "🖼️"}</div>
        <h2>{busy ? "Uploading…" : "Drop a Punjabi image here"}</h2>
        <p>
          or <span className="linkish">browse files</span> · JPG, PNG or WebP · up
          to&nbsp;12&nbsp;MB
        </p>
      </motion.div>
    </motion.section>
  );
}
