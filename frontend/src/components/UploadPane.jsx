import { useRef, useState } from "react";
import { motion } from "framer-motion";

import { uploadImage } from "../lib/api.js";
import { useToast } from "../lib/useToast.jsx";

export default function UploadPane({ onUploaded }) {
  const pickerRef = useRef(); // gallery / files (and the OS menu on mobile)
  const cameraRef = useRef(); // straight to the rear camera on phones
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  async function handleFile(file) {
    if (!file) return;
    const okType =
      /^image\//i.test(file.type) || /\.(png|jpe?g|webp|heic|heif)$/i.test(file.name);
    if (!okType) {
      toast("Please choose an image (JPG, PNG, WebP or HEIC).");
      return;
    }
    setBusy(true);
    try {
      const { id, image_url } = await uploadImage(file);
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

  const openPicker = () => pickerRef.current.click();
  const openVia = (ref) => (e) => {
    e.stopPropagation(); // don't also fire the dropzone's onClick
    ref.current.click();
  };

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
        onClick={openPicker}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && openPicker()}
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
        {/* gallery / files — no capture, so mobile shows camera + library + files */}
        <input
          ref={pickerRef}
          type="file"
          accept="image/*,.heic,.heif"
          hidden
          onChange={(e) => handleFile(e.target.files[0])}
        />
        {/* camera — capture jumps straight to the rear camera on phones */}
        <input
          ref={cameraRef}
          type="file"
          accept="image/*"
          capture="environment"
          hidden
          onChange={(e) => handleFile(e.target.files[0])}
        />

        <div className="dz-icon">{busy ? "⏳" : "🖼️"}</div>
        <h2>{busy ? "Uploading…" : "Add a Punjabi image"}</h2>
        <p className="dz-sub">Tap to use your camera or photo library</p>

        {!busy && (
          <div className="dz-actions">
            <button type="button" className="dz-btn" onClick={openVia(cameraRef)}>
              📷 Camera
            </button>
            <button type="button" className="dz-btn" onClick={openVia(pickerRef)}>
              🖼️ Gallery
            </button>
          </div>
        )}

        <p className="dz-hint">
          or drag &amp; drop · JPG, PNG, WebP or HEIC · up to&nbsp;12&nbsp;MB
        </p>
      </motion.div>
    </motion.section>
  );
}
