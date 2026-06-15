import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";

import { runOcr } from "../lib/api.js";
import { useToast } from "../lib/useToast.jsx";
import BoundingBoxes from "./BoundingBoxes.jsx";
import ResultActions from "./ResultActions.jsx";

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export default function Workspace({ image, onReset }) {
  const [stage, setStage] = useState("idle"); // idle | scanning | reveal | done
  const [ocr, setOcr] = useState(null);
  const [refinedText, setRefinedText] = useState(null);
  const [leftStatus, setLeftStatus] = useState("");
  const [rightStatus, setRightStatus] = useState("");

  const imageRef = useRef(null);
  const boxLayerRef = useRef(null);
  const flyLayerRef = useRef(null);
  const spanRefs = useRef([]);

  async function startAi() {
    setStage("scanning");
    setLeftStatus("Reading the image with AI…");
    setRightStatus("");
    try {
      const data = await runOcr(image.id);
      spanRefs.current = [];
      setRefinedText(null);
      setOcr(data); // triggers the reveal effect below
      setLeftStatus(`${data.words.length} words found`);
    } catch (err) {
      toast(err.message);
      setStage("idle");
      setLeftStatus("");
    }
  }

  const toast = useToast();

  // Orchestrate the draw + fly animation once OCR data is in the DOM.
  useEffect(() => {
    if (!ocr) return;
    let cancelled = false;

    (async () => {
      setStage("reveal");
      await drawBoxes(boxLayerRef.current, ocr);
      if (cancelled) return;
      setRightStatus("Transferring words…");
      await flyWords({
        ocr,
        image: imageRef.current,
        spans: spanRefs.current,
        flyLayer: flyLayerRef.current,
      });
      if (cancelled) return;
      burstSparkles(flyLayerRef.current);
      setStage("done");
      setRightStatus("Saved to backend ✓");
    })();

    return () => {
      cancelled = true;
    };
  }, [ocr]);

  const showSpans = ocr && !refinedText;

  return (
    <motion.section
      className={`workspace${stage !== "idle" ? " active" : ""}`}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 12 }}
      transition={{ duration: 0.4 }}
    >
      {/* LEFT: source image with animated boxes */}
      <div className="pane left-pane">
        <div className="pane-head">
          <h3>Source image</h3>
          <button className="ghost" onClick={onReset}>
            ↺ New image
          </button>
        </div>
        <div className="image-wrap">
          <div className="image-stage">
            <img ref={imageRef} className="source-image" src={image.url} alt="Uploaded document" />
            <BoundingBoxes ref={boxLayerRef} ocr={ocr} />
            {stage === "scanning" && <div className="scan-beam" />}
          </div>
        </div>
        <div className="pane-foot">
          {stage === "done" ? null : (
            <button className="cta" onClick={startAi} disabled={stage !== "idle"}>
              {stage === "idle" ? "✨ Start AI" : "Scanning…"}
            </button>
          )}
          <span className="status">{leftStatus}</span>
        </div>
      </div>

      {/* RIGHT: extracted text */}
      <motion.div
        className="pane right-pane"
        initial={{ flexBasis: 0, opacity: 0, marginLeft: 0 }}
        animate={{ flexBasis: "calc(50% - 9px)", opacity: 1, marginLeft: 18 }}
        transition={{ duration: 0.8, ease: [0.6, 0, 0.2, 1], opacity: { delay: 0.25 } }}
      >
        <div className="pane-head">
          <h3>Extracted text</h3>
          <span className="status">{rightStatus}</span>
        </div>

        {refinedText ? (
          <div className="text-flow refined" lang="pa">
            {refinedText}
          </div>
        ) : showSpans ? (
          <div className="text-flow" lang="pa">
            {ocr.words.map((word, i) => (
              <FlyTarget
                key={i}
                word={word}
                refFn={(el) => (spanRefs.current[i] = el)}
              />
            ))}
          </div>
        ) : (
          <div className="text-flow">
            <span className="placeholder">Press “Start AI” to read this image…</span>
          </div>
        )}

        {stage === "done" && (
          <ResultActions
            id={image.id}
            currentText={refinedText || ocr.full_text}
            refined={!!refinedText}
            onRefined={(t) => {
              setRefinedText(t);
              setRightStatus("Refined with AI ✓");
            }}
            onStatus={setRightStatus}
          />
        )}
      </motion.div>

      <div ref={flyLayerRef} className="fly-layer" aria-hidden="true" />
    </motion.section>
  );
}

function FlyTarget({ word, refFn }) {
  return (
    <>
      <span ref={refFn} className="w pending">
        {word.text}
      </span>
      {(word.suffix || " ").includes("\n") ? <br /> : " "}
    </>
  );
}

/* ---------- animation helpers ---------- */

function perimeter(box) {
  let total = 0;
  for (let i = 0; i < box.length; i++) {
    const [x1, y1] = box[i];
    const [x2, y2] = box[(i + 1) % box.length];
    total += Math.hypot(x2 - x1, y2 - y1);
  }
  return total;
}

function drawBoxes(svg, ocr) {
  if (!svg) return Promise.resolve();
  const polys = Array.from(svg.querySelectorAll(".word-box"));
  const n = polys.length;
  const stagger = clamp(2600 / Math.max(n, 1), 8, 45);

  polys.forEach((poly) => {
    const peri = perimeter(ocr.words[Number(poly.dataset.i)].box);
    poly.style.strokeDasharray = peri;
    poly.style.strokeDashoffset = peri;
    const delay = Math.round(i * stagger);
    poly.style.transition =
      `stroke-dashoffset .45s ease ${delay}ms, fill .4s ease ${delay + 250}ms, opacity .4s ease`;
  });
  svg.getBoundingClientRect(); // flush layout
  requestAnimationFrame(() => {
    polys.forEach((poly) => {
      poly.style.strokeDashoffset = 0;
      poly.classList.add("drawn");
    });
  });
  return wait(n * stagger + 700);
}

function flyWords({ ocr, image, spans, flyLayer }) {
  const n = ocr.words.length;
  const stagger = clamp(4200 / Math.max(n, 1), 12, 70);
  const flights = ocr.words.map(
    (word, i) =>
      new Promise((resolve) => {
        setTimeout(
          () => launchWord({ word, index: i, target: spans[i], image, flyLayer }).then(resolve),
          Math.round(i * stagger)
        );
      })
  );
  return Promise.all(flights);
}

function launchWord({ word, index, target, image, flyLayer }) {
  return new Promise((resolve) => {
    if (!target || !image || !flyLayer) return resolve();
    target.scrollIntoView({ block: "nearest" });
    const t = target.getBoundingClientRect();
    const r = image.getBoundingClientRect();

    let srcLeft, srcTop, srcH;
    if (word.box && word.box.length >= 3) {
      const sx = r.width / (image.naturalWidth || 1);
      const sy = r.height / (image.naturalHeight || 1);
      const xs = word.box.map((p) => p[0]);
      const ys = word.box.map((p) => p[1]);
      srcLeft = r.left + Math.min(...xs) * sx;
      srcTop = r.top + Math.min(...ys) * sy;
      srcH = (Math.max(...ys) - Math.min(...ys)) * sy || t.height;
    } else {
      // No boxes (vision-LLM OCR): the word lifts off a random spot on the image.
      srcLeft = r.left + r.width * (0.18 + Math.random() * 0.64);
      srcTop = r.top + r.height * (0.18 + Math.random() * 0.64);
      srcH = t.height * 1.5;
    }

    const fly = document.createElement("span");
    fly.className = "fly";
    fly.textContent = word.text;
    fly.style.left = `${t.left}px`;
    fly.style.top = `${t.top}px`;
    fly.style.fontSize = getComputedStyle(target).fontSize;
    flyLayer.appendChild(fly);

    const dx = srcLeft - t.left;
    const dy = srcTop - t.top;
    const scale = clamp(srcH / Math.max(t.height, 1), 0.4, 3);

    const poly = flyLayer.ownerDocument.querySelector(`.word-box[data-i="${index}"]`);
    if (poly) poly.classList.add("done");

    const anim = fly.animate(
      [
        { transform: `translate(${dx}px, ${dy}px) scale(${scale})`, opacity: 0.9 },
        { transform: "translate(0,0) scale(1)", opacity: 1 },
      ],
      { duration: 620, easing: "cubic-bezier(.2,.8,.25,1)" }
    );
    anim.onfinish = () => {
      fly.remove();
      target.classList.remove("pending");
      target.classList.add("landed");
      resolve();
    };
  });
}

function burstSparkles(flyLayer) {
  if (!flyLayer) return;
  for (let i = 0; i < 18; i++) {
    const s = document.createElement("span");
    s.className = "sparkle";
    s.style.left = `${55 + Math.random() * 40}vw`;
    s.style.top = `${20 + Math.random() * 60}vh`;
    s.style.animationDelay = `${Math.random() * 0.3}s`;
    flyLayer.appendChild(s);
    setTimeout(() => s.remove(), 1200);
  }
}
