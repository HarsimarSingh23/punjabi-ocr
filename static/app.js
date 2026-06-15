/* Punjabi OCR Studio — front-end flow:
   upload → split panes → Start AI → draw boxes → fly words → download/refine. */

const $ = (sel) => document.querySelector(sel);
const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  id: null,
  data: null,        // OCR payload {width, height, words[], full_text}
  refinedText: null,
};

const els = {
  uploadPane: $("#upload-pane"),
  dropzone: $("#dropzone"),
  fileInput: $("#file-input"),
  workspace: $("#workspace"),
  img: $("#source-image"),
  boxLayer: $("#box-layer"),
  scanBeam: $("#scan-beam"),
  startBtn: $("#start-ai"),
  leftStatus: $("#left-status"),
  rightStatus: $("#right-status"),
  textFlow: $("#text-flow"),
  actions: $("#result-actions"),
  flyLayer: $("#fly-layer"),
  toast: $("#toast"),
};

/* ---------- helpers ---------- */

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

let toastTimer;
function toast(message) {
  els.toast.textContent = message;
  els.toast.hidden = false;
  requestAnimationFrame(() => els.toast.classList.add("show"));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    els.toast.classList.remove("show");
    toastTimer = setTimeout(() => (els.toast.hidden = true), 300);
  }, 4200);
}

async function errorMessage(res) {
  try {
    const body = await res.json();
    if (body.detail) return body.detail;
  } catch (_) { /* not JSON */ }
  return `Request failed (${res.status})`;
}

function perimeter(box) {
  let total = 0;
  for (let i = 0; i < box.length; i++) {
    const [x1, y1] = box[i];
    const [x2, y2] = box[(i + 1) % box.length];
    total += Math.hypot(x2 - x1, y2 - y1);
  }
  return total;
}

/* ---------- step 1: upload ---------- */

function bindUpload() {
  els.dropzone.addEventListener("click", () => els.fileInput.click());
  els.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") els.fileInput.click();
  });
  $("#browse-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    els.fileInput.click();
  });
  els.dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.dropzone.classList.add("over");
  });
  els.dropzone.addEventListener("dragleave", () => els.dropzone.classList.remove("over"));
  els.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("over");
    handleFile(e.dataTransfer.files[0]);
  });
  els.fileInput.addEventListener("change", () => handleFile(els.fileInput.files[0]));
}

async function handleFile(file) {
  if (!file) return;
  if (!/^image\/(png|jpe?g|webp)$/.test(file.type)) {
    toast("Please choose a JPG, PNG or WebP image.");
    return;
  }
  els.dropzone.classList.add("busy");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await errorMessage(res));
    const { id, image_url } = await res.json();
    state.id = id;
    await new Promise((ok, bad) => {
      els.img.onload = ok;
      els.img.onerror = () => bad(new Error("Could not display the image."));
      els.img.src = image_url;
    });
    splitPanes();
  } catch (err) {
    toast(err.message);
    els.dropzone.classList.remove("busy");
  }
}

/* ---------- step 2: one pane becomes two ---------- */

function splitPanes() {
  els.uploadPane.classList.add("leaving");
  setTimeout(() => {
    els.uploadPane.hidden = true;
    els.workspace.hidden = false;
    els.textFlow.innerHTML =
      '<span class="placeholder">Press “Start AI” to read this image…</span>';
    // double rAF so the collapsed right pane paints once before the split animates
    requestAnimationFrame(() =>
      requestAnimationFrame(() => document.body.classList.add("split"))
    );
  }, 380);
}

/* ---------- step 3–6: Start AI → OCR → animations ---------- */

async function startAi() {
  els.startBtn.disabled = true;
  els.startBtn.textContent = "Scanning…";
  els.scanBeam.hidden = false;
  setStatus(els.leftStatus, "Reading the image with AI…");

  let data;
  try {
    const res = await fetch(`/api/ocr/${state.id}`, { method: "POST" });
    if (!res.ok) throw new Error(await errorMessage(res));
    data = await res.json();
  } catch (err) {
    toast(err.message);
    els.scanBeam.hidden = true;
    els.startBtn.disabled = false;
    els.startBtn.textContent = "✨ Start AI";
    setStatus(els.leftStatus, "");
    return;
  }

  els.scanBeam.hidden = true;
  setStatus(els.leftStatus, `${data.words.length} words found`);
  await playResults(data);
}

async function playResults(data) {
  state.data = data;
  state.refinedText = null;
  els.textFlow.classList.remove("refined");

  setStatus(els.rightStatus, "");
  await drawBoxes(data);
  setStatus(els.rightStatus, "Transferring words…");
  await flyWords(data);

  els.startBtn.hidden = true;
  els.actions.hidden = false;
  setStatus(els.rightStatus, "Saved to backend ✓");
}
window.__play = playResults; // dev hook for trying the animations with mock data

/* Draw each word's bounding box as an animated SVG stroke. */
function drawBoxes(data) {
  const vw = data.width || els.img.naturalWidth;
  const vh = data.height || els.img.naturalHeight;
  els.boxLayer.setAttribute("viewBox", `0 0 ${vw} ${vh}`);
  els.boxLayer.innerHTML = "";

  const n = data.words.length;
  const stagger = clamp(2600 / Math.max(n, 1), 8, 45);

  data.words.forEach((word, i) => {
    const poly = document.createElementNS(SVG_NS, "polygon");
    poly.setAttribute("points", word.box.map((p) => p.join(",")).join(" "));
    poly.classList.add("word-box");
    const peri = perimeter(word.box);
    poly.style.strokeDasharray = peri;
    poly.style.strokeDashoffset = peri;
    const delay = Math.round(i * stagger);
    poly.style.transition =
      `stroke-dashoffset .45s ease ${delay}ms, ` +
      `fill .4s ease ${delay + 250}ms, opacity .4s ease`;
    els.boxLayer.appendChild(poly);
    word._poly = poly;
  });

  els.boxLayer.getBoundingClientRect(); // flush layout so transitions fire
  requestAnimationFrame(() => {
    data.words.forEach((word) => {
      word._poly.style.strokeDashoffset = 0;
      word._poly.classList.add("drawn");
    });
  });

  return wait(n * stagger + 700);
}

/* Fly every word from its box on the image to its slot in the text pane. */
async function flyWords(data) {
  els.textFlow.innerHTML = "";
  const spans = [];
  const frag = document.createDocumentFragment();
  data.words.forEach((word) => {
    const span = document.createElement("span");
    span.className = "w pending";
    span.textContent = word.text;
    frag.appendChild(span);
    spans.push(span);
    if ((word.suffix || " ").includes("\n")) frag.appendChild(document.createElement("br"));
    else frag.appendChild(document.createTextNode(" "));
  });
  els.textFlow.appendChild(frag);

  const n = data.words.length;
  const stagger = clamp(4200 / Math.max(n, 1), 12, 70);
  const flights = data.words.map(
    (word, i) =>
      new Promise((resolve) => {
        setTimeout(() => launchWord(word, spans[i], data).then(resolve), Math.round(i * stagger));
      })
  );
  await Promise.all(flights);
}

function launchWord(word, target, data) {
  return new Promise((resolve) => {
    // keep the landing slot visible, then measure it (scroll shifts coordinates)
    target.scrollIntoView({ block: "nearest" });
    const t = target.getBoundingClientRect();
    const r = els.img.getBoundingClientRect();
    const sx = r.width / (data.width || els.img.naturalWidth);
    const sy = r.height / (data.height || els.img.naturalHeight);

    const xs = word.box.map((p) => p[0]);
    const ys = word.box.map((p) => p[1]);
    const srcLeft = r.left + Math.min(...xs) * sx;
    const srcTop = r.top + Math.min(...ys) * sy;
    const srcH = (Math.max(...ys) - Math.min(...ys)) * sy || t.height;

    const fly = document.createElement("span");
    fly.className = "fly";
    fly.textContent = word.text;
    fly.style.left = `${t.left}px`;
    fly.style.top = `${t.top}px`;
    fly.style.fontSize = getComputedStyle(target).fontSize;
    els.flyLayer.appendChild(fly);

    const dx = srcLeft - t.left;
    const dy = srcTop - t.top;
    const scale = clamp(srcH / Math.max(t.height, 1), 0.4, 3);

    if (word._poly) word._poly.classList.add("done");

    const anim = fly.animate(
      [
        { transform: `translate(${dx}px, ${dy}px) scale(${scale})`, opacity: 0.9 },
        { transform: "translate(0, 0) scale(1)", opacity: 1 },
      ],
      { duration: 620, easing: "cubic-bezier(.2, .8, .25, 1)" }
    );
    anim.onfinish = () => {
      fly.remove();
      target.classList.remove("pending");
      target.classList.add("landed");
      resolve();
    };
  });
}

/* ---------- step 9: download / copy / refine ---------- */

function currentText() {
  return state.refinedText || (state.data && state.data.full_text) || "";
}

function bindActions() {
  els.startBtn.addEventListener("click", startAi);
  $("#reset-btn").addEventListener("click", () => location.reload());

  $("#download-btn").addEventListener("click", () => {
    const refined = state.refinedText ? "?refined=true" : "";
    window.location.href = `/api/results/${state.id}/download${refined}`;
  });

  $("#copy-btn").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(currentText());
      toast("Copied to clipboard ✓");
    } catch (_) {
      toast("Could not access the clipboard.");
    }
  });

  $("#refine-btn").addEventListener("click", refineText);
}

async function refineText() {
  const btn = $("#refine-btn");
  btn.disabled = true;
  btn.textContent = "Refining…";
  setStatus(els.rightStatus, "Asking the AI to clean up the text…");
  try {
    const res = await fetch(`/api/refine/${state.id}`, { method: "POST" });
    if (!res.ok) throw new Error(await errorMessage(res));
    const { refined_text } = await res.json();
    state.refinedText = refined_text;
    els.textFlow.classList.add("refined");
    els.textFlow.textContent = refined_text;
    setStatus(els.rightStatus, "Refined with AI ✓");
  } catch (err) {
    toast(err.message);
    setStatus(els.rightStatus, "Saved to backend ✓");
  }
  btn.disabled = false;
  btn.textContent = "🪄 Refine with AI";
}

function setStatus(el, text) {
  el.textContent = text;
}

bindUpload();
bindActions();
