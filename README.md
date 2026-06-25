# Punjabi OCR Studio

A web app that reads Punjabi (Gurmukhi) text out of images using Google Cloud
Vision **or Azure AI Vision**, with animated bounding boxes, a "words fly
across" reveal, text download, and an optional AI cleanup step via OpenAI or
Azure OpenAI.

The frontend is a **React (Vite + framer-motion)** single-page app; the backend
is FastAPI. In production the backend serves the built React bundle, so a single
`uvicorn` process runs the whole thing.

## Deploy

One-click deploy to Render (free tier, all-in-one):

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/HarsimarSingh23/punjabi-ocr)

After it spins up, set `NVIDIA_API_KEY` and `ADMIN_TOKEN` in the Render
dashboard. Full walkthrough (and the Cloudflare Pages split option) in
**[DEPLOY.md](DEPLOY.md)**.

## How it works

1. Drop an image into the upload pane — it splits into two panes.
2. Press **✨ Start AI** in the left pane: the backend calls Google Cloud
   Vision (`DOCUMENT_TEXT_DETECTION`, Punjabi language hint) and the detected
   word bounding boxes are drawn over the image with a staggered stroke
   animation.
3. Each word then flies from its box on the image into its place in the text
   pane on the right.
4. The OCR result is stored in the backend (SQLite). You can download the text
   as a `.txt`, copy it, or run **🪄 Refine with AI** to fix OCR mistakes via
   the configured OpenAI / Azure OpenAI model.

## Example: OpenCV box detection + per-line NVIDIA OCR

Multi-column pages (dictionaries, newspapers) must be read one column at a time,
not straight across. With the NVIDIA engine, **OpenCV finds every text line's
bounding box deterministically** (threshold → connected components → drop
duplicate/overlapping boxes → merge nearby fragments into line boxes) — the
model is never asked to invent coordinates. Each line is then cropped and OCR'd
**in parallel**, and `page_columns` (`auto`/`2`/`3`) reorders the lines into
correct reading order — left column fully, then the right column.

Input — a two-column Punjabi (Gurmukhi) dictionary page:

![Example two-column dictionary page](docs/example-dictionary-page.png)

Running `app/cvboxes.py` on this page finds **109 line boxes** — cleanly
wrapping each line of text in both columns while skipping the illustrations and
the column-divider rule. Each box is cropped and sent to the NVIDIA-hosted
`meta/llama-4-maverick-17b-128e-instruct` vision model in parallel
(`ocr.run_nvidia_ocr_cv`); the boxes and recognized text look like this:

![Detected boxes with OCR'd text overlaid](docs/cv-pipeline-example.jpg)

> Notes:
> - Detecting boxes with classical image processing (not the model) means box
>   geometry is exact and free — no model call can hallucinate or omit a
>   coordinate. The model only has to *read* an already-isolated line.
> - The pipeline filters out non-text page furniture before it can contaminate
>   merging: a hole-filter (`RETR_CCOMP`) drops the spurious "interior of the
>   page border" contour that `RETR_EXTERNAL` would otherwise swallow every
>   glyph into, and a size-outlier filter drops thin column-divider rules and
>   illustrations/photos.
> - Raw text still has OCR errors (it's a general vision model, not a dedicated
>   Gurmukhi OCR engine) — use **Refine with AI**, or switch the OCR engine to
>   Google Cloud Vision for more literal text with the same column-aware
>   reordering.
> - Reproduce locally: `.venv/bin/python scripts/test_cv_pipeline.py <image> <columns>`
>   — logs every box + OCR'd text to `logs/bounding_boxes.log` and writes the
>   annotated image to `docs/cv-pipeline-example.jpg`. For Google/Azure (which
>   return boxes natively), use `scripts/test_columns.py` instead.

## Setup

```bash
# backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# frontend (build once; FastAPI serves the result)
cd frontend && npm install && npm run build && cd ..

# run
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000>.

### Frontend development (hot reload)

For live-reloading UI work, run the Vite dev server alongside the backend. It
proxies `/api` and `/uploads` to FastAPI on port 8000:

```bash
.venv/bin/uvicorn app.main:app --port 8000   # terminal 1
cd frontend && npm run dev                    # terminal 2 → http://localhost:5173
```

Run `npm run build` again when you want the FastAPI-served bundle (port 8000) to
pick up your changes.

## Configure keys (Admin portal)

Open <http://localhost:8000/admin>:

- **OCR engine** — pick one:
  - *Google Cloud Vision*: an API key from the
    [Google Cloud console](https://console.cloud.google.com/apis/credentials)
    with the Cloud Vision API enabled.
  - *Azure AI Vision*: the endpoint + key of an Azure AI Services or Computer
    Vision resource (uses the Image Analysis 4.0 Read API).
  - *NVIDIA vision*: an `nvapi-…` key from
    [integrate.api.nvidia.com](https://integrate.api.nvidia.com) plus a
    vision-capable model (default `meta/llama-4-maverick-17b-128e-instruct`).
    OpenCV detects each text line's bounding box on the server (see the example
    below), then every line is cropped and OCR'd through the vision LLM in
    parallel — so this engine *does* return precise word boxes, despite using a
    general vision model rather than a dedicated OCR API.
- **AI text refinement** — optional, used by the "Refine with AI" button.
  Choose **OpenAI** (API key + model) or **Azure OpenAI** (endpoint, API key,
  deployment name, API version).

Keys are stored in the local `data.db` SQLite file and are only shown back
masked.

**Protecting the admin portal:** set an `ADMIN_TOKEN` environment variable on the
backend. When set, the admin settings endpoints require an `X-Admin-Token`
header, and the admin page shows a lock screen that prompts for the token once
per browser session. With `ADMIN_TOKEN` unset (local dev) the portal stays open.
Always set it when the backend is publicly reachable.

You can also supply any setting through an **environment variable** (the env var
name is the UPPERCASE setting key). DB values from the admin portal take
precedence; env vars fill in the rest — handy for hosted deploys where the DB is
ephemeral:

| Env var | Purpose |
| --- | --- |
| `OCR_PROVIDER` | `google`, `azure`, or `nvidia` |
| `AI_PROVIDER` | `openai`, `azure`, or empty |
| `GOOGLE_API_KEY` | Google Cloud Vision key |
| `AZURE_VISION_ENDPOINT`, `AZURE_VISION_KEY` | Azure AI Vision |
| `NVIDIA_API_KEY`, `NVIDIA_MODEL` | NVIDIA vision OCR |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | OpenAI refinement |
| `AZURE_ENDPOINT`, `AZURE_API_KEY`, `AZURE_DEPLOYMENT`, `AZURE_API_VERSION` | Azure OpenAI refinement |
| `ADMIN_TOKEN` | If set, the admin portal requires this token (recommended in production) |

## Deployment (frontend on Cloudflare Pages, backend elsewhere)

For a free, step-by-step walkthrough (Cloudflare Pages + Render free tier with a
`render.yaml` blueprint), see **[DEPLOY.md](DEPLOY.md)**.

The frontend and backend can be hosted separately.

**Backend** (Render / Railway / Fly.io / any Docker host) — a `Dockerfile` and
`Procfile` are included:

- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Set the provider env vars above, plus `ALLOWED_ORIGINS` = your Pages URL
  (e.g. `https://punjabi-ocr.pages.dev`) so the browser can call the API, and
  `ADMIN_TOKEN` to lock the admin portal.
- Note: the container filesystem is ephemeral, so `uploads/` and `data.db` reset
  on redeploy — fine, since keys come from env vars. Add a persistent disk if you
  want uploads/OCR history to survive restarts.

**Frontend** (Cloudflare Pages):

- Project root: `frontend`  ·  Build command: `npm run build`  ·  Output dir: `dist`
- Build-time env var `VITE_API_BASE` = your backend URL
  (e.g. `https://punjabi-ocr-api.onrender.com`).
- SPA routing is handled by `frontend/public/_redirects` (`/* → /index.html`).

With `VITE_API_BASE` empty (the default), the frontend uses relative paths, so
the single-server setup (FastAPI serving `frontend/dist`) keeps working
unchanged for local development.

## Layout

```
app/
  main.py      FastAPI routes (upload, OCR, refine, download, admin, SPA serving)
  ocr.py       Google Cloud Vision + Azure AI Vision + NVIDIA vision LLM REST calls
  cvboxes.py   OpenCV text-line box detection for the NVIDIA engine (no model call)
  layout.py    column-aware reading-order reconstruction from word/line boxes
  refine.py    OpenAI / Azure OpenAI chat-completion cleanup
  db.py        SQLite storage (settings + results)
frontend/
  src/
    App.jsx               app shell + upload→workspace transition
    components/
      UploadPane.jsx      drag-and-drop uploader
      Workspace.jsx       two panes + box-draw + flying-words orchestration
      BoundingBoxes.jsx   animated SVG word boxes
      ResultActions.jsx   download / copy / refine
      AdminPage.jsx       API keys & provider settings
      AuroraBackground.jsx, TopBar.jsx
    lib/api.js            backend client
    lib/useToast.jsx      toast notifications
  dist/        built bundle served by FastAPI (created by `npm run build`)
static/        legacy vanilla-JS frontend (superseded by frontend/, kept as a fallback)
uploads/       uploaded images (created at runtime)
data.db        SQLite database (created at runtime)
```
