# Punjabi OCR Studio

A web app that reads Punjabi (Gurmukhi) text out of images using Google Cloud
Vision **or Azure AI Vision**, with animated bounding boxes, a "words fly
across" reveal, text download, and an optional AI cleanup step via OpenAI or
Azure OpenAI.

The frontend is a **React (Vite + framer-motion)** single-page app; the backend
is FastAPI. In production the backend serves the built React bundle, so a single
`uvicorn` process runs the whole thing.

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
    vision-capable model (default `meta/llama-3.2-11b-vision-instruct`). This
    engine runs OCR through a vision LLM and returns plain text **without word
    boxes** — words animate by lifting off the scanned image instead of from
    precise boxes.
- **AI text refinement** — optional, used by the "Refine with AI" button.
  Choose **OpenAI** (API key + model) or **Azure OpenAI** (endpoint, API key,
  deployment name, API version).

Keys are stored in the local `data.db` SQLite file and are only shown back
masked. The admin portal has no authentication — run this app locally or put
it behind your own auth before exposing it.

## Layout

```
app/
  main.py      FastAPI routes (upload, OCR, refine, download, admin, SPA serving)
  ocr.py       Google Cloud Vision + Azure AI Vision REST calls + word/box parsing
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
