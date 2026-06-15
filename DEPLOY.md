# Deploying for free: Cloudflare Pages (frontend) + Render (backend)

This guide deploys the React frontend to **Cloudflare Pages** and the FastAPI
backend to a **Render free web service**. Both have free tiers and no credit
card is required.

> Free-tier note: the Render service **sleeps after ~15 min of inactivity**, so
> the first request after idle takes ~30–60s to wake. The Cloudflare Pages
> frontend is always-on. The backend filesystem is ephemeral, so uploaded images
> and OCR history reset on redeploy — your API keys come from env vars, so that's
> fine.

There's a small chicken-and-egg (each side needs the other's URL), so deploy in
this order:

## 1. Backend → Render

1. Push this repo to GitHub (already done).
2. Go to <https://dashboard.render.com> → **New** → **Blueprint**, and pick this
   repo. Render reads [`render.yaml`](render.yaml) and creates the
   `punjabi-ocr-api` web service.
   - (Or **New → Web Service** manually: Runtime **Python 3**,
     Build `pip install -r requirements.txt`,
     Start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, Plan **Free**.)
3. Set the environment variables (Dashboard → the service → **Environment**):
   | Variable | Value |
   | --- | --- |
   | `NVIDIA_API_KEY` | your `nvapi-…` key |
   | `OCR_PROVIDER` | `nvidia` (already defaulted) |
   | `NVIDIA_MODEL` | `meta/llama-3.2-11b-vision-instruct` (already defaulted) |
   | `ADMIN_TOKEN` | a long random string (e.g. `openssl rand -hex 24`) |
   | `ALLOWED_ORIGINS` | `*` for now — tighten in step 3 |
4. Deploy. Note the service URL, e.g. `https://punjabi-ocr-api.onrender.com`.
   Open it — you should get a response (the legacy page or JSON), confirming it's
   up.

> Using Azure/Google/OpenAI instead of NVIDIA? Set those env vars too — see the
> table in [README.md](README.md#configure-keys-admin-portal).

## 2. Frontend → Cloudflare Pages

1. Go to <https://dash.cloudflare.com> → **Workers & Pages** → **Create** →
   **Pages** → **Connect to Git**, and pick this repo.
2. Build settings:
   - **Framework preset:** None (or Vite)
   - **Root directory:** `frontend`
   - **Build command:** `npm run build`
   - **Build output directory:** `dist`
3. Add a build environment variable:
   | Variable | Value |
   | --- | --- |
   | `VITE_API_BASE` | your Render URL from step 1, e.g. `https://punjabi-ocr-api.onrender.com` |
4. Deploy. Note the Pages URL, e.g. `https://punjabi-ocr.pages.dev`.
   SPA routing (`/admin`) is handled by `frontend/public/_redirects`.

## 3. Lock down CORS

Back on Render, set `ALLOWED_ORIGINS` to your exact Pages URL (no trailing
slash), e.g. `https://punjabi-ocr.pages.dev`, and redeploy. If you add a custom
domain, include it comma-separated.

## 4. Use it

- Open the Cloudflare Pages URL.
- Go to **/admin**, enter your `ADMIN_TOKEN` when prompted, and confirm the keys
  show as saved (they come from the Render env vars).
- Upload a Punjabi image and run OCR.

## Updating

Push to `main` → both Render and Cloudflare Pages auto-build and redeploy. After
changing frontend code, no manual rebuild is needed; Pages rebuilds from source.
