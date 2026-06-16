# Deploying for free on Render (all-in-one)

This deploys the **whole app** — React frontend + FastAPI backend — as a single
free Render web service. The Docker image builds the frontend and FastAPI serves
it, so there's one service, one URL, and no CORS to configure.

> Free-tier notes: the service **sleeps after ~15 min of inactivity**, so the
> first visit after idle takes ~30–60s to wake. The filesystem is ephemeral, so
> uploaded images and OCR history reset on redeploy — your API keys come from
> env vars, so that's fine.

## Steps

1. **Push to GitHub** (already done for this repo).

2. **Create the service from the blueprint.**
   Go to <https://dashboard.render.com> → **New** → **Blueprint** → select this
   repo. Render reads [`render.yaml`](render.yaml) and creates the `punjabi-ocr`
   web service (Docker, free plan).

   *Prefer to do it by hand?* **New → Web Service** → connect the repo →
   **Runtime: Docker** → **Plan: Free**. Render uses the [`Dockerfile`](Dockerfile)
   automatically.

3. **Set environment variables** (the service → **Environment**):

   | Variable | Value |
   | --- | --- |
   | `NVIDIA_API_KEY` | your `nvapi-…` key |
   | `ADMIN_TOKEN` | a long random string — `openssl rand -hex 24` |
   | `OCR_PROVIDER` | `nvidia` *(already defaulted by the blueprint)* |
   | `NVIDIA_MODEL` | `meta/llama-3.2-11b-vision-instruct` *(already defaulted)* |

   Using Azure/Google/OpenAI instead? Add those vars too — see the table in
   [README.md](README.md#configure-keys-admin-portal). No `ALLOWED_ORIGINS` or
   `VITE_API_BASE` is needed for the all-in-one setup.

4. **Deploy.** First build takes a few minutes (it installs npm + pip deps and
   builds the SPA). When it's live, open the service URL, e.g.
   `https://punjabi-ocr.onrender.com`.

5. **Use it.**
   - The home page loads the React app.
   - Go to **/admin**, enter your `ADMIN_TOKEN` when prompted — the keys show as
     saved (they come from the env vars).
   - Upload a Punjabi image and run OCR.

## Updating

Push to `main` → Render auto-builds and redeploys. No manual rebuild needed; the
Docker build recompiles the frontend each time.

## Want the frontend always-on (no cold start on first load)?

Split it instead: host the frontend on Cloudflare Pages (always-on CDN) and keep
the backend on Render. That setup is also supported — see the
"Deployment (frontend on Cloudflare Pages, backend elsewhere)" section in
[README.md](README.md). The trade-off is two services to manage instead of one.
