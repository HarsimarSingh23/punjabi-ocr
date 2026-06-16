# All-in-one image: build the React frontend, then serve it (plus the API) from
# FastAPI. One container, one URL — no CORS or separate frontend host needed.

# --- stage 1: build the frontend ---
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# VITE_API_BASE is empty -> the SPA calls the same origin that serves it
RUN npm run build

# --- stage 2: python backend serving the built SPA ---
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY --from=frontend /fe/dist ./frontend/dist

# Providers are configured via env vars (see README / DEPLOY.md).
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
