# Backend-only image (the frontend deploys separately to Cloudflare Pages).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static

# Configure providers via env vars (see README). Optionally set ALLOWED_ORIGINS
# to your Cloudflare Pages URL.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
