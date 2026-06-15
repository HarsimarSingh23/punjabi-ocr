"""Punjabi OCR web app — FastAPI backend."""

import json
import os
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import db, ocr, refine

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
DIST_DIR = ROOT / "frontend" / "dist"
UPLOAD_DIR = ROOT / "uploads"

ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

SECRET_KEYS = {
    "google_api_key",
    "azure_vision_key",
    "nvidia_api_key",
    "openai_api_key",
    "azure_api_key",
}
PLAIN_KEYS = {
    "ocr_provider",
    "azure_vision_endpoint",
    "nvidia_model",
    "ai_provider",
    "openai_model",
    "azure_endpoint",
    "azure_deployment",
    "azure_api_version",
}
VALID_AI_PROVIDERS = {"", "openai", "azure"}
VALID_OCR_PROVIDERS = {"", "google", "azure", "nvidia"}

app = FastAPI(title="Punjabi OCR")

# When the frontend is hosted separately (e.g. Cloudflare Pages), set
# ALLOWED_ORIGINS to its URL(s), comma-separated. Defaults to "*" since the API
# carries no cookies/credentials.
_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init()
UPLOAD_DIR.mkdir(exist_ok=True)


def _spa_index() -> FileResponse:
    """Serve the built React app; fall back to the legacy static page in dev."""
    built = DIST_DIR / "index.html"
    if built.exists():
        return FileResponse(built)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return _spa_index()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = ALLOWED_IMAGE_TYPES.get(file.content_type or "")
    if not ext:
        raise HTTPException(400, "Please upload a JPEG, PNG or WebP image.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "Image is too large (max 12 MB).")
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")

    rid = uuid.uuid4().hex[:12]
    path = UPLOAD_DIR / f"{rid}{ext}"
    path.write_bytes(data)
    db.create_result(rid, file.filename, str(path))
    return {"id": rid, "image_url": f"/uploads/{path.name}"}


@app.post("/api/ocr/{rid}")
async def run_ocr(rid: str):
    row = _get_result_or_404(rid)
    settings = db.get_settings()
    image_bytes = Path(row["image_path"]).read_bytes()

    provider = settings.get("ocr_provider") or "google"
    if provider == "azure":
        endpoint = settings.get("azure_vision_endpoint")
        api_key = settings.get("azure_vision_key")
        if not (endpoint and api_key):
            raise HTTPException(
                400,
                "Azure AI Vision endpoint and key are not configured. "
                "Open the Admin portal (/admin) to set them.",
            )
        result = await ocr.run_azure_ocr(image_bytes, endpoint, api_key)
    elif provider == "nvidia":
        api_key = settings.get("nvidia_api_key")
        if not api_key:
            raise HTTPException(
                400,
                "NVIDIA API key is not configured. Open the Admin portal (/admin) to set it.",
            )
        result = await ocr.run_nvidia_ocr(image_bytes, api_key, settings.get("nvidia_model"))
    else:
        api_key = settings.get("google_api_key")
        if not api_key:
            raise HTTPException(
                400,
                "Google Vision API key is not configured. Open the Admin portal (/admin) to set it.",
            )
        result = await ocr.run_google_ocr(image_bytes, api_key)
    db.save_ocr(rid, json.dumps(result, ensure_ascii=False), result["full_text"])
    return result


@app.post("/api/refine/{rid}")
async def refine_result(rid: str):
    row = _get_result_or_404(rid)
    if not row["full_text"]:
        raise HTTPException(400, "Run OCR on this image first.")
    refined = await refine.refine_text(row["full_text"], db.get_settings())
    db.save_refined(rid, refined)
    return {"refined_text": refined}


@app.get("/api/results/{rid}")
def get_result(rid: str):
    row = _get_result_or_404(rid)
    return {
        "id": row["id"],
        "filename": row["filename"],
        "full_text": row["full_text"],
        "refined_text": row["refined_text"],
        "ocr": json.loads(row["ocr_json"]) if row["ocr_json"] else None,
        "created_at": row["created_at"],
    }


@app.get("/api/results/{rid}/download")
def download_result(rid: str, refined: bool = False):
    row = _get_result_or_404(rid)
    text = (row["refined_text"] if refined else None) or row["full_text"]
    if not text:
        raise HTTPException(404, "No OCR text available for this image yet.")
    headers = {"Content-Disposition": f'attachment; filename="punjabi-ocr-{rid}.txt"'}
    return PlainTextResponse(text, headers=headers)


@app.get("/api/admin/settings")
def get_admin_settings():
    settings = db.get_settings()
    out: dict[str, object] = {}
    for key in SECRET_KEYS:
        value = settings.get(key) or ""
        out[key] = {
            "set": bool(value),
            "hint": f"•••• {value[-4:]}" if len(value) >= 8 else ("set" if value else ""),
        }
    for key in PLAIN_KEYS:
        out[key] = settings.get(key) or ""
    return out


@app.post("/api/admin/settings")
def update_admin_settings(payload: dict = Body(...)):
    updates: dict[str, str] = {}
    for key, value in payload.items():
        if key not in SECRET_KEYS | PLAIN_KEYS or not isinstance(value, str):
            continue
        value = value.strip()
        if key in SECRET_KEYS and not value:
            continue  # empty secret field means "leave unchanged"
        if key == "ai_provider" and value not in VALID_AI_PROVIDERS:
            raise HTTPException(400, "ai_provider must be 'openai', 'azure' or empty.")
        if key == "ocr_provider" and value not in VALID_OCR_PROVIDERS:
            raise HTTPException(400, "ocr_provider must be 'google', 'azure' or empty.")
        updates[key] = value
    if updates:
        db.set_settings(updates)
    return {"ok": True, "updated": sorted(updates)}


def _get_result_or_404(rid: str):
    row = db.get_result(rid)
    if not row:
        raise HTTPException(404, "Unknown result id. Upload the image again.")
    return row


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Built React assets (Vite emits /assets/*). Mounted only when a build exists.
if (DIST_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str) -> FileResponse:
    """Client-side routes (e.g. /admin) resolve to the SPA shell."""
    if full_path.startswith(("api/", "uploads/", "assets/", "static/")):
        raise HTTPException(404, "Not found")
    candidate = DIST_DIR / full_path
    if full_path and candidate.is_file():
        return FileResponse(candidate)
    return _spa_index()
