"""Punjabi OCR web app — FastAPI backend."""

import asyncio
import json
import logging
import os
import secrets
import uuid
from pathlib import Path

from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import db, layout, ocr, refine

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
DIST_DIR = ROOT / "frontend" / "dist"
UPLOAD_DIR = ROOT / "uploads"

# Declared content types we accept. The declaration only gates the request —
# the stored extension comes from the format Pillow actually finds in the bytes.
ALLOWED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
IMAGE_FORMAT_EXTS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
# HEIC/HEIF (common from iPhone cameras) is accepted and converted to JPEG.
HEIC_TYPES = {"image/heic", "image/heif", "image/heic-sequence", "image/heif-sequence"}
HEIC_EXTS = (".heic", ".heif")
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024

SECRET_KEYS = {
    "google_api_key",
    "azure_vision_key",
    "nvidia_api_key",
    "openai_api_key",
    "azure_api_key",
}
PLAIN_KEYS = {
    "ocr_provider",
    "page_columns",
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
VALID_PAGE_COLUMNS = {"", "auto", "1", "2", "3"}

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


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Gate the admin settings endpoints when ADMIN_TOKEN is configured.

    With no ADMIN_TOKEN set (e.g. local dev) the endpoints stay open; once it is
    set, callers must send a matching X-Admin-Token header.
    """
    expected = os.environ.get("ADMIN_TOKEN")
    if expected and not secrets.compare_digest(x_admin_token or "", expected):
        raise HTTPException(401, "Admin token required or invalid.")


def _spa_index() -> FileResponse:
    """Serve the built React app; fall back to the legacy static page in dev."""
    built = DIST_DIR / "index.html"
    if built.is_file():
        return FileResponse(built)
    legacy = STATIC_DIR / "index.html"
    if legacy.is_file():
        return FileResponse(legacy)
    raise HTTPException(
        503, "The web UI is not built. Run `npm run build` in frontend/, or use the API directly."
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return _spa_index()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    filename = (file.filename or "").lower()
    is_heic = content_type in HEIC_TYPES or filename.endswith(HEIC_EXTS)
    if content_type not in ALLOWED_IMAGE_TYPES and not is_heic:
        raise HTTPException(400, "Please upload a JPEG, PNG, WebP or HEIC image.")

    data = await _read_capped(file)
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")

    if is_heic:  # HEIC/HEIF — convert to JPEG so every OCR engine can read it
        data, ext = _heic_to_jpeg(data), ".jpg"
    else:
        # Trust the pixels, not the declared type: the extension decides how the
        # file is later served from /uploads, and a mislabelled upload would be
        # served under the wrong one.
        ext = _verified_extension(data)

    rid = uuid.uuid4().hex[:12]
    path = UPLOAD_DIR / f"{rid}{ext}"
    path.write_bytes(data)
    await asyncio.to_thread(db.create_result, rid, file.filename, str(path))
    return {"id": rid, "image_url": f"/uploads/{path.name}"}


async def _read_capped(file: UploadFile) -> bytes:
    """Read the upload in chunks, rejecting it as soon as it exceeds the cap.

    Reading first and measuring afterwards means a client can make the process
    buffer a file of any size before we ever look at it.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(CHUNK_BYTES):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(400, "Image is too large (max 12 MB).")
        chunks.append(chunk)
    return b"".join(chunks)


def _verified_extension(data: bytes) -> str:
    """Confirm the bytes really are a supported image, and return its extension."""
    import io

    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            img.verify()  # header/structure check; cheap, doesn't decode pixels
            fmt = (img.format or "").upper()
    except Exception as exc:  # noqa: BLE001 — any decode failure is a bad upload
        raise HTTPException(400, "This file isn't a readable image.") from exc

    ext = IMAGE_FORMAT_EXTS.get(fmt)
    if not ext:
        raise HTTPException(400, "Please upload a JPEG, PNG, WebP or HEIC image.")
    return ext


def _heic_to_jpeg(data: bytes) -> bytes:
    import io

    try:
        import pillow_heif
        from PIL import Image

        pillow_heif.register_heif_opener()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=92)
        return out.getvalue()
    except Exception as exc:  # noqa: BLE001 — surface a friendly message for any decode failure
        raise HTTPException(
            400, "Could not read this HEIC image. Please try a JPG or PNG instead."
        ) from exc


@app.post("/api/ocr/{rid}")
async def run_ocr(rid: str):
    row = await asyncio.to_thread(_get_result_or_404, rid)
    settings = await asyncio.to_thread(db.get_settings)
    image_bytes = await asyncio.to_thread(_read_upload, row["image_path"])
    columns = settings.get("page_columns") or "auto"  # auto | 1 | 2 | 3

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
        result = layout.reading_order_from_boxes(result, columns)
    elif provider == "nvidia":
        api_key = settings.get("nvidia_api_key")
        if not api_key:
            raise HTTPException(
                400,
                "NVIDIA API key is not configured. Open the Admin portal (/admin) to set it.",
            )
        model = settings.get("nvidia_model")
        result = await ocr.run_nvidia_ocr_cv(image_bytes, api_key, model)
        result = layout.reading_order_from_boxes(result, columns)
    else:
        api_key = settings.get("google_api_key")
        if not api_key:
            raise HTTPException(
                400,
                "Google Vision API key is not configured. Open the Admin portal (/admin) to set it.",
            )
        result = await ocr.run_google_ocr(image_bytes, api_key)
        result = layout.reading_order_from_boxes(result, columns)
    await asyncio.to_thread(
        db.save_ocr, rid, json.dumps(result, ensure_ascii=False), result["full_text"]
    )
    return result


def _read_upload(image_path: str) -> bytes:
    """Read a stored upload, or explain that it is gone.

    Uploads live on local disk, which on an ephemeral host (Render, Fly, a
    container restart) does not survive a redeploy while the database row does.
    Without this the endpoint raises FileNotFoundError and returns an opaque 500.
    """
    try:
        return Path(image_path).read_bytes()
    except OSError as exc:
        raise HTTPException(
            410, "The uploaded image is no longer on the server. Please upload it again."
        ) from exc


@app.post("/api/refine/{rid}")
async def refine_result(rid: str):
    row = await asyncio.to_thread(_get_result_or_404, rid)
    if not row["full_text"]:
        raise HTTPException(400, "Run OCR on this image first.")
    settings = await asyncio.to_thread(db.get_settings)
    refined = await refine.refine_text(row["full_text"], settings)
    await asyncio.to_thread(db.save_refined, rid, refined)
    return {"refined_text": refined}


@app.get("/api/results/{rid}")
def get_result(rid: str):
    row = _get_result_or_404(rid)
    return {
        "id": row["id"],
        "filename": row["filename"],
        "full_text": row["full_text"],
        "refined_text": row["refined_text"],
        "ocr": _parse_ocr_json(row["ocr_json"]),
        "created_at": row["created_at"],
    }


def _parse_ocr_json(raw: str | None):
    """Stored OCR payload, or None if the row is missing/corrupt.

    A single unparseable row shouldn't 500 the endpoint that reads it — the
    caller can still show the plain text and re-run OCR.
    """
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        log.warning("stored ocr_json is not valid JSON; returning null")
        return None


@app.get("/api/results/{rid}/download")
def download_result(rid: str, refined: bool = False):
    row = _get_result_or_404(rid)
    text = (row["refined_text"] if refined else None) or row["full_text"]
    if not text:
        raise HTTPException(404, "No OCR text available for this image yet.")
    headers = {"Content-Disposition": f'attachment; filename="punjabi-ocr-{rid}.txt"'}
    return PlainTextResponse(text, headers=headers)


@app.get("/api/admin/settings", dependencies=[Depends(require_admin)])
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


@app.post("/api/admin/settings", dependencies=[Depends(require_admin)])
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
            raise HTTPException(400, "ocr_provider must be 'google', 'azure', 'nvidia' or empty.")
        if key == "page_columns" and value not in VALID_PAGE_COLUMNS:
            raise HTTPException(400, "page_columns must be 'auto', '1', '2', '3' or empty.")
        updates[key] = value
    if updates:
        db.set_settings(updates)
    return {"ok": True, "updated": sorted(updates)}


def _get_result_or_404(rid: str):
    row = db.get_result(rid)
    if not row:
        raise HTTPException(404, "Unknown result id. Upload the image again.")
    return row


if STATIC_DIR.is_dir():  # optional: a slim deploy may ship only the built SPA
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
    candidate = _safe_dist_file(full_path)
    if candidate is not None:
        return FileResponse(candidate)
    return _spa_index()


def _safe_dist_file(rel_path: str) -> Path | None:
    """Resolve ``rel_path`` inside the built SPA, or None if it isn't a file there.

    The path segment is attacker-controlled and arrives percent-decoded, so
    ``DIST_DIR / rel_path`` alone would happily walk out of the build directory
    ("../../.." reaches the repo root, and data.db with it). Resolve first, then
    require the result to still live under DIST_DIR.
    """
    if not rel_path:
        return None
    try:
        resolved = (DIST_DIR / rel_path).resolve()
        root = DIST_DIR.resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(root) or resolved == root:
        return None
    return resolved if resolved.is_file() else None
