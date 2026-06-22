"""OCR engines: Google Cloud Vision, Azure AI Vision, and an NVIDIA-hosted
vision LLM, via their REST APIs."""

import base64

import httpx
from fastapi import HTTPException

from . import layout

VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
AZURE_ANALYZE_PATH = "/computervision/imageanalysis:analyze"
AZURE_API_VERSION = "2023-10-01"

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_DEFAULT_MODEL = "google/diffusiongemma-26b-a4b-it"
NVIDIA_OCR_PROMPT = (
    "You are an OCR engine. Transcribe ALL text in this image exactly as it "
    "appears, preserving the original line breaks. The text is primarily Punjabi "
    "in the Gurmukhi script. If the page has multiple columns separated by "
    "vertical gaps or rules, read each column top-to-bottom and the columns "
    "left-to-right — never read straight across the columns. Output only the raw "
    "transcribed text — no translation, no transliteration, no commentary, no markdown."
)
# Structured, column-aware mode: ask the model to ground each line with a box.
NVIDIA_LAYOUT_PROMPT = (
    "You are a layout-aware OCR engine. The page is primarily Punjabi (Gurmukhi). "
    "It may have multiple columns separated by vertical gaps or rules. Return ONLY "
    "valid JSON (no markdown, no commentary) in exactly this schema:\n"
    '{"columns": [{"lines": [{"text": "<line text>", "box": [x0, y0, x1, y1]}]}]}\n'
    "Order columns left-to-right; within each column order lines top-to-bottom. "
    "Each box is the line's bounding box as integers on a 0-1000 grid relative to "
    "the image (x0,y0 = top-left corner, x1,y1 = bottom-right corner, origin at the "
    "image's top-left). Transcribe text exactly; do not translate or transliterate."
)

# How Vision's detectedBreak types translate into text between words.
_BREAKS = {
    "SPACE": " ",
    "SURE_SPACE": " ",
    "EOL_SURE_SPACE": "\n",
    "LINE_BREAK": "\n",
    "HYPHEN": "",
}


async def run_google_ocr(
    image_bytes: bytes,
    api_key: str,
    language_hints: tuple[str, ...] = ("pa",),
) -> dict:
    payload = {
        "requests": [
            {
                "image": {"content": base64.b64encode(image_bytes).decode()},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": list(language_hints)},
            }
        ]
    }
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(VISION_URL, params={"key": api_key}, json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Could not reach Google Vision: {exc}") from exc

    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code != 200:
        msg = body.get("error", {}).get("message", resp.text[:300])
        raise HTTPException(502, f"Google Vision error: {msg}")

    response = (body.get("responses") or [{}])[0]
    if "error" in response:
        raise HTTPException(502, f"Google Vision error: {response['error'].get('message')}")

    annotation = response.get("fullTextAnnotation")
    if not annotation:
        raise HTTPException(422, "No text was detected in this image.")

    return _parse_annotation(annotation)


async def run_azure_ocr(image_bytes: bytes, endpoint: str, api_key: str) -> dict:
    """Azure AI Vision Image Analysis 4.0 with the 'read' feature."""
    url = endpoint.rstrip("/") + AZURE_ANALYZE_PATH
    params = {"features": "read", "api-version": AZURE_API_VERSION}
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/octet-stream",
    }
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(url, params=params, headers=headers, content=image_bytes)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Could not reach Azure AI Vision: {exc}") from exc

    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code != 200:
        error = body.get("error", {})
        msg = error.get("message") or error.get("innererror", {}).get("message") or resp.text[:300]
        raise HTTPException(502, f"Azure AI Vision error: {msg}")

    read = body.get("readResult") or {}
    words = []
    lines_text = []
    for block in read.get("blocks") or []:
        for line in block.get("lines") or []:
            line_words = line.get("words") or []
            lines_text.append(line.get("text", ""))
            for i, word in enumerate(line_words):
                text = word.get("text", "")
                polygon = word.get("boundingPolygon") or []
                box = [[p.get("x", 0), p.get("y", 0)] for p in polygon]
                if not text or len(box) < 3:
                    continue
                suffix = "\n" if i == len(line_words) - 1 else " "
                words.append({"text": text, "suffix": suffix, "box": box})

    if not words:
        raise HTTPException(422, "No text was detected in this image.")

    metadata = body.get("metadata") or {}
    return {
        "width": metadata.get("width", 0),
        "height": metadata.get("height", 0),
        "words": words,
        "full_text": "\n".join(t for t in lines_text if t).strip(),
    }


def _downscale_for_api(image_bytes: bytes, max_side: int = 1500) -> bytes:
    """Shrink very large images before sending — cuts payload and latency. Boxes
    stay correct because the model returns normalized (0-1000) coordinates that
    we later scale to the original pixel size."""
    import io

    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            if max(img.size) <= max_side:
                return image_bytes
            img = img.convert("RGB")
            img.thumbnail((max_side, max_side))
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=88)
            return out.getvalue()
    except Exception:  # noqa: BLE001 — fall back to the original bytes
        return image_bytes


async def _nvidia_chat(image_bytes: bytes, api_key: str, model: str | None, prompt: str) -> str:
    """Send one image+prompt to the NVIDIA vision endpoint, return the reply text."""
    b64 = base64.b64encode(_downscale_for_api(image_bytes)).decode()
    payload = {
        "model": model or NVIDIA_DEFAULT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=240) as client:
            resp = await client.post(NVIDIA_URL, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Could not reach NVIDIA: {exc}") from exc

    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code != 200:
        msg = body.get("detail") or body.get("error", {}).get("message") or resp.text[:300]
        raise HTTPException(502, f"NVIDIA OCR error: {msg}")

    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError):
        raise HTTPException(502, "NVIDIA returned an unexpected response.")


async def run_nvidia_ocr(image_bytes: bytes, api_key: str, model: str | None = None) -> dict:
    """Plain OCR via an NVIDIA vision LLM — returns text only (boxes are None)."""
    text = await _nvidia_chat(image_bytes, api_key, model, NVIDIA_OCR_PROMPT)
    if not text:
        raise HTTPException(422, "No text was detected in this image.")
    return _words_from_text(text)


async def run_nvidia_ocr_structured(
    image_bytes: bytes,
    api_key: str,
    model: str | None,
    width: int,
    height: int,
) -> dict:
    """Column-aware OCR via the vision LLM with grounded line boxes.

    Asks the model for structured JSON (columns -> lines -> {text, box}) and maps
    the normalized boxes to pixels. Falls back to plain text OCR if the model's
    JSON is unusable, so a weak spatial response never breaks OCR entirely.
    """
    reply = await _nvidia_chat(image_bytes, api_key, model, NVIDIA_LAYOUT_PROMPT)

    # 1) clean, fully-valid JSON
    data = layout.extract_json(reply)
    if data:
        result = layout.words_from_vision_json(data, width, height)
        if result:
            return result

    # 2) broken/truncated JSON — salvage the well-formed lines (keeps boxes)
    salvaged = layout.words_from_salvaged(reply, width, height)
    if salvaged:
        return salvaged

    # 3) last resort: readable text only — NEVER surface raw JSON to the user
    text = layout.text_from_reply(reply)
    if text:
        return _words_from_text(text)
    raise HTTPException(422, "No text was detected in this image.")


def _words_from_text(text: str) -> dict:
    """Split a plain-text OCR result into box-less words for the front-end."""
    words = []
    lines = text.split("\n")
    for li, line in enumerate(lines):
        tokens = line.split()
        for ti, token in enumerate(tokens):
            last_in_line = ti == len(tokens) - 1
            suffix = "\n" if last_in_line and li < len(lines) - 1 else " "
            words.append({"text": token, "suffix": suffix, "box": None})
    return {"width": 0, "height": 0, "words": words, "full_text": text.strip()}


def _parse_annotation(annotation: dict) -> dict:
    words = []
    width = height = 0
    for page in annotation.get("pages", []):
        width = page.get("width", width)
        height = page.get("height", height)
        for block in page.get("blocks", []):
            for paragraph in block.get("paragraphs", []):
                for word in paragraph.get("words", []):
                    symbols = word.get("symbols", [])
                    text = "".join(s.get("text", "") for s in symbols)
                    if not text:
                        continue
                    suffix = " "
                    if symbols:
                        brk = (
                            symbols[-1]
                            .get("property", {})
                            .get("detectedBreak", {})
                            .get("type")
                        )
                        suffix = _BREAKS.get(brk, " ")
                    vertices = word.get("boundingBox", {}).get("vertices", [])
                    box = [[v.get("x", 0), v.get("y", 0)] for v in vertices]
                    if len(box) < 3:
                        continue
                    words.append({"text": text, "suffix": suffix, "box": box})
    return {
        "width": width,
        "height": height,
        "words": words,
        "full_text": annotation.get("text", "").strip(),
    }
