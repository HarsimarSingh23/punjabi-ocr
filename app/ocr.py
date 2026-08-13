"""OCR engines: Google Cloud Vision, Azure AI Vision, and an NVIDIA-hosted
vision LLM, via their REST APIs."""

import asyncio
import base64
import logging
import random

import httpx
from fastapi import HTTPException

from . import cvboxes, layout

log = logging.getLogger(__name__)

VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
AZURE_ANALYZE_PATH = "/computervision/imageanalysis:analyze"
AZURE_API_VERSION = "2023-10-01"

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_DEFAULT_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
NVIDIA_OCR_PROMPT = (
    "You are an OCR engine. Transcribe ALL text in this image exactly as it "
    "appears, preserving the original line breaks. The text is primarily Punjabi "
    "in the Gurmukhi script. If the page has multiple columns separated by "
    "vertical gaps or rules, read each column top-to-bottom and the columns "
    "left-to-right — never read straight across the columns. Output only the raw "
    "transcribed text — no translation, no transliteration, no commentary, no markdown."
)
# Per-box mode (see run_nvidia_ocr_cv): OpenCV has already isolated a single
# line, so the model just has to read it — no layout/column instructions needed.
NVIDIA_CROP_PROMPT = (
    "Transcribe the Punjabi (Gurmukhi) text in this image exactly as it appears. "
    "This is a single line cropped from a larger page. Output only the raw "
    "transcribed text on one line — no translation, no transliteration, no "
    "commentary, no markdown. If there is no legible text, output nothing."
)
NVIDIA_CV_MAX_CONCURRENCY = 6
# One model call per detected line, so a page with hundreds of "lines" (a noisy
# scan, or a photo where detection fragments) would be slow and expensive.
# Past this count we stop trusting the detection and OCR the page in one call.
NVIDIA_CV_MAX_BOXES = 200
# A page is only usable if most of its lines came back; below this we fall back
# to whole-page OCR rather than returning a silently half-empty transcription.
NVIDIA_CV_MIN_SUCCESS_RATIO = 0.5

# Retry policy for the upstream OCR APIs — these are rate-limited, and a
# transient 429/503 on one crop should not punch a hole in the page.
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0

# How Vision's detectedBreak types translate into text between words.
_BREAKS = {
    "SPACE": " ",
    "SURE_SPACE": " ",
    "EOL_SURE_SPACE": "\n",
    "LINE_BREAK": "\n",
    "HYPHEN": "",
}


def _json_body(resp: httpx.Response) -> dict:
    """Parse a response body as a JSON object, never raising."""
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _error_message(body: dict, resp: httpx.Response) -> str:
    """Best-effort human-readable error out of an upstream error body.

    Providers disagree on shape (``error.message``, ``error`` as a bare string,
    ``detail``, or nothing at all), so every lookup is defensive — an error
    path that itself raises would mask the real failure.
    """
    error = body.get("error")
    if isinstance(error, dict):
        msg = error.get("message") or error.get("innererror", {}).get("message")
        if msg:
            return str(msg)
    elif isinstance(error, str) and error:
        return error
    detail = body.get("detail") or body.get("message")
    if detail:
        return str(detail)
    return (resp.text or f"HTTP {resp.status_code}")[:300]


async def _post_with_retry(client: httpx.AsyncClient, service: str, **kwargs) -> httpx.Response:
    """POST with bounded retries on transport errors and retryable statuses.

    Honours ``Retry-After`` when the server sends one, otherwise backs off
    exponentially with jitter so parallel per-crop calls don't retry in lockstep
    and re-trigger the rate limit they just hit.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = await client.post(**kwargs)
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt == MAX_ATTEMPTS:
                break
            await asyncio.sleep(_backoff(attempt))
            continue

        if resp.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
            delay = _retry_after(resp) or _backoff(attempt)
            log.warning(
                "%s returned %s; retrying in %.1fs (attempt %d/%d)",
                service, resp.status_code, delay, attempt, MAX_ATTEMPTS,
            )
            await asyncio.sleep(delay)
            continue
        return resp

    raise HTTPException(502, f"Could not reach {service}: {last_exc}")


def _backoff(attempt: int) -> float:
    return RETRY_BASE_DELAY * (2 ** (attempt - 1)) * (0.5 + random.random())


def _retry_after(resp: httpx.Response) -> float | None:
    """Seconds from a ``Retry-After`` header, if it carries a sane delay."""
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        delay = float(raw)
    except ValueError:
        return None  # HTTP-date form — fall back to our own backoff
    return delay if 0 <= delay <= 30 else None


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
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await _post_with_retry(
            client, "Google Vision", url=VISION_URL, params={"key": api_key}, json=payload
        )

    body = _json_body(resp)
    if resp.status_code != 200:
        raise HTTPException(502, f"Google Vision error: {_error_message(body, resp)}")

    responses = body.get("responses")
    response = responses[0] if isinstance(responses, list) and responses else {}
    if not isinstance(response, dict):
        raise HTTPException(502, "Google Vision returned an unexpected response.")
    if response.get("error"):
        raise HTTPException(502, f"Google Vision error: {_error_message(response, resp)}")

    annotation = response.get("fullTextAnnotation")
    if not isinstance(annotation, dict) or not annotation:
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
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await _post_with_retry(
            client,
            "Azure AI Vision",
            url=url,
            params=params,
            headers=headers,
            content=image_bytes,
        )

    body = _json_body(resp)
    if resp.status_code != 200:
        raise HTTPException(502, f"Azure AI Vision error: {_error_message(body, resp)}")

    read = body.get("readResult")
    read = read if isinstance(read, dict) else {}
    words = []
    lines_text = []
    for block in read.get("blocks") or []:
        for line in (block or {}).get("lines") or []:
            line_words = line.get("words") or []
            lines_text.append(line.get("text") or "")
            for i, word in enumerate(line_words):
                text = word.get("text") or ""
                polygon = word.get("boundingPolygon") or []
                box = [[p.get("x", 0), p.get("y", 0)] for p in polygon if isinstance(p, dict)]
                if not text or len(box) < 3:
                    continue
                suffix = "\n" if i == len(line_words) - 1 else " "
                words.append({"text": text, "suffix": suffix, "box": box})

    if not words:
        raise HTTPException(422, "No text was detected in this image.")

    metadata = body.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return {
        "width": metadata.get("width") or 0,
        "height": metadata.get("height") or 0,
        "words": words,
        "full_text": "\n".join(t for t in lines_text if t).strip(),
    }


# Image formats the vision endpoint accepts, keyed by PIL's format name.
_MIME_BY_FORMAT = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}


def _prepare_for_api(image_bytes: bytes, max_side: int = 1500) -> tuple[bytes, str]:
    """Return ``(bytes, mime)`` ready to embed in a data: URI.

    Shrinks oversized images (payload and latency) and applies any EXIF
    orientation, so a phone photo is sent the same way up the browser shows it.
    The mime type describes the bytes actually being sent: labelling a re-encoded
    JPEG as ``image/png`` — as this used to — leaves the endpoint to sniff the
    real format, and it is entitled to reject the mismatch instead.
    """
    import io

    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(image_bytes)) as img:
            original_format = (img.format or "").upper()
            oversized = max(img.size) > max_side
            # 1 == "as stored"; anything else means the viewer would rotate it
            needs_rotation = img.getexif().get(0x0112, 1) not in (1, None)
            if not oversized and not needs_rotation and original_format in _MIME_BY_FORMAT:
                return image_bytes, _MIME_BY_FORMAT[original_format]

            prepared = ImageOps.exif_transpose(img).convert("RGB")
            if oversized:
                prepared.thumbnail((max_side, max_side))
            out = io.BytesIO()
            prepared.save(out, format="JPEG", quality=88)
            return out.getvalue(), "image/jpeg"
    except Exception:  # noqa: BLE001 — unreadable by PIL; let the endpoint try the raw bytes
        log.debug("could not pre-process image for the API; sending raw bytes", exc_info=True)
        return image_bytes, _sniff_mime(image_bytes)


def _sniff_mime(image_bytes: bytes) -> str:
    """Magic-byte mime detection, for when PIL can't open the image."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


async def _nvidia_chat(
    image_bytes: bytes,
    api_key: str,
    model: str | None,
    prompt: str,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Send one image+prompt to the NVIDIA vision endpoint, return the reply text.

    Pass ``client`` to reuse one connection pool across the many per-crop calls
    ``run_nvidia_ocr_cv`` makes; without it a client is created per call.
    """
    data, mime = _prepare_for_api(image_bytes)
    b64 = base64.b64encode(data).decode()
    payload = {
        "model": model or NVIDIA_DEFAULT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    if client is None:
        async with httpx.AsyncClient(timeout=240) as own_client:
            return await _nvidia_request(own_client, payload, headers)
    return await _nvidia_request(client, payload, headers)


async def _nvidia_request(client: httpx.AsyncClient, payload: dict, headers: dict) -> str:
    resp = await _post_with_retry(client, "NVIDIA", url=NVIDIA_URL, headers=headers, json=payload)

    body = _json_body(resp)
    if resp.status_code != 200:
        raise HTTPException(502, f"NVIDIA OCR error: {_error_message(body, resp)}")

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HTTPException(502, "NVIDIA returned an unexpected response.")
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, list):  # some deployments return content parts, not a string
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        raise HTTPException(502, "NVIDIA returned an unexpected response.")
    return content.strip()


async def run_nvidia_ocr(image_bytes: bytes, api_key: str, model: str | None = None) -> dict:
    """Plain OCR via an NVIDIA vision LLM — returns text only (boxes are None)."""
    text = await _nvidia_chat(image_bytes, api_key, model, NVIDIA_OCR_PROMPT)
    if not text.strip():
        raise HTTPException(422, "No text was detected in this image.")
    return _words_from_text(text)


async def run_nvidia_ocr_cv(image_bytes: bytes, api_key: str, model: str | None = None) -> dict:
    """Column-aware OCR via OpenCV-detected line boxes + one model call per box.

    OpenCV finds each text line's pixel box deterministically (see
    ``cvboxes.detect_text_boxes``) — the model is never asked to invent
    coordinates, it only has to read the (small, already-isolated) crop it's
    given, in parallel across all boxes.

    Falls back to whole-image plain-text OCR when the box detection is not
    trustworthy: no boxes, implausibly many boxes, an image OpenCV can't
    decode, or too few crops coming back readable. That last case matters —
    without it a page where most calls were rate-limited would return a
    plausible-looking but badly incomplete transcription.
    """
    try:
        boxes, width, height = cvboxes.detect_text_boxes(image_bytes)
    except ValueError as exc:  # undecodable by OpenCV — the model may still cope
        log.warning("OpenCV could not decode the image (%s); falling back to whole-page OCR", exc)
        return await run_nvidia_ocr(image_bytes, api_key, model)
    except Exception:  # noqa: BLE001 — detection must never take the request down
        log.exception("box detection failed; falling back to whole-page OCR")
        return await run_nvidia_ocr(image_bytes, api_key, model)

    if not boxes:
        return await run_nvidia_ocr(image_bytes, api_key, model)
    if len(boxes) > NVIDIA_CV_MAX_BOXES:
        log.warning(
            "box detection produced %d boxes (max %d); falling back to whole-page OCR",
            len(boxes), NVIDIA_CV_MAX_BOXES,
        )
        return await run_nvidia_ocr(image_bytes, api_key, model)

    try:
        crops = cvboxes.crops_for_boxes(image_bytes, boxes)
    except Exception:  # noqa: BLE001
        log.exception("cropping failed; falling back to whole-page OCR")
        return await run_nvidia_ocr(image_bytes, api_key, model)

    sem = asyncio.Semaphore(NVIDIA_CV_MAX_CONCURRENCY)
    attempted = 0
    failed = 0

    async def _ocr_crop(client: httpx.AsyncClient, crop_bytes: bytes) -> str:
        nonlocal attempted, failed
        if not crop_bytes:
            return ""
        async with sem:
            attempted += 1
            try:
                return await _nvidia_chat(crop_bytes, api_key, model, NVIDIA_CROP_PROMPT, client)
            except HTTPException as exc:
                failed += 1  # one failed crop shouldn't sink the whole page
                log.warning("crop OCR failed: %s", exc.detail)
                return ""
            except Exception:  # noqa: BLE001 — keep gather from cancelling its siblings
                failed += 1
                log.exception("crop OCR raised an unexpected error")
                return ""

    limits = httpx.Limits(max_connections=NVIDIA_CV_MAX_CONCURRENCY)
    async with httpx.AsyncClient(timeout=240, limits=limits) as client:
        texts = await asyncio.gather(*(_ocr_crop(client, c) for c in crops))

    if attempted and (attempted - failed) < attempted * NVIDIA_CV_MIN_SUCCESS_RATIO:
        log.warning(
            "%d/%d crop calls failed; falling back to whole-page OCR", failed, attempted
        )
        return await run_nvidia_ocr(image_bytes, api_key, model)

    lines = [
        {"text": text.strip(), "box": box}
        for box, text in zip(boxes, texts)
        if text.strip()
    ]
    if not lines:
        return await run_nvidia_ocr(image_bytes, api_key, model)

    words = []
    for line in lines:
        toks = layout.split_line_into_words(line["text"], line["box"])
        for i, tok in enumerate(toks):
            tok["suffix"] = "\n" if i == len(toks) - 1 else " "
            words.append(tok)

    full_text = "\n".join(line["text"] for line in lines)
    return {"width": width, "height": height, "words": words, "full_text": full_text}


def _words_from_text(text: str) -> dict:
    """Split a plain-text OCR result into box-less words for the front-end.

    Blank lines carry no tokens of their own, so their newline is folded onto
    the preceding word's suffix — otherwise paragraph breaks vanish when the
    front-end rebuilds the text from the word list.
    """
    words = []
    lines = text.split("\n")
    for li, line in enumerate(lines):
        tokens = line.split()
        if not tokens:
            if words and li < len(lines) - 1:
                words[-1]["suffix"] += "\n"
            continue
        for ti, token in enumerate(tokens):
            last_in_line = ti == len(tokens) - 1
            suffix = "\n" if last_in_line and li < len(lines) - 1 else " "
            words.append({"text": token, "suffix": suffix, "box": None})
    if words:
        words[-1]["suffix"] = ""  # nothing follows the final word
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
