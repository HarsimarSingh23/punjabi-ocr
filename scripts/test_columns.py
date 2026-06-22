"""Run the column-aware OCR pipeline on one image and log every API call and
every bounding box to logs/bounding_boxes.log.

Usage:
    .venv/bin/python scripts/test_columns.py [path/to/image] [columns]

`columns` is auto | 1 | 2 | 3 (default: the page_columns setting, else "2").
The API key is read from data.db / env and is NEVER written to the log.
"""

import asyncio
import io
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from app import db, layout, ocr  # noqa: E402

LOG_PATH = ROOT / "logs" / "bounding_boxes.log"


def _setup_logger(log_path=LOG_PATH):
    log_path.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("bbox")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _fmt_box(box):
    if not box:
        return "None"
    return "[" + ", ".join(f"({round(x)},{round(y)})" for x, y in box) + "]"


async def main():
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "test_image" / "img.png"
    settings = db.get_settings()
    columns = sys.argv[2] if len(sys.argv) > 2 else (settings.get("page_columns") or "2")
    model_override = sys.argv[3] if len(sys.argv) > 3 else None
    model = model_override or settings.get("nvidia_model") or ocr.NVIDIA_DEFAULT_MODEL

    # the default run writes the canonical log; explicit model comparisons get
    # their own file so they don't clobber it
    if model_override:
        safe = model_override.replace("/", "_").replace(".", "-")
        log_path = ROOT / "logs" / f"bbox_{safe}.log"
    else:
        log_path = LOG_PATH
    log = _setup_logger(log_path)

    data = image_path.read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        width, height = im.width, im.height

    provider = settings.get("ocr_provider") or "google"

    log.info("=" * 78)
    log.info("PUNJABI OCR — COLUMN-AWARE BOUNDING-BOX LOG")
    log.info("run at      : %s", datetime.now().isoformat(timespec="seconds"))
    log.info("image       : %s", image_path)
    log.info("image size  : %d x %d px", width, height)
    log.info("ocr provider: %s", provider)
    log.info("page_columns: %s", columns)
    log.info("=" * 78)

    try:
        if provider == "nvidia":
            await _run_nvidia(log, data, settings, model, width, height)
        elif provider == "azure":
            await _run_boxed(
                log,
                columns,
                ocr.run_azure_ocr(
                    data, settings.get("azure_vision_endpoint"), settings.get("azure_vision_key")
                ),
            )
        else:
            await _run_boxed(
                log, columns, ocr.run_google_ocr(data, settings.get("google_api_key"))
            )
    except Exception as exc:  # noqa: BLE001 — log the failure instead of crashing
        log.info("")
        log.info("!! API CALL FAILED: %s: %s", type(exc).__name__, exc)

    log.info("")
    log.info("log written to: %s", log_path)


async def _run_nvidia(log, data, settings, model, width, height):
    key = settings.get("nvidia_api_key")
    if not key:
        log.info("ERROR: NVIDIA API key is not set.")
        return

    # ---- API CALL ----
    log.info("")
    log.info("### API CALL 1 — NVIDIA vision (structured, column-aware)")
    log.info("POST %s", ocr.NVIDIA_URL)
    log.info("model        : %s", model)
    log.info("authorization: Bearer ****redacted****")
    log.info("image payload: data:image/png;base64 (%d bytes raw)", len(data))
    log.info("prompt:\n%s", ocr.NVIDIA_LAYOUT_PROMPT)

    reply = await ocr._nvidia_chat(data, key, model, ocr.NVIDIA_LAYOUT_PROMPT)
    log.info("")
    log.info("--- RAW MODEL RESPONSE ---")
    log.info("%s", reply)
    log.info("--- END RAW RESPONSE (%d chars) ---", len(reply))

    # ---- PARSE ----
    parsed = layout.extract_json(reply)
    if not parsed:
        log.info("")
        log.info("!! Could not parse JSON from the response — falling back to plain text.")
        result = ocr._words_from_text(reply)
        _log_words(log, result)
        return

    cols = parsed.get("columns") or ([{"lines": parsed["lines"]}] if parsed.get("lines") else [])
    log.info("")
    log.info("### PARSED LAYOUT — %d column(s)", len(cols))
    for ci, col in enumerate(cols):
        lines = col.get("lines") if isinstance(col, dict) else col
        lines = lines or []
        log.info("")
        log.info("  COLUMN %d — %d line(s)", ci + 1, len(lines))
        for li, line in enumerate(lines):
            if not isinstance(line, dict):
                continue
            text = (line.get("text") or "").strip()
            raw_box = line.get("box")
            px = layout._scale_box(raw_box, width, height)
            px_str = (
                f"[{round(px[0])},{round(px[1])} -> {round(px[2])},{round(px[3])}]"
                if px
                else "None"
            )
            log.info("    line %02d  norm=%-22s px=%-26s  %s", li + 1, raw_box, px_str, text)

    # ---- FINAL WORDS WITH BOXES ----
    result = layout.words_from_vision_json(parsed, width, height)
    if not result:
        log.info("!! No usable words parsed; falling back to plain text.")
        result = ocr._words_from_text(reply)
    _log_words(log, result)


async def _run_boxed(log, columns, coro):
    log.info("")
    log.info("### API CALL 1 — box-returning OCR engine")
    result = await coro
    log.info("raw words returned: %d", len(result.get("words", [])))
    result = layout.reading_order_from_boxes(result, columns)
    log.info("after column-aware reading order: %d words", len(result.get("words", [])))
    _log_words(log, result)


def _log_words(log, result):
    words = result.get("words", [])
    log.info("")
    log.info("### FINAL WORDS IN READING ORDER — %d words (%d with boxes)",
             len(words), sum(1 for w in words if w.get("box")))
    log.info("%-5s  %-14s  %s", "idx", "box", "text")
    for i, w in enumerate(words):
        log.info("%-5d  %-14s  %s", i, _fmt_box(w.get("box")), w.get("text"))

    log.info("")
    log.info("### RECONSTRUCTED FULL TEXT (reading order)")
    log.info("%s", result.get("full_text", ""))


if __name__ == "__main__":
    asyncio.run(main())
