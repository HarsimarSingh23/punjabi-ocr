"""End-to-end test of the OpenCV box-detection + per-box NVIDIA OCR pipeline.

Runs ``cvboxes.detect_text_boxes`` on an image, OCRs each detected box in
parallel via ``ocr.run_nvidia_ocr_cv``, reorders into column-aware reading
order, and:

* logs every box + its OCR'd text to logs/bounding_boxes.log
* draws each box and its recognized text over a copy of the image, saved to
  docs/cv-pipeline-example.jpg

Usage:
    .venv/bin/python scripts/test_cv_pipeline.py [path/to/image] [columns]

`columns` is auto | 1 | 2 | 3 (default: the page_columns setting, else "auto").
The API key is read from data.db / env and is NEVER written to the log.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app import cvboxes, db, layout, ocr  # noqa: E402

LOG_PATH = ROOT / "logs" / "bounding_boxes.log"
OUT_IMAGE = ROOT / "docs" / "cv-pipeline-example.jpg"

# Common Gurmukhi-capable font locations; falls back to boxes-only if none exist.
GURMUKHI_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Gurmukhi MN.ttc",
    "/System/Library/Fonts/Supplemental/Gurmukhi Sangam MN.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansGurmukhi-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansGurmukhi-Regular.ttf",
    str(Path.home() / "Library/Fonts/NotoSansGurmukhi-VariableFont_wdth,wght.ttf"),
]


def _setup_logger():
    LOG_PATH.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("cvbbox")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _load_font(size=16):
    for path in GURMUKHI_FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return None


async def main():
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "test_image" / "img.png"
    settings = db.get_settings()
    columns = sys.argv[2] if len(sys.argv) > 2 else (settings.get("page_columns") or "auto")
    api_key = settings.get("nvidia_api_key")
    model = settings.get("nvidia_model") or ocr.NVIDIA_DEFAULT_MODEL

    log = _setup_logger()
    data = image_path.read_bytes()

    log.info("=" * 78)
    log.info("PUNJABI OCR — OPENCV BOX DETECTION + PER-BOX NVIDIA OCR")
    log.info("run at      : %s", datetime.now().isoformat(timespec="seconds"))
    log.info("image       : %s", image_path)
    log.info("model       : %s", model)
    log.info("page_columns: %s", columns)
    log.info("=" * 78)

    if not api_key:
        log.info("ERROR: NVIDIA API key is not set (Admin portal or NVIDIA_API_KEY env).")
        return

    boxes, width, height = cvboxes.detect_text_boxes(data)
    log.info("")
    log.info("### STAGE 1 — OpenCV box detection")
    log.info("image size  : %d x %d px", width, height)
    log.info("line boxes  : %d", len(boxes))
    for i, b in enumerate(boxes):
        log.info("  box %03d  [%d,%d -> %d,%d]", i, *b)

    log.info("")
    log.info("### STAGE 2 — per-box OCR (parallel, model=%s)", model)
    result = await ocr.run_nvidia_ocr_cv(data, api_key, model)
    log.info("words returned: %d (%d with boxes)", len(result["words"]),
              sum(1 for w in result["words"] if w.get("box")))

    log.info("")
    log.info("### STAGE 3 — column-aware reading order")
    result = layout.reading_order_from_boxes(result, columns)
    log.info("%-5s  %-26s  %s", "idx", "box", "text")
    for i, w in enumerate(result["words"]):
        box = w.get("box")
        box_str = "[" + ", ".join(f"({round(x)},{round(y)})" for x, y in box) + "]" if box else "None"
        log.info("%-5d  %-26s  %s", i, box_str, w["text"])

    log.info("")
    log.info("### RECONSTRUCTED FULL TEXT (reading order)")
    log.info("%s", result["full_text"])

    _draw_overlay(image_path, boxes, result)
    log.info("")
    log.info("annotated image written to: %s", OUT_IMAGE)
    log.info("log written to: %s", LOG_PATH)


def _draw_overlay(image_path, boxes, result):
    """Draw each detected box and its OCR'd text over a copy of the image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(size=max(12, img.height // 90))
    if font is None:
        print("WARNING: no Gurmukhi-capable font found locally; drawing boxes only.")

    # words carry the per-word split boxes; group back by original line box so
    # each line's full OCR'd text is captioned once, not split word-by-word.
    text_by_box = {}
    for box in boxes:
        words_in_box = [
            w["text"]
            for w in result["words"]
            if w.get("box") and _box_inside(w["box"], box)
        ]
        if words_in_box:
            text_by_box[box] = " ".join(words_in_box)

    for box in boxes:
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        text = text_by_box.get(box)
        if text and font is not None:
            caption_y = max(0, y0 - font.size - 2)
            draw.text((x0, caption_y), text, fill=(0, 100, 255), font=font)

    OUT_IMAGE.parent.mkdir(exist_ok=True)
    img.save(OUT_IMAGE, quality=85, optimize=True)


def _box_inside(word_box, line_box):
    """True if a per-word vertex box's center falls inside a line pixel box."""
    xs = [p[0] for p in word_box]
    ys = [p[1] for p in word_box]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    x0, y0, x1, y1 = line_box
    return x0 <= cx <= x1 and y0 <= cy <= y1


if __name__ == "__main__":
    asyncio.run(main())
