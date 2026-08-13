"""Regression tests for the failure modes the pipeline used to fall over on.

Every case here maps to a bug that was live in the app: a crash, a silent
data loss, or an unhandled upstream response. No network and no API keys —
the OCR endpoints are exercised through httpx's MockTransport.

Usage:
    .venv/bin/python scripts/test_robustness.py
"""

import asyncio
import io
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# point the app at a throwaway DB before importing it — importing app.main
# opens the database at module scope
os.environ["DATA_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "nested" / "test.db")

import cv2  # noqa: E402
import httpx  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from PIL import Image  # noqa: E402

from app import cvboxes, layout, main, ocr  # noqa: E402

ocr.RETRY_BASE_DELAY = 0.01  # keep the retry tests fast

_FAILURES = []


def check(name, fn):
    try:
        fn()
    except Exception:  # noqa: BLE001
        _FAILURES.append(name)
        print(f"FAIL  {name}")
        traceback.print_exc()
    else:
        print(f"ok    {name}")


def png_bytes(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def blank_page(h=200, w=400) -> np.ndarray:
    return np.full((h, w, 3), 255, np.uint8)


def text_page() -> np.ndarray:
    img = blank_page(240, 700)
    cv2.putText(img, "First line of text", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "Second line here", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return img


def poly(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# --------------------------------------------------------------------------- #
# cvboxes
# --------------------------------------------------------------------------- #

def test_blank_image_does_not_crash():
    """A uniform page makes findContours return hierarchy=None, which used to
    raise TypeError and surface as a 500."""
    boxes, w, h = cvboxes.detect_text_boxes(png_bytes(blank_page()))
    assert boxes == [], boxes
    assert (w, h) == (400, 200), (w, h)


def test_undecodable_bytes_raise_valueerror():
    for data in (b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40):
        try:
            cvboxes.detect_text_boxes(data)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {data[:16]!r}")


def test_detection_finds_lines():
    boxes, _, _ = cvboxes.detect_text_boxes(png_bytes(text_page()))
    assert len(boxes) == 2, f"expected 2 line boxes, got {len(boxes)}"
    for x0, y0, x1, y1 in boxes:
        assert x1 > x0 and y1 > y0


def test_dedupe_keeps_touching_glyphs():
    """Two boxes that merely brush each other are distinct glyphs. The old
    threshold dropped anything 30% contained, deleting real text."""
    a, b = (0, 0, 100, 20), (90, 0, 110, 20)
    kept = cvboxes._dedupe_overlapping([a, b], cvboxes.DEDUPE_IOU)
    assert set(kept) == {a, b}, kept


def test_dedupe_drops_nested_box():
    outer, inner = (0, 0, 100, 100), (10, 10, 40, 40)
    kept = cvboxes._dedupe_overlapping([outer, inner], cvboxes.DEDUPE_IOU)
    assert kept == [outer], kept


def test_crops_handle_degenerate_boxes():
    data = png_bytes(text_page())
    crops = cvboxes.crops_for_boxes(data, [(0, 0, 0, 0), (10, 10, 5, 5), (-20, -20, -5, -5)])
    assert crops[0] == b"", "zero-area box should yield no crop"
    assert crops[2] == b"", "fully off-image box should yield no crop"


def test_crop_never_wraps_around_negative_index():
    """A negative coordinate is an offset from the far edge in numpy, so an
    unclamped slice would silently return a different part of the page."""
    img = blank_page(100, 100)
    img[0:20, 0:20] = 0  # black marker in the top-left only
    crop = cvboxes._crop(img, (-50, -50, 20, 20), 100, 100)
    assert crop is not None
    assert crop.min() == 0, "clamped crop should contain the top-left marker"


def test_zero_height_boxes_do_not_divide_by_zero():
    assert cvboxes._y_overlap_ratio((0, 5, 10, 5), (0, 0, 10, 10)) == 0.0


def test_box_count_is_capped():
    many = [(i, i, i + 2, i + 2) for i in range(cvboxes.MAX_RAW_BOXES + 500)]
    assert len(cvboxes._cap_box_count(many)) == cvboxes.MAX_RAW_BOXES


def test_tiny_crop_is_upscaled_and_huge_crop_shrunk():
    small = cvboxes._rescale_for_ocr(blank_page(6, 30))
    assert small.shape[0] == cvboxes.MIN_CROP_HEIGHT, small.shape
    huge = cvboxes._rescale_for_ocr(blank_page(4000, 3000))
    assert max(huge.shape[:2]) == cvboxes.MAX_CROP_SIDE, huge.shape


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #

def test_reordering_keeps_box_less_words():
    """Reordering used to replace the word list with only the boxed words,
    deleting the rest from both `words` and `full_text`."""
    result = {
        "width": 100, "height": 40,
        "words": [
            {"text": "left", "suffix": " ", "box": poly(0, 0, 10, 10)},
            {"text": "ORPHAN", "suffix": " ", "box": None},
            {"text": "right", "suffix": " ", "box": poly(60, 0, 70, 10)},
            {"text": "below", "suffix": " ", "box": poly(0, 20, 10, 30)},
        ],
        "full_text": "left ORPHAN right below",
    }
    out = layout.reading_order_from_boxes(result, "2")
    texts = [w["text"] for w in out["words"]]
    assert "ORPHAN" in texts, texts
    assert "ORPHAN" in out["full_text"], out["full_text"]
    assert texts.index("ORPHAN") == texts.index("left") + 1, texts


def test_column_order_is_applied():
    words = [
        {"text": "L1", "suffix": " ", "box": poly(0, 0, 20, 10)},
        {"text": "R1", "suffix": " ", "box": poly(200, 0, 220, 10)},
        {"text": "L2", "suffix": " ", "box": poly(0, 30, 20, 40)},
        {"text": "R2", "suffix": " ", "box": poly(200, 30, 220, 40)},
    ]
    out = layout.reading_order_from_boxes(
        {"width": 240, "height": 50, "words": words, "full_text": ""}, "2"
    )
    assert [w["text"] for w in out["words"]] == ["L1", "L2", "R1", "R2"]


def test_word_gap_is_not_mistaken_for_a_column():
    """A 5px gap between two words used to register as a column gutter, because
    gutter width was judged against the text's own x-span rather than its
    height. The page came back as 'a\\nb'."""
    words = [
        {"text": "a", "suffix": " ", "box": poly(0, 0, 20, 10)},
        {"text": "b", "suffix": " ", "box": poly(25, 0, 45, 10)},
    ]
    original = {"width": 60, "height": 20, "words": words, "full_text": "a b"}
    out = layout.reading_order_from_boxes(original, "auto")
    assert out["full_text"] == "a b", out["full_text"]


def test_sparse_two_column_page_is_still_detected():
    """The width floor must not be so strict that a short two-column page
    stops being reordered."""
    words = []
    for i, y in enumerate((60, 120, 180)):
        words.append({"text": f"L{i}", "suffix": " ", "box": poly(30, y, 200, y + 30)})
        words.append({"text": f"R{i}", "suffix": " ", "box": poly(500, y, 670, y + 30)})
    out = layout.reading_order_from_boxes(
        {"width": 800, "height": 300, "words": words, "full_text": ""}, "auto"
    )
    assert [w["text"] for w in out["words"]] == ["L0", "L1", "L2", "R0", "R1", "R2"]


def test_real_page_gutter_is_found():
    """Guards the gutter-width calibration against the sample page in the repo."""
    image = ROOT / "test_image" / "img.png"
    if not image.is_file():
        return
    boxes, _, _ = cvboxes.detect_text_boxes(image.read_bytes())
    words = []
    for i, box in enumerate(boxes):
        words.extend(layout.split_line_into_words(f"line{i} word two", box))
    heights = [layout._height(w["box"]) for w in words if layout._height(w["box"]) > 0]
    line_h = sorted(heights)[len(heights) // 2]
    gutters = layout._detect_gutters(words, min_gap_px=line_h * layout.MIN_GUTTER_HEIGHTS)
    assert len(gutters) == 1, f"expected one column gutter, got {gutters}"


def test_layout_survives_empty_and_degenerate_input():
    assert layout.reading_order_from_boxes({"words": [], "full_text": ""}, "auto")["words"] == []
    assert layout.split_line_into_words("a b", (5, 0, 5, 10)) == [
        {"text": "a", "box": None}, {"text": "b", "box": None}
    ]
    assert layout.split_line_into_words("", (0, 0, 10, 10)) == []


# --------------------------------------------------------------------------- #
# ocr — upstream responses
# --------------------------------------------------------------------------- #

def mock_client(handler, **kw):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kw)


def nvidia_ok(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
        return nvidia_ok(" ਸਤ ਸ੍ਰੀ ਅਕਾਲ ")

    async def run():
        async with mock_client(handler) as c:
            return await ocr._nvidia_request(c, {}, {})

    assert asyncio.run(run()) == "ਸਤ ਸ੍ਰੀ ਅਕਾਲ"
    assert calls["n"] == 3, calls


def test_gives_up_with_502_not_a_crash():
    async def run():
        async with mock_client(lambda r: httpx.Response(503, text="upstream down")) as c:
            await ocr._nvidia_request(c, {}, {})

    try:
        asyncio.run(run())
    except HTTPException as exc:
        assert exc.status_code == 502, exc.status_code
        return
    raise AssertionError("expected HTTPException")


def test_transport_error_becomes_502():
    def boom(request):
        raise httpx.ConnectError("no route to host")

    async def run():
        async with mock_client(boom) as c:
            await ocr._nvidia_request(c, {}, {})

    try:
        asyncio.run(run())
    except HTTPException as exc:
        assert exc.status_code == 502
        return
    raise AssertionError("expected HTTPException")


def test_malformed_bodies_never_raise_attributeerror():
    bodies = [
        {}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]}, {"error": "plain string error"},
        [], "not even an object",
    ]
    for body in bodies:
        async def run(body=body):
            async with mock_client(lambda r, b=body: httpx.Response(200, json=b)) as c:
                return await ocr._nvidia_request(c, {}, {})
        try:
            asyncio.run(run())
        except HTTPException as exc:
            assert exc.status_code == 502, (body, exc.status_code)
        else:
            raise AssertionError(f"expected HTTPException for body {body!r}")


def test_content_parts_are_joined():
    body = {"choices": [{"message": {"content": [{"text": "ਸਤ "}, {"text": "ਅਕਾਲ"}]}}]}

    async def run():
        async with mock_client(lambda r: httpx.Response(200, json=body)) as c:
            return await ocr._nvidia_request(c, {}, {})

    assert asyncio.run(run()) == "ਸਤ ਅਕਾਲ"


def test_image_is_labelled_with_its_real_mime():
    png = io.BytesIO(); Image.new("RGB", (50, 50), "white").save(png, "PNG")
    data, mime = ocr._prepare_for_api(png.getvalue())
    assert mime == "image/png", mime
    assert data == png.getvalue(), "small images should pass through untouched"

    big = io.BytesIO(); Image.new("RGB", (3000, 2000), "white").save(big, "PNG")
    data, mime = ocr._prepare_for_api(big.getvalue())
    assert mime == "image/jpeg", mime
    assert max(Image.open(io.BytesIO(data)).size) == 1500

    assert ocr._prepare_for_api(b"garbage")[1] == "application/octet-stream"


def test_exif_rotation_is_applied():
    img = Image.new("RGB", (200, 100), "white")
    exif = img.getexif(); exif[0x0112] = 6  # rotate 90°
    buf = io.BytesIO(); img.save(buf, "JPEG", exif=exif)
    data, _ = ocr._prepare_for_api(buf.getvalue())
    assert Image.open(io.BytesIO(data)).size == (100, 200)


def test_blank_lines_survive_text_splitting():
    out = ocr._words_from_text("a b\n\nc")
    assert out["words"][1]["suffix"] == "\n\n", out["words"]
    assert out["words"][-1]["suffix"] == ""


# --------------------------------------------------------------------------- #
# ocr — the CV pipeline's fallbacks
# --------------------------------------------------------------------------- #

def run_cv(image_bytes, handler):
    """Drive run_nvidia_ocr_cv with every HTTP call served by `handler`."""
    original = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs.pop("transport", None)
        kwargs["transport"] = httpx.MockTransport(handler)
        return original(*args, **kwargs)

    httpx.AsyncClient = patched
    try:
        return asyncio.run(ocr.run_nvidia_ocr_cv(image_bytes, "test-key", None))
    finally:
        httpx.AsyncClient = original


def test_cv_pipeline_reads_every_line():
    result = run_cv(png_bytes(text_page()), lambda r: nvidia_ok("ਲਾਈਨ"))
    assert result["full_text"].count("ਲਾਈਨ") == 2, result["full_text"]
    assert all(w["box"] for w in result["words"]), "per-word boxes should be present"


def test_undecodable_image_falls_back_instead_of_500():
    """OpenCV can't decode it, but the model still might — this used to escape
    as an unhandled ValueError."""
    result = run_cv(b"definitely not an image", lambda r: nvidia_ok("ਪਾਠ"))
    assert result["full_text"] == "ਪਾਠ", result


def test_blank_page_falls_back_to_whole_page_ocr():
    result = run_cv(png_bytes(blank_page()), lambda r: nvidia_ok("ਕੁਝ ਨਹੀਂ"))
    assert result["full_text"] == "ਕੁਝ ਨਹੀਂ"


def test_mostly_failing_crops_fall_back_rather_than_half_a_page():
    """A page where most crops were rate-limited should not be returned as if
    it were complete."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        # fail every per-crop call; the whole-page retry then succeeds
        if calls["n"] <= 2 * ocr.MAX_ATTEMPTS:
            return httpx.Response(500, text="nope")
        return nvidia_ok("ਪੂਰਾ ਪੰਨਾ")

    result = run_cv(png_bytes(text_page()), handler)
    assert result["full_text"] == "ਪੂਰਾ ਪੰਨਾ", result["full_text"]


def test_too_many_boxes_falls_back():
    saved = ocr.NVIDIA_CV_MAX_BOXES
    ocr.NVIDIA_CV_MAX_BOXES = 1
    try:
        result = run_cv(png_bytes(text_page()), lambda r: nvidia_ok("ਪੂਰਾ ਪੰਨਾ"))
        assert result["full_text"] == "ਪੂਰਾ ਪੰਨਾ", result["full_text"]
    finally:
        ocr.NVIDIA_CV_MAX_BOXES = saved


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #

def test_spa_fallback_refuses_to_escape_the_build_dir():
    escapes = [
        "../" * 7 + "etc/hosts",
        "../" * 4 + "punjabi_ocr/data.db",
        "../app/main.py",
        "..",
        "",
    ]
    for path in escapes:
        assert main._safe_dist_file(path) is None, f"{path!r} escaped DIST_DIR"


def test_spa_fallback_still_serves_real_build_files():
    index = main.DIST_DIR / "index.html"
    if not index.is_file():
        return  # frontend isn't built in this checkout
    assert main._safe_dist_file("index.html") == index.resolve()


def api_client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def test_upload_rejects_non_images():
    client = api_client()
    png = io.BytesIO(); Image.new("RGB", (80, 40), "white").save(png, "PNG")
    cases = [
        (("ok.png", png.getvalue(), "image/png"), 200),
        (("lie.png", b"<html>not an image</html>", "image/png"), 400),
        (("empty.png", b"", "image/png"), 400),
        (("doc.pdf", b"%PDF-1.4", "application/pdf"), 400),
        (("huge.png", b"\x89PNG\r\n\x1a\n" + b"x" * (13 * 1024 * 1024), "image/png"), 400),
    ]
    for file_tuple, expected in cases:
        resp = client.post("/api/upload", files={"file": file_tuple})
        assert resp.status_code == expected, (file_tuple[0], resp.status_code, resp.text[:120])


def test_stored_extension_follows_the_real_format():
    """A PNG announced as image/jpeg must not be stored as .jpg — the extension
    is what /uploads serves it as."""
    client = api_client()
    png = io.BytesIO(); Image.new("RGB", (40, 40), "white").save(png, "PNG")
    resp = client.post("/api/upload", files={"file": ("x.jpg", png.getvalue(), "image/jpeg")})
    assert resp.status_code == 200, resp.text
    assert resp.json()["image_url"].endswith(".png"), resp.json()


def test_missing_upload_returns_410_not_500():
    client = api_client()
    png = io.BytesIO(); Image.new("RGB", (40, 40), "white").save(png, "PNG")
    rid = client.post("/api/upload", files={"file": ("g.png", png.getvalue(), "image/png")}).json()["id"]
    (main.UPLOAD_DIR / f"{rid}.png").unlink()
    resp = client.post(f"/api/ocr/{rid}")
    assert resp.status_code == 410, (resp.status_code, resp.text[:120])


def test_unknown_result_id_is_404():
    assert api_client().post("/api/ocr/deadbeef0000").status_code == 404


def test_admin_settings_validation():
    client = api_client()
    assert client.post("/api/admin/settings", json={"ocr_provider": "nvidia"}).status_code == 200
    bad = client.post("/api/admin/settings", json={"ocr_provider": "bogus"})
    assert bad.status_code == 400
    assert "nvidia" in bad.json()["detail"], bad.json()
    # non-string values are ignored, not crashed on
    assert client.post("/api/admin/settings", json={"nvidia_api_key": 12345}).json()["updated"] == []


def test_corrupt_stored_json_does_not_500():
    from app import db
    client = api_client()
    png = io.BytesIO(); Image.new("RGB", (40, 40), "white").save(png, "PNG")
    rid = client.post("/api/upload", files={"file": ("c.png", png.getvalue(), "image/png")}).json()["id"]
    db.save_ocr(rid, "{not valid json", "some text")
    resp = client.get(f"/api/results/{rid}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["ocr"] is None


def main_():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    print(f"running {len(tests)} checks\n")
    for name, fn in tests:
        check(name, fn)
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED: {', '.join(_FAILURES)}")
        return 1
    print(f"all {len(tests)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
