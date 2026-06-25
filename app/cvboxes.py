"""OpenCV-based text-region detection.

Finds bounding boxes for lines of text on a page using classical image
processing only (threshold -> connected components -> dedupe -> merge) —
no model call is involved here. The vision model is only used afterwards, to
read the text inside each detected box (see ``ocr.run_nvidia_ocr_cv``).

Pipeline:

1. ``_raw_boxes``         — Otsu-threshold the page, take each connected
   component's bounding rect (roughly glyph/word fragments).
2. ``_dedupe_overlapping`` — drop boxes that are mostly contained inside a
   bigger box (nested/duplicate contours, e.g. a detached matra dot).
3. ``_merge_close``       — union boxes that are close enough to belong to
   the same text line into a single line-level box. The merge distance is
   derived from the page's own median glyph height so it scales with image
   resolution and font size, but can be overridden explicitly.

``detect_text_boxes`` runs all three steps and returns line-level pixel boxes
in raster order (top-to-bottom, then left-to-right); column-aware reading
order is reconstructed later by ``layout.reading_order_from_boxes``.
"""

import cv2
import numpy as np

MIN_AREA_RATIO = 0.00005  # drop contours smaller than this fraction of the page area (noise)
DEDUPE_IOU = 0.3  # a box >=80%-contained in a bigger one is redundant; see _dedupe_overlapping
MERGE_GAP_X_FACTOR = 1.8  # merge boxes into a line if their horizontal gap < factor * median height
MERGE_GAP_Y_FACTOR = 0.6  # ...and their vertical gap < factor * median height (keeps merges intra-line)
OUTLIER_HEIGHT_FACTOR = 6  # drop raw boxes taller than factor * median glyph height (rule lines)
OUTLIER_AREA_FACTOR = 25  # ...or bigger in area than factor * median glyph area (illustrations/photos)
PAD_PX = 3  # pixels of padding added around each final box (avoids clipping matras/ascenders)
MIN_CROP_HEIGHT = 48  # crops shorter than this are upscaled before OCR


def detect_text_boxes(
    image_bytes: bytes,
    *,
    merge_gap_x: float | None = None,
    merge_gap_y: float | None = None,
    dedupe_iou_thresh: float = DEDUPE_IOU,
    pad: int = PAD_PX,
) -> tuple[list[tuple[int, int, int, int]], int, int]:
    """Return ``(line_boxes, width, height)`` for ``image_bytes``.

    ``merge_gap_x``/``merge_gap_y`` are pixel thresholds controlling how
    aggressively nearby fragments are merged into a line; left at ``None``
    they're derived from the page's median glyph height.
    """
    img = _decode(image_bytes)
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)

    min_area = max(12, int(width * height * MIN_AREA_RATIO))
    boxes = _raw_boxes(binary, min_area)
    if not boxes:
        return [], width, height

    boxes = _drop_size_outliers(boxes)
    if not boxes:
        return [], width, height

    boxes = _dedupe_overlapping(boxes, dedupe_iou_thresh)

    median_h = _median_height(boxes)
    gap_x = merge_gap_x if merge_gap_x is not None else median_h * MERGE_GAP_X_FACTOR
    gap_y = merge_gap_y if merge_gap_y is not None else median_h * MERGE_GAP_Y_FACTOR
    lines = _merge_close(boxes, gap_x, gap_y)
    lines = _dedupe_overlapping(lines, dedupe_iou_thresh)  # merging can create new overlaps

    lines = [_pad_box(b, pad, width, height) for b in lines]
    lines.sort(key=lambda b: (b[1], b[0]))
    return lines, width, height


def crops_for_boxes(image_bytes: bytes, boxes: list[tuple[int, int, int, int]]) -> list[bytes]:
    """PNG-encode the pixel crop for each box (empty bytes for a degenerate box)."""
    img = _decode(image_bytes)
    out = []
    for x0, y0, x1, y1 in boxes:
        crop = img[int(y0) : int(y1), int(x0) : int(x1)]
        if crop.size == 0:
            out.append(b"")
            continue
        crop = _maybe_upscale(crop)
        ok, buf = cv2.imencode(".png", crop)
        out.append(buf.tobytes() if ok else b"")
    return out


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _decode(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def _binarize(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _raw_boxes(binary: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    """Bounding rects of every foreground (text) blob.

    Uses RETR_CCOMP, not RETR_EXTERNAL: a scanned page's border commonly forms
    one continuous ring touching all four edges, and RETR_EXTERNAL would treat
    every glyph inside it as a nested contour and silently discard them all.
    RETR_CCOMP instead gives a 2-level hierarchy (objects, then holes within
    them); we keep only object contours (``parent == -1``) — every glyph
    inside the border's hole is bumped back to that top level by OpenCV, while
    the hole itself (which would otherwise cover almost the whole page) is
    dropped. A box still spanning ~the whole page (the border ring itself) is
    then dropped explicitly too — no real text line is page-sized.
    """
    height, width = binary.shape
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c, h in zip(contours, hierarchy[0]):
        if h[3] != -1:  # has a parent => this is a hole, not a foreground object
            continue
        x, y, w, ht = cv2.boundingRect(c)
        if w * ht < min_area:
            continue
        if w >= width * 0.95 and ht >= height * 0.95:
            continue
        boxes.append((x, y, x + w, y + ht))
    return boxes


def _drop_size_outliers(boxes):
    """Drop raw boxes that are wildly bigger than a typical glyph fragment.

    Catches non-text page furniture (a thin column-divider rule spanning the
    whole page height, an illustration/photo block) before they reach the
    merge step, where their size would otherwise drag in everything nearby.
    """
    if len(boxes) < 3:
        return boxes
    heights = sorted(b[3] - b[1] for b in boxes)
    areas = sorted(_area(b) for b in boxes)
    median_h = heights[len(heights) // 2] or 1
    median_area = areas[len(areas) // 2] or 1
    return [
        b
        for b in boxes
        if (b[3] - b[1]) <= OUTLIER_HEIGHT_FACTOR * median_h
        and _area(b) <= OUTLIER_AREA_FACTOR * median_area
    ]


def _area(b: tuple[int, int, int, int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _iou_and_containment(a, b) -> tuple[float, float]:
    """Return (IoU, intersection / smaller-box-area)."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0, 0.0
    area_a, area_b = _area(a), _area(b)
    union = area_a + area_b - inter
    return inter / union, inter / min(area_a, area_b)


def _dedupe_overlapping(boxes, iou_thresh: float):
    """Drop boxes that mostly overlap a bigger, already-kept box."""
    ordered = sorted(boxes, key=_area, reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for b in ordered:
        redundant = any(
            max(_iou_and_containment(b, k)) >= iou_thresh
            or _iou_and_containment(b, k)[1] >= 0.8
            for k in kept
        )
        if not redundant:
            kept.append(b)
    return kept


def _median_height(boxes) -> float:
    heights = sorted(b[3] - b[1] for b in boxes)
    return heights[len(heights) // 2] if heights else 10.0


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _y_overlap_ratio(a, b) -> float:
    """Vertical overlap as a fraction of the shorter box's height."""
    inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / min(a[3] - a[1], b[3] - b[1])


def _merge_close(boxes, gap_x: float, gap_y: float):
    """Union boxes into line-level boxes.

    Two passes, kept deliberately separate so a merge can never "bleed" across
    text lines (a naive single-pass transitive merge over (x, y) distance
    chains through paragraphs: line N's descender sits close enough to line
    N+1's ascender, which chains to N+2, and so on):

    1. Bucket boxes into rows by vertical center (``gap_y`` tall buckets) —
       this only ever looks at the y-axis, so it can't be dragged sideways
       into merging across a column gutter.
    2. Within each row, union boxes whose horizontal gap is <= ``gap_x``
       (a 1-D sweep over x, not transitive across the whole row) — this is
       what joins separate glyph/word fragments into one line box.

    A box landing right on a row-bucket boundary can get split from its true
    row; a final pass stitches back any two resulting boxes that still
    overlap vertically by a large margin and sit close horizontally.
    """
    if not boxes:
        return []

    bucket = max(gap_y, 1.0)
    rows: dict[int, list] = {}
    for b in boxes:
        cy = (b[1] + b[3]) / 2
        rows.setdefault(round(cy / bucket), []).append(b)

    lines = []
    for _, items in sorted(rows.items()):
        lines.extend(_merge_row(items, gap_x))

    return _stitch_split_rows(lines, gap_x)


def _merge_row(boxes, gap_x: float):
    """1-D union of same-row boxes whose horizontal gap is within ``gap_x``."""
    ordered = sorted(boxes, key=lambda b: b[0])
    merged = [ordered[0]]
    for b in ordered[1:]:
        cur = merged[-1]
        if b[0] - cur[2] <= gap_x:
            merged[-1] = _union(cur, b)
        else:
            merged.append(b)
    return merged


def _stitch_split_rows(lines, gap_x: float, y_overlap_thresh: float = 0.5):
    """Re-merge line boxes that landed in adjacent row buckets by mistake."""
    changed = True
    while changed:
        changed = False
        ordered = sorted(lines, key=lambda b: b[1])
        merged = []
        used = [False] * len(ordered)
        for i, cur in enumerate(ordered):
            if used[i]:
                continue
            used[i] = True
            for j in range(i + 1, len(ordered)):
                if used[j]:
                    continue
                cand = ordered[j]
                dx = max(cur[0] - cand[2], cand[0] - cur[2], 0)
                if dx <= gap_x and _y_overlap_ratio(cur, cand) >= y_overlap_thresh:
                    cur = _union(cur, cand)
                    used[j] = True
                    changed = True
            merged.append(cur)
        lines = merged
    return lines


def _pad_box(b, pad: int, width: int, height: int):
    x0, y0, x1, y1 = b
    return (max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad))


def _maybe_upscale(crop: np.ndarray, min_height: int = MIN_CROP_HEIGHT) -> np.ndarray:
    h = crop.shape[0]
    if h <= 0 or h >= min_height:
        return crop
    scale = min_height / h
    new_w = max(1, int(round(crop.shape[1] * scale)))
    return cv2.resize(crop, (new_w, min_height), interpolation=cv2.INTER_CUBIC)
