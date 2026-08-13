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
DEDUPE_IOU = 0.3  # two boxes overlapping this much (IoU) are the same region
DEDUPE_CONTAINMENT = 0.8  # ...as is a box this fraction swallowed by a bigger one
MERGE_GAP_X_FACTOR = 1.8  # merge boxes into a line if their horizontal gap < factor * median height
MERGE_GAP_Y_FACTOR = 0.6  # ...and their vertical gap < factor * median height (keeps merges intra-line)
OUTLIER_HEIGHT_FACTOR = 6  # drop raw boxes taller than factor * median glyph height (rule lines)
OUTLIER_AREA_FACTOR = 25  # ...or bigger in area than factor * median glyph area (illustrations/photos)
PAD_PX = 3  # pixels of padding added around each final box (avoids clipping matras/ascenders)
MIN_CROP_HEIGHT = 48  # crops shorter than this are upscaled before OCR
MAX_CROP_SIDE = 1600  # crops larger than this are downscaled before OCR (payload/latency)
MAX_RAW_BOXES = 6000  # cap on raw contours carried into the O(n^2) dedupe/merge stages
MAX_STITCH_PASSES = 20  # bound on the _stitch_split_rows fixpoint loop


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
    boxes = _cap_box_count(boxes)
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
    """PNG-encode the pixel crop for each box (empty bytes for a degenerate box).

    Coordinates are clamped to the image and their corners re-ordered, so a
    box reaching outside the frame is trimmed rather than silently wrapping
    (numpy treats a negative index as an offset from the far edge, which would
    crop a completely different part of the page).
    """
    img = _decode(image_bytes)
    height, width = img.shape[:2]
    out = []
    for box in boxes:
        crop = _crop(img, box, width, height)
        if crop is None:
            out.append(b"")
            continue
        try:
            ok, buf = cv2.imencode(".png", crop)
        except cv2.error:
            ok = False
        out.append(buf.tobytes() if ok else b"")
    return out


def _crop(img: np.ndarray, box, width: int, height: int) -> np.ndarray | None:
    try:
        x0, y0, x1, y1 = (int(v) for v in box)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((max(0, min(x0, width)), max(0, min(x1, width))))
    y0, y1 = sorted((max(0, min(y0, height)), max(0, min(y1, height))))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return _rescale_for_ocr(crop)


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _decode(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Could not decode image bytes: empty input")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    try:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error as exc:  # malformed data can trip the decoder itself
        raise ValueError(f"Could not decode image bytes: {exc}") from exc
    if img is None or img.size == 0:
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
    if not contours or hierarchy is None:
        return []  # blank/uniform page — OpenCV returns hierarchy=None, not an empty array
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


def _dedupe_overlapping(boxes, iou_thresh: float, containment_thresh: float = DEDUPE_CONTAINMENT):
    """Drop boxes that duplicate a bigger, already-kept box.

    A box is redundant when it overlaps a kept box by ``iou_thresh`` IoU, or
    when ``containment_thresh`` of it sits inside that kept box (a nested
    contour, e.g. a detached matra dot inside a glyph's rect). The two tests
    are deliberately separate: IoU alone misses a small box fully swallowed by
    a large one, while containment alone would drop a glyph that merely brushes
    a taller neighbour's bounding rect — at ``iou_thresh`` (0.3) that is barely
    a touch, and Gurmukhi's overhanging matras make it a common one.

    Candidates come from a coarse grid index rather than a full scan, so a
    noisy page with thousands of contours stays roughly linear instead of
    quadratic.
    """
    ordered = sorted(boxes, key=_area, reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    index = _GridIndex(_grid_cell_size(ordered))
    for b in ordered:
        redundant = False
        for k in index.candidates(b):
            iou, containment = _iou_and_containment(b, k)
            if iou >= iou_thresh or containment >= containment_thresh:
                redundant = True
                break
        if not redundant:
            index.add(b, len(kept))
            kept.append(b)
    return kept


def _grid_cell_size(boxes) -> float:
    """Cell size for the dedupe index: roughly one typical box across."""
    if not boxes:
        return 32.0
    widths = sorted(max(1, b[2] - b[0]) for b in boxes)
    heights = sorted(max(1, b[3] - b[1]) for b in boxes)
    return float(max(8, widths[len(widths) // 2], heights[len(heights) // 2]))


class _GridIndex:
    """Uniform-grid spatial index over boxes, for neighbour lookup only.

    Boxes spanning an absurd number of cells (a page-wide rule that survived
    the outlier filter) are held in ``_oversized`` and returned for every
    query, so the index never blows up on pathological input.
    """

    MAX_CELLS_PER_BOX = 256

    def __init__(self, cell: float):
        self.cell = max(1.0, cell)
        self.cells: dict[tuple[int, int], list[int]] = {}
        self.boxes: list[tuple[int, int, int, int]] = []
        self._oversized: list[int] = []

    def _keys(self, box):
        x0, y0, x1, y1 = box
        gx0, gx1 = int(x0 // self.cell), int(x1 // self.cell)
        gy0, gy1 = int(y0 // self.cell), int(y1 // self.cell)
        if (gx1 - gx0 + 1) * (gy1 - gy0 + 1) > self.MAX_CELLS_PER_BOX:
            return None
        return [(gx, gy) for gx in range(gx0, gx1 + 1) for gy in range(gy0, gy1 + 1)]

    def add(self, box, idx: int) -> None:
        self.boxes.append(box)
        keys = self._keys(box)
        if keys is None:
            self._oversized.append(idx)
            return
        for key in keys:
            self.cells.setdefault(key, []).append(idx)

    def candidates(self, box):
        seen = set(self._oversized)
        keys = self._keys(box)
        if keys is None:  # oversized query: fall back to a full scan
            yield from self.boxes
            return
        for key in keys:
            for idx in self.cells.get(key, ()):
                if idx not in seen:
                    seen.add(idx)
                    yield self.boxes[idx]
        for idx in self._oversized:
            yield self.boxes[idx]


def _cap_box_count(boxes):
    """Keep only the largest ``MAX_RAW_BOXES`` contours on very noisy pages.

    Speckle and halftone dots produce tens of thousands of tiny contours on a
    bad scan; they are the smallest boxes by area, and letting them through
    would make the later merge stages crawl for no gain in text coverage.
    """
    if len(boxes) <= MAX_RAW_BOXES:
        return boxes
    return sorted(boxes, key=_area, reverse=True)[:MAX_RAW_BOXES]


def _median_height(boxes) -> float:
    heights = sorted(b[3] - b[1] for b in boxes)
    if not heights:
        return 10.0
    return max(1.0, float(heights[len(heights) // 2]))


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _y_overlap_ratio(a, b) -> float:
    """Vertical overlap as a fraction of the shorter box's height."""
    shorter = min(a[3] - a[1], b[3] - b[1])
    if shorter <= 0:  # degenerate box — no meaningful overlap to report
        return 0.0
    inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / shorter


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
    if not boxes:
        return []
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
    """Re-merge line boxes that landed in adjacent row buckets by mistake.

    Iterated to a fixpoint because one stitch can bring two more boxes into
    range, but capped at ``MAX_STITCH_PASSES``: each pass is quadratic in the
    line count, and the result is already usable after the first few.
    """
    changed = True
    passes = 0
    while changed and passes < MAX_STITCH_PASSES:
        passes += 1
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


def _rescale_for_ocr(
    crop: np.ndarray,
    min_height: int = MIN_CROP_HEIGHT,
    max_side: int = MAX_CROP_SIDE,
) -> np.ndarray:
    """Upscale crops too small to read, downscale ones too big to send cheaply."""
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return crop

    if h < min_height:
        scale = min_height / h
        # don't let a very wide, very short strip blow past the payload cap
        scale = min(scale, max_side / w) if w * scale > max_side else scale
        if scale > 1:
            new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
            return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return crop

    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return crop
