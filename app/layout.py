"""Column-aware reading-order reconstruction.

``reading_order_from_boxes`` is the single entry point: given a result with
word/line boxes (from Google, Azure, or OpenCV-detected NVIDIA line boxes —
see ``ocr.run_nvidia_ocr_cv``), it detects columns via a vertical projection
profile and re-sorts into human reading order (column -> line -> x), then
rebuilds the line breaks and full text. The box geometry itself is untouched.

``split_line_into_words`` is a small standalone helper (used by
``ocr.run_nvidia_ocr_cv``) that turns one line's text + pixel box into
per-word sub-boxes, proportional to character length, so the front-end
animation stays fine-grained even though the model OCR'd a whole line at once.

Both produce the same payload shape the rest of the app uses:
``{"width", "height", "words":[{"text","suffix","box"}], "full_text"}``.
"""

import statistics


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #

def _xs(box):
    return [p[0] for p in box]


def _ys(box):
    return [p[1] for p in box]


def _cx(box):
    xs = _xs(box)
    return (min(xs) + max(xs)) / 2


def _cy(box):
    ys = _ys(box)
    return (min(ys) + max(ys)) / 2


def _height(box):
    ys = _ys(box)
    return max(ys) - min(ys)


def _rect(x0, y0, x1, y1):
    """A 4-point polygon in the same vertex format the front-end expects."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# --------------------------------------------------------------------------- #
# column detection (projection profile)
# --------------------------------------------------------------------------- #

def _detect_gutters(boxed, width, force_n=None, bins=160):
    """Return the x positions that separate columns.

    Summed over the whole page height, a column shows up as a band of bins that
    some word covers, while the gutter between columns is a band that no word
    covers. We find those empty interior bands; their centers are the splits.
    """
    left = min(min(_xs(w["box"])) for w in boxed)
    right = max(max(_xs(w["box"])) for w in boxed)
    span = right - left
    if span <= 0:
        return []

    bin_w = span / bins
    coverage = [0] * bins
    for w in boxed:
        xs = _xs(w["box"])
        b0 = int((min(xs) - left) / bin_w)
        b1 = int((max(xs) - left) / bin_w)
        for b in range(max(0, b0), min(bins, b1 + 1)):
            coverage[b] += 1

    peak = max(coverage) or 1
    low = peak * 0.04  # tolerate a little speckle/noise inside a true gutter
    min_run = max(2, round(bins * 0.02))  # ignore narrow intra-line gaps

    # collect interior runs of low-coverage bins (gutters touching a margin
    # are page borders, not column separators)
    gutters = []
    b = 0
    while b < bins:
        if coverage[b] <= low:
            start = b
            while b < bins and coverage[b] <= low:
                b += 1
            end = b - 1
            touches_margin = start == 0 or end == bins - 1
            if not touches_margin and (end - start + 1) >= min_run:
                center = left + (start + end + 1) / 2 * bin_w
                gutters.append((center, end - start + 1))
        else:
            b += 1

    if force_n is not None and force_n >= 2:
        # keep the (force_n - 1) widest gutters; back-fill with even splits
        gutters.sort(key=lambda g: g[1], reverse=True)
        chosen = sorted(c for c, _ in gutters[: force_n - 1])
        while len(chosen) < force_n - 1:
            k = len(chosen) + 1
            chosen.append(left + span * k / force_n)
        return sorted(chosen)[: force_n - 1]

    return sorted(c for c, _ in gutters)


def _column_of(box, boundaries):
    cx = _cx(box)
    col = 0
    for bound in boundaries:
        if cx >= bound:
            col += 1
    return col


# --------------------------------------------------------------------------- #
# reading order from existing boxes (Google / Azure / NVIDIA+OpenCV)
# --------------------------------------------------------------------------- #

def reading_order_from_boxes(result, columns="auto"):
    """Re-sort ``result['words']`` into column-aware reading order.

    ``columns`` is "1" (no-op), "2"/"3" (forced) or "auto" (detect; only
    reorders when >=2 columns are found, so single-column pages are untouched).
    """
    words = result.get("words") or []
    boxed = [w for w in words if w.get("box") and len(w["box"]) >= 3]
    if columns == "1" or len(boxed) < 2:
        return result

    width = result.get("width") or max(max(_xs(w["box"])) for w in boxed)
    force_n = int(columns) if columns in ("2", "3") else None
    boundaries = _detect_gutters(boxed, width, force_n=force_n)

    if not boundaries and columns == "auto":
        return result  # genuinely single column — leave the engine order alone

    heights = [_height(w["box"]) for w in boxed if _height(w["box"]) > 0]
    line_h = statistics.median(heights) if heights else 1
    bucket = max(line_h * 0.6, 1)

    def key(w):
        return (
            _column_of(w["box"], boundaries),
            round(_cy(w["box"]) / bucket),
            _cx(w["box"]),
        )

    ordered = sorted(boxed, key=key)
    return _rebuild(result, ordered, boundaries, bucket)


def _rebuild(result, ordered, boundaries, bucket):
    """Recompute per-word suffixes (space vs newline) and the full text."""
    parts = []
    n = len(ordered)
    for i, w in enumerate(ordered):
        nxt = ordered[i + 1] if i + 1 < n else None
        same_line = (
            nxt is not None
            and _column_of(w["box"], boundaries) == _column_of(nxt["box"], boundaries)
            and round(_cy(w["box"]) / bucket) == round(_cy(nxt["box"]) / bucket)
        )
        w["suffix"] = " " if same_line else "\n"
        if i < n - 1:
            parts.append(w["text"] + w["suffix"])
        else:
            parts.append(w["text"])
    result["words"] = ordered
    result["full_text"] = "".join(parts).strip()
    return result


# --------------------------------------------------------------------------- #
# per-word sub-boxes within one OCR'd line
# --------------------------------------------------------------------------- #

def split_line_into_words(text, line_px):
    """Split a line's text into words, distributing the line box across them
    proportionally to character length (a cheap, deterministic per-word box)."""
    tokens = text.split()
    if not tokens:
        return []
    if not line_px:
        return [{"text": t, "box": None} for t in tokens]

    x0, y0, x1, y1 = line_px
    weights = [len(t) + 1 for t in tokens]  # +1 approximates the trailing space
    total = sum(weights)
    out = []
    cursor = x0
    for tok, wgt in zip(tokens, weights):
        w_px = (x1 - x0) * wgt / total
        out.append({"text": tok, "box": _rect(cursor, y0, cursor + w_px, y1)})
        cursor += w_px
    return out
