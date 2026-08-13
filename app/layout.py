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

# A real column gutter is wide compared to the text: at least this many median
# word-heights of clear space. Measured on the sample dictionary page, the true
# gutter runs ~1.1x the word height while inter-word gaps sit near 0.5x, so this
# sits between the two with margin on either side.
MIN_GUTTER_HEIGHTS = 0.8
# Below this many boxed words, "auto" leaves the engine's own order alone — a
# projection profile over two or three boxes describes those boxes, not a page
# layout. Kept low: the gutter width floor above is the real discriminator, and
# a short page with genuine columns should still be picked up.
MIN_BOXES_FOR_AUTO_COLUMNS = 4


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

def _detect_gutters(boxed, force_n=None, bins=160, min_gap_px=0.0):
    """Return the x positions that separate columns.

    Summed over the whole page height, a column shows up as a band of bins that
    some word covers, while the gutter between columns is a band that no word
    covers. We find those empty interior bands; their centers are the splits.

    A candidate band must also be at least ``min_gap_px`` wide. Width in bins
    alone is relative to the text's own x-span, so on a page with few words an
    ordinary inter-word gap covers a large share of the span and reads as a
    column break; measured against the text height it plainly doesn't.
    """
    if not boxed or bins < 2:
        return []
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
            run = end - start + 1
            touches_margin = start == 0 or end == bins - 1
            if not touches_margin and run >= min_run and run * bin_w >= min_gap_px:
                center = left + (start + end + 1) / 2 * bin_w
                gutters.append((center, run))
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
    boxed = [w for w in words if _has_box(w)]
    if columns == "1" or len(boxed) < 2:
        return result

    force_n = int(columns) if columns in ("2", "3") else None
    if force_n is None and len(boxed) < MIN_BOXES_FOR_AUTO_COLUMNS:
        return result  # too little text for a projection profile to mean anything

    heights = [_height(w["box"]) for w in boxed if _height(w["box"]) > 0]
    line_h = statistics.median(heights) if heights else 1
    bucket = max(line_h * 0.6, 1)

    boundaries = _detect_gutters(
        boxed, force_n=force_n, min_gap_px=line_h * MIN_GUTTER_HEIGHTS
    )

    if not boundaries and columns == "auto":
        return result  # genuinely single column — leave the engine order alone

    def key(w):
        return (
            _column_of(w["box"], boundaries),
            round(_cy(w["box"]) / bucket),
            _cx(w["box"]),
        )

    ordered = _reinsert_unboxed(words, sorted(boxed, key=key))
    return _rebuild(result, ordered, boundaries, bucket)


def _has_box(word) -> bool:
    box = word.get("box")
    return bool(box) and len(box) >= 3


def _reinsert_unboxed(original, ordered_boxed):
    """Splice box-less words back into the reordered sequence.

    Only words with boxes can be placed geometrically, but dropping the rest
    would silently delete text from both the word list and ``full_text`` — a
    partial result that looks complete. Each box-less word instead follows the
    boxed word it originally followed (or leads, if it came first), so it stays
    attached to its neighbour wherever that neighbour lands.
    """
    if len(ordered_boxed) == len(original):
        return ordered_boxed

    trailing: dict[int, list] = {}
    leading = []
    anchor = None
    for word in original:
        if _has_box(word):
            anchor = id(word)
        elif anchor is None:
            leading.append(word)
        else:
            trailing.setdefault(anchor, []).append(word)

    out = list(leading)
    for word in ordered_boxed:
        out.append(word)
        out.extend(trailing.get(id(word), ()))
    return out


def _rebuild(result, ordered, boundaries, bucket):
    """Recompute per-word suffixes (space vs newline) and the full text.

    Box-less words inherit a plain space: they have no geometry to compare, so
    guessing a line break for them would invent structure that isn't there.
    """
    parts = []
    n = len(ordered)
    for i, w in enumerate(ordered):
        nxt = ordered[i + 1] if i + 1 < n else None
        if nxt is None:
            same_line = False
        elif not (_has_box(w) and _has_box(nxt)):
            same_line = True  # unknown geometry — keep it on the current line
        else:
            same_line = _column_of(w["box"], boundaries) == _column_of(
                nxt["box"], boundaries
            ) and round(_cy(w["box"]) / bucket) == round(_cy(nxt["box"]) / bucket)
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
    if not line_px or len(line_px) != 4:
        return [{"text": t, "box": None} for t in tokens]

    x0, y0, x1, y1 = line_px
    if x1 <= x0:  # degenerate line box — better no box than a zero-width one
        return [{"text": t, "box": None} for t in tokens]
    weights = [len(t) + 1 for t in tokens]  # +1 approximates the trailing space
    total = sum(weights)
    out = []
    cursor = x0
    for tok, wgt in zip(tokens, weights):
        w_px = (x1 - x0) * wgt / total
        out.append({"text": tok, "box": _rect(cursor, y0, cursor + w_px, y1)})
        cursor += w_px
    return out
