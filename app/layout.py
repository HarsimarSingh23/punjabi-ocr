"""Column-aware reading-order reconstruction.

Two entry points:

* ``reading_order_from_boxes`` — for engines that already return word boxes
  (Google, Azure). It detects columns from the boxes via a vertical projection
  profile and re-sorts the words into human reading order (column -> line -> x),
  then rebuilds the line breaks and full text. The box geometry is untouched.

* ``words_from_vision_json`` — for the vision-LLM engine, which we ask to return
  structured JSON (columns -> lines -> {text, box}). It scales the model's
  normalized 0-1000 boxes to real pixels and splits each line into per-word
  sub-boxes so the front-end animation stays fine-grained.

Both produce the same payload shape the rest of the app uses:
``{"width", "height", "words":[{"text","suffix","box"}], "full_text"}``.
"""

import json
import re
import statistics

# Vision LLMs are prompted to emit coordinates on a normalized 0-1000 grid.
VISION_COORD_SCALE = 1000.0


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
# reading order from existing boxes (Google / Azure)
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
# structured vision-LLM output -> words with boxes
# --------------------------------------------------------------------------- #

def extract_json(text):
    """Pull the first JSON object out of an LLM reply (handles ``` fences)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start : end + 1] if start != -1 and end > start else None
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


# Matches a single {"text": "...", "box": [x0, y0, x1, y1]} object even when the
# surrounding JSON is malformed or truncated (stops at the box's first ']', so a
# stray extra ']' or a cut-off tail don't matter).
_LINE_RE = re.compile(
    r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"box"\s*:\s*'
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def salvage_lines(text):
    """Recover {text, box} line objects from a broken/truncated JSON reply.

    The model occasionally emits invalid JSON (an extra bracket, a cut-off tail).
    Rather than throw the whole page away, pull out every well-formed line object
    in order. Returns a list of {"text", "box":[x0,y0,x1,y1]} or [] if none."""
    lines = []
    for m in _LINE_RE.finditer(text or ""):
        try:
            label = json.loads('"' + m.group(1) + '"')  # unescape JSON string
        except json.JSONDecodeError:
            label = m.group(1)
        if not label.strip():
            continue
        box = [float(m.group(i)) for i in range(2, 6)]
        lines.append({"text": label, "box": box})
    return lines


def words_from_salvaged(text, width, height):
    """Build the words payload from salvaged line objects (reading order is the
    model's emission order). Returns None if nothing could be recovered."""
    lines = salvage_lines(text)
    if not lines:
        return None
    return words_from_vision_json({"columns": [{"lines": lines}]}, width, height)


_TEXT_FIELD_RE = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"')


def text_from_reply(text):
    """Return clean human-readable text from a model reply, NEVER raw JSON.

    If the reply is (even broken) JSON, pull out the "text" field values and join
    them; otherwise return the reply as-is. Guards the user-facing output against
    ever showing JSON scaffolding."""
    if not text:
        return ""
    matches = _TEXT_FIELD_RE.findall(text)
    if matches:
        out = []
        for raw in matches:
            try:
                value = json.loads('"' + raw + '"')
            except json.JSONDecodeError:
                value = raw
            if value.strip():
                out.append(value.strip())
        if out:
            return "\n".join(out)
    return text.strip()


def _scale_box(box, width, height):
    """Scale a normalized [x0,y0,x1,y1] (0-1000) box to pixel coordinates."""
    if not box or len(box) != 4:
        return None
    sx, sy = width / VISION_COORD_SCALE, height / VISION_COORD_SCALE
    x0, y0, x1, y1 = box
    x0, x1 = sorted((x0 * sx, x1 * sx))
    y0, y1 = sorted((y0 * sy, y1 * sy))
    # clamp into the image
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return (x0, y0, x1, y1)


def _split_line_into_words(text, line_px):
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


def words_from_vision_json(data, width, height):
    """Convert structured vision JSON into ordered words with pixel boxes.

    Accepts ``{"columns":[{"lines":[{"text","box"}]}]}`` and tolerates a few
    shape variants (a bare ``lines`` list, or columns that are line lists).
    Returns ``None`` if no usable text is found so callers can fall back.
    """
    columns = data.get("columns")
    if columns is None and isinstance(data.get("lines"), list):
        columns = [{"lines": data["lines"]}]
    if not isinstance(columns, list):
        return None

    words = []
    for col in columns:
        lines = col.get("lines") if isinstance(col, dict) else col
        if not isinstance(lines, list):
            continue
        for line in lines:
            if not isinstance(line, dict):
                continue
            text = (line.get("text") or "").strip()
            if not text:
                continue
            line_px = _scale_box(line.get("box"), width, height)
            toks = _split_line_into_words(text, line_px)
            for i, tok in enumerate(toks):
                tok["suffix"] = "\n" if i == len(toks) - 1 else " "
                words.append(tok)

    if not words:
        return None

    full_text = "".join(
        w["text"] + (w["suffix"] if i < len(words) - 1 else "")
        for i, w in enumerate(words)
    ).strip()
    return {"width": width, "height": height, "words": words, "full_text": full_text}
