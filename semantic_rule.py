# semantic_rules.py

from bs4 import BeautifulSoup
import re


# ---------------------------------------------------------------------------
# Heading level thresholds
# ---------------------------------------------------------------------------
# How many of the largest distinct font sizes are treated as headings.
# This is relative (top-N), so it adapts to whatever font sizes appear in
# the document — no hardcoded point values.
HEADING_LEVELS = 3

# Minimum word count ceiling for a block to be classified as a heading.
# Blocks with more words than this are never treated as headings even if the
# font size qualifies.
MAX_HEADING_WORD_COUNT = 12

# Bullet / ordered-list marker pattern
_BULLET_PATTERN = re.compile(
    r"^[\u2022\u2023\u2043\u2219\u22C5\u25AA\u25AB\u25CF\u25E6\u25CB"
    r"\-\*•▪●◦oO]\s+"
)
_ORDERED_PATTERN = re.compile(r"^(\d+|[a-zA-Z])[\.\)]\s+")


def _is_list_text(text: str) -> bool:
    """Return True if *text* starts with a recognisable list marker."""
    t = text.strip()
    return bool(_BULLET_PATTERN.match(t) or _ORDERED_PATTERN.match(t))


def _strip_bullet(text: str) -> str:
    """Remove a leading bullet / ordered marker from list-item text."""
    t = text.strip()
    t = _BULLET_PATTERN.sub("", t)
    t = _ORDERED_PATTERN.sub("", t)
    return t.strip()


# ---------------------------------------------------------------------------
# YOLO layout-type → semantic label map
# ---------------------------------------------------------------------------
# All values are lowercase YOLO class names as produced by the model.
# Add entries here whenever the model outputs a new class that should map to
# a specific semantic tag.  "text" / unknown → "P" (paragraph) by default.
YOLO_TO_LABEL = {
    "title": "H1",
    "section-header": "H2",
    "section_header": "H2",
    "paragraph": "P",
    "text": "P",
    "list": "LI",
    "list-item": "LI",
    "list_item": "LI",
    "table": "TABLE",
    "figure": "FIGURE",
    "caption": "P",
    "formula": "P",
    "page-header": "H3",
    "page_header": "H3",
    "page-footer": "P",
    "page_footer": "P",
}


def _yolo_label(layout_type: str) -> str | None:
    """
    Map a YOLO class string to a semantic label.

    Returns ``None`` when no mapping exists so the caller can fall through
    to the heuristic path.
    """
    key = (layout_type or "").strip().lower()
    return YOLO_TO_LABEL.get(key)


def build_semantic_html_from_clean(clean_html: str) -> str:
    """
    Convert positioned-text HTML into a semantic HTML document.

    Input
    -----
    clean_html : str
        HTML produced by the PDF converter containing ``.page-container``
        wrappers and ``.positioned-text`` elements decorated with
        ``data-type`` attributes (YOLO layout class).

    Output
    ------
    str
        A new ``<html><body>…</body></html>`` string using semantic tags
        (``<h1>``–``<h3>``, ``<p>``, ``<ul>/<li>``).

    Design notes
    ------------
    * YOLO ``data-type`` is the primary classification signal.
    * Font-size heuristics are used only as a fallback when YOLO gives a
      generic "text" / unknown label.
    * All thresholds are controlled by module-level constants — nothing is
      hard-coded inline.
    """

    soup = BeautifulSoup(clean_html, "html.parser")
    text_divs = soup.select(".positioned-text")

    # ------------------------------------------------------------------
    # 1. Collect block metadata
    # ------------------------------------------------------------------
    blocks = []
    for div in text_divs:
        style = div.get("style", "")
        text = div.get_text(strip=True)
        layout_type = div.get("data-type", "text")

        # Parse font-size from inline style
        m = re.search(r"font-size:\s*([\d\.]+)px", style)
        font_size = float(m.group(1)) if m else 12.0

        is_bold = "font-weight: bold" in style or "font-weight: 700" in style
        is_italic = "font-style: italic" in style

        blocks.append(
            {
                "text": text,
                "style": style,
                "font_size": font_size,
                "bold": is_bold,
                "italic": is_italic,
                "layout_type": layout_type,
            }
        )

    # ------------------------------------------------------------------
    # 2. Build font-size statistics for the heuristic fallback
    # ------------------------------------------------------------------
    if blocks:
        # Distinct sizes in descending order; the top-N are heading sizes.
        distinct_sizes = sorted(
            {b["font_size"] for b in blocks}, reverse=True
        )
        heading_sizes = distinct_sizes[:HEADING_LEVELS]
    else:
        heading_sizes = []

    # ------------------------------------------------------------------
    # 3. Label every block
    # ------------------------------------------------------------------
    for b in blocks:
        text = b["text"]
        layout_type = b["layout_type"]

        if not text:
            b["label"] = "P"
            continue

        # --- Primary: YOLO class ---
        yolo = _yolo_label(layout_type)
        if yolo is not None and yolo not in ("P",):
            # YOLO gave a definitive non-paragraph label → use it directly.
            b["label"] = yolo
            continue

        # For explicit "paragraph" / "text" from YOLO we still apply the
        # list-regex check, because YOLO sometimes misses list items.

        # --- Secondary: list-marker regex ---
        if _is_list_text(text):
            b["label"] = "LI"
            continue

        # --- Tertiary: font-size heuristic (headings only) ---
        if (
            b["font_size"] in heading_sizes
            and len(text.split()) <= MAX_HEADING_WORD_COUNT
        ):
            idx = heading_sizes.index(b["font_size"])
            level = min(idx + 1, HEADING_LEVELS)
            b["label"] = f"H{level}"
        else:
            b["label"] = "P"

    # ------------------------------------------------------------------
    # 4. Emit semantic HTML
    # ------------------------------------------------------------------
    out_parts: list[str] = ["<html><body>"]
    list_open = False

    for b in blocks:
        text = b["text"]
        label = b["label"]

        # Skip non-content structural labels (tables and figures are
        # rendered separately by the main converter; skip here).
        if label in ("TABLE", "FIGURE"):
            if list_open:
                out_parts.append("</ul>")
                list_open = False
            continue

        if label == "LI":
            if not list_open:
                out_parts.append("<ul>")
                list_open = True
            clean = _strip_bullet(text)
            out_parts.append(f"<li>{clean}</li>")
            continue

        # Close any open list before emitting a non-list element
        if list_open:
            out_parts.append("</ul>")
            list_open = False

        if label.startswith("H") and len(label) == 2 and label[1].isdigit():
            lvl = label[1]
            out_parts.append(f"<h{lvl}>{text}</h{lvl}>")
        else:
            out_parts.append(f"<p>{text}</p>")

    # Close any still-open list at end of document
    if list_open:
        out_parts.append("</ul>")

    out_parts.append("</body></html>")
    return "\n".join(out_parts)