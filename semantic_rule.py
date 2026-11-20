# semantic_rule.py
"""
Module: semantic_rule
=====================
Provides reading-order sort, stack-based section-tree construction, and
two-track HTML/JSON emitters used by app.py.

Track 1 (visual) stays pixel-perfect and absolute-positioned (untouched).
Track 2 (semantic) produces clean, hierarchically nested HTML/JSON.

Known limitations (out of scope for this iteration):
- Complex multi-column layouts with column-spanning headers may mis-order.
- True semantic accuracy deferred to planned LayoutLM/Donut fine-tuning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADING_LEVELS = 3
MAX_HEADING_WORD_COUNT = 12

# Line-tolerance: elements whose y0 values are within this many px of each
# other are treated as the same visual line.
LINE_TOLERANCE_PX = 4

# Gap threshold for column detection (fraction of page width).
COLUMN_GAP_RATIO = 0.15

# YOLO class name -> semantic label (DocLayNet + PubLayNet variants)
YOLO_TO_LABEL: Dict[str, str] = {
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
    "picture": "FIGURE",
    "caption": "P",
    "formula": "P",
    "page-header": "H3",
    "page_header": "H3",
    "page-footer": "FOOTER",
    "page_footer": "FOOTER",
    # PubLayNet labels
    "Figure": "FIGURE",
    "Table": "TABLE",
    "List": "LI",
    "Title": "H1",
    "Text": "P",
}

_BULLET_PATTERN = re.compile(
    r"^[\u2022\u2023\u2043\u2219\u22C5\u25AA\u25AB\u25CF\u25E6\u25CB\-\*\u2013\u2014oO]\s+"
)
_ORDERED_PATTERN = re.compile(r"^(\d+|[a-zA-Z])[.)]\s+")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _yolo_label(layout_type: str) -> Optional[str]:
    return YOLO_TO_LABEL.get((layout_type or "").strip())


def _is_list_text(text: str) -> bool:
    t = (text or "").strip()
    return bool(_BULLET_PATTERN.match(t) or _ORDERED_PATTERN.match(t))


def _strip_bullet(text: str) -> str:
    t = (text or "").strip()
    t = _BULLET_PATTERN.sub("", t)
    t = _ORDERED_PATTERN.sub("", t)
    return t.strip()


def _parse_coord(style: str, prop: str) -> float:
    m = re.search(rf"{prop}:\s*([\d.]+)px", style)
    return float(m.group(1)) if m else 0.0


def _parse_font_size(style: str) -> float:
    return _parse_coord(style, "font-size")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContentElement:
    """A single leaf element (paragraph, list-item, table, figure, etc.)."""
    tag: str        # p | li | table | figure | footer | h1 | h2 | h3
    text: str       # plain text content
    html: str       # rich inner HTML (spans with styling)
    style: str      # original inline style from positioned HTML
    data_type: str  # original data-type attribute
    x0: float = 0.0
    y0: float = 0.0
    rows: List[Any] = field(default_factory=list)   # TABLE rows
    src: str = ""        # FIGURE src
    caption: str = ""    # FIGURE caption


@dataclass
class Section:
    """A section of the document, with optional heading and children."""
    level: int                          # 0 = root; 1/2/3 = h1/h2/h3
    heading: Optional[ContentElement]
    children: List[Any] = field(default_factory=list)   # Section | ContentElement


# ---------------------------------------------------------------------------
# Step 1 -- Reading-order sort (column-aware)
# ---------------------------------------------------------------------------

def _detect_columns(elements: List[ContentElement], page_width: float) -> List[tuple]:
    """
    Detect column x-ranges from element x0 positions using gap-based histogram.
    Returns list of (col_x_min, col_x_max) sorted left-to-right.
    NOTE: column-spanning headers are NOT handled -- flag as future work.
    """
    if not elements or page_width <= 0:
        return [(0, page_width or float("inf"))]

    x0s = sorted(set(e.x0 for e in elements))
    if len(x0s) <= 1:
        return [(0, page_width)]

    gap_threshold = COLUMN_GAP_RATIO * page_width
    column_starts = [x0s[0]]
    for a, b in zip(x0s, x0s[1:]):
        if b - a > gap_threshold:
            column_starts.append(b)

    cols = []
    for i, start in enumerate(column_starts):
        end = column_starts[i + 1] if i + 1 < len(column_starts) else page_width + 1
        cols.append((start, end))
    return cols


def reading_order_sort(
    elements: List[ContentElement],
    page_width: float = 0.0,
) -> List[ContentElement]:
    """
    Sort elements into reading order: left-to-right columns, top-to-bottom rows.
    Within each column, elements are sorted by y0 (with LINE_TOLERANCE_PX clustering).
    """
    cols = _detect_columns(elements, page_width)
    bucketed: Dict[int, List[ContentElement]] = {i: [] for i in range(len(cols))}

    for el in elements:
        placed = False
        for i, (col_min, col_max) in enumerate(cols):
            if col_min <= el.x0 < col_max:
                bucketed[i].append(el)
                placed = True
                break
        if not placed:
            bucketed[len(cols) - 1].append(el)

    result: List[ContentElement] = []
    for i in range(len(cols)):
        col_els = sorted(bucketed[i], key=lambda e: (
            round(e.y0 / max(LINE_TOLERANCE_PX, 1)) * LINE_TOLERANCE_PX,
            e.x0,
        ))
        result.extend(col_els)

    return result


# ---------------------------------------------------------------------------
# Step 2 -- Stack-based section tree construction
# ---------------------------------------------------------------------------

def build_section_tree(
    elements: List[ContentElement],
    page_width: float = 0.0,
) -> Section:
    """
    Convert a flat reading-order list of ContentElements into a nested Section tree.

    - Heading encountered -> open new Section at that level (pop deeper/equal first).
    - Everything else -> child of the current innermost Section.
    - Orphan content before any heading -> attached to root Section.
    - Running headers/footers (tag == 'footer') -> excluded from body tree.
    - Heading level skips (H1 -> H3) handled naturally by the >= pop condition.
    """
    ordered = reading_order_sort(elements, page_width)

    root = Section(level=0, heading=None)
    stack: List[Section] = [root]

    for el in ordered:
        tag = el.tag.lower()

        if tag == "footer":
            continue

        if tag in ("h1", "h2", "h3"):
            level = int(tag[1])
            while stack[-1].level >= level:
                stack.pop()
            new_section = Section(level=level, heading=el)
            stack[-1].children.append(new_section)
            stack.append(new_section)
        else:
            stack[-1].children.append(el)

    return root


# ---------------------------------------------------------------------------
# Step 3a -- Emit clean semantic HTML (Track 2)
# ---------------------------------------------------------------------------

def _apply_semantic_emphasis(html_fragment: str) -> str:
    """Wrap inline-styled bold/italic spans (as emitted by Track 1) with real
    <strong>/<em> tags, so Track 2's clean semantic HTML captures emphasis as
    actual markup rather than only inline CSS."""
    if not html_fragment or "<span" not in html_fragment:
        return html_fragment
    frag = BeautifulSoup(html_fragment, "html.parser")
    for span in frag.find_all("span"):
        style = span.get("style", "") or ""
        compact = style.replace(" ", "")
        is_bold = "font-weight:700" in compact or "font-weight:bold" in compact
        is_italic = "font-style:italic" in compact
        if is_bold:
            strong = frag.new_tag("strong")
            span.wrap(strong)
        if is_italic:
            em = frag.new_tag("em")
            span.wrap(em)
    return str(frag)


def _emit_element_html(el: ContentElement, out: List[str]) -> None:
    """Render a single ContentElement as clean (un-positioned) HTML."""
    tag = el.tag.lower()
    text = _apply_semantic_emphasis(el.html) or el.text

    if tag in ("h1", "h2", "h3"):
        out.append(f"<{tag}>{text}</{tag}>")
    elif tag == "p":
        out.append(f"<p>{text}</p>")
    elif tag == "table":
        out.append("<table>")
        for row in el.rows:
            out.append("<tr>")
            for cell in row:
                if isinstance(cell, dict):
                    cell_text = cell.get("text", "")
                    attrs = ""
                    rowspan = cell.get("rowspan")
                    colspan = cell.get("colspan")
                    if rowspan:
                        attrs += f' rowspan="{rowspan}"'
                    if colspan:
                        attrs += f' colspan="{colspan}"'
                else:
                    cell_text = str(cell)
                    attrs = ""
                out.append(f"<td{attrs}>{cell_text}</td>")
            out.append("</tr>")
        out.append("</table>")
    elif tag == "figure":
        if el.src:
            out.append(f'<figure><img src="{el.src}" alt="figure"/>')
            if el.caption:
                out.append(f"<figcaption>{el.caption}</figcaption>")
            out.append("</figure>")


def _emit_section_html(section: Section, out: List[str]) -> None:
    """Recursively render a Section (and its children) as clean nested HTML."""
    is_root = section.level == 0

    if not is_root:
        out.append("<section>")
        if section.heading:
            _emit_element_html(section.heading, out)

    li_buffer: List[str] = []

    def flush_li_buffer():
        if li_buffer:
            out.append("<ul>")
            out.extend(li_buffer)
            out.append("</ul>")
            li_buffer.clear()

    for child in section.children:
        if isinstance(child, Section):
            flush_li_buffer()
            _emit_section_html(child, out)
        elif isinstance(child, ContentElement):
            tag = child.tag.lower()
            if tag == "li":
                clean = _strip_bullet(child.text)
                li_buffer.append(f"<li>{clean}</li>")
            else:
                flush_li_buffer()
                _emit_element_html(child, out)

    flush_li_buffer()

    if not is_root:
        out.append("</section>")


def build_semantic_html(
    elements: List[ContentElement],
    page_width: float = 0.0,
    page_number: int = 1,
) -> str:
    """Build clean nested semantic HTML fragment for one page."""
    tree = build_section_tree(elements, page_width)
    parts: List[str] = [f'<div class="semantic-page" data-page="{page_number}">']
    _emit_section_html(tree, parts)
    parts.append("</div>")
    return "\n".join(parts)


def build_full_semantic_html(pages: List[Dict[str, Any]]) -> str:
    """
    Build a complete HTML document from a list of page dicts:
        {"elements": List[ContentElement], "page_width": float, "page_number": int}
    """
    body_parts = []
    for page in pages:
        body_parts.append(build_semantic_html(
            page["elements"],
            page.get("page_width", 0.0),
            page.get("page_number", 1),
        ))
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head><meta charset=\"UTF-8\">\n"
        "<title>Semantic Document</title>\n</head>\n<body>\n"
        + "\n".join(body_parts)
        + "\n</body>\n</html>"
    )


# ---------------------------------------------------------------------------
# Step 3b -- Emit nested JSON tree (Track 2)
# ---------------------------------------------------------------------------

def _section_to_dict(section: Section) -> Dict[str, Any]:
    """Recursively convert a Section tree node to a nested dict."""
    result: Dict[str, Any] = {}

    if section.heading:
        result["heading"] = {"level": section.level, "text": section.heading.text}

    elements_list: List[Any] = []
    subsections_list: List[Any] = []
    li_group: List[str] = []

    def flush_li():
        if li_group:
            elements_list.append({
                "type": "list",
                "items": [{"text": t, "children": []} for t in li_group],
            })
            li_group.clear()

    for child in section.children:
        if isinstance(child, Section):
            flush_li()
            subsections_list.append(_section_to_dict(child))
        elif isinstance(child, ContentElement):
            tag = child.tag.lower()
            if tag == "li":
                li_group.append(_strip_bullet(child.text))
            elif tag in ("h1", "h2", "h3"):
                flush_li()
                elements_list.append({"type": "heading", "level": int(tag[1]), "text": child.text})
            elif tag == "p":
                flush_li()
                elements_list.append({"type": "paragraph", "text": child.text})
            elif tag == "table":
                flush_li()
                elements_list.append({
                    "type": "table",
                    "rows": [
                        {"cells": [
                            {
                                "text": c.get("text", "") if isinstance(c, dict) else str(c),
                                "rowspan": c.get("rowspan") if isinstance(c, dict) else None,
                                "colspan": c.get("colspan") if isinstance(c, dict) else None,
                            }
                            for c in row
                        ]}
                        for row in child.rows
                    ],
                })
            elif tag == "figure":
                flush_li()
                elements_list.append({"type": "figure", "src": child.src, "caption": child.caption})

    flush_li()
    result["elements"] = elements_list
    result["subsections"] = subsections_list
    return result


def build_nested_json(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build the nested document JSON from a list of page dicts:
        {"elements": List[ContentElement], "page_width": float, "page_number": int}
    Returns: {"document": {"pages": [{"page_number": N, "sections": [...]}]}}
    """
    pages_data = []
    for page in pages:
        tree = build_section_tree(page["elements"], page.get("page_width", 0.0))
        top_sections = []
        orphan_elements: List[ContentElement] = []

        for child in tree.children:
            if isinstance(child, Section):
                if orphan_elements:
                    orphan_sec = Section(level=0, heading=None)
                    orphan_sec.children = orphan_elements[:]
                    top_sections.append(_section_to_dict(orphan_sec))
                    orphan_elements.clear()
                top_sections.append(_section_to_dict(child))
            else:
                orphan_elements.append(child)

        if orphan_elements:
            orphan_sec = Section(level=0, heading=None)
            orphan_sec.children = orphan_elements
            top_sections.append(_section_to_dict(orphan_sec))

        pages_data.append({"page_number": page.get("page_number", 1), "sections": top_sections})

    return {"document": {"pages": pages_data}}


# ---------------------------------------------------------------------------
# Legacy compatibility -- build_semantic_html_from_clean
# ---------------------------------------------------------------------------

def build_semantic_html_from_clean(clean_html: str) -> str:
    """
    Legacy interface: convert positioned-text HTML (with .positioned-text
    elements + data-type) into clean nested semantic HTML.
    """
    soup = BeautifulSoup(clean_html, "html.parser")
    pages_data = []

    for page_idx, page_div in enumerate(soup.select(".page-container"), start=1):
        page_style = page_div.get("style", "")
        pw_m = re.search(r"width:\s*([\d.]+)px", page_style)
        page_width = float(pw_m.group(1)) if pw_m else 0.0

        elements: List[ContentElement] = []
        for div in page_div.select(".positioned-text"):
            style = div.get("style", "")
            text = div.get_text(strip=True)
            layout_type = div.get("data-type", "text")
            x0 = _parse_coord(style, "left")
            y0 = _parse_coord(style, "top")
            yolo = _yolo_label(layout_type) or "P"
            el = ContentElement(
                tag=yolo.lower(),
                text=text,
                html=str(div.decode_contents()),
                style=style,
                data_type=layout_type,
                x0=x0,
                y0=y0,
            )
            elements.append(el)

        pages_data.append({"elements": elements, "page_width": page_width, "page_number": page_idx})

    return build_full_semantic_html(pages_data)
