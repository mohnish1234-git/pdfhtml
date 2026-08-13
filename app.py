from flask import Flask, request, render_template_string, render_template, jsonify, send_file
import fitz  # PyMuPDF
import pdfplumber
import os
import html
import re
import time
import base64
import math
import threading
import uuid
import zipfile
from collections import Counter
from typing import Any, Dict, List, Optional
from validation import validate_pdf
from io import BytesIO
from fontTools.ttLib import TTFont
from bs4 import BeautifulSoup
import json
import shutil   
from PIL import Image
from ultralytics import YOLO
from icon_extractor import extract_icons_from_pdf
from pdf_analytics import generate_analytics_data

# =====================================================
# CONFIG
# =====================================================

DPI = 200

# Target pixel width for the rendered HTML page.
# Scale for all coordinates is computed as TARGET_WIDTH / page_width_in_points.
# Increase for higher-resolution output; decrease for smaller/faster output.
TARGET_WIDTH = 960

TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "intersection_tolerance": 5,
    "join_tolerance": 5,
    "snap_tolerance": 3,
    "edge_min_length": 20,
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
}

# Minimum area ratio for a filled rectangle to be emitted as a background element.
# Rectangles smaller than this fraction of the page area are ignored.
MIN_BACKGROUND_AREA_RATIO = 0.01

# Minimum coverage for background-color matching on spans/blocks.
FILL_COVERAGE_SPAN = 0.45   # looser: catch partial backgrounds under text
FILL_COVERAGE_CELL = 0.75   # tighter: table cells need a clear fill
FILL_COVERAGE_TABLE = 0.80  # whole-table background

# Ratio of page area at which a rectangle is treated as the full-page background.
PAGE_BG_AREA_RATIO = 0.80

yolo_model = YOLO("yolov12l-doclaynet.pt")
print("YOLO Loaded")


def render_page_image(page, dpi=200):
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def detect_layout_yolo(pil_img):
    results = yolo_model(pil_img, conf=0.3)[0]
    boxes = []
    for b in results.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls = int(b.cls[0])
        boxes.append({
            "bbox": (x1, y1, x2, y2),
            "class": yolo_model.names[cls]
        })
    return boxes


def match_span_to_box(span_bbox, boxes, sx, sy):
    cx = (span_bbox[0] + span_bbox[2]) / 2 * sx
    cy = (span_bbox[1] + span_bbox[3]) / 2 * sy
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return str(b.get("class", "text")).lower()
    return "text"


def match_bbox_to_box(rect_bbox, boxes, sx, sy):
    """Match a PDF-space bbox to the YOLO box with highest overlap."""
    if not rect_bbox or not boxes:
        return None

    rx0, ry0, rx1, ry1 = rect_bbox
    px0, py0, px1, py1 = rx0 * sx, ry0 * sy, rx1 * sx, ry1 * sy
    rect_area = max(0.0, px1 - px0) * max(0.0, py1 - py0)
    if rect_area <= 0:
        return None

    best_box = None
    best_score = 0.0
    for b in boxes:
        bx0, by0, bx1, by1 = b["bbox"]
        ix0 = max(px0, bx0)
        iy0 = max(py0, by0)
        ix1 = min(px1, bx1)
        iy1 = min(py1, by1)
        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            continue
        score = inter / rect_area
        if score > best_score:
            best_score = score
            best_box = b
    return best_box


TEXT_FONT_ITALIC = 2
TEXT_FONT_BOLD = 16


def span_font_style(span, internal_font=""):
    """Return (is_bold, is_italic) from PyMuPDF span flags, then font name."""
    flags = span.get("flags", 0)
    is_bold = bool(flags & TEXT_FONT_BOLD)
    is_italic = bool(flags & TEXT_FONT_ITALIC)
    if not is_bold and internal_font:
        is_bold = "bold" in internal_font.lower()
    if not is_italic and internal_font:
        is_italic = (
            "italic" in internal_font.lower()
            or "oblique" in internal_font.lower()
        )
    return is_bold, is_italic


def span_color_hex(span):
    """Return 6-digit hex color from span."""
    c = span.get("color", 0)
    if isinstance(c, int):
        c = c & 0xFFFFFF
    return "#{:06x}".format(c)


DATA_TYPE_TO_SEMANTIC_TAG = {
    "title": "h1",
    "section-header": "h2",
    "section_header": "h2",
    "text": "p",
    "paragraph": "p",
    "list": "li",
    "list-item": "li",
    "list_item": "li",
    "table": "div",
    "figure": "div",
    "caption": "p",
    "formula": "p",
    "page-header": "h3",
    "page_header": "h3",
    "page-footer": "footer",
    "page_footer": "footer",
}

def _resolve_font_family(internal_font, font_name_map):
    """
    Resolve a PDF internal font name to a CSS font-family string.

    Lookup order:
    1. Exact match on internal_font
    2. Strip common subset prefix (ABCDEF+FontName → FontName)
    3. Strip Bold/Italic/Oblique suffixes and retry
    4. Fall back to the cleaned-up internal name itself
    """
    if not internal_font:
        return "Arial"

    # 1. Exact match
    if internal_font in font_name_map:
        return font_name_map[internal_font]

    # 2. Strip PDF subset prefix (6 uppercase letters + '+')
    stripped = re.sub(r'^[A-Z]{6}\+', '', internal_font)
    if stripped in font_name_map:
        return font_name_map[stripped]

    # 3. Strip style suffixes
    norm = re.sub(r'[,\-_ ]*(bold|italic|oblique|regular|roman)', '', stripped, flags=re.IGNORECASE).strip()
    if norm in font_name_map:
        return font_name_map[norm]

    # 4. Try a case-insensitive scan
    lower = stripped.lower()
    for key, val in font_name_map.items():
        if key.lower() == lower:
            return val

    # 5. Return the cleaned name as a best-effort font-family
    return norm if norm else stripped


def strip_bullet_from_li(text):
    t = (text or "").strip()
    t = _BULLET_PATTERN.sub("", t)
    t = _NUMBERED_PREFIX.sub("", t)
    return t.strip()


def get_semantic_tag_for_data_type(data_type):
    key = (data_type or "text").lower().strip().replace("_", "-")
    return DATA_TYPE_TO_SEMANTIC_TAG.get(key, "p")


def _median(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def block_line_metrics(block_lines):
    metrics = []
    for line_spans in block_lines:
        xs0, ys0, xs1, ys1 = [], [], [], []
        text_len = 0
        for s in line_spans:
            x0, y0, x1, y1 = s.get("bbox", (0, 0, 0, 0))
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)
            text_len += len((s.get("text", "") or "").strip())
        if xs0 and xs1:
            metrics.append({
                "x0": min(xs0), "y0": min(ys0),
                "x1": max(xs1), "y1": max(ys1),
                "text_len": text_len,
            })
    metrics.sort(key=lambda m: m["y0"])
    return metrics


def estimate_line_height_ratio(block_lines, font_size):
    metrics = block_line_metrics(block_lines)
    if len(metrics) < 2:
        return 1.25
    deltas = []
    for i in range(len(metrics) - 1):
        dy = metrics[i + 1]["y0"] - metrics[i]["y0"]
        if dy > 0:
            deltas.append(dy)
    med_dy = _median(deltas)
    if not med_dy:
        return 1.25
    base = max(1e-3, float(font_size))
    ratio = med_dy / base
    return max(1.0, min(2.0, ratio))


def is_justified_paragraph(block_lines, block_bbox):
    metrics = block_line_metrics(block_lines)
    if len(metrics) < 3:
        return False
    bx0, _, bx1, _ = block_bbox
    block_w = max(1e-3, bx1 - bx0)
    candidate_lines = [m for m in metrics[:-1] if m["text_len"] >= 20]
    if len(candidate_lines) < 2:
        return False
    fill_ratios = [max(0.0, (m["x1"] - m["x0"]) / block_w) for m in candidate_lines]
    left_offsets = [abs(m["x0"] - bx0) / block_w for m in candidate_lines]
    right_offsets = [abs(bx1 - m["x1"]) / block_w for m in candidate_lines]
    med_fill = _median(fill_ratios) or 0.0
    med_left = _median(left_offsets) or 1.0
    med_right = _median(right_offsets) or 1.0
    return med_fill >= 0.90 and med_left <= 0.06 and med_right <= 0.08


def visualize_yolo_boxes(pil_img, boxes, save_path=None):
    import cv2
    import numpy as np
    img = np.array(pil_img)
    COLORS = {
        "title": (255, 0, 0),
        "text": (0, 255, 0),
        "list": (0, 0, 255),
        "table": (255, 255, 0),
        "figure": (255, 0, 255),
    }
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        cls = b["class"]
        color = COLORS.get(cls, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, cls, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    out_img = Image.fromarray(img)
    if save_path:
        out_img.save(save_path)
    return out_img


def rgb_tuple_to_hex(rgb):
    if not rgb:
        return None
    r, g, b = [max(0, min(255, int(round(channel * 255)))) for channel in rgb[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def extract_filled_rectangles(page):
    """Collect filled vector rectangles for background-color reuse."""
    filled_rects = []
    try:
        drawings = page.get_drawings()
    except Exception as e:
        print(f"[Filled rect extraction warning] page {page.number + 1}: {e}")
        return filled_rects

    for drawing in drawings:
        rect = drawing.get("rect")
        fill = drawing.get("fill")
        fill_opacity = drawing.get("fill_opacity", 1)
        if not rect or fill is None or fill_opacity <= 0:
            continue
        width = max(0, rect.x1 - rect.x0)
        height = max(0, rect.y1 - rect.y0)
        area = width * height
        if area <= 1:
            continue
        fill_color = rgb_tuple_to_hex(fill)
        if not fill_color:
            continue
        filled_rects.append({
            "bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
            "color": fill_color,
            "opacity": round(fill_opacity, 3),
            "area": area,
        })
    return filled_rects


def pick_fill_for_bbox(target_bbox, filled_rects, min_coverage=0.55):
    """
    Pick the most specific filled rectangle color for a target bbox.

    "Most specific" means the smallest rect that still covers the target with
    at least *min_coverage* of the target area.  When multiple rects fully
    contain the target (e.g. a page-wide tint AND a cell-level band), the
    smallest one wins because it is the most localised styling.  This prevents
    a broad background wash from overriding a narrower panel or cell colour.

    Scoring:  1 / fill_area  (smaller rect → higher score) × opacity
    Coverage gate:  intersection / target_area  ≥  min_coverage
    """
    if not target_bbox or not filled_rects:
        return None

    tx0, ty0, tx1, ty1 = target_bbox
    target_area = max(0.0, tx1 - tx0) * max(0.0, ty1 - ty0)
    if target_area <= 0:
        return None

    best = None
    best_score = 0.0
    for fill in filled_rects:
        fx0, fy0, fx1, fy1 = fill["bbox"]
        fill_area = max(1.0, (fx1 - fx0) * (fy1 - fy0))
        ix0 = max(tx0, fx0)
        iy0 = max(ty0, fy0)
        ix1 = min(tx1, fx1)
        iy1 = min(ty1, fy1)
        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            continue
        coverage = inter / target_area
        if coverage < min_coverage:
            continue
        # Prefer the smallest (most specific) enclosing rect.
        # Divide by fill_area so a tightly-fitted cell beats a page-wide wash.
        score = (coverage / fill_area) * max(fill.get("opacity", 1.0), 0.01)
        if score > best_score:
            best_score = score
            best = fill
    return best


def extract_background_shapes(page, scale):
    """
    Extract filled vector drawings from the page so background colors/design
    are preserved behind semantic text/image elements.

    Returns
    -------
    page_bg_color : str or None
        Hex color of any rectangle that covers ≥ PAGE_BG_AREA_RATIO of the page.
    background_elements : list[str]
        Absolute-positioned <div> strings ready to inject into the page container.
    """
    background_elements = []
    page_bg_color = None

    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return page_bg_color, background_elements

    try:
        drawings = page.get_drawings()
    except Exception as e:
        print(f"[Background extraction warning] page {page.number + 1}: {e}")
        return page_bg_color, background_elements

    # Sort largest-first so full-page backgrounds are detected before panels.
    sorted_drawings = sorted(
        drawings,
        key=lambda d: (
            max(0, (d.get("rect").x1 - d.get("rect").x0)
                * (d.get("rect").y1 - d.get("rect").y0))
            if d.get("rect") else 0
        ),
        reverse=True,
    )

    pending = []

    for drawing in sorted_drawings:
        rect = drawing.get("rect")
        fill = drawing.get("fill")
        fill_opacity = drawing.get("fill_opacity", 1)

        if not rect or fill is None or fill_opacity <= 0:
            continue

        width = max(0, rect.x1 - rect.x0)
        height = max(0, rect.y1 - rect.y0)
        area = width * height
        if area <= 0:
            continue

        fill_color = rgb_tuple_to_hex(fill)
        if not fill_color:
            continue

        area_ratio = area / page_area

        # Detect full-page background (only record the first / largest)
        if area_ratio >= PAGE_BG_AREA_RATIO and page_bg_color is None:
            page_bg_color = fill_color

        # Skip very small decorative elements (lines, dots)
        if area_ratio < MIN_BACKGROUND_AREA_RATIO:
            continue

        pending.append({
            "area": area,
            "html": (
                f'<div class="positioned-background" style="'
                f'top: {round(rect.y0 * scale, 1)}px; '
                f'left: {round(rect.x0 * scale, 1)}px; '
                f'width: {round(width * scale, 1)}px; '
                f'height: {round(height * scale, 1)}px; '
                f'background: {fill_color}; '
                f'opacity: {round(fill_opacity, 3)};'
                f'"></div>'
            ),
        })

    # Sort largest first so large background panels are painted first,
    # then smaller precise shapes paint on top of them.
    pending.sort(key=lambda d: d["area"], reverse=True)
    background_elements = [d["html"] for d in pending]

    return page_bg_color, background_elements


# --- Clear directories before every rerun ---
for folder in ["uploads", "output"]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

conversion_progress = {}


# HTML → JSON HELPER
def html_to_json(html_content, json_path):
    soup = BeautifulSoup(html_content, "html.parser")
    pages_data = []

    for page_idx, page_div in enumerate(soup.select(".page-container"), start=1):
        page_obj = {"page_number": page_idx, "elements": []}

        for text_div in page_div.select(".positioned-text"):
            style = text_div.get("style", "")
            page_obj["elements"].append({
                "type": "text",
                "text": text_div.get_text(),
                "style": style,
            })

        for img_div in page_div.select(".positioned-image"):
            img_tag = img_div.find("img")
            if img_tag:
                style = img_div.get("style", "")
                src = img_tag.get("src", "")
                page_obj["elements"].append({
                    "type": "image",
                    "style": style,
                    "src": src,
                })

        for table in page_div.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                row_data = []
                for td in tr.find_all("td"):
                    row_data.append({
                        "text": td.get_text(strip=True),
                        "rowspan": td.get("rowspan"),
                        "colspan": td.get("colspan"),
                    })
                rows.append(row_data)
            page_obj["elements"].append({"type": "table", "rows": rows})

        pages_data.append(page_obj)

    json_data = {"document": {"pages": pages_data}}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    return json_data


def normalize_header_name(header: str) -> str:
    text = (header or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_header_aliases(headers: List[str]) -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}
    for header in headers:
        normalized = normalize_header_name(header)
        alias_values = {normalized}
        if normalized.endswith(" id"):
            alias_values.add(normalized.replace(" id", " identifier"))
            alias_values.add(normalized.replace(" id", " number"))
        if "amount" in normalized:
            alias_values.add(normalized.replace("amount", "value"))
        aliases[header] = sorted(v for v in alias_values if v)
    return aliases


def build_structured_table_record(
        job_id: str,
        page_number: int,
        table_index: int,
        headers: List[str],
        rows: List[Dict[str, Any]],
        bbox: Optional[Any] = None) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "page": page_number,
        "table_id": f"page_{page_number}_table_{table_index}",
        "sheet_name": f"Page{page_number}_Table{table_index}",
        "bbox": bbox,
        "headers": headers,
        "header_aliases": build_header_aliases(headers),
        "rows": rows,
    }


def choose_table_headers(table_data: List[List[Any]]) -> Dict[str, Any]:
    if not table_data:
        return {"headers": [], "header_row_index": None}

    def build_headers_from_row(row):
        headers = []
        seen = {}
        for idx, value in enumerate(row):
            header = str(value or f"Column {idx + 1}").strip() or f"Column {idx + 1}"
            count = seen.get(header, 0)
            seen[header] = count + 1
            if count:
                header = f"{header}_{count + 1}"
            headers.append(header)
        return headers

    best_index = None
    best_score = -1
    scan_limit = min(3, len(table_data))

    for row_index in range(scan_limit):
        row = table_data[row_index]
        if not row:
            continue
        non_empty = [str(v or "").strip() for v in row if str(v or "").strip()]
        if not non_empty:
            continue
        avg_length = sum(len(v) for v in non_empty) / len(non_empty)
        score = len(non_empty) * 10
        if len(non_empty) == 1:
            score -= 20
        if avg_length > 30:
            score -= 8
        if score > best_score:
            best_score = score
            best_index = row_index

    if best_index is None:
        first_row = table_data[0]
        return {
            "headers": [f"Column {idx + 1}" for idx in range(len(first_row))],
            "header_row_index": None,
        }
    return {
        "headers": build_headers_from_row(table_data[best_index]),
        "header_row_index": best_index,
    }


def convert_table_data_to_rows(table_data: List[List[Any]]) -> Dict[str, Any]:
    if not table_data:
        return {"headers": [], "rows": []}

    header_info = choose_table_headers(table_data)
    headers = header_info.get("headers", [])
    header_row_index = header_info.get("header_row_index")
    use_header_row = (
        header_row_index is not None
        and headers
        and normalize_header_name(headers[0]) != "column 1"
    )

    data_rows = [
        row for idx, row in enumerate(table_data)
        if not use_header_row or idx != header_row_index
    ]

    row_widths = [len(row) for row in data_rows if row]
    common_data_width = 0
    if row_widths:
        width_counts = Counter(row_widths)
        common_data_width = max(width_counts.items(), key=lambda item: (item[1], item[0]))[0]

    effective_headers = list(headers)
    if common_data_width and headers and common_data_width < len(headers):
        effective_headers = headers[-common_data_width:]
        if common_data_width == 2:
            leading_label = None
            for row in data_rows[:3]:
                non_empty = [str(c or "").strip() for c in row if str(c or "").strip()]
                if len(non_empty) == 1 and len(non_empty[0]) <= 40:
                    leading_label = non_empty[0]
                    break
            if leading_label:
                effective_headers[0] = leading_label

    row_dicts: List[Dict[str, Any]] = []
    for row in data_rows:
        row_dict = {}
        raw_values = list(row)
        if effective_headers and len(raw_values) == len(effective_headers):
            aligned = raw_values
        elif effective_headers and common_data_width and len(raw_values) < len(effective_headers):
            aligned = [""] * len(effective_headers)
            if len(raw_values) == 1:
                aligned[0] = raw_values[0]
            else:
                aligned[-len(raw_values):] = raw_values
        else:
            aligned = raw_values
        for idx, header in enumerate(effective_headers):
            row_dict[header] = aligned[idx] if idx < len(aligned) else ""
        row_dicts.append(row_dict)

    return {
        "headers": effective_headers,
        "header_row_index": header_row_index,
        "raw_rows": table_data,
        "rows": row_dicts,
    }


def save_structured_tables_json(job_id: str, tables: List[Dict[str, Any]], output_dir: str = "output") -> str:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{job_id}_tables.json")
    payload = {"job_id": job_id, "tables": tables}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return json_path


def extract_structured_tables_from_document_json(job_id: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    structured_tables: List[Dict[str, Any]] = []
    pages = payload.get("document", {}).get("pages", [])

    for page in pages:
        page_number = page.get("page_number", 1)
        table_index = 0
        for element in page.get("elements", []):
            if element.get("type") != "table":
                continue
            raw_rows = element.get("rows", [])
            table_data: List[List[str]] = []
            for raw_row in raw_rows:
                row_values: List[str] = []
                for cell in raw_row:
                    if isinstance(cell, dict):
                        row_values.append(str(cell.get("text", "") or ""))
                    else:
                        row_values.append(str(cell or ""))
                table_data.append(row_values)

            converted = convert_table_data_to_rows(table_data)
            if not converted.get("headers"):
                continue
            table_index += 1
            table_record = build_structured_table_record(
                job_id=job_id,
                page_number=page_number,
                table_index=table_index,
                headers=converted.get("headers", []),
                rows=converted.get("rows", []),
                bbox=element.get("bbox"),
            )
            table_record["header_row_index"] = converted.get("header_row_index")
            table_record["raw_rows"] = converted.get("raw_rows", [])
            structured_tables.append(table_record)

    return structured_tables


def save_detailed_tables_json(job_id: str, payload: Dict[str, Any], output_dir: str = "output") -> str:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{job_id}_detailed_tables.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return json_path


def load_structured_tables_json(job_id: str, output_dir: str = "output") -> Dict[str, Any]:
    document_json_path = os.path.join(output_dir, f"{job_id}.json")
    if os.path.exists(document_json_path):
        with open(document_json_path, "r", encoding="utf-8") as f:
            document_payload = json.load(f)
        structured_tables = extract_structured_tables_from_document_json(job_id, document_payload)
        if structured_tables:
            detailed_payload = {
                "job_id": job_id,
                "tables": structured_tables,
                "source": "document_json",
            }
            save_detailed_tables_json(job_id, detailed_payload, output_dir=output_dir)
            return detailed_payload
    raise FileNotFoundError(f"Document JSON not found or contains no tables: {document_json_path}")


def parse_excel_export_query(query: str) -> Dict[str, Any]:
    text = (query or "").strip()
    normalized = normalize_header_name(text)
    requested_columns: List[str] = []
    filters: List[Dict[str, str]] = []
    table_hint = None

    column_match = re.search(
        r"(?:only|columns?|give me|export)\s+(.+?)(?:\s+for\s+|\s+where\s+|$)",
        text, re.I
    )
    if column_match:
        raw_columns = re.split(r",| and ", column_match.group(1))
        requested_columns = [col.strip() for col in raw_columns if col.strip()]
    elif text and not re.search(r"\b(where|for)\b", text, re.I):
        raw_columns = re.split(r",| and ", text)
        requested_columns = [col.strip() for col in raw_columns if col.strip()]

    where_match = re.search(r"(?:where|for)\s+(.+)", text, re.I)
    if where_match:
        filter_text = where_match.group(1).strip()
        eq_match = re.match(r"(.+?)\s*(?:=|is)\s*(.+)", filter_text, re.I)
        if eq_match:
            filters.append({
                "column_hint": eq_match.group(1).strip(),
                "operator": "=",
                "value": eq_match.group(2).strip(),
            })

    table_hint_match = re.search(r"\btable\s+(\d+)\b", text, re.I)
    if table_hint_match:
        table_hint = table_hint_match.group(1)

    return {
        "intent": "export_excel",
        "query": text,
        "normalized_query": normalized,
        "requested_columns": requested_columns,
        "filters": filters,
        "table_hint": table_hint,
    }


def resolve_requested_columns(requested_columns: List[str], table_record: Dict[str, Any]) -> List[str]:
    if not requested_columns:
        return list(table_record.get("headers", []))
    resolved: List[str] = []
    alias_map = table_record.get("header_aliases", {})
    for requested in requested_columns:
        norm_requested = normalize_header_name(requested)
        for header in table_record.get("headers", []):
            header_aliases = alias_map.get(header, [])
            if norm_requested == normalize_header_name(header) or norm_requested in header_aliases:
                if header not in resolved:
                    resolved.append(header)
                break
    return resolved


def apply_table_filters(rows, filters, table_record):
    if not filters:
        return rows
    alias_map = table_record.get("header_aliases", {})
    filtered_rows = rows
    for rule in filters:
        column_hint = normalize_header_name(rule.get("column_hint", ""))
        target_header = None
        for header in table_record.get("headers", []):
            if column_hint == normalize_header_name(header) or column_hint in alias_map.get(header, []):
                target_header = header
                break
        if not target_header:
            continue
        target_value = (rule.get("value") or "").strip().lower()
        filtered_rows = [
            row for row in filtered_rows
            if str(row.get(target_header, "")).strip().lower() == target_value
        ]
    return filtered_rows


def project_table_columns(rows, selected_columns):
    if not selected_columns:
        return rows
    return [{col: row.get(col, "") for col in selected_columns} for row in rows]


def summarize_table_candidate(table):
    headers = table.get("headers", [])
    header_preview = ", ".join(headers[:4])
    if len(headers) > 4:
        header_preview += ", ..."
    return f"{table.get('table_id')} (page {table.get('page')}): {header_preview}"


def find_best_table_for_query(request_data, tables):
    table_hint = request_data.get("table_hint")
    scored_candidates = []

    for table in tables:
        if table_hint:
            table_id = str(table.get("table_id", ""))
            if not table_id.endswith(f"_table_{table_hint}"):
                continue

        score = 0
        requested_columns = request_data.get("requested_columns", [])
        resolved_columns = resolve_requested_columns(requested_columns, table)
        if requested_columns and not resolved_columns:
            continue

        score += len(resolved_columns) * 10
        if requested_columns and len(resolved_columns) < len(requested_columns):
            score -= (len(requested_columns) - len(resolved_columns)) * 8

        for rule in request_data.get("filters", []):
            column_hint = normalize_header_name(rule.get("column_hint", ""))
            for header in table.get("headers", []):
                aliases = table.get("header_aliases", {}).get(header, [])
                if column_hint == normalize_header_name(header) or column_hint in aliases:
                    score += 5
                    break

        if score > 0:
            scored_candidates.append({
                "table": table,
                "score": score,
                "resolved_count": len(resolved_columns),
                "row_count": len(table.get("rows", [])),
            })

    if not scored_candidates:
        return None

    scored_candidates.sort(
        key=lambda item: (item["score"], item["resolved_count"], item["row_count"]),
        reverse=True,
    )
    top = scored_candidates[0]
    tied = [
        item for item in scored_candidates
        if item["score"] == top["score"] and item["resolved_count"] == top["resolved_count"]
    ]

    if len(tied) > 1:
        candidate_summaries = "; ".join(
            summarize_table_candidate(item["table"]) for item in tied[:4]
        )
        raise ValueError(
            "Multiple tables match the requested columns. "
            "Add a row filter or specify a table number like 'table 2'. "
            f"Candidates: {candidate_summaries}"
        )

    return top["table"]


def export_rows_to_excel(output_path, sheet_name, headers, rows):
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = (sheet_name or "Export")[:31]
        for col_idx, header in enumerate(headers, start=1):
            ws.cell(row=1, column=col_idx, value=header)
        for row_idx, row_data in enumerate(rows, start=2):
            for col_idx, header in enumerate(headers, start=1):
                ws.cell(row=row_idx, column=col_idx, value=row_data.get(header, ""))
        wb.save(output_path)
        return output_path
    except ImportError:
        pass

    def excel_column_name(index):
        name = ""
        n = index
        while n > 0:
            n, rem = divmod(n - 1, 26)
            name = chr(65 + rem) + name
        return name

    def xml_escape(value):
        text = "" if value is None else str(value)
        return (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;")
        )

    def inline_cell(ref, value):
        text = xml_escape(value)
        return (
            f'<c r="{ref}" t="inlineStr">'
            f'<is><t xml:space="preserve">{text}</t></is>'
            f'</c>'
        )

    sheet_rows_xml: List[str] = []
    all_rows = [headers] + [[row.get(h, "") for h in headers] for row in rows]
    for row_idx, values in enumerate(all_rows, start=1):
        cells_xml = [inline_cell(f"{excel_column_name(c + 1)}{row_idx}", v)
                     for c, v in enumerate(values)]
        sheet_rows_xml.append(f'<row r="{row_idx}">{"".join(cells_xml)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows_xml)}</sheetData>'
        '</worksheet>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>'
        f'<sheet name="{xml_escape((sheet_name or "Export")[:31])}" sheetId="1" r:id="rId1"/>'
        '</sheets></workbook>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1"'
        ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"'
        ' Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2"'
        ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"'
        ' Target="styles.xml"/>'
        '</Relationships>'
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1"'
        ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
        ' Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml"'
        ' ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml"'
        ' ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml"'
        ' ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '</Types>'
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        zf.writestr("xl/styles.xml", styles_xml)
    return output_path


def handle_excel_export_query(job_id, query, output_dir="output"):
    payload = load_structured_tables_json(job_id, output_dir=output_dir)
    tables = payload.get("tables", [])
    if not tables:
        raise ValueError("No structured tables found for this job")

    request_data = parse_excel_export_query(query)
    matched_table = find_best_table_for_query(request_data, tables)
    if not matched_table:
        raise ValueError("No matching table found for the query")

    selected_columns = resolve_requested_columns(
        request_data.get("requested_columns", []), matched_table
    )
    filtered_rows = apply_table_filters(
        matched_table.get("rows", []), request_data.get("filters", []), matched_table
    )
    exported_rows = project_table_columns(filtered_rows, selected_columns)

    output_path = os.path.join(output_dir, f"{job_id}_filtered.xlsx")
    export_rows_to_excel(
        output_path=output_path,
        sheet_name=matched_table.get("sheet_name", "Export"),
        headers=selected_columns,
        rows=exported_rows,
    )
    return {
        "job_id": job_id,
        "matched_table": matched_table.get("table_id"),
        "sheet_name": matched_table.get("sheet_name"),
        "selected_columns": selected_columns,
        "rows_found": len(exported_rows),
        "excel_path": output_path,
    }


# ---------- FONT EXTRACTION ----------
def extract_fonts_as_css(pdf_doc):
    css_rules = []
    seen_fonts = {}
    try:
        for page in pdf_doc:
            try:
                fonts = page.get_fonts(full=True)
            except Exception as e:
                print(f"[Font list error on page {page.number}]: {e}")
                continue

            for font in fonts:
                xref = font[0]
                internal_name = font[3]
                if not internal_name or internal_name in seen_fonts:
                    continue
                try:
                    ext, font_bytes, font_desc, font_name = pdf_doc.extract_font(xref)
                    display_name = internal_name
                    if isinstance(font_bytes, (bytes, bytearray)) and font_bytes:
                        try:
                            font_file = BytesIO(font_bytes)
                            tt = TTFont(font_file)
                            name_record = (
                                tt['name'].getName(4, 3, 1, 1033)
                                or tt['name'].getName(4, 1, 0, 0)
                            )
                            if name_record:
                                display_name = str(name_record)
                        except Exception as e:
                            print(f"[FontTools Error] Could not read full name for {internal_name}: {e}")
                    else:
                        print(f"[Font Extraction Warning] No embedded data for {internal_name}")

                    normalized_display_name = re.sub(
                        r'[,\-_ ]*(bold|italic|oblique|regular|roman)', '',
                        display_name, flags=re.IGNORECASE
                    ).strip()
                    # Detect variant from the internal font name
                    is_bold_font = bool(re.search(r'\bbold\b', internal_name, re.I))
                    is_italic_font = bool(re.search(r'\b(italic|oblique)\b', internal_name, re.I))
                    fw = "700" if is_bold_font else "400"
                    fs = "italic" if is_italic_font else "normal"

                    # Map the raw internal name → base family name (used during rendering)
                    seen_fonts[internal_name] = normalized_display_name
                    # Also map the subset-stripped name for easier lookup
                    stripped_internal = re.sub(r'^[A-Z]{6}\+', '', internal_name)
                    if stripped_internal != internal_name:
                        seen_fonts[stripped_internal] = normalized_display_name

                    extra_descriptors = f"font-weight: {fw}; font-style: {fs};"

                    if isinstance(font_bytes, (bytes, bytearray)) and font_bytes:
                        mime_type = "font/woff" if ext.lower() == "woff" else "font/ttf"
                        b64_font = base64.b64encode(font_bytes).decode("utf-8")
                        css_rules.append(f"""
                            @font-face {{
                                font-family: '{normalized_display_name}';
                                src: url(data:{mime_type};base64,{b64_font}) format('{ext.lower()}');
                                {extra_descriptors}
                            }}
                        """)
                except Exception as e:
                    print(f"[Font Extraction Error] {internal_name}: {e}")

    except Exception as e:
        print(f"[extract_fonts_as_css Error]: {e}")
    return "\n".join(css_rules), seen_fonts


def extract_table_cell_styling(page_mupdf, table_bbox, cell_bbox, font_name_map, scale):
    cell_styles = []
    try:
        _dict_flags = getattr(fitz, "TEXTFLAGS_DICT", None) or 0
        _style_flags = getattr(fitz, "TEXT_COLLECT_STYLES", 0)
        blocks = page_mupdf.get_text(
            "dict", flags=(_dict_flags | _style_flags)
        ).get("blocks", [])
    except Exception as e:
        print(f"[Table style extraction warning] page {page_mupdf.number + 1}: {e}")
        return cell_styles

    tx0, ty0, tx1, ty1 = table_bbox
    cx0, cy0, cx1, cy1 = cell_bbox

    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sx0, sy0, sx1, sy1 = span.get("bbox", (0, 0, 0, 0))
                if sx1 <= tx0 or sx0 >= tx1 or sy1 <= ty0 or sy0 >= ty1:
                    continue
                if sx1 <= cx0 or sx0 >= cx1 or sy1 <= cy0 or sy0 >= cy1:
                    continue
                text_value = span.get("text", "")
                if not text_value.strip():
                    continue

                internal_font = span.get("font", "Arial")
                font_family = _resolve_font_family(internal_font, font_name_map)
                is_bold, is_italic = span_font_style(span, internal_font)
                cell_styles.append({
                    "font_family": font_family,
                    "font_size": round(span.get("size", 12) * scale, 1),
                    "font_color": span_color_hex(span),
                    "is_bold": is_bold,
                    "is_italic": is_italic,
                })

    return cell_styles


def extract_table_cell_spans(page_mupdf, table_bbox, cell_bbox, font_name_map, scale):
    extracted_spans = []
    try:
        _dict_flags = getattr(fitz, "TEXTFLAGS_DICT", None) or 0
        _style_flags = getattr(fitz, "TEXT_COLLECT_STYLES", 0)
        blocks = page_mupdf.get_text(
            "dict", flags=(_dict_flags | _style_flags)
        ).get("blocks", [])
    except Exception as e:
        print(f"[Table span extraction warning] page {page_mupdf.number + 1}: {e}")
        return extracted_spans

    tx0, ty0, tx1, ty1 = table_bbox
    cx0, cy0, cx1, cy1 = cell_bbox

    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sx0, sy0, sx1, sy1 = span.get("bbox", (0, 0, 0, 0))
                if sx1 <= tx0 or sx0 >= tx1 or sy1 <= ty0 or sy0 >= ty1:
                    continue
                if sx1 <= cx0 or sx0 >= cx1 or sy1 <= cy0 or sy0 >= cy1:
                    continue
                raw_text = span.get("text", "")
                if not raw_text.strip():
                    continue

                internal_font = span.get("font", "Arial")
                font_family = _resolve_font_family(internal_font, font_name_map)
                is_bold, is_italic = span_font_style(span, internal_font)
                extracted_spans.append({
                    "text": html.escape(raw_text),
                    "left": round((sx0 - cx0) * scale, 2),
                    "top": round((sy0 - cy0) * scale, 2),
                    "width": round(max(0.0, (sx1 - sx0) * scale), 2),
                    "height": round(max(0.0, (sy1 - sy0) * scale), 2),
                    "font_family": font_family,
                    "font_size": round(span.get("size", 12) * scale, 2),
                    "font_color": span_color_hex(span),
                    "is_bold": is_bold,
                    "is_italic": is_italic,
                })

    extracted_spans.sort(key=lambda item: (item["top"], item["left"]))
    return extracted_spans


def pick_primary_table_style(cell_styles):
    if not cell_styles:
        return None
    freq = {}
    for style in cell_styles:
        key = (
            style.get("font_family", "Arial"),
            round(float(style.get("font_size", 12)), 1),
            style.get("font_color", "#000000"),
            bool(style.get("is_bold", False)),
            bool(style.get("is_italic", False)),
        )
        freq[key] = freq.get(key, 0) + 1
    dominant = max(freq.items(), key=lambda item: item[1])[0]
    return {
        "font_family": dominant[0],
        # Use the actual PDF-derived font size — do NOT clamp it here.
        # fit_table_cell_font_size() will shrink it if the cell is too small.
        "font_size": max(6.0, dominant[1]),
        "font_color": dominant[2],
        "is_bold": dominant[3],
        "is_italic": dominant[4],
    }


def format_table_cell_text(cell_text):
    escaped = html.escape(cell_text or "")
    return escaped.replace("\n", "<br>")


def build_rich_cell_html(cell_spans, fitted_font_size, primary_style, cell_fill, padding_px):
    """
    Build the <td> style string and inner HTML for a table cell.

    Uses per-span styling extracted from the PDF so colours, bold, italic and
    size variations within a single cell are faithfully reproduced.
    """
    # --- inner HTML: one <span> per PDF text run ---
    if cell_spans:
        parts = []
        for sp in cell_spans:
            sp_style_parts = [
                f"font-family: '{sp['font_family']}', Arial, sans-serif",
                f"font-size: {sp['font_size']:.2f}px",
                f"color: {sp['font_color']}",
                "display: inline",
            ]
            if sp["is_bold"]:
                sp_style_parts.append("font-weight: 700")
            if sp["is_italic"]:
                sp_style_parts.append("font-style: italic")
            parts.append(
                f'<span style="{"; ".join(sp_style_parts)}">{sp["text"]}</span>'
            )
        inner_html = "".join(parts)
    else:
        inner_html = ""

    # --- td style: base font from dominant style, background from fill ---
    td_style_parts = [
        "border: 1px solid #000",
        f"padding: {padding_px}px",
        "box-sizing: border-box",
        "vertical-align: top",
        "overflow: visible",
        "white-space: pre-wrap",
        "word-break: break-word",
        "text-align: left",
        "line-height: 1.2",
    ]
    if primary_style:
        td_style_parts.extend([
            f"font-family: '{primary_style['font_family']}', Arial, sans-serif",
            f"font-size: {fitted_font_size:.2f}px",
            f"color: {primary_style['font_color']}",
        ])
        if primary_style["is_bold"]:
            td_style_parts.append("font-weight: 700")
        if primary_style["is_italic"]:
            td_style_parts.append("font-style: italic")
    else:
        td_style_parts.extend([
            "font-family: Arial, sans-serif",
            f"font-size: {fitted_font_size:.2f}px",
            "color: #000000",
        ])
    if cell_fill:
        td_style_parts.append(f"background: {cell_fill['color']}")

    return inner_html, "; ".join(td_style_parts)


def extract_table_cell_text(page_mupdf, cell_bbox, fallback_text=""):
    fallback_text = (fallback_text or "").strip()
    if fallback_text:
        return fallback_text
    try:
        rect = fitz.Rect(*cell_bbox)
        text = page_mupdf.get_textbox(rect) or ""
    except Exception as e:
        print(f"[Table cell text extraction warning] page {page_mupdf.number + 1}: {e}")
        text = ""
    text = (text or "").strip()
    return text if text else fallback_text


def fit_table_cell_font_size(base_font_size, cell_bbox, cell_text, padding=4):
    x0, y0, x1, y1 = cell_bbox
    cell_width = max(1.0, x1 - x0)
    cell_height = max(1.0, y1 - y0)
    usable_width = max(1.0, cell_width - (padding * 2))
    usable_height = max(1.0, cell_height - (padding * 2))
    raw_lines = [line.strip() for line in (cell_text or "").splitlines() if line.strip()]
    lines = max(1, len(raw_lines))
    longest_line_len = max(
        (len(line) for line in raw_lines), default=max(1, len((cell_text or "").strip()))
    )
    base = float(base_font_size)
    max_font_from_height = usable_height / (lines * 1.05)
    avg_char_width_factor = 0.56
    max_font_from_width = usable_width / max(1.0, longest_line_len * avg_char_width_factor)
    fitted = min(base * 1.10, max_font_from_height, max_font_from_width)
    min_floor = min(max(5.0, base * 0.65), base)
    return max(min_floor, min(fitted, base * 1.10))


def estimate_wrapped_line_count(cell_text, cell_width, font_size, padding=2):
    usable_width = max(1.0, cell_width - (padding * 2))
    raw_lines = [line.strip() for line in (cell_text or "").splitlines()] or [""]
    avg_char_width = max(1.0, font_size * 0.56)
    total_lines = 0
    for line in raw_lines:
        line_len = max(1, len(line))
        approx_line_width = line_len * avg_char_width
        total_lines += max(1, int(math.ceil(approx_line_width / usable_width)))
    return max(1, total_lines)


def compute_table_fit_scale(cell_entries, row_heights):
    if not cell_entries or not row_heights:
        return 1.0
    row_demands = [0.0 for _ in row_heights]
    for entry in cell_entries:
        lines = estimate_wrapped_line_count(
            entry["text"], entry["cell_width_px"],
            entry["base_font_size"], padding=entry["padding_px"]
        )
        required_height = (lines * entry["base_font_size"] * 1.08) + (entry["padding_px"] * 2)
        per_row = required_height / max(1, entry["rowspan"])
        for idx in range(entry["row_idx"], min(len(row_heights), entry["row_idx"] + entry["rowspan"])):
            row_demands[idx] = max(row_demands[idx], per_row)
    available_height = sum(row_heights)
    required_height = sum(max(row_heights[idx], row_demands[idx]) for idx in range(len(row_heights)))
    if required_height <= 0:
        return 1.0
    return min(1.0, available_height / required_height)


def build_table_grid(table_data, row_cells=None):
    if not table_data:
        return [], set(), 0, 0

    rows = len(table_data)
    cols = max((len(row) for row in table_data), default=0)
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    occupied = set()

    for row_idx, row in enumerate(table_data):
        col_idx = 0
        for cell_idx, cell in enumerate(row):
            while (row_idx, col_idx) in occupied:
                col_idx += 1
            if col_idx >= cols:
                break
            if cell is None:
                continue

            rowspan = 1
            colspan = 1

            if cell_idx < len(row) - 1 and row[cell_idx + 1] is None:
                next_col = col_idx + 1
                while next_col < cols and (row_idx, next_col) not in occupied and (
                    next_col >= len(row) or row[next_col] is None
                ):
                    colspan += 1
                    next_col += 1

            if row_idx < len(table_data) - 1 and len(table_data[row_idx + 1]) > col_idx and \
                    table_data[row_idx + 1][col_idx] is None:
                next_row = row_idx + 1
                while next_row < rows and (next_row, col_idx) not in occupied and (
                    col_idx >= len(table_data[next_row]) or table_data[next_row][col_idx] is None
                ):
                    rowspan += 1
                    next_row += 1

            for r in range(row_idx, min(rows, row_idx + rowspan)):
                for c in range(col_idx, min(cols, col_idx + colspan)):
                    if r < rows and c < cols:
                        occupied.add((r, c))

            cell_bbox = None
            if row_cells and row_idx < len(row_cells) and col_idx < len(row_cells[row_idx]):
                cell_bbox = row_cells[row_idx][col_idx]

            grid[row_idx][col_idx] = {
                "content": html.escape(cell or ""),
                "rowspan": rowspan if rowspan > 1 else None,
                "colspan": colspan if colspan > 1 else None,
                "bbox": cell_bbox,
            }
            col_idx += colspan

    return grid, occupied, rows, cols


def compute_table_layout_from_bbox(table_bbox, grid, rows, cols, scale):
    x0, y0, x1, y1 = table_bbox
    total_width = max(1.0, (x1 - x0) * scale)
    total_height = max(1.0, (y1 - y0) * scale)
    default_col_width = total_width / max(cols, 1)
    default_row_height = total_height / max(rows, 1)

    col_width_sums = [0.0] * cols
    col_width_counts = [0] * cols
    row_height_sums = [0.0] * rows
    row_height_counts = [0] * rows

    for row_idx in range(rows):
        for col_idx in range(cols):
            cell = grid[row_idx][col_idx]
            if not cell or not cell.get("bbox"):
                continue
            bx0, by0, bx1, by1 = cell["bbox"]
            colspan = cell.get("colspan") or 1
            rowspan = cell.get("rowspan") or 1
            per_col_width = max(1.0, ((bx1 - bx0) * scale) / max(1, colspan))
            per_row_height = max(1.0, ((by1 - by0) * scale) / max(1, rowspan))
            for c in range(col_idx, min(cols, col_idx + colspan)):
                col_width_sums[c] += per_col_width
                col_width_counts[c] += 1
            for r in range(row_idx, min(rows, row_idx + rowspan)):
                row_height_sums[r] += per_row_height
                row_height_counts[r] += 1

    col_widths = [
        (col_width_sums[i] / col_width_counts[i]) if col_width_counts[i] else default_col_width
        for i in range(cols)
    ]
    row_heights = [
        (row_height_sums[i] / row_height_counts[i]) if row_height_counts[i] else default_row_height
        for i in range(rows)
    ]
    col_sum = sum(col_widths) or 1.0
    row_sum = sum(row_heights) or 1.0
    col_widths = [max(1.0, w * total_width / col_sum) for w in col_widths]
    row_heights = [max(1.0, h * total_height / row_sum) for h in row_heights]
    return col_widths, row_heights


def merge_close_positions(values, tolerance=0.75):
    vals = sorted(float(v) for v in values)
    if not vals:
        return []
    merged = [vals[0]]
    for v in vals[1:]:
        if abs(v - merged[-1]) <= tolerance:
            merged[-1] = (merged[-1] + v) / 2.0
        else:
            merged.append(v)
    return merged


def nearest_position_index(positions, value, tolerance=1.5):
    if not positions:
        return None
    best_idx = None
    best_dist = None
    for idx, pos in enumerate(positions):
        dist = abs(pos - float(value))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = idx
    if best_dist is not None and best_dist <= tolerance:
        return best_idx
    return best_idx


def build_geometry_table_model(row_cells, table_data, scale):
    x_positions = []
    y_positions = []
    for row in row_cells or []:
        for cell in row or []:
            if not cell:
                continue
            x0, y0, x1, y1 = cell
            x_positions.extend([x0, x1])
            y_positions.extend([y0, y1])

    xs = merge_close_positions(x_positions)
    ys = merge_close_positions(y_positions)
    if len(xs) < 2 or len(ys) < 2:
        return None

    cols = len(xs) - 1
    rows = len(ys) - 1
    col_widths = [max(1.0, (xs[i + 1] - xs[i]) * scale) for i in range(cols)]
    row_heights = [max(1.0, (ys[i + 1] - ys[i]) * scale) for i in range(rows)]
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    occupied = set()
    seen_bboxes = set()

    for src_row_idx, row in enumerate(row_cells or []):
        row_data = table_data[src_row_idx] if src_row_idx < len(table_data) else []
        for src_col_idx, cell_bbox in enumerate(row or []):
            if not cell_bbox:
                continue
            bbox_key = tuple(round(v, 2) for v in cell_bbox)
            if bbox_key in seen_bboxes:
                continue
            seen_bboxes.add(bbox_key)

            x0, y0, x1, y1 = cell_bbox
            start_col = nearest_position_index(xs, x0)
            end_col = nearest_position_index(xs, x1)
            start_row = nearest_position_index(ys, y0)
            end_row = nearest_position_index(ys, y1)
            if None in {start_col, end_col, start_row, end_row}:
                continue

            colspan = max(1, end_col - start_col)
            rowspan = max(1, end_row - start_row)
            for rr in range(start_row, min(rows, start_row + rowspan)):
                for cc in range(start_col, min(cols, start_col + colspan)):
                    if rr == start_row and cc == start_col:
                        continue
                    occupied.add((rr, cc))

            raw_content = row_data[src_col_idx] if src_col_idx < len(row_data) else ""
            grid[start_row][start_col] = {
                "content": html.escape(raw_content or ""),
                "rowspan": rowspan if rowspan > 1 else None,
                "colspan": colspan if colspan > 1 else None,
                "bbox": tuple(cell_bbox),
            }

    return {
        "grid": grid,
        "occupied": occupied,
        "rows": rows,
        "cols": cols,
        "col_widths": col_widths,
        "row_heights": row_heights,
    }


def is_semantic_table_candidate(table_data):
    if not table_data:
        return False
    rows = len(table_data)
    cols = max((len(r) for r in table_data), default=0)
    if rows < 2 or cols < 2:
        return False
    non_empty_cells = 0
    dense_rows = 0
    for row in table_data:
        filled = sum(1 for c in row if c and str(c).strip())
        non_empty_cells += filled
        if filled >= 2:
            dense_rows += 1
    if dense_rows < 2:
        return False
    occupancy = non_empty_cells / max(1, rows * cols)
    return occupancy >= 0.2


def normalize_table_data(table_data):
    if not table_data:
        return []
    max_cols = max((len(r) for r in table_data), default=0)
    if max_cols <= 0:
        return []
    normalized = []
    for row in table_data:
        r = list(row) if isinstance(row, (list, tuple)) else [row]
        if len(r) < max_cols:
            r.extend([None] * (max_cols - len(r)))
        normalized.append(r)
    return normalized


def detect_tables_pymupdf(page_mupdf):
    tables_out = []
    try:
        finder = page_mupdf.find_tables()
        pm_tables = getattr(finder, "tables", finder) or []
    except Exception as e:
        print(f"[PyMuPDF table detection warning] page {page_mupdf.number + 1}: {e}")
        return tables_out

    for t in pm_tables:
        bbox = getattr(t, "bbox", None)
        if not bbox:
            continue
        try:
            table_data = t.extract()
        except Exception:
            table_data = None
        table_data = normalize_table_data(table_data)
        if not table_data or not is_semantic_table_candidate(table_data):
            continue

        row_cells = []
        for row in getattr(t, "rows", []) or []:
            row_cells.append([
                tuple(cell) if cell else None
                for cell in getattr(row, "cells", []) or []
            ])

        mask_bbox = tuple(bbox)
        header = getattr(t, "header", None)
        header_bbox = getattr(header, "bbox", None) if header else None
        if header_bbox:
            hx0, hy0, hx1, hy1 = header_bbox
            bx0, by0, bx1, by1 = mask_bbox
            mask_bbox = (min(bx0, hx0), min(by0, hy0), max(bx1, hx1), max(by1, hy1))

        tables_out.append({
            "bbox": tuple(bbox),
            "mask_bbox": mask_bbox,
            "data": table_data,
            "row_cells": row_cells,
        })
    return tables_out


def detect_tables_pdfplumber(page_plumber):
    tables_out = []
    if page_plumber is None:
        return tables_out
    try:
        detected_tables = page_plumber.find_tables(table_settings=TABLE_SETTINGS)
    except Exception as e:
        print(f"[pdfplumber table detection warning]: {e}")
        return tables_out

    for t in detected_tables:
        table_data = normalize_table_data(t.extract())
        if not table_data or not is_semantic_table_candidate(table_data):
            continue
        tables_out.append({
            "bbox": tuple(t.bbox),
            "mask_bbox": tuple(t.bbox),
            "data": table_data,
            "row_cells": None,
        })
    return tables_out


# ---------- PDF CONVERSION ----------
def convert_pdf_with_progress(filename, job_id):
    try:
        conversion_progress[job_id] = {
            'status': 'starting',
            'progress': 0,
            'message': 'Initializing conversion...',
            'result': None,
            'error': None,
            'pdf_base64': None,
            'icons': [],
        }

        start_time = time.time()
        full_html = ""
        structured_tables: List[Dict[str, Any]] = []

        with open(filename, 'rb') as pdf_file:
            pdf_base64 = base64.b64encode(pdf_file.read()).decode('utf-8')
            conversion_progress[job_id]['pdf_base64'] = pdf_base64

        pdf_doc = fitz.open(filename)
        pdf_plumber = None
        try:
            pdf_plumber = pdfplumber.open(filename)
        except Exception as e:
            print(f"[pdfplumber open warning]: {e}")
        total_pages = len(pdf_doc)

        # Extract icons — no extra keyword args needed
        try:
            icons = extract_icons_from_pdf(filename)
            conversion_progress[job_id]['icons'] = icons
        except Exception as e:
            print(f"[Icon extraction warning]: {e}")

        font_css, font_name_map = extract_fonts_as_css(pdf_doc)
        conversion_progress[job_id]['progress'] = 5

        for page_num, page_mupdf in enumerate(pdf_doc):
            page_plumber = None
            if pdf_plumber and page_num < len(pdf_plumber.pages):
                page_plumber = pdf_plumber.pages[page_num]

            page_progress = int(5 + (page_num / total_pages) * 85)
            conversion_progress[job_id]['progress'] = page_progress
            conversion_progress[job_id]['message'] = f'Processing page {page_num + 1} of {total_pages}...'

            page_width = page_mupdf.rect.width
            page_height = page_mupdf.rect.height

            # ----------------------------------------------------------------
            # SCALE: compute from TARGET_WIDTH so output adapts to any PDF.
            # ----------------------------------------------------------------
            scale = TARGET_WIDTH / max(page_width, 1.0)
            # Pre-compute scaled page dimensions — used both for clamping block
            # widths during rendering and for the final page container style.
            scaled_page_width = int(page_width * scale)
            scaled_page_height = int(page_height * scale)

            page_bg_color, background_elements = extract_background_shapes(page_mupdf, scale)
            elements = list(background_elements)

            # YOLO layout detection
            page_img = render_page_image(page_mupdf, dpi=DPI)
            yolo_boxes = detect_layout_yolo(page_img)

            debug_dir = "output/yolo_debug"
            os.makedirs(debug_dir, exist_ok=True)
            debug_img_path = os.path.join(debug_dir, f"{job_id}_page_{page_num + 1}.png")
            visualize_yolo_boxes(page_img, yolo_boxes, save_path=debug_img_path)

            img_w, img_h = page_img.size
            sx = img_w / page_width
            sy = img_h / page_height

            valid_tables = detect_tables_pymupdf(page_mupdf)
            if not valid_tables:
                valid_tables = detect_tables_pdfplumber(page_plumber)
            table_bboxes = [t.get("mask_bbox", t["bbox"]) for t in valid_tables]
            filled_rects = extract_filled_rectangles(page_mupdf)

            def is_within_table(x0, y0, x1, y1):
                span_area = max(0, (x1 - x0)) * max(0, (y1 - y0))
                if span_area <= 0:
                    return False
                for tx0, ty0, tx1, ty1 in table_bboxes:
                    inter_x0 = max(x0, tx0)
                    inter_y0 = max(y0, ty0)
                    inter_x1 = min(x1, tx1)
                    inter_y1 = min(y1, ty1)
                    inter_w = max(0, inter_x1 - inter_x0)
                    inter_h = max(0, inter_y1 - inter_y0)
                    if (inter_w * inter_h) / span_area >= 0.35:
                        return True
                return False

            _dict_flags = getattr(fitz, "TEXTFLAGS_DICT", None) or 0
            _style_flags = getattr(fitz, "TEXT_COLLECT_STYLES", 0)
            blocks = page_mupdf.get_text(
                "dict", flags=(_dict_flags | _style_flags)
            )["blocks"]

            for block in blocks:
                if block["type"] != 0:
                    continue

                block_lines = []
                block_spans = []
                for line in block.get("lines", []):
                    line_spans = []
                    for span in line.get("spans", []):
                        x0, y0, x1, y1 = span.get("bbox", (0, 0, 0, 0))
                        if is_within_table(x0, y0, x1, y1):
                            continue
                        raw = span.get("text", "")
                        if not raw.strip():
                            continue
                        line_spans.append(span)
                        block_spans.append(span)
                    if line_spans:
                        block_lines.append(line_spans)

                if not block_spans:
                    continue

                block_bbox = block.get("bbox", block_spans[0].get("bbox", (0, 0, 0, 0)))
                matched_box = match_bbox_to_box(block_bbox, yolo_boxes, sx, sy)
                layout_type = str((matched_box or {}).get("class", "text")).lower()
                if not matched_box:
                    layout_type = match_span_to_box(block_spans[0]["bbox"], yolo_boxes, sx, sy)
                tag = get_semantic_tag_for_data_type(layout_type)

                if tag in {"p", "h1", "h2", "h3", "footer"}:
                    ref_span = block_spans[0]

                    # Build raw_lines for layout calculations (white-space, justify, footer detection)
                    raw_lines = []
                    for line_spans in block_lines:
                        raw_line = "".join(s.get("text", "") for s in line_spans).strip()
                        if raw_line:
                            raw_lines.append(raw_line)
                    if not raw_lines:
                        continue

                    # Build rich HTML preserving per-span colour/bold/italic.
                    html_lines = []
                    for line_spans in block_lines:
                        line_parts = []
                        for sp in line_spans:
                            sp_text = sp.get("text", "")
                            if not sp_text:
                                continue
                            sp_font = sp.get("font", "Arial")
                            sp_family = _resolve_font_family(sp_font, font_name_map)
                            sp_size = round(sp.get("size", 12) * scale, 1)
                            sp_color = span_color_hex(sp)
                            sp_bold, sp_italic = span_font_style(sp, sp_font)
                            sp_style = (
                                f"font-family:'{sp_family}',Arial,sans-serif;"
                                f"font-size:{sp_size:.2f}px;"
                                f"color:{sp_color};"
                            )
                            if sp_bold:
                                sp_style += "font-weight:700;"
                            if sp_italic:
                                sp_style += "font-style:italic;"
                            line_parts.append(
                                f'<span style="{sp_style}">{html.escape(sp_text)}</span>'
                            )
                        if line_parts:
                            html_lines.append("".join(line_parts))

                    if tag in ("p", "footer"):
                        text = " ".join(html_lines)
                    else:
                        text = "<br>".join(html_lines)

                    text = re.sub(
                        r'(https?://[^\s<"]+)',
                        r'<a href="\1" target="_blank">\1</a>',
                        text,
                    )

                    line_height_ratio = estimate_line_height_ratio(
                        block_lines, ref_span.get("size", 12)
                    )

                    bx0, by0, bx1, by1 = block_bbox
                    top = by0 * scale
                    left = bx0 * scale
                    # Clamp width so left+width never exceeds page width
                    raw_width = (bx1 - bx0) * scale
                    max_width = max(1.0, scaled_page_width - left)
                    width = min(raw_width, max_width)
                    height = max(0.0, (by1 - by0) * scale)

                    # Note: we intentionally use the precise PDF block_bbox for all
                    # positioning (top/left/width/height computed above from bx0..by1).
                    # YOLO boxes have integer-snapping and padding that cause text
                    # misalignment when used as layout geometry; YOLO is used only for
                    # semantic classification (tag), not for coordinate placement.

                    if tag in ("p", "footer"):
                        if len(raw_lines) == 1:
                            # Single PDF line — never reflow, trust the PDF layout.
                            # Exception: if the block is wider than the page it must wrap.
                            if raw_width > max_width:
                                should_justify = False
                                text_align = "left"
                                white_space = "nowrap"
                                text_justify = "auto"
                            else:
                                should_justify = False
                                text_align = "left"
                                white_space = "nowrap"
                                text_justify = "auto"
                        else:
                            para_box = (
                                left / max(scale, 1e-6), top / max(scale, 1e-6),
                                (left + width) / max(scale, 1e-6),
                                (top + height) / max(scale, 1e-6),
                            )
                            should_justify = is_justified_paragraph(block_lines, para_box)
                            text_align = "justify" if should_justify else "left"
                            white_space = "normal"
                            text_justify = "inter-word" if should_justify else "auto"
                        # For footer blocks: override text_align based on position
                        if tag == "footer":
                            _combined = " ".join(raw_lines)
                            _block_center = (bx0 + bx1) / 2.0
                            _is_centred = abs(_block_center - (page_width / 2.0)) <= (page_width * 0.15)
                            if len(_combined.split()) <= 4 and _is_centred:
                                text_align = "center"
                    else:
                        # Headings, headers — check centring
                        block_center = (bx0 + bx1) / 2.0
                        page_center = page_width / 2.0
                        is_centered = abs(block_center - page_center) <= (page_width * 0.08)
                        text_align = "center" if is_centered else "left"
                        white_space = "nowrap" if len(raw_lines) == 1 else "normal"
                        text_justify = "auto"

                    # Detect if this text block has a background fill.
                    # Only apply an inline background when the fill rect is small/
                    # specific (e.g. a cell highlight).  Large panel/column fills
                    # are already rendered as positioned-background divs beneath the
                    # text, so duplicating them here causes colour-stacking artefacts.
                    block_fill = pick_fill_for_bbox(block_bbox, filled_rects, min_coverage=FILL_COVERAGE_SPAN)
                    if block_fill:
                        _fx0, _fy0, _fx1, _fy1 = block_fill["bbox"]
                        _fill_area = max(1.0, (_fx1 - _fx0) * (_fy1 - _fy0))
                        _page_area = max(1.0, page_width * page_height)
                        # Suppress fills that cover more than 15 % of the page —
                        # those are large panel/column backgrounds already rendered
                        # as positioned-background divs beneath the text layer.
                        if _fill_area / _page_area > 0.15:
                            block_fill = None
                    bg_style = f"background: {block_fill['color']}; " if block_fill else ""

                    # Font size is taken directly from the PDF span — no capping.
                    # The PDF renderer already laid out text at this size; changing
                    # it here would cause misalignment with neighbouring elements.

                    style = (
                        f"top: {round(top, 2)}px; left: {round(left, 2)}px; "
                        f"width: {round(max(0.0, width), 2)}px; "
                        f"min-height: {round(height, 2)}px; height: auto; "
                        f"margin: 0; padding: 0; "
                        f"line-height: {line_height_ratio:.3f}; "
                        f"text-align: {text_align}; "
                        f"text-justify: {text_justify}; "
                        f"white-space: {white_space}; "
                        f"overflow-wrap: break-word; "
                        f"word-wrap: break-word; "
                        f"{bg_style}"
                    )

                    # Resolve semantic output tag and data-role attribute.
                    # "footer" → <p> with data-role to distinguish page-number from footer-text.
                    # Page number = the entire block content is just digits (possibly with spaces).
                    emit_tag = tag if tag != "footer" else "p"
                    data_role = ""
                    if tag == "footer":
                        _combined = " ".join(raw_lines).strip()
                        _is_page_num = bool(re.match(r'^\d+\s*$', _combined))
                        if _is_page_num:
                            data_role = ' data-role="page-number"'
                        else:
                            data_role = ' data-role="footer-text"'

                    elements.append(
                        f'<{emit_tag} class="positioned-text" '
                        f'data-type="{layout_type}"{data_role} '
                        f'style="{style}">{text}</{emit_tag}>'
                    )
                    continue

                # Span-level rendering for list items and other types
                for line in block_lines:
                    for span in line:
                        x0, y0, x1, y1 = span.get("bbox", (0, 0, 0, 0))
                        raw = span.get("text", "")
                        if not raw.strip():
                            continue

                        span_layout_type = match_span_to_box(span["bbox"], yolo_boxes, sx, sy)
                        span_tag = get_semantic_tag_for_data_type(span_layout_type)

                        if span_tag == "li":
                            raw = strip_bullet_from_li(raw)
                            if not raw.strip():
                                continue

                        text = html.escape(raw)
                        text = re.sub(
                            r'(https?://[^\s<]+)',
                            r'<a href="\1" target="_blank">\1</a>',
                            text,
                        )

                        font_size = round(span.get("size", 12) * scale, 1)
                        font_color = span_color_hex(span)

                        internal_font = span.get("font", "Arial")
                        font_family = _resolve_font_family(internal_font, font_name_map)

                        is_bold, is_italic = span_font_style(span, internal_font)
                        weight_style = "font-weight: 700; " if is_bold else ""
                        slant_style = "font-style: italic; " if is_italic else ""

                        left = round(x0 * scale, 2)
                        top_pos = round(y0 * scale, 2)
                        width = max(0.0, round((x1 - x0) * scale, 2))
                        height = max(0.0, round((y1 - y0) * scale, 2))

                        span_fill = pick_fill_for_bbox(
                            (x0, y0, x1, y1), filled_rects, min_coverage=FILL_COVERAGE_SPAN
                        )
                        if span_fill:
                            _sfx0, _sfy0, _sfx1, _sfy1 = span_fill["bbox"]
                            _sf_area = max(1.0, (_sfx1 - _sfx0) * (_sfy1 - _sfy0))
                            if _sf_area / max(1.0, page_width * page_height) > 0.15:
                                span_fill = None
                        bg_style = f"background: {span_fill['color']}; " if span_fill else ""

                        li_extra = ""
                        if span_tag == "li":
                            li_extra = (
                                "list-style-type: disc; "
                                "list-style-position: inside; "
                                "display: list-item; "
                            )
                        style = (
                            f"top: {top_pos}px; left: {left}px; "
                            f"width: {width}px; height: {height}px; "
                            f"margin: 0; padding: 0; "
                            f"font-size: {font_size:.2f}px; "
                            f"line-height: 1; "
                            f"color: {font_color}; "
                            f"font-family: '{font_family}', Arial, sans-serif; "
                            f"{weight_style}{slant_style}"
                            f"{li_extra}"
                            f"{bg_style}"
                            f"white-space: nowrap;"
                        )

                        elements.append(
                            f'<{span_tag} class="positioned-text" '
                            f'data-type="{span_layout_type}" '
                            f'style="{style}">{text}</{span_tag}>'
                        )

            # Images
            for img_index, img in enumerate(page_mupdf.get_images(full=True)):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(pdf_doc, xref)
                    if pix.n >= 5:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes = pix.tobytes("png")
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    rects = page_mupdf.get_image_rects(xref)
                    for rect in rects:
                        left = round(rect.x0 * scale, 1)
                        top_pos = round(rect.y0 * scale, 1)
                        width = round((rect.x1 - rect.x0) * scale, 1)
                        height = round((rect.y1 - rect.y0) * scale, 1)
                        elements.append(
                            f'<div class="positioned-image" style="'
                            f'top: {top_pos}px; left: {left}px; '
                            f'width: {width}px; height: {height}px;">'
                            f'<img src="data:image/png;base64,{img_base64}" '
                            f'style="width: 100%; height: 100%; object-fit: contain;">'
                            f'</div>'
                        )
                except Exception as e:
                    print(f"[Image extraction warning] page {page_num + 1}, xref {xref}: {e}")

            # Tables
            for table in valid_tables:
                x0, top_t, x1, bottom_t = table["bbox"]
                width = (x1 - x0) * scale
                height = (bottom_t - top_t) * scale
                top_scaled = top_t * scale
                left_scaled = x0 * scale

                table_data = table.get("data")
                if not table_data:
                    continue

                structured_table = convert_table_data_to_rows(table_data)
                structured_tables.append(
                    build_structured_table_record(
                        job_id=job_id,
                        page_number=page_num + 1,
                        table_index=len(structured_tables) + 1,
                        headers=structured_table["headers"],
                        rows=structured_table["rows"],
                        bbox=table["bbox"],
                    )
                )

                rows_count = len(table_data)
                cols_count = max(len(row) for row in table_data) if table_data else 0
                row_cells = table.get("row_cells")
                table_fill = pick_fill_for_bbox(table["bbox"], filled_rects, min_coverage=FILL_COVERAGE_TABLE)

                if row_cells:
                    grid, occupied, rows_count, cols_count = build_table_grid(
                        table_data, row_cells=row_cells
                    )
                    col_widths, row_heights = compute_table_layout_from_bbox(
                        table["bbox"], grid, rows_count, cols_count, scale
                    )
                    prepared_cells = []

                    for row_idx in range(rows_count):
                        for col_idx in range(cols_count):
                            cell_meta = grid[row_idx][col_idx]
                            if not cell_meta:
                                continue
                            cell_bbox = cell_meta.get("bbox")
                            if not cell_bbox:
                                continue

                            raw_cell_value = (
                                table_data[row_idx][col_idx]
                                if col_idx < len(table_data[row_idx]) else ""
                            )
                            cell_text = extract_table_cell_text(page_mupdf, cell_bbox, raw_cell_value)
                            cell_spans = extract_table_cell_spans(
                                page_mupdf, table["bbox"], cell_bbox, font_name_map, scale
                            )
                            cell_styles = extract_table_cell_styling(
                                page_mupdf, table["bbox"], cell_bbox, font_name_map, scale
                            )
                            primary_style = pick_primary_table_style(cell_styles)
                            cell_fill = pick_fill_for_bbox(
                                cell_bbox, filled_rects, min_coverage=FILL_COVERAGE_CELL
                            )
                            padding_px = 2
                            colspan = cell_meta["colspan"] or 1
                            rowspan = cell_meta["rowspan"] or 1
                            effective_cell_width_px = sum(
                                col_widths[col_idx:min(cols_count, col_idx + colspan)]
                            )
                            effective_cell_height_px = sum(
                                row_heights[row_idx:min(rows_count, row_idx + rowspan)]
                            )
                            effective_cell_bbox = (
                                0.0, 0.0,
                                effective_cell_width_px / max(scale, 1e-6),
                                effective_cell_height_px / max(scale, 1e-6),
                            )
                            base_font_size = primary_style['font_size'] if primary_style else 12
                            fitted_font_size = fit_table_cell_font_size(
                                base_font_size, effective_cell_bbox, cell_text, padding=padding_px
                            )
                            prepared_cells.append({
                                "row_idx": row_idx,
                                "rowspan": rowspan,
                                "text": cell_text,
                                "base_font_size": fitted_font_size,
                                "cell_width_px": effective_cell_width_px,
                                "padding_px": padding_px,
                            })
                            grid[row_idx][col_idx]["prepared"] = {
                                "cell_text": cell_text,
                                "cell_spans": cell_spans,
                                "primary_style": primary_style,
                                "cell_fill": cell_fill,
                                "fitted_font_size": fitted_font_size,
                                "padding_px": padding_px,
                            }

                    table_fit_scale = compute_table_fit_scale(prepared_cells, row_heights)

                    for row_idx in range(rows_count):
                        for col_idx in range(cols_count):
                            cell_meta = grid[row_idx][col_idx]
                            if not cell_meta or not cell_meta.get("prepared"):
                                continue
                            prepared = cell_meta["prepared"]
                            cell_text = prepared["cell_text"]
                            cell_spans = prepared.get("cell_spans", [])
                            primary_style = prepared["primary_style"]
                            cell_fill = prepared["cell_fill"]
                            padding_px = prepared["padding_px"]
                            fitted_font_size = prepared["fitted_font_size"] * table_fit_scale

                            cell_html, cell_td_style = build_rich_cell_html(
                                cell_spans, fitted_font_size, primary_style, cell_fill, padding_px
                            )
                            # Fall back to plain escaped text if no spans extracted
                            if not cell_html:
                                cell_html = format_table_cell_text(cell_text)

                            grid[row_idx][col_idx] = {
                                "content": cell_html,
                                "rowspan": cell_meta["rowspan"],
                                "colspan": cell_meta["colspan"],
                                "style": cell_td_style,
                                "bbox": cell_meta.get("bbox"),
                            }

                    colgroup_html = "".join(
                        f'<col style="width: {round(col_w, 2)}px;">'
                        for col_w in col_widths
                    )
                    table_html = (
                        f'<table style="border-collapse: collapse; width: {round(width, 2)}px; '
                        f'height: {round(height, 2)}px; table-layout: fixed;">'
                        f'<colgroup>{colgroup_html}</colgroup>'
                    )

                    for row_idx in range(rows_count):
                        table_html += f'<tr style="height: {round(row_heights[row_idx], 2)}px;">'
                        for col_idx in range(cols_count):
                            if (row_idx, col_idx) in occupied and grid[row_idx][col_idx] is None:
                                continue
                            cell = grid[row_idx][col_idx]
                            if not cell:
                                table_html += (
                                    '<td style="border: 1px solid #000; padding: 2px; '
                                    'box-sizing: border-box;"></td>'
                                )
                                continue
                            attributes = ""
                            if cell.get("rowspan"):
                                attributes += f' rowspan="{cell["rowspan"]}"'
                            if cell.get("colspan"):
                                attributes += f' colspan="{cell["colspan"]}"'
                            table_html += f'<td{attributes} style="{cell["style"]}">{cell["content"]}</td>'
                        table_html += "</tr>"
                    table_html += "</table>"

                    table_background = f"background: {table_fill['color']}; " if table_fill else ""
                    elements.append(
                        f'<div style="position: absolute; top: {top_scaled}px; left: {left_scaled}px; '
                        f'width: {width}px; height: {height}px; z-index: 20; '
                        f'{table_background}box-sizing: border-box; overflow: visible;">'
                        f'{table_html}'
                        f'</div>'
                    )
                    continue

                # Fallback: no per-cell bbox data
                grid, occupied, rows_count, cols_count = build_table_grid(table_data)
                for row_idx in range(rows_count):
                    for col_idx in range(cols_count):
                        cell = grid[row_idx][col_idx]
                        if not cell:
                            continue
                        rowspan = cell["rowspan"] or 1
                        colspan = cell["colspan"] or 1
                        cell_width = width / cols_count
                        cell_height = height / rows_count
                        cell_x0 = x0 + (col_idx * cell_width) / scale
                        cell_y0 = top_t + (row_idx * cell_height) / scale
                        cell_x1 = x0 + ((col_idx + colspan) * cell_width) / scale
                        cell_y1 = top_t + ((row_idx + rowspan) * cell_height) / scale
                        cell_bbox = (cell_x0, cell_y0, cell_x1, cell_y1)
                        cell_styles = extract_table_cell_styling(
                            page_mupdf, table["bbox"], cell_bbox, font_name_map, scale
                        )
                        cell["content"] = format_table_cell_text(html.unescape(cell["content"]))
                        cell["styles"] = cell_styles
                        cell["fill"] = pick_fill_for_bbox(
                            cell_bbox, filled_rects, min_coverage=FILL_COVERAGE_CELL
                        )

                table_html = '<table style="border-collapse: collapse; width: 100%; height: 100%;">'
                for r_idx in range(rows_count):
                    table_html += "<tr>"
                    for c_idx in range(cols_count):
                        if (r_idx, c_idx) in occupied and grid[r_idx][c_idx] is None:
                            continue
                        cell = grid[r_idx][c_idx]
                        if cell:
                            attributes = ""
                            if cell['rowspan']:
                                attributes += f' rowspan="{cell["rowspan"]}"'
                            if cell['colspan']:
                                attributes += f' colspan="{cell["colspan"]}"'
                            cell_styles = cell.get('styles', [])
                            primary_style = pick_primary_table_style(cell_styles)
                            if primary_style:
                                style_parts = [
                                    "border: 1px solid #000", "padding: 4px",
                                    "vertical-align: top", "line-height: 1.25",
                                    "overflow: visible", "white-space: normal",
                                    "word-break: break-word",
                                    f"font-family: '{primary_style['font_family']}', Arial, sans-serif",
                                    f"font-size: {primary_style['font_size']}px",
                                    f"color: {primary_style['font_color']}",
                                ]
                                if primary_style['is_bold']:
                                    style_parts.append("font-weight: 700")
                                if primary_style['is_italic']:
                                    style_parts.append("font-style: italic")
                            else:
                                style_parts = [
                                    "border: 1px solid #000", "padding: 4px",
                                    "vertical-align: top", "font-size: 12px",
                                    "line-height: 1.25", "overflow: visible",
                                    "white-space: normal", "word-break: break-word",
                                ]
                            if cell.get("fill"):
                                style_parts.append(f"background: {cell['fill']['color']}")
                            table_html += (
                                f'<td{attributes} style="{"; ".join(style_parts)}">'
                                f'{cell["content"]}</td>'
                            )
                        else:
                            table_html += (
                                '<td style="border: 1px solid #000; padding: 4px; '
                                'vertical-align: top; font-size: 12px; line-height: 1.25; '
                                'white-space: normal; word-break: break-word"></td>'
                            )
                    table_html += "</tr>"
                table_html += "</table>"

                table_background = f"background: {table_fill['color']}; " if table_fill else ""
                elements.append(
                    f'<div style="position: absolute; top: {top_scaled}px; left: {left_scaled}px; '
                    f'width: {width}px; height: {height}px; z-index: 20; '
                    f'{table_background}box-sizing: border-box; overflow: visible;">'
                    f'{table_html}'
                    f'</div>'
                )

            # scaled_page_width / scaled_page_height already computed above.

            page_container_style = (
                f"width: {scaled_page_width}px; "
                f"height: {scaled_page_height}px; "
                f"overflow: visible;"
            )
            if page_bg_color:
                page_container_style += f" background: {page_bg_color};"

            full_html += f'''
            <div class="page-wrapper" style="width: {scaled_page_width}px; height: {scaled_page_height}px;">
                <div class="page-container" style="{page_container_style}">
                    {"".join(elements)}
                </div>
            </div>
            '''

        # Finalizing
        conversion_progress[job_id]['progress'] = 95
        conversion_progress[job_id]['message'] = 'Finalizing HTML output...'
        if pdf_plumber:
            pdf_plumber.close()

        conversion_time = round(time.time() - start_time, 2)
        icon_count = len(conversion_progress.get(job_id, {}).get('icons', []))

        excel_panel_html = f"""
            <div class="excel-panel">
                <form action="/export_excel/{job_id}" method="post" target="_blank" class="excel-form">
                    <label for="excel-query"><strong>Excel Export</strong></label>
                    <input
                        id="excel-query"
                        name="query"
                        type="text"
                        placeholder="Example: export Bill Date and Due Date where Account ID = 7067817787"
                        required
                    />
                    <button type="submit">Generate Excel</button>
                    <a href="/analysis/{job_id}" target="_blank" class="analysis-btn">
                        &#x1F4CA;&nbsp; View Document Analysis
                    </a>
                </form>
            </div>
            <style>
                .analysis-btn {{
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                    padding: 10px 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #fff;
                    font-weight: 700;
                    font-size: 14px;
                    border-radius: 8px;
                    text-decoration: none;
                    letter-spacing: 0.3px;
                    box-shadow: 0 4px 14px rgba(102,126,234,0.45);
                    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
                }}
                .analysis-btn:hover {{
                    transform: translateY(-2px) scale(1.03);
                    box-shadow: 0 8px 24px rgba(102,126,234,0.6);
                    filter: brightness(1.08);
                }}
                .analysis-btn:active {{
                    transform: translateY(0);
                    box-shadow: 0 3px 10px rgba(102,126,234,0.4);
                }}
            </style>
        """

        # ----------------------------------------------------------------
        # CSS for the comparison / clean views.
        # NOTE: .positioned-text does NOT set white-space here — each element
        # sets its own white-space via inline style so paragraph wrapping works
        # correctly while headings / list items still get the right behaviour.
        # ----------------------------------------------------------------
        shared_css = f"""
            {font_css}
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background: #f5f5f5;
            }}
            .page-wrapper {{
                position: relative;
                margin: 30px auto;
                border: 1px solid #ccc;
                box-shadow: 0 2px 12px rgba(0,0,0,0.18);
                overflow: hidden;
                /* block + explicit width (set inline) ensures centering works */
                display: block;
            }}
            .page-container {{
                position: relative;
                background: white;
                /* overflow visible so positioned children near edges aren't clipped
                   by the container itself — the wrapper handles clipping */
                overflow: visible;
                isolation: isolate;
            }}
            .positioned-background {{
                position: absolute;
                pointer-events: none;
                z-index: 0;
            }}
            .positioned-text {{
                position: absolute;
                margin: 0;
                text-decoration: none;
                z-index: 2;
                box-sizing: border-box;
                overflow: visible;
            }}
            .positioned-text a {{
                color: blue;
                text-decoration: underline;
            }}
            .positioned-image {{
                position: absolute;
                object-fit: contain;
                z-index: 1;
            }}
            .semantic-only {{
                position: absolute;
                width: 1px;
                height: 1px;
                padding: 0;
                margin: -1px;
                overflow: hidden;
                clip: rect(0, 0, 0, 0);
                white-space: nowrap;
                border: 0;
            }}
            table {{
                border-collapse: collapse;
            }}
            table td {{
                border: 1px solid #000;
                padding: 4px;
                vertical-align: top;
            }}
        """

        final_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>PDF to HTML Comparison</title>
            <style>
                {shared_css}
                .header {{
                    padding: 15px;
                    background-color: #2c3e50;
                    color: white;
                    text-align: center;
                    font-size: 18px;
                    font-weight: bold;
                }}
                .info-bar {{
                    padding: 12px;
                    font-size: 14px;
                    background-color: #ecf0f1;
                    border-bottom: 1px solid #bdc3c7;
                    font-family: monospace;
                    text-align: center;
                }}
                .comparison-container {{
                    display: flex;
                    height: calc(100vh - 120px);
                }}
                .excel-panel {{
                    padding: 12px;
                    background: #f8fafc;
                    border-bottom: 1px solid #d5dde7;
                }}
                .excel-form {{
                    display: flex;
                    gap: 10px;
                    align-items: center;
                    justify-content: center;
                    flex-wrap: wrap;
                }}
                .excel-form input {{
                    min-width: 420px;
                    max-width: 760px;
                    width: 60%;
                    padding: 10px 12px;
                    border: 1px solid #b9c4d0;
                    border-radius: 6px;
                    font-size: 14px;
                }}
                .excel-form button {{
                    padding: 10px 16px;
                    border: none;
                    border-radius: 6px;
                    background: #1f5faa;
                    color: white;
                    font-weight: 600;
                    cursor: pointer;
                }}
                .pdf-panel, .html-panel {{
                    width: 50%;
                    border: 2px solid #34495e;
                    overflow: auto;
                }}
                .panel-header {{
                    background-color: #34495e;
                    color: white;
                    padding: 10px;
                    text-align: center;
                    font-weight: bold;
                    position: sticky;
                    top: 0;
                    z-index: 100;
                }}
                .pdf-content {{
                    padding: 20px;
                    text-align: center;
                    background: white;
                }}
                .pdf-embed {{
                    width: 100%;
                    height: 800px;
                    border: none;
                }}
                .html-content {{
                    background: #eee;
                    min-height: 100%;
                }}
            </style>
            <script>
                const t0 = performance.now();
                window.onload = () => {{
                    const t1 = performance.now();
                    document.getElementById("render-time").innerText = (t1 - t0).toFixed(2) + " ms";
                    const pdfPanel = document.querySelector('.pdf-panel');
                    const htmlPanel = document.querySelector('.html-panel');
                    let isScrollingPdf = false;
                    let isScrollingHtml = false;
                    pdfPanel.addEventListener('scroll', () => {{
                        if (isScrollingHtml) return;
                        isScrollingPdf = true;
                        const ratio = pdfPanel.scrollTop / (pdfPanel.scrollHeight - pdfPanel.clientHeight);
                        htmlPanel.scrollTop = ratio * (htmlPanel.scrollHeight - htmlPanel.clientHeight);
                        setTimeout(() => isScrollingPdf = false, 50);
                    }});
                    htmlPanel.addEventListener('scroll', () => {{
                        if (isScrollingPdf) return;
                        isScrollingHtml = true;
                        const ratio = htmlPanel.scrollTop / (htmlPanel.scrollHeight - htmlPanel.clientHeight);
                        pdfPanel.scrollTop = ratio * (pdfPanel.scrollHeight - pdfPanel.clientHeight);
                        setTimeout(() => isScrollingHtml = false, 50);
                    }});
                }};
            </script>
        </head>
        <body>
            <div class="header">PDF to HTML Conversion Comparison</div>
            <div class="info-bar">
                <b>Conversion time:</b> {conversion_time} seconds |
                <b>Render time:</b> <span id="render-time">...</span> |
                <b>Pages:</b> {total_pages} |
                <b>Icons detected:</b> {icon_count}
            </div>
            {excel_panel_html}
            <div class="comparison-container">
                <div class="pdf-panel">
                    <div class="panel-header">Original PDF</div>
                    <div class="pdf-content">
                        <embed class="pdf-embed" src="data:application/pdf;base64,{pdf_base64}" type="application/pdf" />
                    </div>
                </div>
                <div class="html-panel">
                    <div class="panel-header">Converted HTML</div>
                    <div class="html-content">
                        {full_html}
                    </div>
                </div>
            </div>
        </body>
        </html>
        '''

        clean_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>Converted PDF (Clean)</title>
                    <style>
                        {shared_css}
                        body {{
                            padding: 20px;
                        }}
                        .excel-panel {{
                            padding: 12px;
                            background: #f8fafc;
                            border-bottom: 1px solid #d5dde7;
                            margin-bottom: 16px;
                        }}
                        .excel-form {{
                            display: flex;
                            gap: 10px;
                            align-items: center;
                            flex-wrap: wrap;
                        }}
                        .excel-form input {{
                            min-width: 420px;
                            max-width: 760px;
                            width: 60%;
                            padding: 10px 12px;
                            border: 1px solid #b9c4d0;
                            border-radius: 6px;
                            font-size: 14px;
                        }}
                        .excel-form button {{
                            padding: 10px 16px;
                            border: none;
                            border-radius: 6px;
                            background: #1f5faa;
                            color: white;
                            font-weight: 600;
                            cursor: pointer;
                        }}
                    </style>
                </head>
                <body>
                    {excel_panel_html}
                    {full_html}
                </body>
                </html>
                """

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        clean_html_path = os.path.join(output_dir, f"{job_id}_clean.html")
        with open(clean_html_path, "w", encoding="utf-8") as f:
            f.write(clean_html)

        output_filename = f"{job_id}.html"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(clean_html)

        json_path = os.path.join(output_dir, f"{job_id}.json")
        json_payload = html_to_json(clean_html, json_path)
        detailed_tables_payload = {
            "job_id": job_id,
            "tables": extract_structured_tables_from_document_json(job_id, json_payload),
            "source": "document_json",
        }
        save_detailed_tables_json(job_id, detailed_tables_payload, output_dir=output_dir)

        conversion_progress[job_id]['progress'] = 100
        conversion_progress[job_id]['status'] = 'completed'
        conversion_progress[job_id]['message'] = f'Conversion completed in {conversion_time} seconds'
        conversion_progress[job_id]['result'] = final_html

    except Exception as e:
        import traceback
        traceback.print_exc()
        conversion_progress[job_id]['status'] = 'error'
        conversion_progress[job_id]['error'] = str(e)
        conversion_progress[job_id]['message'] = f'Error: {str(e)}'


@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/convert', methods=['POST'])
def convert_pdf():
    uploaded_file = request.files['pdf']
    filename = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
    uploaded_file.save(filename)
    errors = validate_pdf(filename)
    if errors:
        return jsonify({'status': 'error', 'message': ' | '.join(errors)}), 400
    job_id = str(uuid.uuid4())
    thread = threading.Thread(target=convert_pdf_with_progress, args=(filename, job_id))
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'ok', 'job_id': job_id})


@app.route('/progress/<job_id>')
def get_progress(job_id):
    if job_id in conversion_progress:
        return jsonify(conversion_progress[job_id])
    return jsonify({'status': 'not_found', 'error': 'Job not found'}), 404


@app.route('/edit/<job_id>', methods=['GET', 'POST'])
def edit_html(job_id):
    output_dir = "output"
    html_file = os.path.join(output_dir, f"{job_id}_clean.html")
    json_file = os.path.join(output_dir, f"{job_id}.json")

    if request.method == 'POST':
        try:
            edited_html = request.form['edited_html']
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(edited_html)
            if os.path.exists(json_file):
                try:
                    html_to_json(edited_html, json_file)
                except Exception as e:
                    print(f"Warning: Could not update JSON: {e}")
            if request.headers.get('Content-Type') == 'application/json' or request.is_json:
                return jsonify({'status': 'success', 'message': 'Changes saved successfully!'}), 200
            return f"""
                <html><body>
                    <h2>Changes saved successfully!</h2>
                    <a href='/edit/{job_id}'>← Back to Editor</a> |
                    <a href='/result/{job_id}' target='_blank'>View Result</a>
                    <script>setTimeout(() => window.location.href = '/edit/{job_id}', 2000);</script>
                </body></html>
            """
        except Exception as e:
            if request.headers.get('Content-Type') == 'application/json' or request.is_json:
                return jsonify({'status': 'error', 'message': f'Failed to save: {str(e)}'}), 500
            return f"<h2>Error saving file: {str(e)}</h2><a href='/edit/{job_id}'>← Back</a>", 500

    if os.path.exists(html_file):
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()
        except Exception as e:
            html_content = f"<!-- Error loading file: {str(e)} -->"
    else:
        html_content = f"<!-- File not found: {html_file} -->"

    return render_template('editor.html', job_id=job_id, html_content=html_content)


@app.route('/debug/<job_id>')
def debug_files(job_id):
    output_dir = "output"
    html_file = os.path.join(output_dir, f"{job_id}_clean.html")
    json_file = os.path.join(output_dir, f"{job_id}.json")
    output_files = os.listdir(output_dir) if os.path.exists(output_dir) else []

    rows_html = "".join(
        f"<li><code>{f}</code> ({os.path.getsize(os.path.join(output_dir, f))} bytes)</li>"
        for f in output_files
    )
    return f"""
    <html><head><title>Debug Info - Job {job_id}</title></head>
    <body>
        <h2>Debug Information for Job ID: {job_id}</h2>
        <h3>Expected Files:</h3>
        <ul>
            <li>HTML File: <code>{html_file}</code> - {'EXISTS' if os.path.exists(html_file) else 'NOT FOUND'}</li>
            <li>JSON File: <code>{json_file}</code> - {'EXISTS' if os.path.exists(json_file) else 'NOT FOUND'}</li>
        </ul>
        <h3>All files in output directory:</h3>
        <ul>{rows_html}</ul>
        <h3>Quick Actions:</h3>
        <ul>
            <li><a href="/edit/{job_id}">Edit HTML</a></li>
            <li><a href="/">Back to Upload</a></li>
        </ul>
    </body></html>
    """


@app.route('/compare/<job_id>')
def compare_view(job_id):
    if job_id in conversion_progress:
        progress = conversion_progress[job_id]
        if progress['status'] == 'completed':
            return render_template(
                'compare.html',
                pdf_base64=progress['pdf_base64'],
                html_content=progress['result'],
                job_id=job_id,
            )
        return f"Conversion not completed. Status: {progress['status']}", 400
    return "Job not found", 404


@app.route('/export_excel/<job_id>', methods=['POST'])
def export_excel(job_id):
    try:
        query = request.form.get('query', '').strip()
        if not query:
            return "Please enter a query for the Excel export.", 400
        result = handle_excel_export_query(job_id, query, output_dir="output")
        excel_path = result["excel_path"]
        if not os.path.exists(excel_path):
            return "Excel file was not created.", 500
        return send_file(
            excel_path,
            as_attachment=True,
            download_name=os.path.basename(excel_path),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        return f"Excel export failed: {str(e)}", 500


@app.route('/result/<job_id>')
def get_result(job_id):
    if job_id in conversion_progress:
        progress = conversion_progress[job_id]
        if progress['status'] == 'completed':
            del conversion_progress[job_id]
            return render_template_string(progress['result'])
        return f"Conversion not completed. Status: {progress['status']}", 400
    return "Job not found", 404


@app.route('/analysis/<job_id>')
def analysis_dashboard(job_id):
    return render_template('analysis.html', job_id=job_id)


@app.route('/api/analysis_data/<job_id>')
def analysis_data(job_id):
    output_dir = "output"
    result = generate_analytics_data(job_id, output_dir=output_dir)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)