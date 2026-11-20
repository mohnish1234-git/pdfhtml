# PDF → HTML Conversion Platform — Project Details

## 1. Overview

A Flask web application that converts digitally-authored PDFs into:
1. **Pixel-positioned HTML** ("Track 1") — absolutely-positioned divs replicating the PDF's visual layout.
2. **Clean semantic HTML + nested JSON** ("Track 2") — a hierarchical document tree (headings, paragraphs, lists, tables, figures) derived from Track 1.

It also provides a side-by-side PDF/HTML comparison viewer, an in-browser HTML editor, a natural-language → Excel table export feature, and a document analytics dashboard.

**Entry point:** `app.py` (single Flask app, ~4,000 lines). Supporting modules: `semantic_rule.py`, `icon_extractor.py`, `pdf_analytics.py`, `validation.py`.

## 2. Problem Statement

Converting PDFs to HTML typically forces a tradeoff between **visual fidelity** (pixel-accurate layout) and **semantic structure** (clean, meaningful markup usable by screen readers, search, or downstream data extraction). Off-the-shelf tools usually deliver one or the other — fixed-layout "print to HTML" dumps, or lossy text extraction. This project's objective is to produce both from one pass: a visually faithful rendering *and* a structured, semantically tagged document (HTML + JSON), including table and figure extraction, for arbitrary digital (non-scanned) PDFs.

## 3. Methodology

For each page:
1. Render the page to a raster image (PyMuPDF, `DPI=200`) for layout detection.
2. Run a YOLO document-layout model over the raster image to classify regions (title, section-header, text, list, table, figure, page-header/footer).
3. Extract text spans/blocks and vector geometry (rects, drawings) directly from the PDF via PyMuPDF (`get_text("dict")`, `get_drawings()`) — this is the source of truth for coordinates, not the YOLO boxes (YOLO is used only for semantic classification, per an explicit in-code design note).
4. Detect tables via PyMuPDF's `find_tables()` first, falling back to `pdfplumber` if none are found.
5. Reconcile layout against ground-truth PDF vector graphics: a geometric check (`table_has_ruled_grid`, `_has_real_vertical/horizontal_boundary`) verifies whether a detected row/column boundary is backed by an actual drawn line or a large background panel, versus being an artifact of the table-finder's text-clustering. This same evidence is used both to accept/reject whole tables and to merge spuriously over-segmented rows/columns.
6. Classify text blocks into semantic tags (`h1`/`h2`/`h3`/`p`/`li`/`footer`) using a hybrid of the YOLO label, a font-size histogram heuristic, and list-marker regex detection, with YOLO-asserted headings sanity-checked against actual font size.
7. Emit absolutely-positioned HTML (Track 1), then re-parse that HTML into `ContentElement` objects, column-aware reading-order sort them, and build a nested `Section` tree (Track 2) emitted as both clean HTML and JSON (`semantic_rule.py`).
8. Detect vector-drawn icons (not embedded raster images) by clustering nearby small vector-path groups and rasterizing the cluster's bounding region out of the pre-rendered page image.

## 4. Architecture

```
PDF upload
   │
   ▼
validation.py ── size/page-count/digital-text checks
   │
   ▼
app.py: convert_pdf_with_progress()  (per page, per PDF)
   ├─ PyMuPDF: text spans, rects, drawings, images, font extraction
   ├─ pdfplumber: table fallback
   ├─ YOLO (yolov12l-doclaynet.pt, + optional publaynet model): layout classes
   ├─ icon_extractor.py: raster-icon heuristic (_is_likely_icon)
   └─ geometric table verification (ruled-grid / boundary checks)
   │
   ├──► Track 1: positioned HTML  (output/<job_id>.html, _clean.html)
   │
   └──► html_to_json() ──► semantic_rule.py ──► Track 2: semantic HTML + JSON
                                                  (output/<job_id>_semantic.html, .json)
   │
   ▼
pdf_analytics.py ── per-page heuristic analytics (word/image/table density, "quality score")
   │
   ▼
Flask routes (templates/*.html) ── upload UI, progress polling, compare view,
                                    in-browser editor, Excel export, analysis dashboard
```

**Data flow note:** Track 2 (semantic HTML/JSON) is *derived by re-parsing* Track 1's rendered HTML (`html_to_json`), not computed independently — correctness of the semantic output is therefore bounded by correctness of the positioned-HTML pass.

## 5. Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask (templates in `templates/`, Jinja2 + inline f-string HTML generation) |
| PDF parsing | PyMuPDF (`fitz`) — primary; `pdfplumber` — table-detection fallback |
| Layout detection | Ultralytics YOLO, model `yolov12l-doclaynet.pt` (DocLayNet-trained); optional second `publaynet_yolo_best.pt` model (**not present in repo** — code degrades gracefully if missing) |
| Font handling | `fontTools` (TTFont) for embedded font metadata/style detection; base64-embedded `@font-face` when font bytes are available, generic family fallback otherwise |
| HTML parsing | BeautifulSoup |
| Imaging | Pillow (PIL) |
| Analytics | pandas, NumPy, SciPy, scikit-learn (`RandomForestRegressor`, `LinearRegression`, `r2_score`) |
| Frontend | Plain HTML/CSS/JS templates (`templates/*.html`, `static/styles.css`) — no JS framework |
| No `requirements.txt` found in repo | **Not found** (dependencies inferred from imports only) |

## 6. Implementation Highlights

- **Positioning math:** scale factor = `TARGET_WIDTH (960px) / page_width_pt`; all element coordinates derive from PDF-point bboxes × this scale, not from YOLO pixel boxes.
- **Table pipeline:** `detect_tables_pymupdf` → `table_has_ruled_grid` (geometry-gated acceptance, overriding a word-count-based prose-panel filter when real ruling is confirmed) → `cleanup_table_rows` / `cleanup_table_columns` (merge over-segmented rows/columns using drawn-line evidence, with text re-derived from merged cell bboxes via `get_textbox` rather than naive string concatenation) → per-cell span/style/image extraction → rendered `<table>`.
- **Vector icon extraction:** `detect_vector_icon_regions` clusters `page.get_drawings()` path groups by proximity/size, excluding large single fills (backgrounds/borders); qualifying clusters are cropped from the pre-rendered page raster and embedded as `<img>`, routed into table cells or page-level divs depending on position.
- **Font-family resolution:** `_resolve_font_family` / `_generic_font_family_fallback` map PDF-internal font names to CSS families, with a style-suffix-stripping step applied only to raw/embedded names (not to final fallback names, to avoid corrupting names like "Times New Roman").
- **NL → Excel export:** `parse_excel_export_query` / `find_best_table_for_query` do keyword/column matching against extracted table JSON to answer free-text export requests (e.g., "export X and Y where Z = value"), writing results via a hand-rolled minimal XLSX writer (`export_rows_to_excel`).
- **Analytics dashboard:** per-page word/image/table/heading counts plus a **heuristic** (not learned/validated) "conversion quality score" with injected random noise (`np.random.uniform(-3, 3)`) "for realism" — see Limitations.
- **Background wipe on startup:** `uploads/` and `output/` are deleted and recreated every time `app.py` is imported/started (line ~572) — intentional for a dev/demo environment, but destructive of prior results on restart.

## 7. Results / Metrics

No formal accuracy benchmark, test suite, or ground-truth evaluation set exists in the repo. Validation to date has been manual/visual, against two sample PDFs during development:

- A richly-formatted, multi-font educational guide (custom fonts, colored panel layout, multi-column tables) — table column/row over- and under-segmentation, spurious badge/background overlap, missing vector icons, and font-fallback bugs were identified and fixed through direct pixel comparison against the source PDF.
- A plain black-and-white academic document with non-embedded standard fonts (Times New Roman, Arial) — confirmed the same fixes generalize, and surfaced/fixed an additional font-name-corruption bug specific to non-embedded-font fallback.

No quantitative metrics (accuracy %, IoU, text-extraction precision/recall, processing time benchmarks) are recorded anywhere in the codebase or output. The in-app "conversion_score"/"consistency_score" in `pdf_analytics.py` are self-referential heuristics over the tool's own output, not an external accuracy measure.

## 8. Limitations

- **No OCR support** — `validation.py` explicitly rejects PDFs with no extractable text layer (scanned/image-only documents).
- **File limits** — 100 MB, 51 pages (`validation.py`); not configurable without code changes.
- **No automated tests** — no `tests/` directory or CI configuration found.
- **No dependency manifest** — no `requirements.txt`/`pyproject.toml`; environment is not reproducible from the repo alone.
- **Font fidelity** — when a PDF doesn't embed its fonts, CSS falls back to a generic system font; glyph-width differences between the fallback and the true original font can still cause minor text clipping/overlap in edge cases (observed and partially, not fully, mitigated).
- **Table detection depends on PyMuPDF/pdfplumber's own table-finding**; the geometric correction layer can only merge/accept what those libraries detect — it cannot discover a table neither library found at all.
- **Analytics "quality score" is a heuristic with injected randomness**, not a validated model — not suitable as a real accuracy metric.
- **Single-process, in-memory job state** (`conversion_progress` dict) — no persistence/queue; not production/multi-worker safe as written.
- **Startup wipes `uploads/`/`output/`** — any restart discards prior jobs' input/output files.
- **Secondary PubLayNet model file is optional and absent** from the repo; only the DocLayNet model ships.

## 9. Future Improvements

- Add an automated regression suite (golden-file HTML/JSON diffs per sample PDF) to prevent silent regressions — today, verification is manual/visual per change.
- Add OCR fallback (e.g., Tesseract) for scanned PDFs.
- Replace in-memory `conversion_progress` with persistent job storage (DB/Redis) and a real task queue for multi-user/production use.
- Pin and publish a `requirements.txt`/lockfile for reproducible environments.
- Extend the geometric table-boundary verification to recover tables missed entirely by PyMuPDF/pdfplumber (e.g., borderless tables inferred purely from text alignment).
- Replace the heuristic analytics "quality score" with a metric validated against human-rated or ground-truth conversions.
- Add a real font-embedding pipeline (subset the actual PDF font as WOFF) for non-embedded fonts where extractable, instead of relying on system-font fallback.

---
*This document was generated from static analysis of the repository as of the current commit (`68746ab`, single "Initial" commit — no further git history available) plus manual verification performed during this development session. Anything not directly observable in code or session evidence is marked **Not found** rather than assumed.*
