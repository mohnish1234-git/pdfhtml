import os
import base64
import fitz  # PyMuPDF


def _is_likely_icon(width, height, colorspace_name=None):
    """
    Heuristic icon detector for embedded images.
    Returns True if the image dimensions and colorspace suggest an icon/logo
    rather than a full-page illustration or photo.
    """
    if width <= 0 or height <= 0:
        return False

    # Small/medium graphics are usually icons or decorative elements
    if width <= 192 and height <= 192:
        return True

    # Slim decorative symbols (wide but short, or tall but narrow)
    if width <= 320 and height <= 120:
        return True
    if height <= 320 and width <= 120:
        return True

    # Monochrome/grayscale images often represent line-icons
    if colorspace_name and colorspace_name.lower() in {"devicegray", "gray"}:
        return width <= 512 and height <= 512

    return False


def extract_icons_from_pdf(pdf_path, output_dir=None):
    """
    Extract likely icons/logos from a PDF.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file to extract icons from.
    output_dir : str or None
        If provided, save each extracted icon as a PNG file in this directory.

    Returns
    -------
    list[dict]
        Metadata for each detected icon, including:
        - page: 1-based page number
        - xref: PDF cross-reference index
        - width, height: pixel dimensions
        - colorspace: colorspace name string
        - data_uri: base64-encoded data URI for inline HTML use
        - file_path: (only if output_dir given) saved file path
    """
    doc = fitz.open(pdf_path)
    icons = []
    seen_xrefs = set()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for page_index, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]

            # Deduplicate by xref — same embedded image can appear on multiple pages
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            width = int(img[2]) if len(img) > 2 else 0
            height = int(img[3]) if len(img) > 3 else 0
            colorspace_name = str(img[5]) if len(img) > 5 else None

            if not _is_likely_icon(width, height, colorspace_name):
                continue

            try:
                pix = fitz.Pixmap(doc, xref)
                # Convert CMYK or alpha-channel images to RGB for consistent output
                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                png_bytes = pix.tobytes("png")
            except Exception as e:
                print(f"[Icon extraction warning] page {page_index + 1}, xref {xref}: {e}")
                continue

            data_uri = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('utf-8')}"

            icon_record = {
                "page": page_index + 1,
                "xref": xref,
                "width": width,
                "height": height,
                "colorspace": colorspace_name,
                "data_uri": data_uri,
            }

            if output_dir:
                out_path = os.path.join(
                    output_dir,
                    f"page_{page_index + 1}_icon_{img_index + 1}.png"
                )
                try:
                    with open(out_path, "wb") as f:
                        f.write(png_bytes)
                    icon_record["file_path"] = out_path
                except Exception as e:
                    print(f"[Icon save warning] {out_path}: {e}")

            icons.append(icon_record)

    doc.close()
    return icons