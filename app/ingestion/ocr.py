"""OCR fallback for scanned/image-only pages. See app/ingestion/docling_ingest.py
for where this gets triggered (only on pages Docling extracted near-zero text from)."""
import re
from typing import Dict, Any, List, Optional
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
import ftfy
from app.tables.helpers import dedupe_columns


def page_text_is_thin(markdown_text: str, min_chars: int = 40) -> bool:
    """A page Docling 'converted' but returned almost no text is very likely
    scanned/image-only, not actually blank — this is the OCR trigger condition."""
    stripped = re.sub(r"\s+", "", markdown_text or "")
    return len(stripped) < min_chars


def ocr_page(path: str, page_no: int, dpi: int = 300) -> Dict[str, Any]:
    """Rasterizes ONE page and runs Tesseract with image_to_data (word boxes),
    not image_to_string, so row/column structure can be reconstructed."""
    images = convert_from_path(path, dpi=dpi, first_page=page_no, last_page=page_no)
    if not images:
        return {"text": "", "words": [], "avg_conf": 0, "n_words": 0}

    data = pytesseract.image_to_data(images[0], output_type=pytesseract.Output.DICT)
    words, lines = [], {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue
        conf = float(data["conf"][i]) if data["conf"][i] not in ("-1", "") else -1
        words.append({"text": word, "conf": conf, "left": data["left"][i],
                       "top": data["top"][i], "line_num": data["line_num"][i],
                       "block_num": data["block_num"][i]})
        lines.setdefault((data["block_num"][i], data["line_num"][i]), []).append(word)

    full_text = ftfy.fix_text("\n".join(" ".join(v) for v in lines.values()))
    confs = [w["conf"] for w in words if w["conf"] >= 0]
    avg_conf = round(sum(confs) / len(confs), 1) if confs else 0
    return {"text": full_text, "words": words, "avg_conf": avg_conf, "n_words": len(words)}


def ocr_words_to_table(words: List[Dict[str, Any]], col_gap_px: int = 40) -> Optional[pd.DataFrame]:
    """Heuristic table reconstruction from OCR word boxes. Always tag output
    attrs['source']='ocr' so callers treat it as lower-confidence than a
    native-extracted table."""
    if not words:
        return None
    lefts = sorted(w["left"] for w in words)
    col_bounds = [lefts[0]]
    for l in lefts[1:]:
        if l - col_bounds[-1] > col_gap_px:
            col_bounds.append(l)

    def col_index(left):
        for i in range(len(col_bounds) - 1, -1, -1):
            if left >= col_bounds[i] - col_gap_px // 2:
                return i
        return 0

    rows: Dict[int, Dict[int, List[str]]] = {}
    for w in words:
        r = w["line_num"] + w["block_num"] * 1000
        rows.setdefault(r, {}).setdefault(col_index(w["left"]), []).append(w["text"])

    n_cols = len(col_bounds)
    table_rows = [[" ".join(rows[r].get(c, [])) for c in range(n_cols)] for r in sorted(rows)]
    if len(table_rows) < 2:
        return None
    df = pd.DataFrame(table_rows[1:], columns=dedupe_columns(table_rows[0]))
    df.attrs["source"] = "ocr"
    return df
