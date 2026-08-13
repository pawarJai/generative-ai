"""File-format exporters — csv/excel/docx/pptx/chart. All write into
app.config.OUTPUT_DIR so the API layer can serve/download them predictably."""
import os
import re
import traceback
from typing import Optional, List
import pandas as pd
from app.config import OUTPUT_DIR
from app.models import QueryPlan
from app.export.schema_map import prep_tables, validation_note


def _out(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)


def _remember(path: str, file_id: str, tables: List[pd.DataFrame]) -> None:
    """Record what produced a file, so it can be modified later.

    A user who asks "now add the header details to that excel" is talking
    about a file that exists. Without this the only way to honour that is to
    re-derive the whole export from a prompt they should not have to repeat —
    which is exactly what failed, three turns running, in the production log.
    """
    from app import state
    first = tables[0] if tables else None
    state.WRITTEN_EXPORTS[os.path.basename(path)] = {
        "file_id": file_id,
        "context": (first.attrs.get("context") if first is not None else None),
        "pages": (first.attrs.get("pages") or
                  ([first.attrs["page"]] if first is not None
                   and first.attrs.get("page") else None)) if first is not None else None,
        "sheets": len(tables),
    }


def band_offset(path: str, sheet_name=0) -> int:
    """How many rows sit above a sheet's real header row.

    Detected from the shape of the sheet, not passed in by the caller, so
    every reader of an exported workbook agrees on where the table starts —
    including the several places that re-verify a file after the exporter
    already did. A context-band row is one merged cell across the table, so
    openpyxl stores a value in its first column only; the header row is the
    first row with two or more populated cells. A genuinely single-column
    table has no such row, and correctly gets an offset of 0.
    """
    try:
        probe = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=40)
    except Exception:
        return 0
    for i in range(len(probe)):
        if int(probe.iloc[i].notna().sum()) >= 2:
            return i
    return 0


def verify_export(path: str, expected_rows: int, format_type: str = "csv",
                  skiprows=None) -> tuple[bool, str]:
    """Hard verification that a file was actually written with plausible data.
    Returns (success: bool, message: str).

    For CSV: parses the actual file to count rows (handles embedded newlines in quoted fields).
    For other formats: checks file exists and size is reasonable.

    ``skiprows`` is how many context-band rows sit above the header row — an
    int for every sheet, or a list in sheet order. Without it a band would be
    counted as data, and a short export could hide behind its own letterhead.
    """
    if not os.path.exists(path):
        return False, f"FAILED — no file at {path}"

    size = os.path.getsize(path)
    if size < 20:
        return False, f"FAILED — {path} is empty/corrupt ({size} bytes)"

    # For CSV: parse to get true row count (handles quoted fields with newlines)
    if format_type == "csv":
        try:
            df_check = pd.read_csv(path)
            actual_rows = len(df_check)
            if actual_rows < 1:
                return False, f"FAILED — {path} has no data rows"
            # Allow 10% variance
            if actual_rows < expected_rows * 0.9:
                return False, (f"FAILED — {path} has {actual_rows} rows, "
                              f"expected ~{expected_rows}. File may be incomplete.")
            return True, f"✓ Verified: {actual_rows} rows saved to {os.path.basename(path)} ({size} bytes)"
        except Exception as e:
            return False, f"FAILED — could not parse {path}: {e}"

    # For Excel/DOCX/PPTX: verify row count if expected_rows is provided
    elif format_type == "excel":
        try:
            # Every sheet, not just the first. A multi-document export writes
            # one sheet per source file, and counting only sheet 1 against a
            # whole-workbook expected_rows failed verification on a workbook
            # that was in fact complete.
            names = pd.ExcelFile(path).sheet_names
            offsets = (list(skiprows) if isinstance(skiprows, (list, tuple))
                       else [skiprows] * len(names) if skiprows is not None
                       else [band_offset(path, n) for n in names])
            offsets += [0] * (len(names) - len(offsets))
            sheets = {n: pd.read_excel(path, sheet_name=n, skiprows=off)
                      for n, off in zip(names, offsets)}
            actual_rows = sum(len(d) for d in sheets.values())
            if actual_rows < 1:
                return False, f"FAILED — {path} has no data rows"
            if actual_rows < expected_rows * 0.9:
                return False, (f"FAILED — {path} has {actual_rows} rows, "
                              f"expected ~{expected_rows}. File may be incomplete.")
            where = (f" across {len(sheets)} sheets ({', '.join(sheets)})"
                     if len(sheets) > 1 else "")
            return True, (f"✓ Verified: {actual_rows} rows saved to "
                          f"{os.path.basename(path)}{where} ({size} bytes)")
        except Exception as e:
            return False, f"FAILED — could not read {path}: {e}"

    elif format_type == "docx":
        # For DOCX, we can't easily verify row count without parsing the structure
        # Just confirm it exists and has reasonable size
        return True, f"✓ Verified: saved to {os.path.basename(path)} ({size} bytes)"

    elif format_type == "pptx":
        # For PPTX, just verify it exists and is reasonable size
        return True, f"✓ Verified: saved to {os.path.basename(path)} ({size} bytes)"

    elif format_type == "chart":
        # For image files, just verify they exist and are non-empty
        if size < 100:
            return False, f"FAILED — {path} is too small to be a valid image ({size} bytes)"
        return True, f"✓ Verified: chart saved to {os.path.basename(path)} ({size} bytes)"

    return True, f"✓ Verified: saved to {os.path.basename(path)} ({size} bytes)"


def safe_export(fn):
    """A tool bug should never crash the request — log the traceback, return
    a plain-English message so the API call stays usable."""
    def wrapper(file_id, plan, tables=None):
        try:
            return fn(file_id, plan, tables=tables)
        except Exception as e:
            traceback.print_exc()
            return f"Export failed ({type(e).__name__}: {e}). Try naming the specific columns/tables you want."
    wrapper.__name__ = fn.__name__
    return wrapper


@safe_export
def export_csv(file_id: str, plan: QueryPlan, tables: Optional[List[pd.DataFrame]] = None) -> str:
    tables = tables if tables is not None else prep_tables(file_id, plan)
    if not tables:
        return "No matching tables found to export."
    out = _out(plan.filename or "export.csv")
    df = pd.concat(tables, ignore_index=True, sort=False)
    expected_rows = len(df)
    df.to_csv(out, index=False)
    # A CSV is a data-interchange format: a context band written into one
    # breaks every parser that reads it. The context is remembered against the
    # file so it can still be added if the user asks for it in a format that
    # can carry it.
    df.attrs.setdefault("context", tables[0].attrs.get("context"))
    _remember(out, file_id, [df])

    # Verify file was actually written with correct data (parses CSV to handle embedded newlines)
    success, message = verify_export(out, expected_rows, format_type="csv")
    if not success:
        return message
    return message


_SHEET_SAFE_RE = re.compile(r"[\[\]:*?/\\]")


def _sheet_names(tables: List[pd.DataFrame]) -> List[str]:
    """Sheet titles for a workbook — a caller-supplied name via
    df.attrs["sheet_name"] (used by multi-document exports to label each
    source document) or the plain Table_N fallback. Excel rejects a
    workbook outright if any name repeats, exceeds 31 chars, or contains
    []:*?/\\, so both are enforced here rather than at every call site."""
    used, out = set(), []
    for i, d in enumerate(tables, 1):
        fallback = f"Table_{i}"
        name = _SHEET_SAFE_RE.sub("-", str(d.attrs.get("sheet_name") or fallback))
        name = name.strip()[:31] or fallback
        base, n = name, 2
        while name.lower() in used:
            suffix = f"_{n}"
            name = base[: 31 - len(suffix)] + suffix
            n += 1
        used.add(name.lower())
        out.append(name)
    return out


_BAND_GREY = "FF595959"
_MAX_COL_WIDTH = 60


def _write_band(worksheet, styled: List[tuple], width: int) -> None:
    """Write the context band above a table, each line merged across it."""
    from openpyxl.styles import Alignment, Font

    for row, (text, role) in enumerate(styled, start=1):
        cell = worksheet.cell(row=row, column=1, value=text)
        if width > 1:
            worksheet.merge_cells(start_row=row, start_column=1,
                                  end_row=row, end_column=width)
        cell.font = Font(bold=role in ("title", "heading"),
                         italic=role == "note",
                         size=12 if role == "title" else 11,
                         color=_BAND_GREY if role == "note" else None)
        cell.alignment = Alignment(horizontal="left", vertical="center")


def _fit_columns(worksheet, df: pd.DataFrame, header_row: int) -> None:
    """Widen columns to their content so the band does not force a
    single-column-wide sheet with everything truncated."""
    from openpyxl.utils import get_column_letter

    for i, name in enumerate(df.columns, start=1):
        longest = max([len(str(name))] +
                      [len(str(v)) for v in df[name].head(200).tolist()] or [0])
        worksheet.column_dimensions[get_column_letter(i)].width = \
            min(_MAX_COL_WIDTH, max(10, longest + 2))


@safe_export
def export_excel(file_id: str, plan: QueryPlan, tables: Optional[List[pd.DataFrame]] = None) -> str:
    from app.tables.context import styled_lines

    tables = tables if tables is not None else prep_tables(file_id, plan)
    if not tables:
        return "No matching tables found to export."
    out = _out(plan.filename or "export.xlsx")
    total_rows = sum(len(t) for t in tables)
    names = _sheet_names(tables)
    offsets = []
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        for d, sheet in zip(tables, names):
            # The letterhead, the section heading and the provenance line go
            # above the header row. Three consecutive turns in the production
            # log asked for exactly this and failed, because until now no code
            # path could put anything above row 1 of a sheet.
            band = [] if plan.no_context else styled_lines(d)
            start = len(band) + 1 if band else 0
            offsets.append(start)
            d.to_excel(xl, sheet_name=sheet, index=False, startrow=start)
            worksheet = xl.sheets[sheet]
            if band:
                _write_band(worksheet, band, max(1, d.shape[1]))
            _fit_columns(worksheet, d, start)

    _remember(out, file_id, tables)

    # Verify file was actually written with correct data
    success, message = verify_export(out, total_rows, format_type="excel")
    if not success:
        return message
    return message


@safe_export
def export_docx(file_id: str, plan: QueryPlan, tables: Optional[List[pd.DataFrame]] = None) -> str:
    from docx import Document as DocxDocument
    from app.tables.context import styled_lines

    tables = tables if tables is not None else prep_tables(file_id, plan)
    doc = DocxDocument()
    doc.add_heading(f"Extracted data — {file_id}", level=1)
    for i, df in enumerate(tables, 1):
        # Multi-document exports label each frame with the document it came
        # from; without this every section reads "Table 3" and the reader has
        # no way to tell which of five uploads produced it.
        doc.add_heading(str(df.attrs.get("sheet_name") or f"Table {i}"), level=2)
        # The source document's own letterhead and section heading, verbatim,
        # so a table lifted out of a tender still says which tender it is.
        for text, role in ([] if plan.no_context else styled_lines(df)):
            para = doc.add_paragraph()
            run = para.add_run(text)
            run.bold = role in ("title", "heading")
            run.italic = role == "note"
        t = doc.add_table(rows=1, cols=len(df.columns))
        t.style = "Light Grid Accent 1"
        for j, col in enumerate(df.columns):
            t.rows[0].cells[j].text = str(col)
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for j, val in enumerate(row):
                cells[j].text = str(val)
    out = _out(plan.filename or "export.docx")
    doc.save(out)

    # Verify file was actually written
    success, message = verify_export(out, len(tables), format_type="docx")
    if not success:
        return message
    return message


@safe_export
def export_pptx(file_id: str, plan: QueryPlan, tables: Optional[List[pd.DataFrame]] = None) -> str:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from app.tables.context import styled_lines

    tables = tables if tables is not None else prep_tables(file_id, plan)
    prs = Presentation()
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    # A deck built from several documents was titled with one file_id — the
    # first of the sources — which misrepresents the other four. Name them.
    sources = []
    for df in tables:
        s = str(df.attrs.get("source_file") or df.attrs.get("sheet_name") or "")
        if s and s not in sources:
            sources.append(s)
    title_slide.shapes.title.text = (
        f"Extracted data — {len(sources)} documents" if len(sources) > 1
        else f"Extracted data — {file_id}")
    if len(sources) > 1 and len(title_slide.placeholders) > 1:
        title_slide.placeholders[1].text = "\n".join(sources[:8])

    blank = prs.slide_layouts[6]
    exported_tables = 0
    for i, df in enumerate(tables[:15], 1):
        heading = str(df.attrs.get("sheet_name") or f"Table {i}")
        total_rows = len(df)
        # Wide source tables (data-file-5 has 76 columns) render as unreadable
        # slivers and silently lose the rest, so a slide carries the first
        # columns and says so rather than pretending it shows everything.
        shown = df.iloc[:12, :8]
        slide = prs.slides.add_slide(blank)
        # The blank layout has no title placeholder — a multi-document deck
        # needs each slide to name its source document, so the heading is a
        # real textbox.
        tb = slide.shapes.add_textbox(Inches(0.4), Inches(0.15), Inches(9), Inches(0.4))
        note = f"{heading} — rows 1-{min(12, total_rows)} of {total_rows}"
        if df.shape[1] > 8:
            note += f", first 8 of {df.shape[1]} columns"
        tb.text_frame.text = note
        # The document's own identification, under the slide heading. A deck
        # is the format most likely to be read away from the source file, so
        # it is the one that can least afford to drop it.
        band = [] if plan.no_context else styled_lines(df)
        top = 0.7
        if band:
            ctx_box = slide.shapes.add_textbox(Inches(0.4), Inches(0.55),
                                               Inches(9), Inches(0.22 * len(band)))
            frame = ctx_box.text_frame
            frame.text = band[0][0]
            for text, _role in band[1:]:
                frame.add_paragraph().text = text
            for para in frame.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(10)
            top = 0.6 + 0.22 * len(band)
        df = shown
        rows, cols = df.shape[0] + 1, df.shape[1]
        table_shape = slide.shapes.add_table(
            rows, cols, Inches(0.4), Inches(top), Inches(9), Inches(0.4 * rows)
        ).table
        for j, col in enumerate(df.columns):
            table_shape.cell(0, j).text = str(col)
        for r in range(df.shape[0]):
            for j in range(cols):
                table_shape.cell(r + 1, j).text = str(df.iat[r, j])
        exported_tables = i
    out = _out(plan.filename or "export.pptx")
    prs.save(out)

    # Verify file was actually written
    success, message = verify_export(out, exported_tables, format_type="pptx")
    if not success:
        return message
    return message


@safe_export
def export_chart(file_id: str, plan: QueryPlan, tables: Optional[List[pd.DataFrame]] = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tables = tables if tables is not None else prep_tables(file_id, plan)
    if not tables:
        return "No tables found to chart."
    df = tables[0]
    numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.6]
    out = _out(plan.filename or "export.png")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(df))))
    if numeric_cols:
        label_col = next((c for c in df.columns if c not in numeric_cols), df.columns[0])
        df.plot(x=label_col, y=numeric_cols[0], kind="bar", ax=ax, legend=False)
        ax.set_ylabel(numeric_cols[0])
    else:
        ax.axis("off")
        ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # Verify chart image was actually written
    success, message = verify_export(out, len(df), format_type="chart")
    if not success:
        return message
    return message


EXPORTERS = {"csv": export_csv, "excel": export_excel, "docx": export_docx,
             "pptx": export_pptx, "chart": export_chart}
