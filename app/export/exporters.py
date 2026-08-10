"""File-format exporters — csv/excel/docx/pptx/chart. All write into
app.config.OUTPUT_DIR so the API layer can serve/download them predictably."""
import os
import traceback
from typing import Optional, List
import pandas as pd
from app.config import OUTPUT_DIR
from app.models import QueryPlan
from app.export.schema_map import prep_tables, validation_note


def _out(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)


def verify_export(path: str, expected_rows: int, format_type: str = "csv") -> tuple[bool, str]:
    """Hard verification that a file was actually written with plausible data.
    Returns (success: bool, message: str).

    For CSV: parses the actual file to count rows (handles embedded newlines in quoted fields).
    For other formats: checks file exists and size is reasonable.
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
            df_check = pd.read_excel(path)
            actual_rows = len(df_check)
            if actual_rows < 1:
                return False, f"FAILED — {path} has no data rows"
            if actual_rows < expected_rows * 0.9:
                return False, (f"FAILED — {path} has {actual_rows} rows, "
                              f"expected ~{expected_rows}. File may be incomplete.")
            return True, f"✓ Verified: {actual_rows} rows saved to {os.path.basename(path)} ({size} bytes)"
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

    # Verify file was actually written with correct data (parses CSV to handle embedded newlines)
    success, message = verify_export(out, expected_rows, format_type="csv")
    if not success:
        return message
    return message


@safe_export
def export_excel(file_id: str, plan: QueryPlan, tables: Optional[List[pd.DataFrame]] = None) -> str:
    tables = tables if tables is not None else prep_tables(file_id, plan)
    if not tables:
        return "No matching tables found to export."
    out = _out(plan.filename or "export.xlsx")
    total_rows = sum(len(t) for t in tables)
    with pd.ExcelWriter(out) as xl:
        for i, d in enumerate(tables, 1):
            d.to_excel(xl, sheet_name=f"Table_{i}"[:31], index=False)

    # Verify file was actually written with correct data
    success, message = verify_export(out, total_rows, format_type="excel")
    if not success:
        return message
    return message


@safe_export
def export_docx(file_id: str, plan: QueryPlan, tables: Optional[List[pd.DataFrame]] = None) -> str:
    from docx import Document as DocxDocument
    tables = tables if tables is not None else prep_tables(file_id, plan)
    doc = DocxDocument()
    doc.add_heading(f"Extracted data — {file_id}", level=1)
    for i, df in enumerate(tables, 1):
        doc.add_heading(f"Table {i}", level=2)
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
    from pptx.util import Inches
    tables = tables if tables is not None else prep_tables(file_id, plan)
    prs = Presentation()
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = f"Extracted data — {file_id}"

    blank = prs.slide_layouts[6]
    exported_tables = 0
    for i, df in enumerate(tables[:15], 1):
        df = df.iloc[:12]
        slide = prs.slides.add_slide(blank)
        rows, cols = df.shape[0] + 1, df.shape[1]
        table_shape = slide.shapes.add_table(
            rows, cols, Inches(0.4), Inches(0.4), Inches(9), Inches(0.4 * rows)
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
