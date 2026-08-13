
# Fix: Horizontal Merge + Column Combine Feature

## READ FIRST

```bash
cat app/graph/agent.py | grep -n "horizontal\|axis=1\|side.by.side\|hmerge\|combine_columns" | head -20
cat app/graph/tools.py | grep -n "def " | head -30
cat app/tables/assembly.py | grep -n "def " | head -20
```

## CONFIRMED FROM OUTPUTS FOLDER

Individual file exports work correctly:

- f1-123.xlsx: 64 rows, real headers (SL no, Item, Spec No., Material code, Qty)
- span-all.xlsx: 64 rows, real headers (Evaluation Schedules, Item/Category, etc.)

Horizontal merge fails:

- test_f1-2f.xlsx: 98 rows x 12 cols — file-1 data correct, file-2 columns
  (col_0,col_1,col_2,col_3) all NaN — file-2 was never actually fetched

## BUG: Horizontal merge tool does not exist

There is no `hmerge` or `horizontal_merge` or `axis=1` tool in tools.py.
When user asks for horizontal merge, the agent uses `export_data` which
only knows how to do vertical (append) operations, not side-by-side.

## FEATURE 1: Add horizontal merge tool to app/graph/tools.py

Add this new tool right after the export_data tool:

```python
@tool
def merge_files_side_by_side(
    description: str,
    file1_id: str = None,
    file1_pages: list = None,
    file2_id: str = None,
    file2_pages: list = None,
    output_filename: str = "merged.xlsx"
) -> str:
    """Merge data from two files SIDE BY SIDE — left columns from file 1,
    right columns from file 2. Like a SQL JOIN but without a key column.
    Row 1 of file1 sits next to Row 1 of file2 in the same spreadsheet row.

    Use when user says: 'side by side', 'horizontal merge', 'left side right
    side', 'SQL join style', 'put both tables next to each other'.
  
    DO NOT use for stacking rows on top of each other — that is export_data.

    Args:
        description: What data to extract from each file
        file1_id: First file's file_id (left side)
        file1_pages: Page numbers to use from file 1 (e.g. [8,9,10])
        file2_id: Second file's file_id (right side)
        file2_pages: Page numbers to use from file 2 (e.g. [6,7,8,9,10])
        output_filename: Output file name (must end in .xlsx)
    """
    from app import state as app_state
    from app.graph.agent import assemble_pages
    from app.config import OUTPUT_DIR
    import pandas as pd
    import os

    # Resolve file IDs
    if not file1_id and len(app_state.FILE_ORDER) >= 1:
        file1_id = app_state.FILE_ORDER[-2] if len(app_state.FILE_ORDER) >= 2 else app_state.FILE_ORDER[0]
    if not file2_id and len(app_state.FILE_ORDER) >= 2:
        file2_id = app_state.FILE_ORDER[-1]

    if not file1_id or not file2_id:
        return ("Need two files to merge side by side. "
                f"Currently loaded: {[app_state.FILE_ORIGINAL_NAME.get(f,f) for f in app_state.FILE_ORDER]}")

    name1 = app_state.FILE_ORIGINAL_NAME.get(file1_id, file1_id)
    name2 = app_state.FILE_ORIGINAL_NAME.get(file2_id, file2_id)

    # Extract data from each file
    try:
        if file1_pages:
            df1, _, _ = assemble_pages(file1_id, file1_pages)
        else:
            from app.tables.helpers import get_all_real_tables
            tables1 = get_all_real_tables(file1_id)
            df1 = pd.concat(tables1, ignore_index=True) if tables1 else pd.DataFrame()
    except Exception as e:
        return f"Could not extract data from {name1}: {e}"

    try:
        if file2_pages:
            df2, _, _ = assemble_pages(file2_id, file2_pages)
        else:
            from app.tables.helpers import get_all_real_tables
            tables2 = get_all_real_tables(file2_id)
            df2 = pd.concat(tables2, ignore_index=True) if tables2 else pd.DataFrame()
    except Exception as e:
        return f"Could not extract data from {name2}: {e}"

    if df1.empty:
        return f"No data found in {name1} for pages {file1_pages}"
    if df2.empty:
        return f"No data found in {name2} for pages {file2_pages}"

    # Add source suffix to ALL columns to distinguish them
    # Derive short name from original filename (e.g. "data-file-1" -> "f1")
    def short_suffix(name):
        import re
        m = re.search(r'data-file-(\d+)', name.lower())
        return f"f{m.group(1)}" if m else re.sub(r'[^a-z0-9]', '', name.lower())[:6]

    sfx1 = short_suffix(name1)
    sfx2 = short_suffix(name2)

    df1.columns = [f"{c}_{sfx1}" if not str(c).startswith('_') else c
                   for c in df1.columns]
    df2.columns = [f"{c}_{sfx2}" if not str(c).startswith('_') else c
                   for c in df2.columns]

    # Remove internal tracking columns
    df1 = df1[[c for c in df1.columns if not c.startswith('_source')]]
    df2 = df2[[c for c in df2.columns if not c.startswith('_source')]]

    # HORIZONTAL merge = axis=1 (side by side, not stacked)
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    merged = pd.concat([df1, df2], axis=1)

    # Save
    if not output_filename.endswith('.xlsx'):
        output_filename = output_filename + '.xlsx'
    out_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged.to_excel(out_path, index=False)

    # Verify
    if not os.path.exists(out_path):
        return f"FAILED: file not created at {out_path}"

    verify_df = pd.read_excel(out_path)
    f1_cols = [c for c in verify_df.columns if c.endswith(f'_{sfx1}')]
    f2_cols = [c for c in verify_df.columns if c.endswith(f'_{sfx2}')]

    return (
        f"Horizontal merge complete: {output_filename}\n"
        f"Shape: {verify_df.shape[0]} rows × {verify_df.shape[1]} columns\n"
        f"Left side ({name1}): {len(f1_cols)} columns — {f1_cols[:4]}...\n"
        f"Right side ({name2}): {len(f2_cols)} columns — {f2_cols[:4]}...\n"
        f"File ready to download."
    )
```

## FEATURE 2: Add column combine tool to app/graph/tools.py

Add this tool after merge_files_side_by_side:

```python
@tool
def combine_columns(
    source_filename: str,
    columns_to_combine: list,
    new_column_name: str,
    separator: str = " ",
    output_filename: str = None
) -> str:
    """Combine two or more columns into one new column.
  
    Examples:
    - first_name + last_name → full_name
    - city + state + country → address_combined
    - code + description → item_label
  
    Use when user says: 'combine columns', 'merge columns', 'join columns',
    'concatenate columns', 'create a new column from'.
  
    Args:
        source_filename: The xlsx file to modify (from outputs folder)
        columns_to_combine: List of column names to combine
        new_column_name: Name for the new combined column
        separator: Character to put between values (default is a space)
        output_filename: Save as this filename (default: same file with _combined)
    """
    from app.config import OUTPUT_DIR
    import pandas as pd
    import os

    # Find the file
    source_path = os.path.join(OUTPUT_DIR, source_filename)
    if not os.path.exists(source_path):
        available = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.xlsx')]
        return (f"File '{source_filename}' not found in outputs.\n"
                f"Available files: {available[:10]}")

    df = pd.read_excel(source_path)

    # Check all requested columns exist
    missing = [c for c in columns_to_combine if c not in df.columns]
    if missing:
        return (f"These columns were not found: {missing}\n"
                f"Available columns: {list(df.columns)}")

    # Create the combined column
    df[new_column_name] = df[columns_to_combine].astype(str).apply(
        lambda row: separator.join(v for v in row if v not in ('nan', 'None', '')),
        axis=1
    )

    # Save
    out_filename = output_filename or source_filename.replace('.xlsx', '_combined.xlsx')
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    df.to_excel(out_path, index=False)

    # Show sample
    sample = df[[*columns_to_combine, new_column_name]].head(3).to_string(index=False)

    return (
        f"Column combine complete: {out_filename}\n"
        f"Combined {columns_to_combine} → '{new_column_name}'\n"
        f"Sample:\n{sample}"
    )
```

## STEP 2: Register both tools in the agent

In app/graph/agent.py, find the TOOLS list and add both new tools:

```python
from app.graph.tools import (
    search_documents, query_table_data, list_uploaded_files,
    export_data, get_file_overview, generate_quotation,
    analyze_past_contracts,
    merge_files_side_by_side,   # ADD THIS
    combine_columns,             # ADD THIS
)

TOOLS = [
    search_documents,
    query_table_data,
    list_uploaded_files,
    export_data,
    get_file_overview,
    generate_quotation,
    analyze_past_contracts,
    merge_files_side_by_side,   # ADD THIS
    combine_columns,             # ADD THIS
]
```

Also update SYSTEM_PROMPT — add these rules:

```
7. HORIZONTAL MERGE: When user says 'side by side', 'left side right side',
   'SQL join style', 'horizontal merge', or 'put both files next to each
   other' — use merge_files_side_by_side tool. NEVER use export_data for
   horizontal merges. export_data only stacks rows vertically.

8. COLUMN COMBINE: When user says 'combine columns', 'merge first_name and
   last_name', 'create username from', 'concatenate columns' — use
   combine_columns tool on the relevant output file.
```

## STEP 3: Test both features

Test 1 — Horizontal merge:
Upload data-file-1.pdf and data-file-2.pdf then say:
"merge data file 1 pages 8 9 10 and data file 2 pages 6 7 8 9 10
side by side save as mrg-04.xlsx"

Expected result: mrg-04.xlsx with ~64 rows and 10+ columns
Left half ends in _f1, right half ends in _f2
MORE columns than rows (not 128 rows with 6 columns)

Test 2 — Column combine:
After getting mrg-04.xlsx, say:
"in mrg-04.xlsx combine the Item_f1 and Evaluation_Schedules_f2 columns
into one column called full_description save as mrg-04-combined.xlsx"

Expected result: mrg-04-combined.xlsx with a new full_description column

Test 3 — Username example (the first_name + last_name example):
"in mrg-04.xlsx combine SL_no_f1 and Item_f1 columns with a dash separator
to create a column called item_code save as mrg-04-combined.xlsx"

Expected result: item_code column has values like "1-40NB PN10 EN FF..."

Show me the shape and first 3 rows of mrg-04.xlsx after Test 1.

```

```
