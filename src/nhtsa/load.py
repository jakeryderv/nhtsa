"""Load NHTSA complaints from the raw flat file, cached as Parquet.

The flat file has one row per (complaint, product, component), so a complaint
(ODINO) with several components or vehicles spans several rows.
load_complaints() collapses those into one row per complaint.
"""

import re
from pathlib import Path

import polars as pl

from nhtsa.paths import PROCESSED_DIR, RAW_DIR

FLAT_FILE = RAW_DIR / "FLAT_CMPL.txt"
FIELDS_FILE = RAW_DIR / "CMPL.txt"
ROWS_PARQUET = PROCESSED_DIR / "complaint_rows.parquet"

PRODUCT_FIELDS = ("MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT")

# e.g. "7       CRASH             CHAR(1)       WAS VEHICLE INVOLVED IN A CRASH, 'Y' OR 'N'"
_FIELD_LINE = re.compile(
    r"^\s*\d+\s+(\w+)\s+(CHAR|NUMBER)\(\d+\)\s+(.*)$", re.MULTILINE
)


def read_fields(path: Path = FIELDS_FILE) -> dict[str, pl.DataType]:
    """Map each flat-file column to a dtype, using the types in NHTSA's field list."""
    fields: dict[str, pl.DataType] = {}
    for name, kind, description in _FIELD_LINE.findall(path.read_text()):
        if kind == "NUMBER":
            fields[name] = pl.Int32()
        elif "YYYYMMDD" in description:
            fields[name] = pl.Date()
        elif "'Y' OR 'N'" in description:
            fields[name] = pl.Boolean()
        else:
            fields[name] = pl.String()
    return fields


def load_rows(refresh: bool = False) -> pl.DataFrame:
    """One row per (complaint, product, component), as in the flat file."""
    if refresh or _is_stale(ROWS_PARQUET, FLAT_FILE):
        _build_rows_parquet()
    return pl.read_parquet(ROWS_PARQUET)


def load_complaints(refresh: bool = False) -> pl.DataFrame:
    """One row per complaint (ODINO).

    - CDESCR: the complaint's distinct texts, newline-joined (old complaints
      sometimes have a separate note per component)
    - COMPONENTS: sorted distinct component descriptions
    - PRODUCTS: distinct (MFR_NAME, MAKETXT, MODELTXT, YEARTXT) structs
    - Y/N flags: true if any row says Y; INJURED/DEATHS: max; dates: earliest
    - everything else: the value from the complaint's first row
    """
    rows = load_rows(refresh)
    handled = {"ODINO", "CMPLID", "CDESCR", "COMPDESC", *PRODUCT_FIELDS}
    aggs = []
    for name, dtype in rows.schema.items():
        if name in handled:
            continue
        col = pl.col(name)
        if dtype == pl.Boolean:
            aggs.append(col.any())
        elif name in ("INJURED", "DEATHS"):
            aggs.append(col.max())
        elif dtype == pl.Date:
            aggs.append(col.min())
        else:
            aggs.append(col.first())

    return rows.group_by("ODINO", maintain_order=True).agg(
        pl.col("CDESCR").drop_nulls().unique(maintain_order=True).str.join("\n"),
        pl.col("COMPDESC").drop_nulls().unique().sort().alias("COMPONENTS"),
        pl.struct(PRODUCT_FIELDS).unique(maintain_order=True).alias("PRODUCTS"),
        pl.len().alias("N_ROWS"),
        *aggs,
    )


def _build_rows_parquet() -> None:
    fields = read_fields()
    raw = pl.read_csv(
        FLAT_FILE,
        separator="\t",
        has_header=False,
        new_columns=list(fields),
        quote_char=None,
        infer_schema=False,
    )
    # CDESCR keeps its original whitespace; other fields are padded inconsistently.
    padded = pl.col(pl.String).exclude("CDESCR")
    rows = raw.with_columns(padded.str.strip_chars().replace("", None)).with_columns(
        _cast(name, dtype) for name, dtype in fields.items()
    )
    rows = rows.with_columns(
        pl.when(pl.col("YEARTXT") != "9999").then(pl.col("YEARTXT"))  # 9999 = unknown
    )
    ROWS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(ROWS_PARQUET)


def _cast(name: str, dtype: pl.DataType) -> pl.Expr:
    col = pl.col(name)
    if dtype == pl.Date:
        return col.str.to_date("%Y%m%d", strict=False)  # invalid dates become null
    if dtype == pl.Boolean:
        return col.replace_strict({"Y": True, "N": False}, default=None)
    return col.cast(dtype, strict=False)


def _is_stale(target: Path, source: Path) -> bool:
    return not target.exists() or target.stat().st_mtime < source.stat().st_mtime
