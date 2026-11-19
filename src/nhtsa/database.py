"""Build the DuckDB database that query tools read.

The database holds three curated tables with plain column names, plus a
one-row dataset_info table. It contains only what a query tool may see: VINs,
cities, dealer contacts, and the mostly-empty tire and child-seat fields are
left out. sources/complaints.yaml describes the tables for such tools;
check_manifest() keeps the two in step.
"""

import json
import tempfile
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import duckdb
import polars as pl
import yaml

from nhtsa.labels import component_label
from nhtsa.load import load_complaints
from nhtsa.paths import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR

DATABASE = PROCESSED_DIR / "nhtsa.duckdb"
MANIFEST = PROJECT_ROOT / "sources" / "complaints.yaml"
DOWNLOAD_MANIFEST = RAW_DIR / "manifest.json"

# CMPL_TYPE codes (CMPL.txt field 21); anything else is "other".
SOURCES = {
    "IVOQ": "website",
    "EVOQ": "hotline",
    "LETR": "letter",
    "VOQ": "questionnaire",
    "SVOQ": "questionnaire",
    "MVOQ": "questionnaire",
    "MIVQ": "mobile_app",
    "MAVQ": "mobile_app",
}
# PROD_TYPE codes (CMPL.txt field 46).
PRODUCT_TYPES = {
    "V": "vehicle",
    "T": "tires",
    "E": "equipment",
    "C": "child_restraint",
}


def build_database(path: Path = DATABASE) -> dict[str, int]:
    """Build the database from the downloaded complaints; returns rows per table."""
    tables = curated_tables(load_complaints())
    write_database(tables, data_version(), path)
    check_manifest(path)
    return {name: table.height for name, table in tables.items()}


def curated_tables(complaints: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """The curated tables, from load_complaints() output."""
    complaint_id = pl.col("ODINO").cast(pl.Int64).alias("complaint_id")
    vehicles = (
        complaints.select(complaint_id, "PRODUCTS")
        .explode("PRODUCTS")
        .unnest("PRODUCTS")
        .select(
            "complaint_id",
            manufacturer=_tidy(pl.col("MFR_NAME")),
            make=_tidy(pl.col("MAKETXT")),
            model=_tidy(pl.col("MODELTXT")),
            model_year=pl.col("YEARTXT").cast(pl.Int32, strict=False),
        )
    )
    # Makes first, so models are merged within the canonical make.
    vehicles = (
        vehicles.with_columns(make=_canonical("make"))
        .with_columns(model=_canonical("model", within="make"))
        .unique(maintain_order=True)
    )
    components = (
        complaints.select(complaint_id, "COMPONENTS")
        .explode("COMPONENTS")
        .drop_nulls("COMPONENTS")
        .select(
            "complaint_id",
            component=component_label(pl.col("COMPONENTS")),
            component_detail=pl.col("COMPONENTS"),
        )
    )
    return {
        "complaints": complaints.select(
            complaint_id,
            received_date=pl.col("LDATE"),
            received_year=pl.col("LDATE").dt.year(),
            incident_date=pl.col("FAILDATE"),
            source=pl.col("CMPL_TYPE").replace_strict(SOURCES, default="other"),
            product_type=pl.col("PROD_TYPE").replace_strict(
                PRODUCT_TYPES, default=None
            ),
            crash=pl.col("CRASH"),
            fire=pl.col("FIRE"),
            injuries=pl.col("INJURED").cast(pl.Int32),
            deaths=pl.col("DEATHS").cast(pl.Int32),
            medical_attention=pl.col("MEDICAL_ATTN"),
            police_report=pl.col("POLICE_RPT_YN"),
            # 0 is how a blank mileage reads since the 2021 system change.
            mileage=pl.col("MILES").cast(pl.Int32).replace(0, None),
            consumer_state=pl.col("STATE"),
            description=pl.col("CDESCR"),
        ),
        "complaint_vehicles": vehicles,
        "complaint_components": components,
    }


def data_version(download_manifest: Path = DOWNLOAD_MANIFEST) -> str:
    """The date NHTSA last modified the complaints file, as YYYY-MM-DD."""
    entry = json.loads(download_manifest.read_text())["FLAT_CMPL.zip"]
    return parsedate_to_datetime(entry["last_modified"]).date().isoformat()


def write_database(tables: dict[str, pl.DataFrame], version: str, path: Path) -> None:
    """Write the tables and a dataset_info row to a new database at path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    building = path.with_name(path.name + ".building")
    building.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as tmp, duckdb.connect(building) as con:
        for name, table in tables.items():
            parquet = Path(tmp) / f"{name}.parquet"
            table.write_parquet(parquet)
            con.execute(
                f"CREATE TABLE {name} AS SELECT * FROM read_parquet(?)", [str(parquet)]
            )
        con.execute(
            "CREATE TABLE dataset_info AS SELECT ?::VARCHAR AS data_version,"
            " ?::TIMESTAMP AS built_at",
            [version, datetime.now(UTC).replace(tzinfo=None)],
        )
    building.replace(path)  # readers never see a half-built file


def check_manifest(path: Path = DATABASE, manifest: Path = MANIFEST) -> None:
    """Raise ValueError unless the manifest matches the database.

    Tables, columns, and types must agree exactly; every value of a column
    that lists `values` must be listed; every example query must run.
    """
    spec = yaml.safe_load(manifest.read_text())
    problems = []
    with duckdb.connect(path, read_only=True) as con:
        actual = {}
        for table, column, dtype in con.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns"
            " WHERE table_name != 'dataset_info' ORDER BY table_name, ordinal_position"
        ).fetchall():
            actual.setdefault(table, {})[column] = dtype
        declared = {
            table["name"]: {c["name"]: c["type"] for c in table["columns"]}
            for table in spec["tables"]
        }
        if declared != actual:
            problems.append(f"schema differs: manifest {declared}, database {actual}")
        for table in spec["tables"]:
            for column in table["columns"]:
                if "values" not in column or table["name"] not in actual:
                    continue
                found = con.execute(
                    f"SELECT DISTINCT {column['name']} FROM {table['name']}"
                    f" WHERE {column['name']} IS NOT NULL"
                ).fetchall()
                unlisted = {row[0] for row in found} - set(column["values"])
                if unlisted:
                    problems.append(
                        f"{table['name']}.{column['name']} has unlisted values:"
                        f" {sorted(unlisted)}"
                    )
        for query in [spec["data_version_sql"], *(e["sql"] for e in spec["examples"])]:
            try:
                con.execute(query).fetchall()
            except duckdb.Error as error:
                problems.append(f"query fails: {query!r}: {error}")
    if problems:
        raise ValueError("manifest does not match database:\n" + "\n".join(problems))


def _tidy(text: pl.Expr) -> pl.Expr:
    """Uppercase with single spaces."""
    return text.str.to_uppercase().str.replace_all(r"\s+", " ").str.strip_chars()


def _canonical(column: str, within: str | None = None) -> pl.Expr:
    """The most common spelling among values that differ only in punctuation.

    "MERCEDES BENZ" and "MERCEDES-BENZ" share the key "MERCEDESBENZ"; both
    become whichever is more common (ties go to the alphabetically first).
    With `within`, spellings are only merged inside the same value of that
    column, so models are merged per make.
    """
    key = pl.col(column).str.replace_all(r"[^A-Z0-9]", "")
    group = [key] if within is None else [pl.col(within), key]
    count = pl.len().over([*group, pl.col(column)])
    best = pl.col(column).sort_by([count, pl.col(column)], descending=[True, False])
    # Values with no letters or digits have an empty key; leave them alone.
    return pl.when(key != "").then(best.first().over(group)).otherwise(pl.col(column))
