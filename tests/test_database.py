import json
from datetime import date

import duckdb
import polars as pl
import pytest

from nhtsa.database import check_manifest, curated_tables, data_version, write_database


def product(make, model, year="2020", mfr="MAKER INC"):
    return {"MFR_NAME": mfr, "MAKETXT": make, "MODELTXT": model, "YEARTXT": year}


def complaints(**overrides):
    """Three complaints shaped like load_complaints() output."""
    columns = {
        "ODINO": ["100", "200", "300"],
        "LDATE": [date(2019, 5, 1), date(2023, 1, 2), date(2023, 6, 3)],
        "FAILDATE": [date(2019, 4, 1), None, date(2023, 6, 1)],
        "CMPL_TYPE": ["IVOQ", "EVOQ", "RC"],
        "PROD_TYPE": ["V", "T", "V"],
        "CRASH": [True, False, False],
        "FIRE": [False, False, True],
        "INJURED": [1, 0, 0],
        "DEATHS": [0, 0, 0],
        "MEDICAL_ATTN": [True, False, False],
        "POLICE_RPT_YN": [True, False, False],
        "MILES": [50_000, 0, None],
        "STATE": ["CA", "TX", "NY"],
        "CDESCR": ["BRAKES FAILED", "Tire blew out", "Engine fire"],
        "COMPONENTS": [
            ["SERVICE BRAKES", "SERVICE BRAKES, HYDRAULIC:PEDALS"],
            ["TIRES"],
            [],
        ],
        "PRODUCTS": [
            [product("Mercedes  Benz", "c class"), product("FORD", "F-150")],
            [product("MERCEDES-BENZ", "C-CLASS", year=None)],
            [product("MERCEDES-BENZ", "C-CLASS"), product("FORD", "F150")],
        ],
    }
    return pl.DataFrame({**columns, **overrides})


def test_complaints_table_decodes_codes_and_keeps_one_row_per_complaint():
    table = curated_tables(complaints())["complaints"]
    assert table["complaint_id"].to_list() == [100, 200, 300]
    assert table["received_year"].to_list() == [2019, 2023, 2023]
    assert table["source"].to_list() == ["website", "hotline", "other"]
    assert table["product_type"].to_list() == ["vehicle", "tires", "vehicle"]
    assert table["mileage"].to_list() == [50_000, None, None]  # 0 means not recorded


def test_vehicles_merge_spellings_that_differ_only_in_punctuation():
    vehicles = curated_tables(complaints())["complaint_vehicles"]
    assert set(vehicles["make"]) == {"MERCEDES-BENZ", "FORD"}
    mercedes = vehicles.filter(pl.col("make") == "MERCEDES-BENZ")
    assert set(mercedes["model"]) == {"C-CLASS"}
    # One spelling each: the tie goes to the alphabetically first.
    assert set(vehicles.filter(pl.col("make") == "FORD")["model"]) == {"F-150"}
    assert vehicles.filter(pl.col("complaint_id") == 200)["model_year"].to_list() == [
        None
    ]


def test_components_have_one_row_per_detail_with_merged_labels():
    components = curated_tables(complaints())["complaint_components"]
    assert components.rows() == [
        (100, "brakes", "SERVICE BRAKES"),
        (100, "brakes", "SERVICE BRAKES, HYDRAULIC:PEDALS"),
        (200, "tires", "TIRES"),
    ]


def test_data_version_is_the_date_nhtsa_modified_the_file(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"FLAT_CMPL.zip": {"last_modified": "Tue, 20 Oct 2026 10:20:26 GMT"}}
        )
    )
    assert data_version(manifest) == "2026-10-20"


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "nhtsa.duckdb"
    write_database(curated_tables(complaints()), "2026-10-20", path)
    return path


def test_written_database_matches_the_manifest_and_is_complete(database):
    check_manifest(database)
    with duckdb.connect(database, read_only=True) as con:
        assert con.execute("SELECT data_version FROM dataset_info").fetchall() == [
            ("2026-10-20",)
        ]
        assert con.execute("SELECT count(*) FROM complaints").fetchone() == (3,)
    assert not database.with_name("nhtsa.duckdb.building").exists()


def test_check_manifest_reports_schema_and_value_differences(tmp_path):
    tables = curated_tables(complaints())
    tables["complaints"] = tables["complaints"].with_columns(
        source=pl.lit("carrier_pigeon"), extra=pl.lit(1)
    )
    path = tmp_path / "nhtsa.duckdb"
    write_database(tables, "2026-10-20", path)
    with pytest.raises(ValueError, match="schema differs") as error:
        check_manifest(path)
    assert "carrier_pigeon" in str(error.value)
