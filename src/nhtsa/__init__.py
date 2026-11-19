import argparse
from pathlib import Path

from nhtsa.database import DATABASE, build_database
from nhtsa.download import DEFAULT_DATA_DIR, download_complaints


def main() -> None:
    parser = argparse.ArgumentParser(prog="nhtsa")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download NHTSA complaint data")
    download.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    download.add_argument(
        "--force", action="store_true", help="redownload even if files are unchanged"
    )

    build = commands.add_parser(
        "build-db", help="build the DuckDB database from downloaded complaints"
    )
    build.add_argument("--path", type=Path, default=DATABASE)

    args = parser.parse_args()
    if args.command == "download":
        download_complaints(args.data_dir, force=args.force)
    elif args.command == "build-db":
        for table, rows in build_database(args.path).items():
            print(f"{table}: {rows:,} rows")
        print(f"wrote {args.path}")
