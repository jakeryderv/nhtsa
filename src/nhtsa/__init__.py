import argparse
from pathlib import Path

from nhtsa.download import DEFAULT_DATA_DIR, download_complaints


def main() -> None:
    parser = argparse.ArgumentParser(prog="nhtsa")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download NHTSA complaint data")
    download.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    download.add_argument(
        "--force", action="store_true", help="redownload even if files are unchanged"
    )

    args = parser.parse_args()
    if args.command == "download":
        download_complaints(args.data_dir, force=args.force)
