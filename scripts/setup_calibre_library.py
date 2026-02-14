#!/usr/bin/env python3
"""
setup_calibre_library.py

Creates a fresh Calibre library and imports the downloaded Gutenberg EPUBs,
setting metadata/tags from the curated CSV.

Usage:
  python scripts/setup_calibre_library.py \
    --csv data/gutenberg_demo_books.csv \
    --epub-dir gutenberg_epubs \
    --library-dir ~/Calibre_Demo_Library

Requirements:
- Calibre installed locally so `calibredb` is available on PATH.
  https://calibre-ebook.com/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


LOG = logging.getLogger("calibre_setup")


@dataclass(frozen=True)
class BookRow:
    gutenberg_id: int
    title: str
    author: str
    category: str
    year_published: str
    rationale: str


def read_books(csv_path: Path) -> list[BookRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"gutenberg_id", "title", "author", "category", "year_published", "rationale"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        out: list[BookRow] = []
        for i, row in enumerate(reader, start=2):
            gid = int(str(row["gutenberg_id"]).strip())
            out.append(
                BookRow(
                    gutenberg_id=gid,
                    title=str(row["title"] or "").strip(),
                    author=str(row["author"] or "").strip(),
                    category=str(row["category"] or "").strip(),
                    year_published=str(row["year_published"] or "").strip(),
                    rationale=str(row["rationale"] or "").strip(),
                )
            )
        return out


def run_cmd(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    LOG.debug("RUN: %s", " ".join(cmd))
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


CALIBREDB_MACOS = "/Applications/calibre.app/Contents/MacOS/calibredb"


def find_calibredb() -> str:
    """Find calibredb binary, checking PATH and macOS app bundle."""
    path = shutil.which("calibredb")
    if path:
        return path
    if Path(CALIBREDB_MACOS).exists():
        return CALIBREDB_MACOS
    raise SystemExit(
        "Could not find `calibredb` on PATH or in /Applications/calibre.app.\n"
        "Install Calibre from https://calibre-ebook.com/ and ensure calibredb is available.\n"
    )


def ensure_calibre_available() -> None:
    find_calibredb()


def ensure_library(library_dir: Path, calibredb: str) -> None:
    library_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [calibredb, "--with-library", str(library_dir), "list", "--limit", "0"],
        check=True,
    )


def find_epub(epub_dir: Path, gutenberg_id: int) -> Optional[Path]:
    matches = sorted(epub_dir.glob(f"{gutenberg_id}_*.epub"))
    if matches:
        return matches[0]
    matches = sorted(epub_dir.glob(f"{gutenberg_id}*.epub"))
    return matches[0] if matches else None


def tags_for(book: BookRow) -> list[str]:
    tags = ["LibTrails Demo", book.category]

    fiction_cats = {
        "Victorian British fiction",
        "American classics",
        "Russian literature",
        "Early sci-fi & gothic",
        "World classics",
        "Drama",
    }
    if book.category in fiction_cats:
        tags.append("Fiction")
    if book.category == "Drama":
        tags.append("Plays")
    if book.category == "Philosophy & essays":
        tags.append("Philosophy")
        tags.append("Essays")
        tags.append("Nonfiction")
    if book.category == "Essays & nonfiction":
        tags.append("Nonfiction")

    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        t = t.strip()
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def parse_added_book_id(output: str) -> Optional[int]:
    """Extract the Calibre book ID from calibredb add output."""
    nums = re.findall(r"\b(\d+)\b", output)
    return int(nums[-1]) if nums else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create a Calibre library from downloaded Gutenberg EPUBs."
    )
    ap.add_argument("--csv", type=Path, required=True, help="Curated CSV used for metadata.")
    ap.add_argument(
        "--epub-dir",
        type=Path,
        default=Path("./gutenberg_epubs"),
        help="Directory containing downloaded EPUBs.",
    )
    ap.add_argument(
        "--library-dir",
        type=Path,
        default=Path.home() / "Calibre_Demo_Library",
        help="Target Calibre library directory (default: ~/Calibre_Demo_Library).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    calibredb = find_calibredb()
    LOG.info("Using calibredb: %s", calibredb)
    ensure_library(args.library_dir, calibredb)

    books = read_books(args.csv)

    ok = 0
    missing_files = 0
    failed = 0

    for idx, book in enumerate(books, start=1):
        epub_path = find_epub(args.epub_dir, book.gutenberg_id)
        if epub_path is None:
            missing_files += 1
            LOG.error(
                "[%s/%s] Missing EPUB for gutenberg_id=%s (expected in %s)",
                idx,
                len(books),
                book.gutenberg_id,
                args.epub_dir,
            )
            continue

        tags = ",".join(tags_for(book))

        add_cmd = [
            calibredb,
            "--with-library",
            str(args.library_dir),
            "add",
            "--automerge",
            "ignore",
            "--title",
            book.title,
            "--authors",
            book.author,
            "--tags",
            tags,
            "--identifier",
            f"gutenberg:{book.gutenberg_id}",
            str(epub_path),
        ]

        if args.dry_run:
            LOG.info("DRY-RUN: %s", " ".join(add_cmd))
            ok += 1
            continue

        try:
            proc = run_cmd(add_cmd, check=True)
            book_id = parse_added_book_id(proc.stdout + "\n" + proc.stderr)

            if book_id is not None and book.rationale:
                set_cmd = [
                    calibredb,
                    "--with-library",
                    str(args.library_dir),
                    "set_metadata",
                    str(book_id),
                    "--field",
                    f"comments:{book.rationale}",
                ]
                run_cmd(set_cmd, check=False)

            ok += 1
            LOG.info(
                "[%s/%s] Imported gutenberg_id=%s as calibre_id=%s",
                idx,
                len(books),
                book.gutenberg_id,
                book_id,
            )

        except subprocess.CalledProcessError as e:
            failed += 1
            LOG.error(
                "[%s/%s] FAIL gutenberg_id=%s\nSTDOUT:\n%s\nSTDERR:\n%s",
                idx,
                len(books),
                book.gutenberg_id,
                e.stdout,
                e.stderr,
            )

    LOG.info("Import complete. OK=%s missing_epub=%s failed=%s", ok, missing_files, failed)

    try:
        proc = run_cmd(
            [
                calibredb,
                "--with-library",
                str(args.library_dir),
                "list",
                "--fields",
                "id,title,authors,tags,identifiers",
                "--for-machine",
            ],
            check=True,
        )
        data = json.loads(proc.stdout)
        LOG.info("Library now contains %s books.", len(data))
        for row in data[:5]:
            LOG.info(
                "Sample: id=%s title=%r authors=%r tags=%r",
                row.get("id"),
                row.get("title"),
                row.get("authors"),
                row.get("tags"),
            )
    except Exception as e:
        LOG.warning("Could not read library summary via calibredb list --for-machine (%s)", e)

    return 0 if failed == 0 and missing_files == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
