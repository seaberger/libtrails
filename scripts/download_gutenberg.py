#!/usr/bin/env python3
"""
download_gutenberg.py

Polite bulk downloader for Project Gutenberg EPUBs based on a curated CSV.

Usage:
  python scripts/download_gutenberg.py --csv data/gutenberg_demo_books.csv

Notes:
- Polite rate limiting (>= 2s between *every* HTTP request).
- Tries multiple EPUB URL patterns in order.
- Idempotent by default: if the target file exists and looks like an EPUB, it is skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import httpx
except ImportError as e:
    raise SystemExit(
        "Missing dependency: httpx\n"
        "Install with: uv add httpx  (or: pip install httpx)"
    ) from e


LOG = logging.getLogger("gutenberg_downloader")


EPUB_URL_PATTERNS: tuple[str, ...] = (
    "https://www.gutenberg.org/ebooks/{id}.epub3.images",
    "https://www.gutenberg.org/ebooks/{id}.epub.images",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}-images.epub",
)


@dataclass(frozen=True)
class BookRow:
    gutenberg_id: int
    title: str
    author: str
    category: str
    year_published: str
    rationale: str

    @property
    def author_lastname(self) -> str:
        a = self.author.strip()
        if "," in a:
            last = a.split(",", 1)[0].strip()
        else:
            last = a.split()[-1] if a.split() else a
        return slugify(last)

    @property
    def short_title(self) -> str:
        t = self.title
        for sep in [":", ";", "\u2014", "-", "(", "["]:
            if sep in t:
                t = t.split(sep, 1)[0].strip()
        s = slugify(t)
        return s[:60].strip("_") or f"book_{self.gutenberg_id}"

    @property
    def filename(self) -> str:
        return f"{self.gutenberg_id}_{self.author_lastname}_{self.short_title}.epub"


def slugify(text: str) -> str:
    """ASCII-ish slug safe for filenames (lowercase, underscores, no punctuation)."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def read_books(csv_path: Path) -> list[BookRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"gutenberg_id", "title", "author", "category", "year_published", "rationale"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        out: list[BookRow] = []
        for i, row in enumerate(reader, start=2):
            try:
                gid = int(str(row["gutenberg_id"]).strip())
            except Exception as e:
                raise ValueError(
                    f"Invalid gutenberg_id on line {i}: {row.get('gutenberg_id')!r}"
                ) from e

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


def is_probably_epub(path: Path) -> bool:
    """EPUB is a ZIP container => starts with 'PK'. Lightweight sanity check."""
    try:
        if not path.exists() or path.stat().st_size < 1024:
            return False
        with path.open("rb") as f:
            return f.read(2) == b"PK"
    except Exception:
        return False


class RateLimiter:
    """Enforces >= min_delay seconds between calls to wait()."""

    def __init__(self, min_delay: float) -> None:
        self.min_delay = float(min_delay)
        self._last_ts: Optional[float] = None

    def wait(self) -> None:
        now = time.time()
        if self._last_ts is not None:
            elapsed = now - self._last_ts
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
        self._last_ts = time.time()


def download_one(
    client: httpx.Client,
    limiter: RateLimiter,
    book: BookRow,
    out_dir: Path,
    *,
    force: bool,
    max_retries: int,
    timeout_s: float,
) -> tuple[bool, str]:
    out_path = out_dir / book.filename

    if not force and is_probably_epub(out_path):
        return True, f"SKIP exists: {out_path.name}"

    tmp_path = out_path.with_suffix(".epub.part")

    for url_tmpl in EPUB_URL_PATTERNS:
        url = url_tmpl.format(id=book.gutenberg_id)

        for attempt in range(1, max_retries + 1):
            try:
                limiter.wait()
                with client.stream("GET", url, timeout=timeout_s, follow_redirects=True) as resp:
                    status = resp.status_code

                    if status == 404:
                        LOG.debug("404 for %s (%s) on %s", book.gutenberg_id, book.title, url)
                        break  # try next URL pattern

                    if status >= 500:
                        raise httpx.HTTPStatusError(
                            f"Server error {status}", request=resp.request, response=resp
                        )

                    if status != 200:
                        raise httpx.HTTPStatusError(
                            f"Unexpected status {status}", request=resp.request, response=resp
                        )

                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    with tmp_path.open("wb") as f:
                        for chunk in resp.iter_bytes():
                            if chunk:
                                f.write(chunk)

                if not is_probably_epub(tmp_path):
                    snippet = tmp_path.read_bytes()[:200]
                    tmp_path.unlink(missing_ok=True)
                    raise ValueError(f"Downloaded content is not an EPUB (starts with {snippet!r})")

                tmp_path.replace(out_path)
                return True, f"OK {out_path.name} ({url})"

            except (
                httpx.TimeoutException,
                httpx.TransportError,
                httpx.HTTPStatusError,
                ValueError,
            ) as e:
                if attempt >= max_retries:
                    LOG.warning(
                        "FAIL id=%s title=%r url=%s err=%s",
                        book.gutenberg_id,
                        book.title,
                        url,
                        e,
                    )
                    break

                backoff = (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                LOG.info(
                    "Retry %s/%s id=%s url=%s in %.2fs (%s)",
                    attempt,
                    max_retries,
                    book.gutenberg_id,
                    url,
                    backoff,
                    e,
                )
                time.sleep(backoff)

    return False, f"FAILED id={book.gutenberg_id} title={book.title!r}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Project Gutenberg EPUBs from a curated CSV.")
    ap.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Curated CSV (columns: gutenberg_id,title,author,category,year_published,rationale)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./gutenberg_epubs"),
        help="Output directory for EPUBs (default: ./gutenberg_epubs)",
    )
    ap.add_argument("--force", action="store_true", help="Re-download even if file already exists")
    ap.add_argument(
        "--min-delay",
        type=float,
        default=2.0,
        help="Minimum delay (seconds) between every request (default: 2.0)",
    )
    ap.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds")
    ap.add_argument("--retries", type=int, default=3, help="Max retries per URL pattern")
    ap.add_argument(
        "--user-agent",
        type=str,
        default="LibTrailsDemo/1.0 (+https://github.com/user/libtrails; polite-bot)",
        help="User-Agent header",
    )
    ap.add_argument("--limit", type=int, default=0, help="Only download first N books (0 = all)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    books = read_books(args.csv)
    if args.limit and args.limit > 0:
        books = books[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.out_dir / "download_log.jsonl"
    report_path = args.out_dir / "download_report.csv"

    headers = {
        "User-Agent": args.user_agent,
        "Accept": "application/epub+zip,application/octet-stream;q=0.9,*/*;q=0.1",
    }

    successes: list[BookRow] = []
    failures: list[BookRow] = []

    limiter = RateLimiter(min_delay=args.min_delay)

    with httpx.Client(headers=headers) as client, log_path.open("a", encoding="utf-8") as log_f:
        for idx, book in enumerate(books, start=1):
            ok, msg = download_one(
                client,
                limiter,
                book,
                args.out_dir,
                force=args.force,
                max_retries=args.retries,
                timeout_s=args.timeout,
            )

            record = {
                "gutenberg_id": book.gutenberg_id,
                "title": book.title,
                "author": book.author,
                "category": book.category,
                "ok": ok,
                "message": msg,
                "filename": book.filename,
            }
            log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_f.flush()

            if ok:
                successes.append(book)
                LOG.info("[%s/%s] %s", idx, len(books), msg)
            else:
                failures.append(book)
                LOG.error("[%s/%s] %s", idx, len(books), msg)

    with report_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["status", "gutenberg_id", "title", "author", "category", "filename"])
        for b in successes:
            w.writerow(["OK", b.gutenberg_id, b.title, b.author, b.category, b.filename])
        for b in failures:
            w.writerow(["FAIL", b.gutenberg_id, b.title, b.author, b.category, b.filename])

    LOG.info("Done. OK=%s FAIL=%s", len(successes), len(failures))
    LOG.info("Logs: %s", log_path)
    LOG.info("Report: %s", report_path)

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
