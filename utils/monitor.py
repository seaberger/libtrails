#!/usr/bin/env python3
"""
Extraction run monitor — parses libtrails index --all log output
and displays a compact dashboard.

Usage:
    python3 utils/monitor.py <logfile>
    python3 utils/monitor.py <logfile> --watch   # auto-refresh every 5s
    python3 utils/monitor.py <logfile> -w -n 3   # refresh every 3s
"""

import os
import re
import sys
import time
from pathlib import Path


def parse_log(path: str) -> dict:
    """Parse a libtrails extraction log file into structured stats."""
    raw_lines = Path(path).read_text(errors="replace").splitlines()
    # Join wrapped lines — if a line doesn't start with a known log prefix
    # or indentation, it's a continuation of the previous line
    _PREFIX = re.compile(
        r"^(\(|  |Indexing:|Author:|EPUB:|PDF:|Skipping:|Error |Extracted |Created |"
        r"Resuming:|Using |Pass \d|Book done|Top topics|Extraction |Will |Theme |Chunk |$)"
    )
    lines = []
    for line in raw_lines:
        if lines and not _PREFIX.match(line):
            lines[-1] += " " + line
        else:
            lines.append(line)

    stats = {
        "total_books": 0,
        "books_completed": 0,
        "books_skipped": 0,
        "books_resumed": 0,
        "books_errored": 0,
        "total_chunks": 0,
        "total_topics": 0,
        "avg_rate": 0.0,
        "elapsed_seconds": 0.0,
        "eta_minutes": 0,
        "current_book": None,
        "current_author": None,
        "current_progress": None,
        "current_number": 0,
        "completed_books": [],
        "skipped_oversize": 0,
    }

    # Extract total from "Indexing N books..."
    full_text = "\n".join(lines)
    m = re.search(r"Indexing (\d+) books\.\.\.", full_text)
    if m:
        stats["total_books"] = int(m.group(1))

    # Parse book-level events
    last_book_title = None
    last_book_author = None
    last_book_number = 0
    total_time = 0.0

    for line in lines:
        # Book number: (N/TOTAL)
        m = re.match(r"\((\d+)/(\d+)\)", line)
        if m:
            last_book_number = int(m.group(1))

        # Indexing: Title
        m = re.match(r"Indexing: (.+)", line)
        if m:
            last_book_title = m.group(1).strip()

        # Author: Name
        m = re.match(r"Author: (.+)", line)
        if m:
            last_book_author = m.group(1).strip()

        # Skipping (oversize)
        if re.match(r"Skipping: .+ exceeds --max-words", line):
            stats["books_skipped"] += 1
            stats["skipped_oversize"] += 1

        # Error
        if re.match(r"Error indexing", line):
            stats["books_errored"] += 1
            stats["books_skipped"] += 1

        # Resuming
        m = re.match(r"Resuming: (\d+)/(\d+) chunks already done", line)
        if m:
            stats["books_resumed"] += 1

        # Processing chunks (current progress)
        m = re.search(r"Processing chunks: (\d+)/(\d+) \((\d+)%\)", line)
        if m:
            stats["current_progress"] = {
                "done": int(m.group(1)),
                "total": int(m.group(2)),
                "pct": int(m.group(3)),
            }
            stats["current_book"] = last_book_title
            stats["current_author"] = last_book_author
            stats["current_number"] = last_book_number

        # Extracted N unique topics
        m = re.search(r"Extracted (\d+) unique topics", line)
        if m:
            stats["total_topics"] += int(m.group(1))

        # Book done: 264.2s (269 chunks, 0.98s/chunk) | Total: 2/514 books, 809 chunks, 0.35s/chunk avg
        m = re.search(
            r"Book done: ([\d.]+)s \((\d+) chunks, .+\) \| Total: (\d+)/(\d+) books, ([\d,]+) chunks,\s+([\d.]+)s/chunk avg",
            line,
        )
        if m:
            book_time = float(m.group(1))
            book_chunks = int(m.group(2))
            total_time += book_time
            stats["books_completed"] = int(m.group(3))
            stats["total_chunks"] = int(m.group(5).replace(",", ""))
            stats["avg_rate"] = float(m.group(6))
            stats["completed_books"].append(
                {
                    "title": last_book_title,
                    "time": book_time,
                    "chunks": book_chunks,
                }
            )

        # ETA
        m = re.search(r"ETA: (\d+)m", line)
        if m:
            stats["eta_minutes"] = int(m.group(1))

    stats["elapsed_seconds"] = total_time
    return stats


def format_time(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m {seconds % 60:.0f}s"
    hours = minutes / 60
    return f"{hours:.1f}h"


def format_eta(minutes: int) -> str:
    """Format ETA minutes into human-readable duration."""
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def render_bar(pct: int, width: int = 30) -> str:
    """Render a progress bar."""
    filled = int(width * pct / 100)
    return f"[{'=' * filled}{' ' * (width - filled)}]"


def dashboard(stats: dict) -> str:
    """Render the monitoring dashboard."""
    lines = []
    lines.append("")
    lines.append("  LIBTRAILS EXTRACTION MONITOR")
    lines.append("  " + "=" * 50)

    # Current book
    if stats["current_book"]:
        lines.append("")
        lines.append(f"  Now:    {stats['current_book'][:50]}")
        if stats["current_author"]:
            lines.append(f"          by {stats['current_author'][:45]}")
        if stats["current_progress"]:
            p = stats["current_progress"]
            bar = render_bar(p["pct"])
            lines.append(f"          {bar} {p['done']}/{p['total']} chunks ({p['pct']}%)")

    # Overall progress
    lines.append("")
    processed = stats["books_completed"] + stats["books_skipped"]
    total = stats["total_books"]
    if total > 0:
        overall_pct = processed * 100 // total
        bar = render_bar(overall_pct)
        lines.append(f"  Books:  {bar} {processed}/{total} ({overall_pct}%)")
    lines.append(f"          {stats['books_completed']} extracted, {stats['books_skipped']} skipped, {stats['books_resumed']} resumed")
    if stats["books_errored"] > 0:
        lines.append(f"          {stats['books_errored']} errors")

    # Chunks & topics
    lines.append("")
    lines.append(f"  Chunks: {stats['total_chunks']:,} processed")
    lines.append(f"  Topics: {stats['total_topics']:,} extracted")

    # Timing
    lines.append("")
    lines.append(f"  Rate:    {stats['avg_rate']:.2f}s/chunk avg")
    lines.append(f"  Elapsed: {format_time(stats['elapsed_seconds'])}")
    if stats["eta_minutes"] > 0:
        lines.append(f"  ETA:     {format_eta(stats['eta_minutes'])}")

    # Recent books
    if stats["completed_books"]:
        lines.append("")
        lines.append("  Recent books:")
        for book in stats["completed_books"][-5:]:
            rate = book["time"] / book["chunks"] if book["chunks"] > 0 else 0
            lines.append(f"    {book['title'][:42]:42s}  {book['chunks']:4d} chunks  {book['time']:6.1f}s  ({rate:.2f}s/ch)")

    lines.append("")
    return "\n".join(lines)


def find_latest_log() -> str | None:
    """Find the most recent libtrails extraction log in Claude Code's task output directory."""
    task_dirs = [
        Path("/private/tmp") / f"claude-{os.getuid()}",
        Path("/tmp") / f"claude-{os.getuid()}",
    ]
    for task_dir in task_dirs:
        if not task_dir.exists():
            continue
        # Search all project task directories for .output files
        candidates = sorted(task_dir.rglob("*.output"), key=lambda p: p.stat().st_mtime, reverse=True)
        for candidate in candidates:
            try:
                # Quick check: does it look like a libtrails extraction log?
                head = candidate.read_text(errors="replace")[:500]
                if "libtrails" in head.lower() or "Indexing" in head or "Extraction mode" in head:
                    return str(candidate)
            except OSError:
                continue
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor libtrails extraction run",
        epilog="If no logfile is given, auto-discovers the most recent extraction log.",
    )
    parser.add_argument("logfile", nargs="?", default=None, help="Path to the extraction log file (auto-detected if omitted)")
    parser.add_argument("-w", "--watch", action="store_true", help="Auto-refresh")
    parser.add_argument("-n", "--interval", type=int, default=5, help="Refresh interval in seconds (default: 5)")
    args = parser.parse_args()

    logfile = args.logfile
    if logfile is None:
        logfile = find_latest_log()
        if logfile is None:
            print("No extraction log found. Pass a log file path or start a run first.")
            sys.exit(1)
        print(f"  Auto-detected: {logfile}")
        print()

    if not Path(logfile).exists():
        print(f"Log file not found: {logfile}")
        sys.exit(1)

    if args.watch:
        try:
            while True:
                stats = parse_log(logfile)
                # Clear screen
                print("\033[2J\033[H", end="")
                print(dashboard(stats))
                print(f"  Auto-detected: {logfile}")
                print(f"  Refreshing every {args.interval}s — Ctrl-C to stop")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n  Stopped.")
    else:
        stats = parse_log(logfile)
        print(dashboard(stats))


if __name__ == "__main__":
    main()
