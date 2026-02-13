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
    # Strip optional HH:MM:SS timestamp prefix from all lines first
    stripped_lines = [re.sub(r"^\d{2}:\d{2}:\d{2}\s*", "", line) for line in raw_lines]
    # Join wrapped lines — if a line doesn't start with a known log prefix
    # or indentation, it's a continuation of the previous line
    _PREFIX = re.compile(
        r"^(\(|  |Indexing:|Author:|EPUB:|PDF:|Skipping:|Error |Extracted |Created |"
        r"Resuming:|Using |Pass \d|Book done|Book themes|Top topics|Extraction |Will |Theme |Chunk |Processing |$)"
    )
    lines = []
    for line in stripped_lines:
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
        "current_book_chunks": 0,
        "current_book_topics": 0,
        "avg_rate": 0.0,
        "elapsed_seconds": 0.0,
        "eta_minutes": 0,
        "current_book": None,
        "current_author": None,
        "current_progress": None,
        "current_resume": None,
        "current_number": 0,
        "completed_books": [],
        "skipped_oversize": 0,
        "start_time": None,
        "wall_elapsed_seconds": 0.0,
    }

    # Extract wall-clock start time from first timestamp in raw log
    from datetime import datetime

    for raw_line in raw_lines:
        m = re.match(r"^(\d{2}:\d{2}:\d{2})", raw_line)
        if m:
            stats["start_time"] = m.group(1)
            break

    # Compute wall-clock elapsed from first to last timestamp
    if stats["start_time"]:
        last_ts = None
        for raw_line in reversed(raw_lines):
            m = re.match(r"^(\d{2}:\d{2}:\d{2})", raw_line)
            if m:
                last_ts = m.group(1)
                break
        if last_ts:
            try:
                t0 = datetime.strptime(stats["start_time"], "%H:%M:%S")
                t1 = datetime.strptime(last_ts, "%H:%M:%S")
                delta = (t1 - t0).total_seconds()
                if delta < 0:
                    delta += 86400  # crossed midnight
                stats["wall_elapsed_seconds"] = delta
            except ValueError:
                pass

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

        # Indexing: Title (new book resets resume context and current counters)
        m = re.match(r"Indexing: (.+)", line)
        if m:
            last_book_title = m.group(1).strip()
            stats["current_book"] = last_book_title
            stats["current_number"] = last_book_number
            stats["current_progress"] = None
            stats["current_resume"] = None
            stats["current_book_chunks"] = 0
            stats["current_book_topics"] = 0

        # Author: Name
        m = re.match(r"Author: (.+)", line)
        if m:
            last_book_author = m.group(1).strip()
            stats["current_author"] = last_book_author

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
            stats["current_resume"] = {
                "done": int(m.group(1)),
                "total": int(m.group(2)),
            }

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

        # Created N chunks (track for current in-progress book)
        m = re.search(r"Created (\d+) chunks", line)
        if m:
            stats["current_book_chunks"] = int(m.group(1))

        # Extracted N unique topics (track for current book; rolled into total on Book done)
        m = re.search(r"Extracted (\d+) unique topics", line)
        if m:
            stats["current_book_topics"] = int(m.group(1))

        # Book done with cumulative totals (--all mode):
        # Book done: 264.2s (269 chunks, 0.98s/chunk) | Total: 2/514 books, 809 chunks, 0.35s/chunk avg
        m = re.search(
            r"Book done: ([\d.]+)s \((\d+) chunks, .+\) \| Total: (\d+)/(\d+) books,\s+([\d,]+)\s+chunks,\s+([\d.]+)s/chunk avg",
            line,
        )
        if m:
            book_time = float(m.group(1))
            book_chunks = int(m.group(2))
            total_time += book_time
            stats["books_completed"] = int(m.group(3))
            stats["total_chunks"] = int(m.group(5).replace(",", ""))
            stats["avg_rate"] = float(m.group(6))
            stats["total_topics"] += stats["current_book_topics"]
            stats["current_book_chunks"] = 0
            stats["current_book_topics"] = 0
            stats["completed_books"].append(
                {
                    "title": last_book_title or "Unknown",
                    "time": book_time,
                    "chunks": book_chunks,
                }
            )
            continue

        # Book done without cumulative totals (single-book / --id mode):
        # Book done: 193.5s (35 chunks, 5.53s/chunk)
        m = re.search(r"Book done: ([\d.]+)s \((\d+) chunks, ([\d.]+)s/chunk\)", line)
        if m:
            book_time = float(m.group(1))
            book_chunks = int(m.group(2))
            total_time += book_time
            stats["books_completed"] += 1
            stats["total_chunks"] += book_chunks
            stats["total_topics"] += stats["current_book_topics"]
            stats["current_book_chunks"] = 0
            stats["current_book_topics"] = 0
            stats["avg_rate"] = float(m.group(3))
            stats["completed_books"].append(
                {
                    "title": last_book_title or "Unknown",
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
            r = stats["current_resume"]
            if r:
                # Show both resume context and pending progress
                total_done = r["done"] + p["done"]
                overall_pct = total_done * 100 // r["total"]
                bar = render_bar(overall_pct)
                lines.append(f"          {bar} {total_done}/{r['total']} chunks ({overall_pct}%)")
                lines.append(f"          ({r['done']} previously done, {p['done']}/{p['total']} remaining)")
            else:
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

    # Chunks & topics (include in-progress book)
    lines.append("")
    cur_chunks = stats["current_book_chunks"]
    cur_topics = stats["current_book_topics"]
    display_chunks = stats["total_chunks"] + cur_chunks
    display_topics = stats["total_topics"] + cur_topics
    chunk_suffix = f" (+{cur_chunks} current)" if cur_chunks and stats["total_chunks"] else ""
    topic_suffix = f" (+{cur_topics} current)" if cur_topics and stats["total_topics"] else ""
    lines.append(f"  Chunks: {display_chunks:,} processed{chunk_suffix}")
    lines.append(f"  Topics: {display_topics:,} extracted{topic_suffix}")

    # Timing
    lines.append("")
    lines.append(f"  Rate:    {stats['avg_rate']:.2f}s/chunk avg")
    if stats["start_time"]:
        lines.append(f"  Started: {stats['start_time']}")
        lines.append(f"  Elapsed: {format_time(stats['wall_elapsed_seconds'])} (wall clock)")
    else:
        lines.append(f"  Elapsed: {format_time(stats['elapsed_seconds'])} (extraction time)")
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


def query_db_stats(db_path: str) -> dict:
    """Query extraction progress directly from the database (real-time, no log parsing)."""
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)

    total_books = conn.execute("SELECT COUNT(*) FROM books WHERE calibre_id IS NOT NULL").fetchone()[0]
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    chunks_with_topics = conn.execute("SELECT COUNT(DISTINCT chunk_id) FROM chunk_topics").fetchone()[0]
    total_topics = conn.execute("SELECT COUNT(*) FROM chunk_topics").fetchone()[0]

    # Books fully done (all chunks have topics)
    books_done = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT c.book_id, COUNT(DISTINCT c.id) as total, COUNT(DISTINCT ct.chunk_id) as done
            FROM chunks c LEFT JOIN chunk_topics ct ON c.id = ct.chunk_id
            GROUP BY c.book_id
        ) WHERE total = done AND total > 0
    """).fetchone()[0]

    # Books with chunks (started)
    books_started = conn.execute("SELECT COUNT(DISTINCT book_id) FROM chunks").fetchone()[0]

    # Current in-progress book (has chunks but not all have topics)
    # Order by book_id DESC to get the most recently started book
    cur = conn.execute("""
        SELECT b.title, b.author, COUNT(DISTINCT c.id) as total, COUNT(DISTINCT ct.chunk_id) as done
        FROM chunks c
        JOIN books b ON c.book_id = b.id
        LEFT JOIN chunk_topics ct ON c.id = ct.chunk_id
        GROUP BY c.book_id
        HAVING done < total AND done > 0
        ORDER BY c.book_id DESC
        LIMIT 1
    """).fetchone()

    # Recent completed books (last 5)
    completed = conn.execute("""
        SELECT b.title, COUNT(DISTINCT c.id) as chunks
        FROM chunks c
        JOIN books b ON c.book_id = b.id
        LEFT JOIN chunk_topics ct ON c.id = ct.chunk_id
        GROUP BY c.book_id
        HAVING COUNT(DISTINCT ct.chunk_id) = COUNT(DISTINCT c.id) AND COUNT(DISTINCT c.id) > 0
        ORDER BY c.book_id DESC
        LIMIT 5
    """).fetchall()

    # Estimate total chunks (including unchunked books)
    books_not_chunked = conn.execute("""
        SELECT COUNT(*) FROM books
        WHERE calibre_id IS NOT NULL
        AND id NOT IN (SELECT DISTINCT book_id FROM chunks)
    """).fetchone()[0]
    avg_chunks_per_book = conn.execute(
        "SELECT AVG(cnt) FROM (SELECT COUNT(*) as cnt FROM chunks GROUP BY book_id)"
    ).fetchone()[0] or 0

    conn.close()

    estimated_total = total_chunks + int(books_not_chunked * avg_chunks_per_book)

    stats = {
        "total_books": total_books,
        "books_completed": books_done,
        "books_skipped": 0,
        "books_resumed": 0,
        "books_errored": 0,
        "total_chunks": chunks_with_topics,
        "total_topics": total_topics,
        "estimated_total_chunks": estimated_total,
        "current_book_chunks": 0,
        "current_book_topics": 0,
        "avg_rate": 0.0,
        "elapsed_seconds": 0.0,
        "eta_minutes": 0,
        "current_book": None,
        "current_author": None,
        "current_progress": None,
        "current_resume": None,
        "current_number": books_done + 1,
        "completed_books": [{"title": r[0], "time": 0, "chunks": r[1]} for r in completed],
        "skipped_oversize": 0,
        "start_time": None,
        "wall_elapsed_seconds": 0.0,
    }

    if cur:
        stats["current_book"] = cur[0]
        stats["current_author"] = cur[1]
        total_ch = cur[2]
        done_ch = cur[3]
        pct = done_ch * 100 // total_ch if total_ch else 0
        stats["current_progress"] = {"done": done_ch, "total": total_ch, "pct": pct}

    return stats


DB_ALIASES = {
    "demo": "data/demo_library.db",
    "v2": "data/ipad_library_v2.db",
    "v1": "data/ipad_library.db",
    "default": "data/ipad_library.db",
}


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
        epilog=(
            "If no logfile is given, auto-discovers the most recent extraction log.\n"
            "Use --db to query the database directly for real-time progress.\n"
            "DB aliases: demo, v2, v1 (or pass a path to any .db file)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("logfile", nargs="?", default=None, help="Path to the extraction log file (auto-detected if omitted)")
    parser.add_argument("-w", "--watch", action="store_true", help="Auto-refresh")
    parser.add_argument("-n", "--interval", type=int, default=5, help="Refresh interval in seconds (default: 5)")
    parser.add_argument("--db", default=None, help="Query DB directly instead of parsing logs (alias: demo, v2, v1; or path)")
    args = parser.parse_args()

    # DB mode: query database directly for real-time stats
    if args.db is not None:
        db_path = DB_ALIASES.get(args.db, args.db)
        if not Path(db_path).exists():
            print(f"Database not found: {db_path}")
            sys.exit(1)

        if args.watch:
            prev_chunks = None
            prev_time = None
            rate_history = []  # rolling window of (chunks/sec) samples
            try:
                while True:
                    try:
                        stats = query_db_stats(db_path)
                    except Exception:
                        time.sleep(args.interval)
                        continue

                    now = time.time()
                    cur_chunks = stats["total_chunks"]

                    # Compute rolling rate from recent samples
                    if prev_chunks is not None and prev_time is not None:
                        dt = now - prev_time
                        if dt > 0 and cur_chunks > prev_chunks:
                            sample_rate = (cur_chunks - prev_chunks) / dt
                            rate_history.append(sample_rate)
                            if len(rate_history) > 12:  # ~60s of history at 5s intervals
                                rate_history.pop(0)

                    prev_chunks = cur_chunks
                    prev_time = now

                    if rate_history:
                        avg_rate = sum(rate_history) / len(rate_history)
                        stats["avg_rate"] = 1.0 / avg_rate if avg_rate > 0 else 0
                        remaining = stats.get("estimated_total_chunks", cur_chunks) - cur_chunks
                        if avg_rate > 0 and remaining > 0:
                            stats["eta_minutes"] = int(remaining / avg_rate / 60)

                    print("\033[2J\033[H", end="")
                    print(dashboard(stats))
                    print(f"  DB: {db_path}")
                    est = stats.get("estimated_total_chunks", 0)
                    if est > cur_chunks:
                        print(f"  Est. total chunks: ~{est:,} ({est - cur_chunks:,} remaining)")
                    print(f"  Refreshing every {args.interval}s — Ctrl-C to stop")
                    sys.stdout.flush()
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n  Stopped.")
        else:
            stats = query_db_stats(db_path)
            print(dashboard(stats))
        return

    # Log mode: parse extraction log
    logfile = args.logfile
    if logfile is None:
        logfile = find_latest_log()
        if logfile is None:
            print("No extraction log found. Pass a log file path, use --db, or start a run first.")
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
                print(f"  Log: {logfile}")
                print(f"  Refreshing every {args.interval}s — Ctrl-C to stop")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n  Stopped.")
    else:
        stats = parse_log(logfile)
        print(dashboard(stats))


if __name__ == "__main__":
    main()
