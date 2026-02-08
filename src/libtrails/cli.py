"""Command-line interface for libtrails."""

import re
import subprocess
import time
from collections import Counter

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from . import __version__
from .chunker import chunk_text
from .config import BATCH_SIZE, CHUNK_MODEL, CHUNK_TARGET_WORDS, DEFAULT_MODEL, THEME_MODEL
from .database import (
    get_all_books,
    get_book,
    get_book_by_title,
    get_book_path,
    get_calibre_book_metadata,
    get_indexing_status,
    get_topics_without_embeddings,
    init_chunks_table,
    migrate_raw_topics_to_normalized,
    save_book_themes,
    save_chunk_topics,
    save_chunks,
    save_topic_embedding,
)
from .document_parser import extract_text
from .stats import refresh_all_stats
from .topic_extractor import (
    check_ollama_available,
    extract_book_themes,
    extract_topics_batch,
    extract_topics_batched,
    get_available_models,
)

console = Console()


def _refresh_and_report_stats() -> None:
    """Refresh materialized stats and print summary."""
    console.print("\n[bold cyan]Refreshing materialized stats...[/bold cyan]")
    stats_result = refresh_all_stats()
    console.print(f"  [green]Stats refreshed in {stats_result['elapsed_seconds']:.1f}s[/green]")


@click.group()
@click.version_option(version=__version__)
def main():
    """libtrails - Trail-finding across your book library."""
    pass


@main.command()
def status():
    """Show current indexing status."""
    stats = get_indexing_status()

    table = Table(title="Library Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Books (with Calibre match)", str(stats["total_books"]))
    table.add_row("Indexed Books", str(stats["indexed_books"]))
    table.add_row("Total Chunks", str(stats["total_chunks"]))
    table.add_row("Unique Topics (raw)", str(stats["unique_topics"]))
    table.add_row("─" * 20, "─" * 10)
    table.add_row("Normalized Topics", str(stats["normalized_topics"]))
    table.add_row("Topics with Embeddings", str(stats["topics_with_embeddings"]))
    table.add_row("Clustered Topics", str(stats["clustered_topics"]))

    if stats["total_books"] > 0:
        pct = (stats["indexed_books"] / stats["total_books"]) * 100
        table.add_row("Progress", f"{pct:.1f}%")

    console.print(table)

    # Show embedding model info
    try:
        from .embeddings import get_model_info

        model_info = get_model_info()
        console.print(
            f"\n[dim]Embedding model: {model_info['name']} ({model_info['dimension']} dims)[/dim]"
        )
        if model_info["cached_locally"]:
            console.print("[dim]Model cached locally ✓[/dim]")
    except Exception:
        pass

    # Check Ollama
    models = get_available_models()
    if models:
        console.print(f"\n[green]Ollama models available:[/green] {', '.join(models)}")
    else:
        console.print("\n[yellow]Warning: Ollama not available or no models loaded[/yellow]")


@main.command()
@click.option(
    "--ipad", "-i", default=None, help="iPad MapleRead server URL (e.g., http://192.168.1.124:8082)"
)
@click.option("--dry-run", is_flag=True, help="Show what would be added without making changes")
@click.option("--skip-index", is_flag=True, help="Add books to database but skip indexing")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Ollama model for indexing new books")
@click.option("--save-url", is_flag=True, help="Save the iPad URL to config for future use")
def sync(ipad: str, dry_run: bool, skip_index: bool, model: str, save_url: bool):
    """Sync new books from iPad MapleRead library."""
    from .config import get_ipad_url, set_ipad_url
    from .sync import sync_ipad_library

    # Get iPad URL from argument or config
    if not ipad:
        ipad = get_ipad_url()
        if not ipad:
            console.print("[red]Error:[/red] No iPad URL provided.")
            console.print("Use: [cyan]libtrails sync --ipad http://192.168.1.124:8082[/cyan]")
            console.print("Or save a default: [cyan]libtrails sync --ipad <url> --save-url[/cyan]")
            return

    # Save URL if requested
    if save_url:
        set_ipad_url(ipad)
        console.print("[green]Saved iPad URL to ~/.libtrails/config.yaml[/green]")

    console.print(f"\n[bold]Syncing from iPad[/bold]: {ipad}")

    if dry_run:
        console.print("[yellow]Dry run mode - no changes will be made[/yellow]\n")

    def progress_callback(message: str):
        console.print(f"  {message}")

    try:
        result = sync_ipad_library(
            ipad_url=ipad,
            dry_run=dry_run,
            skip_index=skip_index,
            model=model,
            progress_callback=progress_callback,
        )
    except ConnectionError as e:
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("[dim]Make sure MapleRead server is running on iPad[/dim]")
        return

    # Show results
    console.print("\n[bold]Sync Summary[/bold]")
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Books on iPad", str(result["total_on_ipad"]))
    table.add_row("New books found", str(result["new_books"]))
    table.add_row("Matched to Calibre", str(result["matched_to_calibre"]))
    table.add_row("Added to database", str(result["added_to_db"]))

    if not skip_index and not dry_run:
        table.add_row("Indexed", str(result.get("indexed", 0)))
        if result.get("index_failed", 0) > 0:
            table.add_row("Index failed", str(result["index_failed"]))

    console.print(table)

    if dry_run and result.get("new_book_titles"):
        console.print("\n[bold]New books that would be added:[/bold]")
        for title in result["new_book_titles"][:20]:
            console.print(f"  • {title}")
        if len(result["new_book_titles"]) > 20:
            console.print(f"  ... and {len(result['new_book_titles']) - 20} more")

    # Summary message for indexing
    if not dry_run and result.get("indexed", 0) > 0:
        console.print(f"\n[green]Successfully indexed {result['indexed']} new books![/green]")

    # If there are unindexed books (skip_index was used or indexing failed)
    if not dry_run and skip_index and result.get("books_to_index"):
        console.print(f"\n[bold]{len(result['books_to_index'])} new books ready to index[/bold]")
        console.print(f"Run: [cyan]libtrails index --all --model {model}[/cyan]")


@main.command()
@click.argument("book_id", type=int, required=False)
@click.option("--title", "-t", help="Search by title")
@click.option("--all", "index_all", is_flag=True, help="Index all books")
@click.option("--model", "-m", default=None, help="Ollama model (sets both theme and chunk model)")
@click.option("--theme-model", default=THEME_MODEL, help=f"Model for book themes (default: {THEME_MODEL})")
@click.option("--chunk-model", default=CHUNK_MODEL, help=f"Model for chunk topics (default: {CHUNK_MODEL})")
@click.option("--batch-size", type=int, default=BATCH_SIZE, help=f"Chunks per batch (default: {BATCH_SIZE})")
@click.option("--legacy", is_flag=True, help="Use legacy single-call extraction (no themes, no batching)")
@click.option("--dry-run", is_flag=True, help="Parse and chunk without topic extraction")
@click.option("--reindex", is_flag=True, help="Re-index books that are already indexed")
@click.option(
    "--max-words", type=int, default=None, help="Skip books with more than N words (e.g., 500000)"
)
@click.option(
    "--chunk-size-words",
    "chunk_size",
    type=int,
    default=None,
    help="Target words per chunk (default: 500, use 2000-3000 for large books)",
)
@click.option(
    "--min-battery", type=int, default=15, help="Pause if battery drops below this % (default: 15)"
)
def index(
    book_id: int,
    title: str,
    index_all: bool,
    model: str,
    theme_model: str,
    chunk_model: str,
    batch_size: int,
    legacy: bool,
    dry_run: bool,
    reindex: bool,
    max_words: int,
    chunk_size: int,
    min_battery: int,
):
    """Index a book (parse, chunk, extract topics).

    Two-pass extraction (default):
      Pass 1: gemma3:27b extracts 5-8 book-level themes (1 call/book)
      Pass 2: gemma3:4b extracts chunk topics in batches, contextualized with themes

    Use --legacy for old single-call extraction per chunk.
    """
    init_chunks_table()

    # If --model is set, use it for both theme and chunk model
    if model:
        theme_model = model
        chunk_model = model

    if index_all:
        _index_all_books(
            theme_model, chunk_model, batch_size, legacy,
            dry_run, reindex, max_words, chunk_size, min_battery,
        )
        return

    # Find the book
    book = None
    if book_id:
        book = get_book(book_id)
    elif title:
        book = get_book_by_title(title)

    if not book:
        console.print("[red]Book not found[/red]")
        return

    _index_single_book(
        book, theme_model, chunk_model, batch_size, legacy,
        dry_run, chunk_size=chunk_size,
    )


def _index_single_book(
    book: dict,
    theme_model: str,
    chunk_model: str,
    batch_size: int,
    legacy: bool,
    dry_run: bool,
    max_words: int = None,
    chunk_size: int = None,
):
    """Index a single book. Raises exception on failure. Returns 'skipped' if over max_words."""
    console.print(f"[bold]Indexing:[/bold] {book['title'][:60]}")
    console.print(f"[dim]Author: {book['author'] or 'Unknown'}[/dim]")

    # Get book file path (EPUB or PDF)
    if not book.get("calibre_id"):
        raise ValueError("No Calibre match for this book")

    book_path = get_book_path(book["calibre_id"])
    if not book_path:
        raise FileNotFoundError("No EPUB or PDF found in Calibre library")

    file_format = book_path.suffix.upper().lstrip(".")
    console.print(f"[dim]{file_format}: {book_path.name}[/dim]")

    # Extract text
    with console.status(f"Extracting text from {file_format}..."):
        text = extract_text(book_path)

    word_count = len(text.split())

    # Check minimum content
    if word_count < 100:
        raise ValueError(f"Insufficient content: only {word_count} words extracted")

    # Check maximum content (for skipping huge collected works)
    if max_words and word_count > max_words:
        console.print(
            f"[yellow]Skipping: {word_count:,} words exceeds --max-words {max_words:,}[/yellow]"
        )
        return "skipped"

    console.print(f"[green]Extracted {word_count:,} words[/green]")

    # Chunk
    target_words = chunk_size if chunk_size else CHUNK_TARGET_WORDS
    chunks = chunk_text(text, target_words)
    console.print(
        f"[green]Created {len(chunks)} chunks[/green] [dim](~{target_words} words/chunk)[/dim]"
    )

    # Save chunks
    save_chunks(book["id"], chunks)

    if dry_run:
        console.print("[yellow]Dry run - skipping topic extraction[/yellow]")
        # Show sample chunk
        if chunks:
            console.print(f"\n[bold]Sample chunk ({len(chunks[0].split())} words):[/bold]")
            console.print(chunks[0][:500] + "...")
        return

    # Check Ollama availability
    if not check_ollama_available(chunk_model):
        console.print(f"[red]Model {chunk_model} not available in Ollama[/red]")
        return

    from .database import get_db

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM chunks WHERE book_id = ? ORDER BY chunk_index", (book["id"],)
        )
        chunk_ids = [row[0] for row in cursor.fetchall()]

    book_themes = []

    if legacy:
        # Legacy mode: single-call extraction, no context
        console.print(f"\n[bold]Extracting topics with {chunk_model} (legacy mode)...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing chunks", total=len(chunks))

            def update_progress(completed: int, total: int):
                progress.update(task, completed=completed)

            topics_per_chunk = extract_topics_batch(
                chunks, chunk_model, progress_callback=update_progress
            )
    else:
        # Two-pass extraction
        # Pass 1: Book-level themes
        console.print(f"\n[bold]Pass 1: Extracting book themes with {theme_model}...[/bold]")

        calibre_meta = get_calibre_book_metadata(book["calibre_id"])
        sample_text = text[:5000]

        with console.status("Extracting book themes..."):
            book_themes = extract_book_themes(
                title=book["title"],
                author=book.get("author", "Unknown"),
                tags=calibre_meta.get("tags"),
                description=calibre_meta.get("description"),
                sample_text=sample_text,
                model=theme_model,
            )

        if book_themes:
            save_book_themes(book["id"], book_themes)
            console.print(f"[green]Book themes:[/green] {', '.join(book_themes)}")
        else:
            console.print("[yellow]No book themes extracted (continuing without context)[/yellow]")

        # Pass 2: Batched chunk extraction with context
        console.print(
            f"\n[bold]Pass 2: Extracting chunk topics with {chunk_model} "
            f"(batches of {batch_size})...[/bold]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing chunks", total=len(chunks))

            def update_progress(completed: int, total: int):
                progress.update(task, completed=completed)

            topics_per_chunk = extract_topics_batched(
                chunks,
                book_title=book["title"],
                author=book.get("author", "Unknown"),
                book_themes=book_themes,
                model=chunk_model,
                batch_size=batch_size,
                progress_callback=update_progress,
            )

    # Save topics
    all_topics = []
    for chunk_id, topics in zip(chunk_ids, topics_per_chunk):
        if topics:
            save_chunk_topics(chunk_id, topics)
            all_topics.extend(topics)

    unique_topics = set(all_topics)
    console.print(f"\n[green]Extracted {len(unique_topics)} unique topics[/green]")

    # Show top topics
    if unique_topics:
        topic_counts = Counter(all_topics)
        console.print("\n[bold]Top topics:[/bold]")
        for topic, count in topic_counts.most_common(10):
            console.print(f"  {topic} ({count})")


def _get_battery_level() -> int | None:
    """Get current battery percentage on macOS. Returns None if not on battery/not macOS."""
    try:
        result = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5)
        # Parse output like: "-InternalBattery-0 (id=...)	51%; charging;"
        match = re.search(r"(\d+)%", result.stdout)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None


def _index_all_books(
    theme_model: str,
    chunk_model: str,
    batch_size: int,
    legacy: bool,
    dry_run: bool,
    reindex: bool = False,
    max_words: int = None,
    chunk_size: int = None,
    min_battery: int = 15,
):
    """Index all books with Calibre matches, with resume support."""
    from .database import get_book_path, get_db

    books = get_all_books(with_calibre_match=True)

    mode_desc = "legacy" if legacy else f"two-pass ({theme_model} + {chunk_model})"
    console.print(f"[dim]Extraction mode: {mode_desc}[/dim]")
    if max_words:
        console.print(f"[dim]Skipping books with more than {max_words:,} words[/dim]")
    if chunk_size:
        console.print(f"[dim]Using {chunk_size:,} words per chunk[/dim]")
    console.print(f"[dim]Will pause if battery drops below {min_battery}%[/dim]")

    # Get already indexed book IDs
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT book_id FROM chunks")
        indexed_ids = {row[0] for row in cursor.fetchall()}

    # Filter books to process
    if reindex:
        to_process = books
        console.print(f"[bold]Re-indexing all {len(books)} books...[/bold]")
    else:
        to_process = [b for b in books if b["id"] not in indexed_ids]
        skipped = len(books) - len(to_process)
        if skipped > 0:
            console.print(f"[dim]Skipping {skipped} already indexed books[/dim]")
        console.print(f"[bold]Indexing {len(to_process)} books...[/bold]")

    if not to_process:
        console.print("[green]All books already indexed![/green]")
        return

    # Filter out books without available files
    processable = []
    no_file = []
    for book in to_process:
        if book.get("calibre_id") and get_book_path(book["calibre_id"]):
            processable.append(book)
        else:
            no_file.append(book)

    if no_file:
        console.print(f"[yellow]Skipping {len(no_file)} books without EPUB/PDF[/yellow]")

    # Process books
    successful = 0
    skipped_large = 0
    failed = []
    start_time = time.time()

    for i, book in enumerate(processable, 1):
        elapsed = time.time() - start_time
        if successful > 0:
            avg_time = elapsed / successful
            remaining = avg_time * (len(processable) - i + 1)
            eta = f"ETA: {int(remaining // 60)}m {int(remaining % 60)}s"
        else:
            eta = ""

        console.print(f"\n[dim]({i}/{len(processable)}) {eta}[/dim]")

        # Check battery level
        battery = _get_battery_level()
        if battery is not None and battery < min_battery:
            console.print(
                f"\n[bold yellow]Battery at {battery}% (below {min_battery}%). Pausing...[/bold yellow]"
            )
            console.print("[dim]Will auto-resume when battery reaches 50%[/dim]")

            # Wait for battery to charge to 50%
            while True:
                time.sleep(300)  # Check every 5 minutes
                battery = _get_battery_level()
                if battery is None:
                    console.print(
                        "[green]Can't read battery - assuming plugged in. Resuming...[/green]"
                    )
                    break
                console.print(f"[dim]Battery at {battery}%... waiting for 50%[/dim]")
                if battery >= 50:
                    console.print(f"[green]Battery at {battery}%. Resuming![/green]")
                    break

        try:
            result = _index_single_book(
                book, theme_model, chunk_model, batch_size, legacy,
                dry_run, max_words, chunk_size,
            )
            if result == "skipped":
                skipped_large += 1
                continue
            successful += 1
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted! Progress saved. Run again to resume.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error indexing '{book['title'][:40]}': {e}[/red]")
            failed.append((book["id"], book["title"], str(e)))

    # Summary
    total_time = time.time() - start_time
    console.print(f"\n[bold]{'─' * 40}[/bold]")
    console.print("[bold]Batch complete![/bold]")
    console.print(f"  [green]Successful: {successful}[/green]")
    if skipped_large:
        console.print(f"  [yellow]Skipped (too large): {skipped_large}[/yellow]")
    if failed:
        console.print(f"  [red]Failed: {len(failed)}[/red]")
    console.print(f"  [dim]Time: {int(total_time // 60)}m {int(total_time % 60)}s[/dim]")

    # Log failures
    if failed:
        console.print("\n[bold red]Failed books:[/bold red]")
        for book_id, title, error in failed[:10]:
            console.print(f"  [dim]{book_id}:[/dim] {title[:40]} - {error[:50]}")
        if len(failed) > 10:
            console.print(f"  [dim]... and {len(failed) - 10} more[/dim]")


@main.command()
@click.argument("book_id", type=int, required=False)
@click.option("--title", "-t", help="Search by title")
def topics(book_id: int, title: str):
    """Show topics for a book."""
    book = None
    if book_id:
        book = get_book(book_id)
    elif title:
        book = get_book_by_title(title)

    if not book:
        console.print("[red]Book not found[/red]")
        return

    console.print(f"\n[bold]{book['title']}[/bold]")
    console.print(f"[dim]{book['author']}[/dim]\n")

    from .database import get_db

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT ct.topic, COUNT(*) as count
            FROM chunk_topics ct
            JOIN chunks c ON ct.chunk_id = c.id
            WHERE c.book_id = ?
            GROUP BY ct.topic
            ORDER BY count DESC
        """,
            (book["id"],),
        )

        topics = cursor.fetchall()

        if not topics:
            console.print("[yellow]No topics extracted yet. Run 'libtrails index' first.[/yellow]")
            return

        table = Table(title="Topics")
        table.add_column("Topic", style="cyan")
        table.add_column("Occurrences", style="green")

        for topic, count in topics:
            table.add_row(topic, str(count))

        console.print(table)


@main.command("book-clusters")
@click.argument("book_id", type=int, required=False)
@click.option("--title", "-t", help="Search by title")
@click.option("--limit", "-n", default=20, help="Max clusters to show")
def book_clusters(book_id: int, title: str, limit: int):
    """Show which theme clusters a book belongs to.

    Lists clusters containing topics from this book, ranked by
    how many of the book's topics are in each cluster.

    Example:
        libtrails book-clusters --title "Anathem"
        libtrails book-clusters 123
    """
    book = None
    if book_id:
        book = get_book(book_id)
    elif title:
        book = get_book_by_title(title)

    if not book:
        console.print("[red]Book not found[/red]")
        return

    console.print(f"\n[bold]{book['title']}[/bold]")
    console.print(f"[dim]{book['author']}[/dim]\n")

    from .database import get_db

    with get_db() as conn:
        cursor = conn.cursor()

        # Get clusters this book belongs to, ranked by topic count
        cursor.execute(
            """
            SELECT
                t.cluster_id,
                COUNT(DISTINCT t.id) as topic_count,
                GROUP_CONCAT(t.label, ', ') as sample_topics
            FROM chunks c
            JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
            JOIN topics t ON t.id = ctl.topic_id
            WHERE c.book_id = ? AND t.cluster_id IS NOT NULL
            GROUP BY t.cluster_id
            ORDER BY topic_count DESC
            LIMIT ?
        """,
            (book["id"], limit),
        )

        clusters = cursor.fetchall()

        if not clusters:
            console.print(
                "[yellow]No cluster assignments found. Run 'libtrails cluster' first.[/yellow]"
            )
            return

        # Get total unique clusters
        cursor.execute(
            """
            SELECT COUNT(DISTINCT t.cluster_id)
            FROM chunks c
            JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
            JOIN topics t ON t.id = ctl.topic_id
            WHERE c.book_id = ? AND t.cluster_id IS NOT NULL
        """,
            (book["id"],),
        )
        total_clusters = cursor.fetchone()[0]

        console.print(
            f"[dim]Book spans {total_clusters} clusters (showing top {min(limit, len(clusters))})[/dim]\n"
        )

        table = Table(title="Theme Clusters")
        table.add_column("Cluster", style="dim", justify="right")
        table.add_column("Topics", style="green", justify="right")
        table.add_column("Sample Topics", style="cyan", max_width=60)

        for cluster_id, topic_count, sample_topics in clusters:
            # Truncate sample topics list
            topics_list = sample_topics.split(", ")[:5]
            topics_display = ", ".join(topics_list)
            if len(sample_topics.split(", ")) > 5:
                topics_display += "..."

            table.add_row(str(cluster_id), str(topic_count), topics_display)

        console.print(table)


@main.command()
@click.argument("query")
def search(query: str):
    """Search for books by topic or text."""
    from .database import get_db

    with get_db() as conn:
        cursor = conn.cursor()

        # Search in chunk topics
        cursor.execute(
            """
            SELECT DISTINCT b.id, b.title, b.author, COUNT(ct.topic) as matches
            FROM books b
            JOIN chunks c ON b.id = c.book_id
            JOIN chunk_topics ct ON c.id = ct.chunk_id
            WHERE ct.topic LIKE ?
            GROUP BY b.id
            ORDER BY matches DESC
            LIMIT 20
        """,
            (f"%{query}%",),
        )

        results = cursor.fetchall()

        if not results:
            # Fallback to full-text search on descriptions
            cursor.execute(
                """
                SELECT id, title, author
                FROM books_fts
                WHERE books_fts MATCH ?
                LIMIT 20
            """,
                (query,),
            )
            results = cursor.fetchall()

        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        table = Table(title=f"Search: {query}")
        table.add_column("ID", style="dim")
        table.add_column("Title", style="cyan")
        table.add_column("Author", style="green")

        for row in results:
            table.add_row(str(row[0]), row[1][:50], row[2] or "")

        console.print(table)


@main.command()
def models():
    """List available Ollama models."""
    models = get_available_models()
    if models:
        console.print("[bold]Available models:[/bold]")
        for model in models:
            console.print(f"  {model}")
    else:
        console.print("[red]No Ollama models found. Install with 'ollama pull <model>'[/red]")


@main.command()
def formats():
    """Show format distribution of books in library."""
    from .database import get_book_path, get_db

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, calibre_id, format
            FROM books
            WHERE calibre_id IS NOT NULL
        """)
        books = cursor.fetchall()

    format_counts = Counter()
    available_counts = Counter()
    unavailable = []

    for book in books:
        format_counts[book["format"]] += 1
        path = get_book_path(book["calibre_id"])
        if path:
            available_counts[path.suffix.lower().lstrip(".")] += 1
        else:
            unavailable.append((book["id"], book["title"]))

    table = Table(title="Book Formats")
    table.add_column("Format", style="cyan")
    table.add_column("In iPad DB", style="dim")
    table.add_column("Available in Calibre", style="green")

    for fmt in sorted(set(format_counts.keys()) | set(available_counts.keys())):
        table.add_row(
            fmt.upper(), str(format_counts.get(fmt, 0)), str(available_counts.get(fmt, 0))
        )

    console.print(table)

    total_available = sum(available_counts.values())
    console.print(f"\n[green]Ready to process: {total_available} books[/green]")

    if unavailable:
        console.print(f"[yellow]Unavailable (no EPUB/PDF): {len(unavailable)} books[/yellow]")
        if len(unavailable) <= 5:
            for id, title in unavailable:
                console.print(f"  [dim]{id}: {title[:50]}[/dim]")
        else:
            for id, title in unavailable[:3]:
                console.print(f"  [dim]{id}: {title[:50]}[/dim]")
            console.print(f"  [dim]... and {len(unavailable) - 3} more[/dim]")


# ============================================================================
# New commands for post-processing pipeline
# ============================================================================


@main.command()
@click.option("--force", is_flag=True, help="Regenerate all embeddings (use after model change)")
def embed(force: bool):
    """Generate embeddings for all topics."""
    from .database import get_db
    from .embeddings import embed_texts, embedding_to_bytes, get_model_info

    # Ensure tables exist
    init_chunks_table()

    # Show model info
    model_info = get_model_info()
    console.print(f"[dim]Using model: {model_info['name']} ({model_info['dimension']} dims)[/dim]")
    if model_info["cached_locally"]:
        console.print("[dim]Model cached locally[/dim]")

    # First, migrate raw topics to normalized form
    console.print("\n[bold]Step 1: Normalizing topics...[/bold]")
    with console.status("Migrating raw topics..."):
        migrated = migrate_raw_topics_to_normalized()
    console.print(f"[green]Migrated {migrated} topics[/green]")

    # If force, clear existing embeddings
    if force:
        console.print("\n[yellow]Force mode: clearing existing embeddings...[/yellow]")
        with get_db() as conn:
            conn.execute("UPDATE topics SET embedding = NULL")
            conn.commit()

    # Get topics without embeddings
    topics = get_topics_without_embeddings()
    if not topics:
        console.print("[yellow]All topics already have embeddings[/yellow]")
        return

    console.print(f"\n[bold]Step 2: Generating embeddings for {len(topics)} topics...[/bold]")

    # Generate embeddings in batches
    labels = [t["label"] for t in topics]
    topic_ids = [t["id"] for t in topics]

    with console.status("Loading embedding model..."):
        from .embeddings import get_model

        get_model()  # Pre-load model

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding topics", total=len(topics))

        batch_size = 64
        for i in range(0, len(labels), batch_size):
            batch_labels = labels[i : i + batch_size]
            batch_ids = topic_ids[i : i + batch_size]

            embeddings = embed_texts(batch_labels)

            for topic_id, embedding in zip(batch_ids, embeddings):
                save_topic_embedding(topic_id, embedding_to_bytes(embedding))

            progress.advance(task, len(batch_labels))

    console.print(f"\n[green]Generated embeddings for {len(topics)} topics[/green]")

    # Build vector index (force recreate if using --force to get cosine distance)
    console.print("\n[bold]Step 3: Building vector index (cosine distance)...[/bold]")
    from .vector_search import get_vec_db, rebuild_vector_index

    with console.status("Rebuilding vector index..."):
        conn = get_vec_db()
        count = rebuild_vector_index(conn, force_recreate=force)
        conn.close()
    console.print(f"[green]Indexed {count} topic vectors[/green]")


@main.command("search-semantic")
@click.argument("query")
@click.option("--limit", "-n", default=20, help="Number of results")
def search_semantic(query: str, limit: int):
    """Semantic search for topics using embeddings."""
    from .vector_search import search_books_by_topic_semantic, search_topics_semantic

    console.print(f"\n[bold]Semantic search:[/bold] {query}\n")

    # Search topics
    topics = search_topics_semantic(query, limit=limit)

    if not topics:
        console.print("[yellow]No topics found. Run 'libtrails embed' first.[/yellow]")
        return

    table = Table(title="Matching Topics")
    table.add_column("Topic", style="cyan")
    table.add_column("Similarity", style="green")
    table.add_column("Occurrences", style="dim")

    for t in topics[:10]:
        table.add_row(t["label"], f"{t['similarity']:.3f}", str(t["occurrence_count"]))

    console.print(table)

    # Search books
    console.print("\n[bold]Books with matching topics:[/bold]")
    books = search_books_by_topic_semantic(query, limit=10)

    if books:
        table = Table()
        table.add_column("Title", style="cyan")
        table.add_column("Author", style="green")
        table.add_column("Relevance", style="yellow")

        for b in books:
            table.add_row(b["title"][:40], b["author"] or "", f"{b['relevance']:.3f}")

        console.print(table)
    else:
        console.print("[dim]No books found with these topics[/dim]")


@main.command()
@click.option(
    "--threshold", "-t", default=0.94, help="Similarity threshold (0.0-1.0). Higher=stricter"
)
@click.option("--dry-run", is_flag=True, help="Preview without making changes")
@click.option(
    "--sample-size",
    "-s",
    default=None,
    type=int,
    help="Process only N topics (for testing speed)",
)
@click.option(
    "--batch-size",
    "-b",
    default=5000,
    type=int,
    help="Batch size for numpy processing (tune for memory)",
)
@click.option(
    "--progress-file",
    "-p",
    default=None,
    help="File to write progress updates (for background runs)",
)
def dedupe(
    threshold: float,
    dry_run: bool,
    sample_size: int,
    batch_size: int,
    progress_file: str,
):
    """Deduplicate similar topics based on embeddings.

    Uses fast numpy batch processing (~15-20 seconds for 175K topics).

    Threshold guide:
        0.97 = Very strict (only near-exact duplicates like "behavior"/"behaviors")
        0.94 = Recommended (catches plurals, spelling variants, word order) [DEFAULT]
        0.95 = Strict (only near-exact matches)
        0.90 = Looser (may create transitive chains)

    Examples:
        libtrails dedupe --dry-run              # Preview with default threshold
        libtrails dedupe --dry-run -t 0.97      # Preview with strict threshold
        libtrails dedupe                        # Run deduplication
        libtrails dedupe -s 5000                # Test with 5K topic sample
    """
    from .deduplication import deduplicate_topics, get_deduplication_preview

    # Show settings
    settings = [f"threshold={threshold}"]
    if sample_size:
        settings.append(f"sample={sample_size:,}")
    console.print(f"[dim]Settings: {', '.join(settings)}[/dim]\n")

    if dry_run:
        console.print(f"[bold]Preview: Deduplication with threshold {threshold}[/bold]\n")
        preview = get_deduplication_preview(
            threshold,
            limit=20,
            sample_size=sample_size,
            batch_size=batch_size,
            progress_file=progress_file,
        )

        if not preview:
            console.print("[yellow]No duplicates found[/yellow]")
            return

        for group in preview:
            console.print(
                f"[green]{group['canonical']}[/green] ({group['canonical_count']} occurrences)"
            )
            for dup in group["duplicates"]:
                console.print(f"  └─ [dim]{dup['label']}[/dim] ({dup['count']})")
            console.print()

        console.print(
            f"[yellow]Found {len(preview)} duplicate groups. Run without --dry-run to merge.[/yellow]"
        )
    else:
        console.print(f"[bold]Deduplicating topics with threshold {threshold}...[/bold]\n")
        result = deduplicate_topics(
            threshold,
            dry_run=False,
            sample_size=sample_size,
            batch_size=batch_size,
            progress_file=progress_file,
        )

        console.print(
            f"[green]Merged {result['topics_merged']} topics into {result['duplicate_groups']} canonical topics[/green]"
        )


@main.command()
@click.argument("topic", required=False)
def tree(topic: str):
    """Browse the topic hierarchy."""
    from .clustering import get_topic_tree

    if topic:
        # Show topics in a specific cluster/category
        from .database import get_db

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, label, cluster_id, occurrence_count
                FROM topics
                WHERE label LIKE ?
                ORDER BY occurrence_count DESC
                LIMIT 20
            """,
                (f"%{topic}%",),
            )

            results = cursor.fetchall()

        if not results:
            console.print(f"[yellow]No topics matching '{topic}'[/yellow]")
            return

        table = Table(title=f"Topics matching: {topic}")
        table.add_column("Topic", style="cyan")
        table.add_column("Cluster", style="dim")
        table.add_column("Occurrences", style="green")

        for row in results:
            table.add_row(row[1], str(row[2] or "-"), str(row[3]))

        console.print(table)
    else:
        # Show full tree
        tree_data = get_topic_tree()

        if not tree_data.get("children"):
            console.print(
                "[yellow]No topic tree available. Run 'libtrails cluster' first.[/yellow]"
            )
            return

        rich_tree = Tree("[bold]Topics[/bold]")

        for cluster in tree_data["children"][:15]:
            branch = rich_tree.add(f"[cyan]{cluster['name']}[/cyan] ({cluster['size']} topics)")
            for child in cluster.get("children", [])[:5]:
                branch.add(f"[dim]{child['name']}[/dim] ({child['count']})")

        console.print(rich_tree)


@main.command("diagnose-hubs")
@click.option(
    "--top-n",
    type=int,
    default=50,
    help="Number of top hubs to display",
)
def diagnose_hubs(top_n):
    """Analyze hub topics that may be causing clustering problems.

    Hub topics are highly connected nodes that create "short circuits"
    between unrelated topic communities. This diagnostic shows:

    - Degree distribution (connections per topic)
    - Top hub topics by connection count
    - Recommendations for hub removal

    Example:
        libtrails diagnose-hubs --top-n 100
    """
    from .clustering import diagnose_hubs as _diagnose_hubs
    from .config import CLUSTER_KNN_K, COOCCURRENCE_MIN_COUNT, PMI_MIN_THRESHOLD
    from .topic_graph import build_topic_graph_knn

    console.print("[bold]Building topic graph for hub analysis...[/bold]\n")

    # Build the graph
    g = build_topic_graph_knn(
        cooccurrence_min=COOCCURRENCE_MIN_COUNT,
        pmi_min=PMI_MIN_THRESHOLD,
        k=CLUSTER_KNN_K,
    )

    if g.vcount() == 0:
        console.print("[red]Error: No topics in graph. Run 'libtrails process' first.[/red]")
        return

    console.print(f"[dim]Graph: {g.vcount():,} nodes, {g.ecount():,} edges[/dim]\n")

    # Run diagnosis
    result = _diagnose_hubs(g, top_n=top_n)

    # Display degree distribution
    stats = result["degree_stats"]
    console.print("[bold cyan]Degree Distribution (connections per topic)[/bold cyan]")
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="green")

    table.add_row("Minimum", str(stats["min"]))
    table.add_row("Median", str(stats["median"]))
    table.add_row("Mean", f"{stats['mean']:.1f}")
    table.add_row("95th percentile", str(stats["p95"]))
    table.add_row("99th percentile", str(stats["p99"]))
    table.add_row("Maximum", str(stats["max"]))

    console.print(table)

    # Display top hubs
    console.print(f"\n[bold cyan]Top {top_n} Hub Topics (by degree)[/bold cyan]")

    hub_table = Table()
    hub_table.add_column("Degree", style="yellow", justify="right")
    hub_table.add_column("Topic Label", style="cyan")

    for hub in result["top_hubs"]:
        hub_table.add_row(str(hub["degree"]), hub["label"])

    console.print(hub_table)

    # Recommendations
    p95_threshold = stats["p95"]
    sum(1 for h in result["top_hubs"] if h["degree"] > p95_threshold)

    console.print("\n[bold]Recommendations:[/bold]")
    console.print(
        f"  • At 95th percentile ({p95_threshold} edges), ~{int(g.vcount() * 0.05):,} topics are hubs"
    )
    console.print(
        "  • Try: [dim]libtrails cluster --remove-hubs --hub-percentile 95 --dry-run[/dim]"
    )
    console.print("  • For tighter clusters: [dim]--hub-percentile 90[/dim] (removes top 10%)")
    console.print("  • Also try: [dim]--hub-method both[/dim] to include generic term patterns")


@main.command("label-clusters")
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Maximum clusters to label (default: all)",
)
@click.option(
    "--min-size",
    type=int,
    default=10,
    help="Minimum cluster size to label (default: 10)",
)
@click.option(
    "--model",
    type=str,
    default="gemma3:4b",
    help="Ollama model to use (default: gemma3:4b)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-label clusters that already have labels",
)
@click.option(
    "--progress-file",
    "-p",
    default=None,
    help="File to write progress updates (for background runs)",
)
def label_clusters(limit, min_size, model, force, progress_file):
    """Generate LLM-based labels for topic clusters.

    Uses Ollama to analyze the top topics in each cluster and generate
    a concise 2-4 word descriptive label.

    Example:
        libtrails label-clusters --limit 50
        libtrails label-clusters --model gemma3:27b
        libtrails label-clusters --force  # Re-label all
    """
    from .clustering import label_clusters_batch

    console.print("[bold]Generating cluster labels with LLM...[/bold]\n")
    console.print(f"[dim]Model: {model}, Min size: {min_size}[/dim]\n")

    result = label_clusters_batch(
        limit=limit,
        min_size=min_size,
        model=model,
        skip_existing=not force,
        progress_file=progress_file,
    )

    console.print(f"\n[green]Labeled {result['labeled']} clusters[/green]")
    if result["failed"] > 0:
        console.print(f"[yellow]Failed: {result['failed']}[/yellow]")


@main.command()
@click.argument("topic")
@click.option("--limit", "-n", default=10, help="Number of related topics")
def related(topic: str, limit: int):
    """Find topics related to the given topic via graph connections."""
    from .topic_graph import get_related_topics

    results = get_related_topics(topic, limit=limit)

    if not results:
        console.print(f"[yellow]No related topics found for '{topic}'[/yellow]")
        return

    table = Table(title=f"Topics related to: {topic}")
    table.add_column("Topic", style="cyan")
    table.add_column("Connection", style="dim")
    table.add_column("Weight", style="green")

    for r in results:
        table.add_row(r["label"], r["connection_type"], f"{r['connection_weight']:.3f}")

    console.print(table)


@main.command()
@click.argument("topic")
def cooccur(topic: str):
    """Show topics that co-occur with the given topic in chunks."""
    from .database import get_db

    with get_db() as conn:
        cursor = conn.cursor()

        # Find the topic
        cursor.execute("SELECT id FROM topics WHERE label LIKE ?", (f"%{topic}%",))
        row = cursor.fetchone()
        if not row:
            console.print(f"[yellow]Topic '{topic}' not found[/yellow]")
            return

        topic_id = row[0]

        # Find co-occurring topics
        cursor.execute(
            """
            SELECT t.label, tc.count, tc.pmi
            FROM topic_cooccurrences tc
            JOIN topics t ON (
                CASE WHEN tc.topic1_id = ? THEN tc.topic2_id ELSE tc.topic1_id END = t.id
            )
            WHERE tc.topic1_id = ? OR tc.topic2_id = ?
            ORDER BY tc.count DESC
            LIMIT 20
        """,
            (topic_id, topic_id, topic_id),
        )

        results = cursor.fetchall()

        if not results:
            console.print(
                "[yellow]No co-occurring topics found. Run 'libtrails cluster' first.[/yellow]"
            )
            return

        table = Table(title=f"Topics co-occurring with: {topic}")
        table.add_column("Topic", style="cyan")
        table.add_column("Co-occurrences", style="green")
        table.add_column("PMI", style="dim")

        for label, count, pmi in results:
            table.add_row(label, str(count), f"{pmi:.2f}" if pmi else "-")

        console.print(table)


@main.command()
def process():
    """Run the full post-processing pipeline (embed, dedupe, cluster)."""
    from .clustering import cluster_topics
    from .deduplication import deduplicate_topics
    from .embeddings import embed_texts, embedding_to_bytes
    from .topic_graph import compute_cooccurrences
    from .vector_search import get_vec_db, rebuild_vector_index

    console.print("[bold]Running full post-processing pipeline...[/bold]\n")

    # Ensure tables exist
    init_chunks_table()

    # Step 1: Embed
    console.print("[bold cyan]Step 1/3: Generating embeddings[/bold cyan]")

    # Normalize topics
    console.print("  Normalizing topics...")
    migrated = migrate_raw_topics_to_normalized()
    console.print(f"  [green]Migrated {migrated} topics[/green]")

    # Get topics without embeddings
    topics = get_topics_without_embeddings()
    if topics:
        console.print(f"  Generating embeddings for {len(topics)} topics...")
        labels = [t["label"] for t in topics]
        topic_ids = [t["id"] for t in topics]

        from .embeddings import get_model

        get_model()  # Pre-load model

        batch_size = 64
        for i in range(0, len(labels), batch_size):
            batch_labels = labels[i : i + batch_size]
            batch_ids = topic_ids[i : i + batch_size]
            embeddings = embed_texts(batch_labels)
            for topic_id, embedding in zip(batch_ids, embeddings):
                save_topic_embedding(topic_id, embedding_to_bytes(embedding))

        console.print(f"  [green]Generated {len(topics)} embeddings[/green]")

        # Build vector index
        conn = get_vec_db()
        count = rebuild_vector_index(conn)
        conn.close()
        console.print(f"  [green]Indexed {count} topic vectors[/green]")
    else:
        console.print("  [yellow]All topics already have embeddings[/yellow]")

    # Step 2: Dedupe
    console.print("\n[bold cyan]Step 2/3: Deduplicating topics[/bold cyan]")
    result = deduplicate_topics(threshold=0.85, dry_run=False)
    console.print(f"  [green]Merged {result['topics_merged']} topics[/green]")

    # Step 3: Cluster
    console.print("\n[bold cyan]Step 3/3: Clustering topics[/bold cyan]")
    console.print("  Computing co-occurrences...")
    cooccur_stats = compute_cooccurrences()
    console.print(
        f"  [green]Found {cooccur_stats['cooccurrence_pairs']} co-occurrence pairs[/green]"
    )

    console.print("  Running Leiden clustering...")
    cluster_result = cluster_topics()
    if "error" not in cluster_result:
        console.print(f"  [green]Created {cluster_result['num_clusters']} clusters[/green]")
    else:
        console.print(f"  [red]Error: {cluster_result['error']}[/red]")

    console.print("\n[bold green]Pipeline complete![/bold green]")

    _refresh_and_report_stats()

    # Show final stats
    stats = get_indexing_status()
    console.print(
        f"\n[dim]Final stats: {stats['normalized_topics']} topics, {stats['topics_with_embeddings']} embedded, {stats['clustered_topics']} clustered[/dim]"
    )


@main.command()
@click.option(
    "--mode",
    type=click.Choice(["cooccurrence", "knn", "full"]),
    default=None,
    help="Graph construction mode (default: knn)",
)
@click.option(
    "--partition-type",
    type=click.Choice(["modularity", "surprise", "cpm"]),
    default=None,
    help="Leiden partition type (default: cpm)",
)
@click.option(
    "--min-cooccur",
    type=int,
    default=None,
    help="Minimum co-occurrence count for edges (default: 2)",
)
@click.option(
    "--resolution",
    type=float,
    default=None,
    help="Resolution parameter (default: 0.1). Guide: 0.01=mega-themes, 0.1=medium, 0.5=focused",
)
@click.option(
    "--knn-k",
    type=int,
    default=None,
    help="Number of neighbors for k-NN mode (default: 10)",
)
@click.option(
    "--skip-cooccur",
    is_flag=True,
    help="Skip co-occurrence computation (use existing data)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run clustering but don't save results to database",
)
@click.option(
    "--sample-size",
    type=int,
    default=None,
    help="Test on a random sample of N topics (implies --dry-run)",
)
@click.option(
    "--remove-hubs",
    is_flag=True,
    help="Remove hub topics before clustering to prevent transitivity chains",
)
@click.option(
    "--hub-percentile",
    type=float,
    default=95,
    help="Hub detection threshold: top N% by degree are hubs (default: 95 = top 5%)",
)
@click.option(
    "--hub-method",
    type=click.Choice(["degree", "generic", "both"]),
    default="degree",
    help="Hub detection method (default: degree)",
)
@click.option(
    "--progress-file",
    "-p",
    default=None,
    help="File to write progress updates (for background runs)",
)
def cluster(
    mode,
    partition_type,
    min_cooccur,
    resolution,
    knn_k,
    skip_cooccur,
    dry_run,
    sample_size,
    remove_hubs,
    hub_percentile,
    hub_method,
    progress_file,
):
    """Run topic clustering with configurable options.

    Defaults are optimized for ~300-400 coherent topic clusters.

    Examples:
        libtrails cluster                              # Use optimized defaults
        libtrails cluster --skip-cooccur               # Reuse existing co-occurrences
        libtrails cluster --resolution 0.1             # More, smaller clusters
        libtrails cluster --mode cooccurrence          # Fast, sparse clustering
        libtrails cluster --sample-size 5000           # Test on 5K topics (dry run)
        libtrails cluster --dry-run --resolution 0.2   # Test params without saving
        libtrails cluster --remove-hubs --dry-run      # Test hub removal approach
    """
    from .clustering import cluster_topics
    from .database import get_db
    from .topic_graph import compute_cooccurrences

    console.print(f"[bold]Running clustering (mode={mode}, partition={partition_type})...[/bold]\n")

    # Check if embeddings exist
    stats = get_indexing_status()
    if stats["topics_with_embeddings"] == 0:
        console.print("[red]Error: No topic embeddings found. Run 'libtrails process' first.[/red]")
        return

    console.print(f"[dim]Found {stats['topics_with_embeddings']} topics with embeddings[/dim]\n")

    # Compute or check co-occurrences
    if skip_cooccur:
        # Check if co-occurrences exist
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM topic_cooccurrences")
            cooccur_count = cursor.fetchone()[0]

        if cooccur_count == 0:
            console.print("[yellow]Warning: No co-occurrences found. Computing now...[/yellow]")
            cooccur_stats = compute_cooccurrences(progress_file=progress_file)
            console.print(
                f"  [green]Found {cooccur_stats['cooccurrence_pairs']} co-occurrence pairs[/green]"
            )
        else:
            console.print("[bold cyan]Step 1/2: Using existing co-occurrences[/bold cyan]")
            console.print(f"  [green]Found {cooccur_count} existing co-occurrence pairs[/green]")
    else:
        console.print("[bold cyan]Step 1/2: Computing co-occurrences[/bold cyan]")
        cooccur_stats = compute_cooccurrences(progress_file=progress_file)
        console.print(
            f"  [green]Found {cooccur_stats['cooccurrence_pairs']} co-occurrence pairs[/green]"
        )

    # Run clustering
    if dry_run or sample_size:
        console.print("\n[bold yellow]Step 2/2: Clustering topics (DRY RUN)[/bold yellow]")
    else:
        console.print("\n[bold cyan]Step 2/2: Clustering topics[/bold cyan]")

    cluster_result = cluster_topics(
        mode=mode,
        partition_type=partition_type,
        cooccurrence_min=min_cooccur,
        resolution=resolution,
        knn_k=knn_k,
        dry_run=dry_run,
        sample_size=sample_size,
        remove_hubs=remove_hubs,
        hub_percentile=hub_percentile,
        hub_method=hub_method,
        progress_file=progress_file,
    )

    if "error" not in cluster_result:
        console.print(f"\n  [green]Created {cluster_result['num_clusters']} clusters[/green]")
        console.print(f"  [dim]Quality score: {cluster_result['modularity']:.4f}[/dim]")
        console.print(
            f"  [dim]Edges: {cluster_result['total_edges']} ({cluster_result.get('edge_types', {})})[/dim]"
        )
        console.print(f"  [dim]Time: {cluster_result.get('leiden_time_seconds', 0):.1f}s[/dim]")
        console.print(f"  [dim]Resolution: {cluster_result.get('resolution', 'N/A')}[/dim]")

        # Show hub removal stats
        if cluster_result.get("hubs_removed"):
            console.print(
                f"  [dim]Hubs removed: {cluster_result['hubs_removed']} (method={cluster_result.get('hub_method')})[/dim]"
            )

        # Show top clusters
        if cluster_result.get("cluster_sizes"):
            console.print("\n  [bold]Top clusters by size:[/bold]")
            for cluster_id, size in list(cluster_result["cluster_sizes"].items())[:5]:
                console.print(f"    Cluster {cluster_id}: {size} topics")

        # Show estimates for dry run
        if cluster_result.get("sample_size"):
            console.print("\n  [bold yellow]Dry Run Estimates:[/bold yellow]")
            console.print(
                f"    Sampled: {cluster_result['sample_size']:,} of {cluster_result['full_node_count']:,} topics"
            )
            console.print(f"    Scale factor: {cluster_result['scale_factor']:.1f}x")
            est_time = cluster_result.get("estimated_full_time_seconds", 0)
            console.print(f"    Estimated full run time: {est_time / 60:.1f} minutes")
            if cluster_result.get("memory_used_mb"):
                console.print(
                    f"    Memory used (sample): {cluster_result['memory_used_mb']:.0f} MB"
                )

        if cluster_result.get("dry_run"):
            console.print("\n[bold yellow]DRY RUN - no changes saved to database[/bold yellow]")
        else:
            console.print("\n[bold green]Clustering complete![/bold green]")
            _refresh_and_report_stats()
    else:
        console.print(f"  [red]Error: {cluster_result['error']}[/red]")


@main.command("load-domains")
@click.option(
    "--json-file",
    "-f",
    default="experiments/domain_labels_final.json",
    help="Path to domain labels JSON file",
)
def load_domains(json_file: str):
    """Load domain (super-cluster) labels into the database."""
    from pathlib import Path

    from .database import get_all_domains, init_chunks_table, load_domains_from_json

    json_path = Path(json_file)
    if not json_path.exists():
        console.print(f"[red]Error: File not found: {json_file}[/red]")
        console.print(
            "Generate domain labels first with: [cyan]uv run python experiments/domain_labels_final.py[/cyan]"
        )
        return

    # Ensure tables exist
    init_chunks_table()

    console.print(f"Loading domains from [cyan]{json_file}[/cyan]...")
    count = load_domains_from_json(json_path)
    console.print(f"[green]Loaded {count} domains[/green]")

    _refresh_and_report_stats()

    # Show summary
    table = Table(title="Domains")
    table.add_column("ID", style="dim", width=4)
    table.add_column("Clusters", justify="right", width=8)
    table.add_column("Label", style="cyan")

    for d in get_all_domains():
        table.add_row(str(d["id"]), str(d["cluster_count"]), d["label"])

    console.print(table)


@main.command("regenerate-domains")
@click.option(
    "--n-domains",
    "-n",
    default=25,
    type=int,
    help="Number of super-clusters to generate",
)
@click.option(
    "--output",
    "-o",
    default="experiments/super_clusters_new.json",
    help="Output JSON file path",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show super-clusters without saving",
)
def regenerate_domains(n_domains: int, output: str, dry_run: bool):
    """
    Regenerate domains (super-clusters) from current Leiden clusters.

    This uses K-means on cluster centroids to group similar clusters.
    After reviewing the auto-generated labels, update REFINED_LABELS
    in experiments/domain_labels_final.py and run load-domains.

    Example workflow:
      1. uv run libtrails regenerate-domains  # generates super_clusters_new.json
      2. Review output and update REFINED_LABELS mapping
      3. uv run python experiments/domain_labels_final.py
      4. uv run libtrails load-domains
    """
    from pathlib import Path

    from .domains import generate_super_clusters

    console.print(f"[bold]Generating {n_domains} super-clusters...[/bold]")

    super_clusters = generate_super_clusters(n_domains=n_domains)

    console.print(f"\n[bold]=== {len(super_clusters)} Super-Clusters ===[/bold]\n")

    table = Table(title="Super-Clusters (Auto-labeled)")
    table.add_column("ID", style="dim", width=4)
    table.add_column("Clusters", justify="right", width=8)
    table.add_column("Auto Label", style="cyan")

    for sc in super_clusters:
        table.add_row(
            str(sc["super_cluster_id"]),
            str(len(sc["leiden_clusters"])),
            sc["auto_label"][:50] + "..." if len(sc["auto_label"]) > 50 else sc["auto_label"],
        )

    console.print(table)

    total = sum(len(sc["leiden_clusters"]) for sc in super_clusters)
    console.print(f"\nTotal clusters mapped: {total}")

    if dry_run:
        console.print("\n[bold yellow]DRY RUN - not saving to file[/bold yellow]")
    else:
        output_path = Path(output)
        # Convert to format expected by domain_labels_final.py
        import json

        with open(output_path, "w") as f:
            json.dump(super_clusters, f, indent=2)
        console.print(f"\n[green]Saved to {output}[/green]")
        console.print("\nNext steps:")
        console.print(
            "  1. Review auto-labels and update REFINED_LABELS in experiments/domain_labels_final.py"
        )
        console.print("  2. Run: [cyan]uv run python experiments/domain_labels_final.py[/cyan]")
        console.print("  3. Run: [cyan]uv run libtrails load-domains[/cyan]")


@main.command("generate-universe")
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Output JSON file path (default: data/universe_coords.json)",
)
@click.option(
    "--n-neighbors",
    default=15,
    type=int,
    help="UMAP n_neighbors parameter",
)
@click.option(
    "--min-dist",
    default=0.3,
    type=float,
    help="UMAP min_dist parameter",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show cluster/domain counts without generating",
)
def generate_universe(output: str | None, n_neighbors: int, min_dist: float, dry_run: bool):
    """
    Generate UMAP projection of clusters for the Galaxy visualization.

    Produces a JSON file with 2D coordinates for every Leiden cluster,
    colored by domain assignment. Used by the frontend galaxy homepage.
    """
    import sqlite3
    from pathlib import Path

    from .config import IPAD_DB_PATH, UNIVERSE_JSON_PATH

    if dry_run:
        conn = sqlite3.connect(IPAD_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(DISTINCT cluster_id) as n
            FROM topics
            WHERE cluster_id IS NOT NULL AND cluster_id >= 0
        """)
        n_clusters = cursor.fetchone()["n"]

        cursor.execute("SELECT COUNT(*) as n FROM domains")
        n_domains = cursor.fetchone()["n"]

        conn.close()

        console.print(f"[bold]Clusters:[/bold] {n_clusters}")
        console.print(f"[bold]Domains:[/bold] {n_domains}")
        console.print(f"[bold]UMAP params:[/bold] n_neighbors={n_neighbors}, min_dist={min_dist}")
        console.print("\n[bold yellow]DRY RUN — not generating[/bold yellow]")
        return

    from .universe import generate_universe_data

    output_path = Path(output) if output else UNIVERSE_JSON_PATH

    console.print("[bold]Generating universe coordinates...[/bold]")
    console.print(f"  UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}")

    result = generate_universe_data(
        output_path=output_path,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )

    n_clusters = len(result["clusters"])
    n_domains = len(result["domains"])

    console.print(f"\n[green]Saved {n_clusters} clusters across {n_domains} domains[/green]")
    console.print(f"  Output: [cyan]{output_path}[/cyan]")

    # Show sample
    table = Table(title="Sample Clusters")
    table.add_column("ID", style="dim", width=6)
    table.add_column("Label", style="cyan")
    table.add_column("Books", justify="right", width=6)
    table.add_column("Domain", style="green")
    table.add_column("Position")

    for cd in result["clusters"][:10]:
        table.add_row(
            str(cd["cluster_id"]),
            cd["label"][:30],
            str(cd["book_count"]),
            cd["domain_label"][:20],
            f"({cd['x']:.3f}, {cd['y']:.3f})",
        )

    console.print(table)


@main.command("refresh-stats")
def refresh_stats():
    """Refresh materialized stats tables for fast API responses.

    Pre-computes cluster→book mappings, per-cluster stats, and per-domain
    stats so that /domains and /themes endpoints respond in <100ms.

    Runs automatically after 'cluster', 'load-domains', and 'process'.
    """
    console.print("[bold]Refreshing materialized stats...[/bold]")
    result = refresh_all_stats()
    console.print(f"  [green]cluster_books rows: {result['cluster_book_rows']:,}[/green]")
    console.print(f"  [green]clusters with stats: {result['clusters_with_stats']:,}[/green]")
    console.print(f"  [green]domains with stats: {result['domains_with_stats']:,}[/green]")
    console.print(f"  [dim]Completed in {result['elapsed_seconds']:.1f}s[/dim]")


@main.command("prepare-v2")
def prepare_v2():
    """Prepare a v2 database for topic extraction experiments.

    Copies the current database, clears all topic-related tables,
    and keeps books + chunks intact (no need to re-parse EPUBs).

    Usage:
        uv run libtrails prepare-v2
        LIBTRAILS_DB=v2 uv run libtrails index --title "Siddhartha"
        LIBTRAILS_DB=v2 uv run libtrails process
    """
    import shutil

    from .config import DATA_DIR

    source = DATA_DIR / "ipad_library.db"
    dest = DATA_DIR / "ipad_library_v2.db"

    if not source.exists():
        console.print(f"[red]Source database not found: {source}[/red]")
        return

    if dest.exists():
        console.print(f"[yellow]v2 database already exists: {dest}[/yellow]")
        console.print("[dim]Delete it manually to recreate.[/dim]")
        return

    # Copy database
    console.print("[bold]Copying database...[/bold]")
    console.print(f"  {source} → {dest}")
    shutil.copy2(source, dest)

    # Clear topic-related tables
    import sqlite3

    conn = sqlite3.connect(dest)
    cursor = conn.cursor()

    tables_to_clear = [
        "chunk_topics",
        "chunk_topic_links",
        "topics",
        "topic_cooccurrences",
        "topic_cluster_memberships",
        "cluster_labels",
        "cluster_books",
        "cluster_stats",
        "domain_stats",
        "domains",
        "cluster_domains",
    ]

    console.print("\n[bold]Clearing topic tables...[/bold]")
    for table in tables_to_clear:
        try:
            cursor.execute(f"DELETE FROM {table}")
            count = cursor.rowcount
            if count > 0:
                console.print(f"  [dim]{table}: cleared {count:,} rows[/dim]")
        except sqlite3.OperationalError:
            pass  # Table doesn't exist

    # Drop and recreate topic_vectors virtual table
    try:
        cursor.execute("DROP TABLE IF EXISTS topic_vectors")
        console.print("  [dim]topic_vectors: dropped[/dim]")
    except sqlite3.OperationalError:
        pass

    # Add book_themes column if missing
    cursor.execute("PRAGMA table_info(books)")
    columns = [row[1] for row in cursor.fetchall()]
    if "book_themes" not in columns:
        cursor.execute("ALTER TABLE books ADD COLUMN book_themes TEXT")
        console.print("  [dim]Added book_themes column[/dim]")

    conn.commit()
    conn.close()

    console.print(f"\n[green]v2 database ready: {dest}[/green]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  [cyan]LIBTRAILS_DB=v2 uv run libtrails index --title 'Siddhartha'[/cyan]")
    console.print("  [cyan]LIBTRAILS_DB=v2 uv run libtrails process[/cyan]")
    console.print("  [cyan]LIBTRAILS_DB=v2 uv run libtrails serve --port 8001[/cyan]")


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Start the API server."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Error: uvicorn not installed.[/red]")
        console.print("Install API dependencies with: [cyan]uv pip install -e '.[api]'[/cyan]")
        return

    console.print("[bold]Starting LibTrails API server...[/bold]")
    console.print(f"  API: [cyan]http://{host}:{port}/api/v1[/cyan]")
    console.print(f"  Docs: [cyan]http://{host}:{port}/docs[/cyan]")
    console.print()

    uvicorn.run(
        "libtrails.api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
