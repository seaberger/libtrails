"""Command-line interface for libtrails."""

import click
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from pathlib import Path

from . import __version__
from .config import DEFAULT_MODEL, CHUNK_TARGET_WORDS
from .database import (
    get_book, get_book_by_title, get_all_books, get_epub_path,
    init_chunks_table, save_chunks, save_chunk_topics, get_indexing_status,
    get_all_topics, get_topics_without_embeddings, save_topic_embedding,
    migrate_raw_topics_to_normalized, get_topic_stats
)
from .epub_parser import extract_text_from_epub
from .chunker import chunk_text
from .topic_extractor import extract_topics, check_ollama_available, get_available_models

console = Console()


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

    table.add_row("Total Books (with Calibre match)", str(stats['total_books']))
    table.add_row("Indexed Books", str(stats['indexed_books']))
    table.add_row("Total Chunks", str(stats['total_chunks']))
    table.add_row("Unique Topics (raw)", str(stats['unique_topics']))
    table.add_row("─" * 20, "─" * 10)
    table.add_row("Normalized Topics", str(stats['normalized_topics']))
    table.add_row("Topics with Embeddings", str(stats['topics_with_embeddings']))
    table.add_row("Clustered Topics", str(stats['clustered_topics']))

    if stats['total_books'] > 0:
        pct = (stats['indexed_books'] / stats['total_books']) * 100
        table.add_row("Progress", f"{pct:.1f}%")

    console.print(table)

    # Check Ollama
    models = get_available_models()
    if models:
        console.print(f"\n[green]Ollama models available:[/green] {', '.join(models)}")
    else:
        console.print("\n[yellow]Warning: Ollama not available or no models loaded[/yellow]")


@main.command()
@click.argument('book_id', type=int, required=False)
@click.option('--title', '-t', help='Search by title')
@click.option('--all', 'index_all', is_flag=True, help='Index all books')
@click.option('--model', '-m', default=DEFAULT_MODEL, help='Ollama model to use')
@click.option('--dry-run', is_flag=True, help='Parse and chunk without topic extraction')
def index(book_id: int, title: str, index_all: bool, model: str, dry_run: bool):
    """Index a book (parse, chunk, extract topics)."""
    init_chunks_table()

    if index_all:
        _index_all_books(model, dry_run)
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

    _index_single_book(book, model, dry_run)


def _index_single_book(book: dict, model: str, dry_run: bool):
    """Index a single book."""
    console.print(f"\n[bold]Indexing:[/bold] {book['title']}")
    console.print(f"[dim]Author: {book['author']}[/dim]")

    # Get EPUB path
    if not book.get('calibre_id'):
        console.print("[red]No Calibre match for this book[/red]")
        return

    epub_path = get_epub_path(book['calibre_id'])
    if not epub_path:
        console.print("[red]EPUB not found in Calibre library[/red]")
        return

    console.print(f"[dim]EPUB: {epub_path.name}[/dim]")

    # Extract text
    with console.status("Extracting text from EPUB..."):
        text = extract_text_from_epub(epub_path)

    word_count = len(text.split())
    console.print(f"[green]Extracted {word_count:,} words[/green]")

    # Chunk
    chunks = chunk_text(text, CHUNK_TARGET_WORDS)
    console.print(f"[green]Created {len(chunks)} chunks[/green]")

    # Save chunks
    save_chunks(book['id'], chunks)

    if dry_run:
        console.print("[yellow]Dry run - skipping topic extraction[/yellow]")
        # Show sample chunk
        if chunks:
            console.print(f"\n[bold]Sample chunk ({len(chunks[0].split())} words):[/bold]")
            console.print(chunks[0][:500] + "...")
        return

    # Extract topics
    if not check_ollama_available(model):
        console.print(f"[red]Model {model} not available in Ollama[/red]")
        return

    console.print(f"\n[bold]Extracting topics with {model}...[/bold]")

    from .database import get_db

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Processing chunks", total=len(chunks))

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM chunks WHERE book_id = ? ORDER BY chunk_index",
                (book['id'],)
            )
            chunk_ids = [row[0] for row in cursor.fetchall()]

            all_topics = []
            for chunk_id, chunk_content in zip(chunk_ids, chunks):
                topics = extract_topics(chunk_content, model)
                if topics:
                    save_chunk_topics(chunk_id, topics)
                    all_topics.extend(topics)
                progress.advance(task)

    unique_topics = set(all_topics)
    console.print(f"\n[green]Extracted {len(unique_topics)} unique topics[/green]")

    # Show top topics
    if unique_topics:
        from collections import Counter
        topic_counts = Counter(all_topics)
        console.print("\n[bold]Top topics:[/bold]")
        for topic, count in topic_counts.most_common(10):
            console.print(f"  {topic} ({count})")


def _index_all_books(model: str, dry_run: bool):
    """Index all books with Calibre matches."""
    books = get_all_books(with_calibre_match=True)
    console.print(f"[bold]Indexing {len(books)} books...[/bold]\n")

    for i, book in enumerate(books, 1):
        console.print(f"\n[dim]({i}/{len(books)})[/dim]")
        try:
            _index_single_book(book, model, dry_run)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@main.command()
@click.argument('book_id', type=int, required=False)
@click.option('--title', '-t', help='Search by title')
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
        cursor.execute("""
            SELECT ct.topic, COUNT(*) as count
            FROM chunk_topics ct
            JOIN chunks c ON ct.chunk_id = c.id
            WHERE c.book_id = ?
            GROUP BY ct.topic
            ORDER BY count DESC
        """, (book['id'],))

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


@main.command()
@click.argument('query')
def search(query: str):
    """Search for books by topic or text."""
    from .database import get_db

    with get_db() as conn:
        cursor = conn.cursor()

        # Search in chunk topics
        cursor.execute("""
            SELECT DISTINCT b.id, b.title, b.author, COUNT(ct.topic) as matches
            FROM books b
            JOIN chunks c ON b.id = c.book_id
            JOIN chunk_topics ct ON c.id = ct.chunk_id
            WHERE ct.topic LIKE ?
            GROUP BY b.id
            ORDER BY matches DESC
            LIMIT 20
        """, (f"%{query}%",))

        results = cursor.fetchall()

        if not results:
            # Fallback to full-text search on descriptions
            cursor.execute("""
                SELECT id, title, author
                FROM books_fts
                WHERE books_fts MATCH ?
                LIMIT 20
            """, (query,))
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


# ============================================================================
# New commands for post-processing pipeline
# ============================================================================

@main.command()
def embed():
    """Generate embeddings for all topics."""
    from .embeddings import embed_texts, embedding_to_bytes

    # Ensure tables exist
    init_chunks_table()

    # First, migrate raw topics to normalized form
    console.print("[bold]Step 1: Normalizing topics...[/bold]")
    with console.status("Migrating raw topics..."):
        migrated = migrate_raw_topics_to_normalized()
    console.print(f"[green]Migrated {migrated} topics[/green]")

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
        console=console
    ) as progress:
        task = progress.add_task("Embedding topics", total=len(topics))

        batch_size = 64
        for i in range(0, len(labels), batch_size):
            batch_labels = labels[i:i + batch_size]
            batch_ids = topic_ids[i:i + batch_size]

            embeddings = embed_texts(batch_labels)

            for topic_id, embedding in zip(batch_ids, embeddings):
                save_topic_embedding(topic_id, embedding_to_bytes(embedding))

            progress.advance(task, len(batch_labels))

    console.print(f"\n[green]Generated embeddings for {len(topics)} topics[/green]")

    # Build vector index
    console.print("\n[bold]Step 3: Building vector index...[/bold]")
    from .vector_search import get_vec_db, rebuild_vector_index
    with console.status("Rebuilding vector index..."):
        conn = get_vec_db()
        count = rebuild_vector_index(conn)
        conn.close()
    console.print(f"[green]Indexed {count} topic vectors[/green]")


@main.command("search-semantic")
@click.argument('query')
@click.option('--limit', '-n', default=20, help='Number of results')
def search_semantic(query: str, limit: int):
    """Semantic search for topics using embeddings."""
    from .vector_search import search_topics_semantic, search_books_by_topic_semantic

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
        table.add_row(
            t["label"],
            f"{t['similarity']:.3f}",
            str(t["occurrence_count"])
        )

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
            table.add_row(
                b["title"][:40],
                b["author"] or "",
                f"{b['relevance']:.3f}"
            )

        console.print(table)
    else:
        console.print("[dim]No books found with these topics[/dim]")


@main.command()
@click.option('--threshold', '-t', default=0.85, help='Similarity threshold (0.0-1.0)')
@click.option('--dry-run', is_flag=True, help='Preview without making changes')
def dedupe(threshold: float, dry_run: bool):
    """Deduplicate similar topics based on embeddings."""
    from .deduplication import deduplicate_topics, get_deduplication_preview

    if dry_run:
        console.print(f"[bold]Preview: Deduplication with threshold {threshold}[/bold]\n")
        preview = get_deduplication_preview(threshold, limit=20)

        if not preview:
            console.print("[yellow]No duplicates found[/yellow]")
            return

        for group in preview:
            console.print(f"[green]{group['canonical']}[/green] ({group['canonical_count']} occurrences)")
            for dup in group["duplicates"]:
                console.print(f"  └─ [dim]{dup['label']}[/dim] ({dup['count']})")
            console.print()

        console.print(f"[yellow]Found {len(preview)} duplicate groups. Run without --dry-run to merge.[/yellow]")
    else:
        console.print(f"[bold]Deduplicating topics with threshold {threshold}...[/bold]\n")
        result = deduplicate_topics(threshold, dry_run=False)

        console.print(f"[green]Merged {result['topics_merged']} topics into {result['duplicate_groups']} canonical topics[/green]")


@main.command()
def cluster():
    """Cluster topics using Leiden algorithm."""
    from .clustering import cluster_topics, get_cluster_summary
    from .topic_graph import compute_cooccurrences

    console.print("[bold]Step 1: Computing topic co-occurrences...[/bold]")
    with console.status("Analyzing chunk co-occurrences..."):
        cooccur_stats = compute_cooccurrences()
    console.print(f"[green]Found {cooccur_stats['cooccurrence_pairs']} co-occurrence pairs[/green]")

    console.print("\n[bold]Step 2: Clustering topics...[/bold]")
    with console.status("Running Leiden clustering..."):
        result = cluster_topics()

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return

    console.print(f"[green]Created {result['num_clusters']} clusters from {result['total_topics']} topics[/green]")
    console.print(f"[dim]Modularity: {result['modularity']:.4f}[/dim]")
    console.print(f"[dim]Cluster sizes: {result['min_cluster_size']} - {result['max_cluster_size']}[/dim]")

    # Show cluster summary
    console.print("\n[bold]Top clusters:[/bold]")
    summary = get_cluster_summary()[:10]

    for cluster in summary:
        topics_str = ", ".join([t["label"] for t in cluster["top_topics"][:3]])
        console.print(f"  [cyan]Cluster {cluster['cluster_id']}[/cyan] ({cluster['size']} topics): {topics_str}")


@main.command()
@click.argument('topic', required=False)
def tree(topic: str):
    """Browse the topic hierarchy."""
    from .clustering import get_topic_tree, get_cluster_topics

    if topic:
        # Show topics in a specific cluster/category
        from .database import get_db
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, label, cluster_id, occurrence_count
                FROM topics
                WHERE label LIKE ?
                ORDER BY occurrence_count DESC
                LIMIT 20
            """, (f"%{topic}%",))

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
            console.print("[yellow]No topic tree available. Run 'libtrails cluster' first.[/yellow]")
            return

        rich_tree = Tree("[bold]Topics[/bold]")

        for cluster in tree_data["children"][:15]:
            branch = rich_tree.add(f"[cyan]{cluster['name']}[/cyan] ({cluster['size']} topics)")
            for child in cluster.get("children", [])[:5]:
                branch.add(f"[dim]{child['name']}[/dim] ({child['count']})")

        console.print(rich_tree)


@main.command()
@click.argument('topic')
@click.option('--limit', '-n', default=10, help='Number of related topics')
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
        table.add_row(
            r["label"],
            r["connection_type"],
            f"{r['connection_weight']:.3f}"
        )

    console.print(table)


@main.command()
@click.argument('topic')
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
        cursor.execute("""
            SELECT t.label, tc.count, tc.pmi
            FROM topic_cooccurrences tc
            JOIN topics t ON (
                CASE WHEN tc.topic1_id = ? THEN tc.topic2_id ELSE tc.topic1_id END = t.id
            )
            WHERE tc.topic1_id = ? OR tc.topic2_id = ?
            ORDER BY tc.count DESC
            LIMIT 20
        """, (topic_id, topic_id, topic_id))

        results = cursor.fetchall()

        if not results:
            console.print(f"[yellow]No co-occurring topics found. Run 'libtrails cluster' first.[/yellow]")
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
    from .embeddings import embed_texts, embedding_to_bytes
    from .deduplication import deduplicate_topics
    from .clustering import cluster_topics, get_cluster_summary
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
            batch_labels = labels[i:i + batch_size]
            batch_ids = topic_ids[i:i + batch_size]
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
    console.print(f"  [green]Found {cooccur_stats['cooccurrence_pairs']} co-occurrence pairs[/green]")

    console.print("  Running Leiden clustering...")
    cluster_result = cluster_topics()
    if "error" not in cluster_result:
        console.print(f"  [green]Created {cluster_result['num_clusters']} clusters[/green]")
    else:
        console.print(f"  [red]Error: {cluster_result['error']}[/red]")

    console.print("\n[bold green]Pipeline complete![/bold green]")

    # Show final stats
    stats = get_indexing_status()
    console.print(f"\n[dim]Final stats: {stats['normalized_topics']} topics, {stats['topics_with_embeddings']} embedded, {stats['clustered_topics']} clustered[/dim]")


if __name__ == '__main__':
    main()
