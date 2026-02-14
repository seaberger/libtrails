# LibTrails Public Demo — Gutenberg Classics Library

This document captures the **one-shot setup plan** for the LibTrails public demo library: a curated set of **100 Project Gutenberg, public-domain, English-language (or English translation) works** designed to yield **legible, cross-era, cross-genre thematic clusters** in LibTrails.

It includes:

- A curated CSV (100 books)
- A polite bulk downloader script for Gutenberg EPUBs
- A Calibre library bootstrap + bulk import script
- Predicted thematic clusters you should see emerge after topic extraction + Leiden clustering

---

## Repository layout (suggested)

```text
repo/
  docs/
    gutenberg_demo_library.md        # this document
  data/
    libtrails_gutenberg_demo_books.csv
  scripts/
    download_gutenberg.py
    setup_calibre_library.py
  gutenberg_epubs/                   # created by downloader
  libtrails-demo-library/            # created by calibre setup script
```

You can keep the scripts at repo root if you prefer; the scripts accept paths via CLI flags.

---

## Requirements

### Python
- Python **3.11+**

Downloader dependency:
- `httpx` (install via `uv add httpx` or `pip install httpx`)

### Calibre
- Calibre installed locally so `calibredb` is on your `PATH`.

---

## Quick start

Assuming you’ve placed files as above:

```bash
python scripts/download_gutenberg.py --csv data/libtrails_gutenberg_demo_books.csv
python scripts/setup_calibre_library.py --csv data/libtrails_gutenberg_demo_books.csv --epub-dir gutenberg_epubs
```

---

## What the scripts do

### 1) `download_gutenberg.py`

- Reads the curated CSV (`gutenberg_id,title,author,category,year_published,rationale`)
- Downloads each EPUB from Gutenberg with **polite rate limiting** (>= 2 seconds between **every** HTTP request)
- Saves to `./gutenberg_epubs/`
- Names files as: `{gutenberg_id}_{author_lastname}_{short_title}.epub`
- Uses retries + logs success/failure
- Produces:
  - `gutenberg_epubs/download_log.jsonl`
  - `gutenberg_epubs/download_report.csv`

**EPUB URL patterns tried (in order):**
1. `https://www.gutenberg.org/ebooks/{id}.epub3.images`
2. `https://www.gutenberg.org/ebooks/{id}.epub.images`
3. `https://www.gutenberg.org/cache/epub/{id}/pg{id}-images.epub`

### 2) `setup_calibre_library.py`

- Creates a fresh Calibre library at `./libtrails-demo-library/`
- Uses `calibredb` to add all downloaded EPUBs
- Sets:
  - Title
  - Author
  - Tags: category tag + `LibTrails Demo`
  - Identifier: `gutenberg:<id>` (so you can search by Gutenberg ID)
  - Comments: stores the CSV `rationale` (useful for provenance)

---

## Curated CSV (100 books)

File: `data/libtrails_gutenberg_demo_books.csv`

Schema:

```csv
gutenberg_id,title,author,category,year_published,rationale
```

Notes:
- Some `year_published` values are **negative** for BCE approximations (Homer/Plato/etc.).
- All works are intended to be **English-language texts or English translations** available on Project Gutenberg.

### Full CSV content

```csv
gutenberg_id,title,author,category,year_published,rationale
1342,Pride and Prejudice,"Austen, Jane",Victorian British fiction,1813,"Manners, class, marriage markets; strong clustering for social mobility and gender constraint."
161,Sense and Sensibility,"Austen, Jane",Victorian British fiction,1811,Emotion vs restraint; family economics and reputation themes.
158,Emma,"Austen, Jane",Victorian British fiction,1815,"Social manipulation, self-knowledge, class boundaries; complements other marriage-plot novels."
141,Mansfield Park,"Austen, Jane",Victorian British fiction,1814,"Morality, duty, wealth, and imperial backdrop; richer ethical vocabulary for topic extraction."
1400,Great Expectations,"Dickens, Charles",Victorian British fiction,1861,"Class aspiration, guilt, transformation; bridges crime, identity, and social critique clusters."
1023,Bleak House,"Dickens, Charles",Victorian British fiction,1853,"Institutional critique (law), fog-of-bureaucracy themes; ideal for governance/justice clusters."
98,A Tale of Two Cities,"Dickens, Charles",Victorian British fiction,1859,"Revolution, sacrifice, mob politics; connects to war/political upheaval clusters."
766,David Copperfield,"Dickens, Charles",Victorian British fiction,1850,"Bildungsroman, memory, labor, family; broad theme coverage for stable clusters."
1260,Jane Eyre,"Brontë, Charlotte",Victorian British fiction,1847,"Autonomy, morality, desire vs duty; strong gender/constraint + identity themes."
768,Wuthering Heights,"Brontë, Emily",Victorian British fiction,1847,"Obsession, isolation, revenge; connects to psychological torment clusters."
145,Middlemarch,"Eliot, George",Victorian British fiction,1871,"Marriage, vocation, reform politics; high thematic density across ethics and society."
599,Vanity Fair,"Thackeray, William Makepeace",Victorian British fiction,1848,"Ambition, hypocrisy, social climbing; strengthens satire + class clusters."
583,The Woman in White,"Collins, Wilkie",Victorian British fiction,1860,"Identity, deception, legal confinement; bridges mystery with social constraint themes."
4276,North and South,"Gaskell, Elizabeth Cleghorn",Victorian British fiction,1855,"Industrial class conflict, labor/capital; anchors social critique and work/life themes."
110,Tess of the d'Urbervilles,"Hardy, Thomas",Victorian British fiction,1891,"Fate, sexuality, judgment; connects morality, gender, rural life clusters."
143,The Mayor of Casterbridge,"Hardy, Thomas",Victorian British fiction,1886,"Pride, downfall, community; complements tragedy and reputation clusters."
2852,The Hound of the Baskervilles,"Doyle, Arthur Conan",Victorian British fiction,1902,Rationalism vs superstition; detective logic links to science/unknown clusters.
174,The Picture of Dorian Gray,"Wilde, Oscar",Victorian British fiction,1890,"Hedonism, corruption, identity; gothic + moral philosophy bridge work."
15,"Moby Dick; Or, The Whale","Melville, Herman",American classics,1851,"Obsession, nature, metaphysics; bridges wilderness, madness, and the sublime."
76,Adventures of Huckleberry Finn,"Twain, Mark",American classics,1884,"Race, freedom, moral growth; key for identity and social critique clusters."
7193,The Adventures of Tom Sawyer,"Twain, Mark",American classics,1876,"Boyhood, community, mischief; contrasts with darker American themes."
33,The Scarlet Letter,"Hawthorne, Nathaniel",American classics,1850,"Shame, law, community judgment; overlaps gender constraint and moral psychology."
77,The House of the Seven Gables,"Hawthorne, Nathaniel",American classics,1851,"Ancestral guilt, property, decay; gothic-inflected American history themes."
514,Little Women,"Alcott, Louisa May",American classics,1868,"Domestic life, ambition, gender roles; strong family/constraint vocabulary."
215,The Call of the Wild,"London, Jack",American classics,1903,"Wilderness, instinct, domination; connects frontier survival with animality themes."
910,White Fang,"London, Jack",American classics,1906,"Civilization vs wild, violence, belonging; complements Call of the Wild for nature clusters."
73,The Red Badge of Courage,"Crane, Stephen",American classics,1895,"Fear, courage, war psychology; anchors war/aftermath clusters."
64317,The Great Gatsby,"Fitzgerald, F. Scott",American classics,1925,"Wealth, desire, disillusionment; sharp class/mobility themes in modern America."
4517,Ethan Frome,"Wharton, Edith",American classics,1911,"Constraint, duty, rural poverty; tight tragedy for isolation/entrapment clusters."
541,The Age of Innocence,"Wharton, Edith",American classics,1920,"Social codes, marriage, status; complements Austen/Dickens class clusters cross-era."
408,The Souls of Black Folk,"Du Bois, W. E. B.",American classics,1903,"Double consciousness, race, sociology; links identity, democracy, and justice clusters."
23,"Narrative of the Life of Frederick Douglass, an American Slave","Douglass, Frederick",American classics,1845,"Freedom, violence, literacy; strengthens oppression/liberation thematic domain."
205,"Walden, and On The Duty Of Civil Disobedience","Thoreau, Henry David",American classics,1854,"Self-reliance, nature, dissent; bridges philosophy with American wilderness."
140,The Jungle,"Sinclair, Upton",American classics,1906,"Labor, corruption, capitalism; anchors industrial exploitation clusters."
160,The Awakening,"Chopin, Kate",American classics,1899,"Desire, autonomy, social pressure; strong gender constraint themes."
148,The Autobiography of Benjamin Franklin,"Franklin, Benjamin",American classics,1791,"Self-making, virtue, pragmatism; connects ethics, industry, and civic life."
2554,Crime and Punishment,"Dostoyevsky, Fyodor",Russian literature,1866,"Guilt, redemption, moral reasoning; core psychological torment cluster."
28054,The Brothers Karamazov,"Dostoyevsky, Fyodor",Russian literature,1880,"Faith, doubt, responsibility; dense ethical and existential themes."
2638,The Idiot,"Dostoyevsky, Fyodor",Russian literature,1869,"Innocence, society, nihilism; bridges morality and social critique."
600,Notes from Underground,"Dostoyevsky, Fyodor",Russian literature,1864,"Alienation, spite, consciousness; high-signal existential vocabulary."
2600,War and Peace,"Tolstoy, Leo",Russian literature,1869,"History, war, family, meaning; anchors war/aftermath + philosophy clusters."
1399,Anna Karenina,"Tolstoy, Leo",Russian literature,1877,"Desire, duty, society; strong cross-link to Austen/Wharton social constraint."
1938,Resurrection,"Tolstoy, Leo",Russian literature,1899,"Justice, repentance, prison system; bridges law, ethics, and social reform."
47935,Fathers and Sons,"Turgenev, Ivan",Russian literature,1862,"Generational conflict, nihilism; connects ideology and family themes."
1081,Dead Souls,"Gogol, Nikolai Vasilevich",Russian literature,1842,"Bureaucracy, greed, satire; complements Dickens/Thackeray social critique."
913,A Hero of Our Time,"Lermontov, Mikhail Yuryevich",Russian literature,1840,"Byronic antihero, ennui; bridges romanticism with existential dread."
54700,Oblomov,"Goncharov, Ivan Aleksandrovich",Russian literature,1859,"Inertia, class, meaning; adds interiority and social stagnation themes."
84,"Frankenstein; Or, The Modern Prometheus","Shelley, Mary Wollstonecraft",Early sci-fi & gothic,1818,"Creation, responsibility, monstrosity; bridge between science anxiety and ethics."
345,Dracula,"Stoker, Bram",Early sci-fi & gothic,1897,"Contagion, sexuality, invasion; strong fear/otherness cluster."
42,Strange Case of Dr Jekyll and Mr Hyde,"Stevenson, Robert Louis",Early sci-fi & gothic,1886,"Duality, addiction, respectability; psychological + morality bridge."
35,The Time Machine,"Wells, H. G.",Early sci-fi & gothic,1895,"Deep time, class evolution; connects social critique to speculative futurism."
36,The War of the Worlds,"Wells, H. G.",Early sci-fi & gothic,1898,"Invasion, technology, panic; clusters with imperialism and apocalypse themes."
5230,The Invisible Man,"Wells, H. G.",Early sci-fi & gothic,1897,"Power without accountability; links to ethics, identity, and social breakdown."
159,The Island of Doctor Moreau,"Wells, H. G.",Early sci-fi & gothic,1896,"Vivisection, cruelty, personhood; bridges science, ethics, and the monstrous."
164,Twenty Thousand Leagues under the Sea,"Verne, Jules",Early sci-fi & gothic,1870,"Technology wonder, isolation at sea; connects exploration + science clusters."
18857,Journey to the Center of the Earth,"Verne, Jules",Early sci-fi & gothic,1864,"Exploration, geology, unknown worlds; feeds 'science & the unknown' cluster."
1013,The First Men in the Moon,"Wells, H. G.",Early sci-fi & gothic,1901,"Alien society, imperial analogy; complements other Wells works."
25439,"Looking Backward, 2000-1887","Bellamy, Edward",Early sci-fi & gothic,1888,"Utopian economics, equality; anchors utopia/dystopia domain."
32,Herland,"Gilman, Charlotte Perkins",Early sci-fi & gothic,1915,"Feminist utopia, social design; bridges gender constraint with utopian politics."
150,The Republic,Plato,Philosophy & essays,-380,"Justice, governance, virtue; key for power/governance thematic domain."
1600,Symposium,Plato,Philosophy & essays,-385,"Love, desire, beauty; links eros to ethics and social bonds clusters."
8438,Nicomachean Ethics,Aristotle,Philosophy & essays,-340,"Virtue ethics, flourishing; stabilizes 'good life' conceptual clusters."
2680,Meditations,Marcus Aurelius,Philosophy & essays,180,"Stoicism, resilience, self-discipline; connects to self-cultivation clusters."
1232,The Prince,"Machiavelli, Niccolò",Philosophy & essays,1532,"Realpolitik, power; anchors governance and manipulation themes."
3207,Leviathan,"Hobbes, Thomas",Philosophy & essays,1651,"State, sovereignty, fear; complements Plato/Machiavelli governance clusters."
7370,Second Treatise of Government,"Locke, John",Philosophy & essays,1689,"Rights, consent, property; supports liberal governance discourse."
59,Discourse on the Method,"Descartes, René",Philosophy & essays,1637,"Skepticism, method, mind-body; bridges epistemology to science themes."
34901,On Liberty,"Mill, John Stuart",Philosophy & essays,1859,"Freedom, harm principle; intersects politics, individuality, morality."
11224,Utilitarianism,"Mill, John Stuart",Philosophy & essays,1863,"Consequences, happiness calculus; links ethics to social policy."
4363,Beyond Good and Evil,"Nietzsche, Friedrich Wilhelm",Philosophy & essays,1886,"Morality critique, will to power; drives existential/nihilism cluster."
1998,Thus Spake Zarathustra: A Book for All and None,"Nietzsche, Friedrich Wilhelm",Philosophy & essays,1883,"Meaning-making, overcoming; complements existential and self-creation themes."
996,Don Quixote,"Cervantes Saavedra, Miguel de","World classics (French, Spanish, Italian, etc.)",1605,"Reality vs illusion, idealism; connects to identity and narrative selfhood."
8800,The Divine Comedy,Dante Alighieri,"World classics (French, Spanish, Italian, etc.)",1320,"Moral cosmology, justice; links sin/virtue themes across eras."
135,Les Misérables,"Hugo, Victor","World classics (French, Spanish, Italian, etc.)",1862,"Justice, poverty, redemption; bridges social critique and moral philosophy."
1184,The Count of Monte Cristo,"Dumas, Alexandre","World classics (French, Spanish, Italian, etc.)",1844,"Revenge, identity, power; connects to justice and transformation clusters."
1257,The Three Musketeers,"Dumas, Alexandre","World classics (French, Spanish, Italian, etc.)",1844,"Honor, loyalty, politics; links adventure to governance/social networks."
2413,Madame Bovary,"Flaubert, Gustave","World classics (French, Spanish, Italian, etc.)",1856,"Desire, boredom, social constraint; cross-links with Austen/Wharton gender themes."
44747,The Red and the Black,"Stendhal, Marie-Henri","World classics (French, Spanish, Italian, etc.)",1830,"Ambition, hypocrisy; reinforces class mobility and moral compromise."
19942,Candide,Voltaire,"World classics (French, Spanish, Italian, etc.)",1759,"Satire, optimism vs suffering; bridges philosophy with war/disaster themes."
3160,The Odyssey,Homer,"World classics (French, Spanish, Italian, etc.)",-700,"Homecoming, identity, trials; anchors epic journey and fate clusters."
2199,The Iliad,Homer,"World classics (French, Spanish, Italian, etc.)",-750,"War, honor, rage; anchors war/aftermath and heroism clusters."
228,The Aeneid,Virgil,"World classics (French, Spanish, Italian, etc.)",-19,"Empire, duty, destiny; links governance with war and founding myths."
28621,Metamorphoses,Ovid,"World classics (French, Spanish, Italian, etc.)",8,"Transformation myths; feeds identity, change, and metamorphosis clusters."
1122,"Hamlet, Prince of Denmark","Shakespeare, William",Drama,1603,"Indecision, revenge, madness; overlaps psychological torment clusters."
1533,Macbeth,"Shakespeare, William",Drama,1606,"Ambition, guilt, fate; links power and moral corruption themes."
1128,King Lear,"Shakespeare, William",Drama,1606,"Authority, family betrayal; connects governance with suffering and madness."
1127,Othello,"Shakespeare, William",Drama,1603,"Jealousy, race, manipulation; bridges identity, trust, and tragedy."
15492,A Doll's House,"Ibsen, Henrik",Drama,1879,"Gender roles, autonomy, marriage; core gender/constraint cluster."
4093,Hedda Gabler,"Ibsen, Henrik",Drama,1890,"Control, boredom, social pressure; complements psychological + constraint themes."
844,The Importance of Being Earnest,"Wilde, Oscar",Drama,1895,"Social performance, hypocrisy; sharp satire for class and identity clusters."
1754,The Cherry Orchard,"Chekhov, Anton Pavlovich",Drama,1904,"Decline, class change, memory; bridges social transition and nostalgia themes."
1228,On the Origin of Species,"Darwin, Charles",Essays & nonfiction,1859,"Evolution, natural selection; anchors science discourse and nature clusters."
944,The Voyage of the Beagle,"Darwin, Charles",Essays & nonfiction,1839,"Exploration, observation, nature; links travel narrative to scientific method."
61,The Communist Manifesto,"Marx, Karl",Essays & nonfiction,1848,"Class struggle, ideology; bridges political philosophy and labor clusters."
18,The Federalist Papers,"Hamilton, Alexander",Essays & nonfiction,1788,"Governance design, faction; strengthens power/governance thematic domain."
147,Common Sense,"Paine, Thomas",Essays & nonfiction,1776,Revolutionary argumentation; connects rhetoric and political change clusters.
3742,Rights of Man,"Paine, Thomas",Essays & nonfiction,1791,"Rights discourse, legitimacy; complements Locke/Mill governance themes."
132,The Art of War,"Sunzi, active 6th century B.C.",Essays & nonfiction,-500,"Strategy, power, conflict; bridges war and governance clusters."
815,Democracy in America — Volume 1,"Tocqueville, Alexis de",Essays & nonfiction,1835,"Democracy, equality, culture; connects governance to social psychology."
2376,Up from Slavery: An Autobiography,"Washington, Booker T.",Essays & nonfiction,1901,"Education, resilience, race; complements Douglass/Du Bois on identity and progress."
```

---

## Script: `download_gutenberg.py`

```python
#!/usr/bin/env python3
"""
download_gutenberg.py

Polite bulk downloader for Project Gutenberg EPUBs based on a curated CSV.

Usage:
  python download_gutenberg.py --csv ./libtrails_gutenberg_demo_books.csv

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
except ImportError as e:  # pragma: no cover
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
        # Prefer the first clause to avoid subtitles in filenames.
        t = self.title
        for sep in [":", ";", "—", "-", "(", "["]:
            if sep in t:
                t = t.split(sep, 1)[0].strip()
        s = slugify(t)
        return (s[:60].strip("_") or f"book_{self.gutenberg_id}")

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
                raise ValueError(f"Invalid gutenberg_id on line {i}: {row.get('gutenberg_id')!r}") from e

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
                        raise httpx.HTTPStatusError(f"Server error {status}", request=resp.request, response=resp)

                    if status != 200:
                        raise httpx.HTTPStatusError(f"Unexpected status {status}", request=resp.request, response=resp)

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

            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError, ValueError) as e:
                if attempt >= max_retries:
                    LOG.warning("FAIL id=%s title=%r url=%s err=%s", book.gutenberg_id, book.title, url, e)
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

        # next URL pattern

    return False, f"FAILED id={book.gutenberg_id} title={book.title!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="Curated CSV (columns: gutenberg_id,title,author,category,year_published,rationale)")
    ap.add_argument("--out-dir", type=Path, default=Path("./gutenberg_epubs"), help="Output directory for EPUBs")
    ap.add_argument("--force", action="store_true", help="Re-download even if file already exists")
    ap.add_argument("--min-delay", type=float, default=2.0, help="Minimum delay (seconds) between every request (politeness)")
    ap.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds")
    ap.add_argument("--retries", type=int, default=3, help="Max retries per URL pattern")
    ap.add_argument("--user-agent", type=str, default="LibTrailsDemo/1.0 (+https://example.com; contact: you@example.com)", help="User-Agent header")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for testing (0 = no limit)")
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
```

---

## Script: `setup_calibre_library.py`

```python
#!/usr/bin/env python3
"""
setup_calibre_library.py

Creates a fresh Calibre library and imports the downloaded Gutenberg EPUBs,
setting metadata/tags from the curated CSV.

Usage:
  python setup_calibre_library.py --csv ./libtrails_gutenberg_demo_books.csv --epub-dir ./gutenberg_epubs

Requirements:
- Calibre installed locally so `calibredb` is available on PATH.
  https://calibre-ebook.com/

Design:
- Idempotent-ish: uses `calibredb add --automerge ignore` and sets a Gutenberg identifier.
  Re-running should not create duplicates if title/author are consistent.
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


def ensure_calibre_available() -> None:
    if shutil.which("calibredb") is None:
        raise SystemExit(
            "Could not find `calibredb` on PATH.\n"
            "Install Calibre from https://calibre-ebook.com/ and ensure calibredb is available.\n"
        )


def ensure_library(library_dir: Path) -> None:
    library_dir.mkdir(parents=True, exist_ok=True)
    # Per Calibre docs, `calibredb --with-library <dir> list` will initialize a new library if needed.
    # We use --limit 0 to keep output tiny.
    run_cmd(["calibredb", "--with-library", str(library_dir), "list", "--limit", "0"], check=True)


def find_epub(epub_dir: Path, gutenberg_id: int) -> Optional[Path]:
    matches = sorted(epub_dir.glob(f"{gutenberg_id}_*.epub"))
    if matches:
        return matches[0]
    # Fallback: any epub whose name starts with the id (rare)
    matches = sorted(epub_dir.glob(f"{gutenberg_id}*.epub"))
    return matches[0] if matches else None


def tags_for(book: BookRow) -> list[str]:
    tags = ["LibTrails Demo", book.category]

    fiction_cats = {
        "Victorian British fiction",
        "American classics",
        "Russian literature",
        "Early sci-fi & gothic",
        "World classics (French, Spanish, Italian, etc.)",
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

    # De-dupe but preserve order
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        t = t.strip()
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def parse_added_book_id(output: str) -> Optional[int]:
    """
    calibredb add output commonly includes lines like:
      Added book ids: 123
    or merge messages depending on --automerge.
    We'll grab the last integer in the output as a best-effort.
    """
    nums = re.findall(r"\b(\d+)\b", output)
    return int(nums[-1]) if nums else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="Curated CSV used for metadata.")
    ap.add_argument("--epub-dir", type=Path, default=Path("./gutenberg_epubs"), help="Directory containing downloaded EPUBs.")
    ap.add_argument("--library-dir", type=Path, default=Path("./libtrails-demo-library"), help="Target Calibre library directory.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    ensure_calibre_available()
    ensure_library(args.library_dir)

    books = read_books(args.csv)

    ok = 0
    missing_files = 0
    failed = 0

    for idx, book in enumerate(books, start=1):
        epub_path = find_epub(args.epub_dir, book.gutenberg_id)
        if epub_path is None:
            missing_files += 1
            LOG.error("[%s/%s] Missing EPUB for gutenberg_id=%s (expected in %s)", idx, len(books), book.gutenberg_id, args.epub_dir)
            continue

        tags = ",".join(tags_for(book))

        add_cmd = [
            "calibredb",
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

            # Add rationale as comments (nice-to-have for UI/debug).
            # Uses `calibredb set_metadata --field` so no OPF file needed.
            if book_id is not None and book.rationale:
                set_cmd = [
                    "calibredb",
                    "--with-library",
                    str(args.library_dir),
                    "set_metadata",
                    str(book_id),
                    "--field",
                    f"comments:{book.rationale}",
                ]
                run_cmd(set_cmd, check=False)

            ok += 1
            LOG.info("[%s/%s] Imported gutenberg_id=%s as calibre_id=%s", idx, len(books), book.gutenberg_id, book_id)

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

    # Summary
    LOG.info("Import complete. OK=%s missing_epub=%s failed=%s", ok, missing_files, failed)

    # Print a quick library summary as JSON for machine readability.
    try:
        proc = run_cmd(
            ["calibredb", "--with-library", str(args.library_dir), "list", "--fields", "id,title,authors,tags,identifiers", "--for-machine"],
            check=True,
        )
        data = json.loads(proc.stdout)
        LOG.info("Library now contains %s books.", len(data))
        # Print first few lines
        for row in data[:5]:
            LOG.info("Sample: id=%s title=%r authors=%r tags=%r", row.get("id"), row.get("title"), row.get("authors"), row.get("tags"))
    except Exception as e:
        LOG.warning("Could not read library summary via calibredb list --for-machine (%s)", e)

    return 0 if failed == 0 and missing_files == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## Expected clustering outcomes (10–15 likely Leiden clusters)

These are the clusters that should “snap into place” quickly given the selection mix and the presence of several **bridge works** (e.g., *Frankenstein*, utopian novels, Wells, Darwin, political philosophy):

1. **Class, manners, and marriage markets**  
   Austen, Eliot (*Middlemarch*), Gaskell, Wharton, Flaubert

2. **Ambition, social climbing, hypocrisy**  
   Thackeray, Dickens (*Great Expectations*), Fitzgerald, Stendhal, Wilde

3. **Gender constraint and autonomy**  
   Brontës, Hardy (*Tess*), Chopin, Ibsen, Gilman

4. **Crime, guilt, confession, redemption**  
   Dostoyevsky, Hugo, Dickens, Shakespearean tragedy

5. **Existential interiority and alienation**  
   *Notes from Underground*, Lermontov, Goncharov, Chekhov, Shakespeare (*Hamlet*)

6. **Institutions: law, bureaucracy, the state**  
   *Bleak House*, *Dead Souls*, *Resurrection*, Hobbes, Federalist Papers, Tocqueville

7. **Revolution, legitimacy, political rupture**  
   *A Tale of Two Cities*, Paine, Marx, Machiavelli

8. **War psychology, honor, and aftermath**  
   Tolstoy, Crane, Homer, Virgil, Shakespeare

9. **Wilderness, survival, and the nonhuman**  
   Melville, Thoreau, London, Darwin travel narrative

10. **Science, evolution, and “the unknown”**  
   Darwin, Verne, Wells, “mad science” adjacency via *Moreau*

11. **Utopia / social engineering / speculative political economy**  
   Bellamy, Gilman, Wells (*Time Machine*), Plato (*Republic* as attractor)

12. **Monstrosity, contagion, invasion, fear of the Other**  
   *Frankenstein*, *Dracula*, Wells invasion/visibility/identity works, Stevenson

13. **Identity, masks, reinvention**  
   *Monte Cristo*, *Gatsby*, *Dorian Gray*, *Huck Finn*, *Great Expectations*

14. **Satire as epistemology**  
   Voltaire, Thackeray, Gogol, Wilde

15. **Journeys, trials, and homecoming myth-structures**  
   Homeric epics, Cervantes, Verne explorations, Darwin voyages

---

## Troubleshooting

- **HTTP 403 / 429 / network flakiness:**  
  The downloader uses retries and enforces a minimum 2s delay per request. If you still see blocks, increase delay:
  ```bash
  python scripts/download_gutenberg.py --csv data/libtrails_gutenberg_demo_books.csv --min-delay 3.0
  ```

- **Calibre not found (`calibredb` missing):**  
  Install Calibre and ensure `calibredb` is on PATH. You can verify:
  ```bash
  calibredb --version
  ```

- **Duplicate imports:**  
  The Calibre script uses `--automerge ignore` and sets a Gutenberg identifier. If you reorganize titles/authors, reruns may add duplicates; wipe the library folder and rerun for a clean rebuild.

---

## Provenance

- Gutenberg catalog feed: `https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz`
- Gutenberg book page pattern: `https://www.gutenberg.org/ebooks/<id>`
