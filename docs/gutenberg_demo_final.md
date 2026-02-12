# Gutenberg Demo Library

100 Project Gutenberg classics in a separate Calibre library and LibTrails DB, designed to produce rich Leiden clusters for the website demo.

## Quick Start

```bash
# 1. Download EPUBs (polite: 2s between requests, ~4 min)
python scripts/download_gutenberg.py --csv data/gutenberg_demo_books.csv

# 2. Create Calibre library
python scripts/setup_calibre_library.py \
  --csv data/gutenberg_demo_books.csv \
  --epub-dir gutenberg_epubs \
  --library-dir ~/Calibre_Demo_Library

# 3. Create LibTrails demo DB
python scripts/create_demo_db.py --library-dir ~/Calibre_Demo_Library

# 4. Index all 100 books
CALIBRE_LIBRARY_PATH=~/Calibre_Demo_Library \
LIBTRAILS_DB=demo \
uv run libtrails index --all \
  --theme-model lm_studio/google/gemma-3-27b \
  --chunk-model lm_studio/google/gemma-3-12b

# 5. Post-processing
CALIBRE_LIBRARY_PATH=~/Calibre_Demo_Library \
LIBTRAILS_DB=demo \
uv run libtrails process
```

## Requirements

- Python 3.11+
- `httpx` (`uv add httpx` or `pip install httpx`) for the downloader
- Calibre installed with `calibredb` on PATH

## Design Principles

1. **Recognizable titles** — website visitors should see books they know
2. **Thematic bridge density** — books that connect across categories for rich Leiden clusters
3. **Author diversity** — max 4 per author, each thematically distinct
4. **Era span** — Homer (~800 BC) to Fitzgerald (1925)
5. **Genre balance** — fiction, philosophy, drama, nonfiction

## Book List (100 titles, 8 categories)

### Victorian British Fiction (15)

| ID | Title | Author | Year |
|------|-------|--------|------|
| 1342 | Pride and Prejudice | Jane Austen | 1813 |
| 161 | Sense and Sensibility | Jane Austen | 1811 |
| 1400 | Great Expectations | Charles Dickens | 1861 |
| 98 | A Tale of Two Cities | Charles Dickens | 1859 |
| 1023 | Bleak House | Charles Dickens | 1853 |
| 1260 | Jane Eyre | Charlotte Bronte | 1847 |
| 768 | Wuthering Heights | Emily Bronte | 1847 |
| 145 | Middlemarch | George Eliot | 1872 |
| 599 | Vanity Fair | W. M. Thackeray | 1848 |
| 583 | The Woman in White | Wilkie Collins | 1860 |
| 4276 | North and South | Elizabeth Gaskell | 1855 |
| 110 | Tess of the d'Urbervilles | Thomas Hardy | 1891 |
| 143 | The Mayor of Casterbridge | Thomas Hardy | 1886 |
| 174 | The Picture of Dorian Gray | Oscar Wilde | 1890 |
| 2852 | The Hound of the Baskervilles | A. C. Doyle | 1902 |

### American Classics (16)

| ID | Title | Author | Year |
|------|-------|--------|------|
| 2701 | Moby Dick | Herman Melville | 1851 |
| 76 | Adventures of Huckleberry Finn | Mark Twain | 1884 |
| 25344 | The Scarlet Letter | Nathaniel Hawthorne | 1850 |
| 77 | The House of the Seven Gables | Nathaniel Hawthorne | 1851 |
| 64317 | The Great Gatsby | F. Scott Fitzgerald | 1925 |
| 215 | The Call of the Wild | Jack London | 1903 |
| 910 | White Fang | Jack London | 1906 |
| 73 | The Red Badge of Courage | Stephen Crane | 1895 |
| 160 | The Awakening | Kate Chopin | 1899 |
| 541 | The Age of Innocence | Edith Wharton | 1920 |
| 4517 | Ethan Frome | Edith Wharton | 1911 |
| 514 | Little Women | Louisa May Alcott | 1868 |
| 140 | The Jungle | Upton Sinclair | 1906 |
| 233 | Sister Carrie | Theodore Dreiser | 1900 |
| 205 | Walden | Henry David Thoreau | 1854 |
| 23 | Narrative of Frederick Douglass | Frederick Douglass | 1845 |

### Russian Literature (10)

| ID | Title | Author | Year |
|------|-------|--------|------|
| 2554 | Crime and Punishment | Fyodor Dostoevsky | 1866 |
| 28054 | The Brothers Karamazov | Fyodor Dostoevsky | 1880 |
| 600 | Notes from Underground | Fyodor Dostoevsky | 1864 |
| 2638 | The Idiot | Fyodor Dostoevsky | 1869 |
| 2600 | War and Peace | Leo Tolstoy | 1869 |
| 1399 | Anna Karenina | Leo Tolstoy | 1877 |
| 1938 | Resurrection | Leo Tolstoy | 1899 |
| 47935 | Fathers and Sons | Ivan Turgenev | 1862 |
| 1081 | Dead Souls | Nikolai Gogol | 1842 |
| 913 | A Hero of Our Time | Mikhail Lermontov | 1840 |

### Early Sci-Fi & Gothic (13)

| ID | Title | Author | Year |
|------|-------|--------|------|
| 84 | Frankenstein | Mary Shelley | 1818 |
| 345 | Dracula | Bram Stoker | 1897 |
| 43 | Jekyll and Hyde | Robert Louis Stevenson | 1886 |
| 35 | The Time Machine | H. G. Wells | 1895 |
| 36 | The War of the Worlds | H. G. Wells | 1898 |
| 159 | The Island of Doctor Moreau | H. G. Wells | 1896 |
| 5230 | The Invisible Man | H. G. Wells | 1897 |
| 164 | Twenty Thousand Leagues Under the Sea | Jules Verne | 1870 |
| 18857 | Journey to the Center of the Earth | Jules Verne | 1864 |
| 2147 | Works of Edgar Allan Poe, Vol. 1 | Edgar Allan Poe | ~1840s |
| 624 | Looking Backward, 2000-1887 | Edward Bellamy | 1888 |
| 32 | Herland | Charlotte Perkins Gilman | 1915 |
| 120 | Treasure Island | Robert Louis Stevenson | 1883 |

### Philosophy & Essays (12)

| ID | Title | Author | Year |
|------|-------|--------|------|
| 1497 | The Republic | Plato | ~380 BC |
| 1600 | Symposium | Plato | ~385 BC |
| 8438 | Nicomachean Ethics | Aristotle | ~340 BC |
| 2680 | Meditations | Marcus Aurelius | ~180 |
| 1232 | The Prince | Machiavelli | 1532 |
| 3207 | Leviathan | Thomas Hobbes | 1651 |
| 7370 | Second Treatise of Government | John Locke | 1689 |
| 34901 | On Liberty | J. S. Mill | 1859 |
| 11224 | Utilitarianism | J. S. Mill | 1863 |
| 4363 | Beyond Good and Evil | Friedrich Nietzsche | 1886 |
| 1998 | Thus Spake Zarathustra | Friedrich Nietzsche | 1885 |
| 2944 | Essays, First Series | Ralph Waldo Emerson | 1841 |

### World Classics (14)

| ID | Title | Author | Year |
|------|-------|--------|------|
| 8800 | The Divine Comedy | Dante Alighieri | ~1320 |
| 996 | Don Quixote | Miguel de Cervantes | 1605 |
| 135 | Les Miserables | Victor Hugo | 1862 |
| 1184 | The Count of Monte Cristo | Alexandre Dumas | 1844 |
| 1257 | The Three Musketeers | Alexandre Dumas | 1844 |
| 2413 | Madame Bovary | Gustave Flaubert | 1857 |
| 1237 | Father Goriot | Honore de Balzac | 1835 |
| 44747 | The Red and the Black | Stendhal | 1830 |
| 19942 | Candide | Voltaire | 1759 |
| 1727 | The Odyssey | Homer | ~800 BC |
| 2199 | The Iliad | Homer | ~800 BC |
| 228 | The Aeneid | Virgil | ~19 BC |
| 14591 | Faust, Part 1 | Goethe | 1808 |
| 28621 | Metamorphoses | Ovid | ~8 AD |

### Drama (8)

| ID | Title | Author | Year |
|------|-------|--------|------|
| 1524 | Hamlet | Shakespeare | ~1601 |
| 1533 | Macbeth | Shakespeare | ~1606 |
| 1128 | King Lear | Shakespeare | ~1606 |
| 1127 | Othello | Shakespeare | ~1603 |
| 2542 | A Doll's House | Henrik Ibsen | 1879 |
| 4093 | Hedda Gabler | Henrik Ibsen | 1891 |
| 844 | The Importance of Being Earnest | Oscar Wilde | 1895 |
| 1754 | The Cherry Orchard | Anton Chekhov | 1904 |

### Essays & Nonfiction (12)

| ID | Title | Author | Year |
|------|-------|--------|------|
| 1228 | On the Origin of Species | Charles Darwin | 1859 |
| 944 | The Voyage of the Beagle | Charles Darwin | 1839 |
| 147 | Common Sense | Thomas Paine | 1776 |
| 3742 | Rights of Man | Thomas Paine | 1791 |
| 3420 | A Vindication of the Rights of Woman | Mary Wollstonecraft | 1792 |
| 61 | The Communist Manifesto | Karl Marx | 1848 |
| 18 | The Federalist Papers | Alexander Hamilton | 1788 |
| 408 | The Souls of Black Folk | W. E. B. Du Bois | 1903 |
| 2376 | Up from Slavery | Booker T. Washington | 1901 |
| 132 | The Art of War | Sun Tzu | ~500 BC |
| 3300 | The Wealth of Nations | Adam Smith | 1776 |
| 3600 | Essays of Montaigne | Michel de Montaigne | 1580 |

## Expected Leiden Clusters (12-15 communities)

1. **Marriage, class, social mobility** — Austen, Eliot, Wharton, Flaubert, Stendhal
2. **Crime, guilt, redemption** — Dostoevsky, Hugo, Shakespeare tragedies
3. **Gender constraint & autonomy** — Brontes, Hardy, Chopin, Ibsen, Gilman, Wollstonecraft
4. **Existential interiority & alienation** — Notes from Underground, Lermontov, Chekhov, Hamlet
5. **Social critique, labor, poverty** — Dickens, Gaskell, Sinclair, Dreiser, Marx
6. **Political philosophy & governance** — Plato, Hobbes, Locke, Mill, Machiavelli, Federalist Papers
7. **War, honor, heroism** — Homer, Virgil, Tolstoy, Crane, Shakespeare
8. **Science, evolution, the unknown** — Darwin, Wells, Verne, Bellamy
9. **Gothic horror & monstrosity** — Shelley, Stoker, Stevenson, Poe, Wells (Moreau)
10. **Adventure, journey, exploration** — Melville, Verne, Dumas, Cervantes, Stevenson
11. **Identity, transformation, masks** — Wilde, Fitzgerald, Dumas (Monte Cristo), Ovid, Hawthorne
12. **Race, freedom, justice** — Douglass, Du Bois, Washington, Twain
13. **Wilderness, nature, self-reliance** — Thoreau, Emerson, London, Darwin (Beagle)
14. **Utopia & social engineering** — Bellamy, Gilman, Plato (Republic), Wells (Time Machine)

## Estimated Processing Time

- ~100 books x ~200 chunks avg = ~20,000 chunks
- At ~1.5s/chunk (gemma-3-12b) = ~8.3 hours
- With themes pass: ~9 hours total

## Verification

After running the full pipeline:
1. Check cluster count is 12-15 (not 5 and not 50)
2. Verify expected thematic clusters appear
3. Spot-check topic quality on a few books
4. Generate domain labels and validate against predicted outcomes

## Files

| File | Purpose |
|------|---------|
| `data/gutenberg_demo_books.csv` | Final 100-book CSV with IDs, metadata, rationales |
| `scripts/download_gutenberg.py` | Polite EPUB downloader |
| `scripts/setup_calibre_library.py` | Calibre library import |
| `scripts/create_demo_db.py` | LibTrails DB bootstrap |
| `src/libtrails/config.py` | `CALIBRE_LIBRARY_PATH` env var + `demo` DB option |

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `CALIBRE_LIBRARY_PATH` | Override Calibre library location | `~/Calibre_Demo_Library` |
| `LIBTRAILS_DB` | Select database (`v2`, `demo`) | `demo` |
