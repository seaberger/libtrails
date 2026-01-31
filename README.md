# libtrails

Trail-finding across your Calibre book library.

## Installation

```bash
uv pip install -e .
```

## Usage

```bash
# Check indexing status
libtrails status

# Index a single book
libtrails index 123
libtrails index --title "Some Book"

# Index all books
libtrails index --all

# View topics for a book
libtrails topics 123

# Search by topic
libtrails search "philosophy"

# List available Ollama models
libtrails models
```
