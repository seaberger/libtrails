"""Pydantic schemas for API responses."""

from pydantic import BaseModel


class BookSummary(BaseModel):
    """Brief book info for lists."""

    id: int
    title: str
    author: str
    calibre_id: int | None = None


class TopicInfo(BaseModel):
    """Topic with occurrence count."""

    id: int
    label: str
    count: int
    cluster_id: int | None = None


class BookDetail(BookSummary):
    """Full book info with topics."""

    description: str | None = None
    topics: list[TopicInfo] = []
    theme_ids: list[int] = []
    chunk_count: int = 0


class ThemeSummary(BaseModel):
    """Brief theme info for lists."""

    cluster_id: int
    label: str
    size: int
    book_count: int
    sample_books: list[BookSummary] = []


class ThemeDetail(BaseModel):
    """Full theme info with all books."""

    cluster_id: int
    label: str
    size: int
    topics: list[TopicInfo] = []
    books: list[BookSummary] = []


class RelatedBook(BookSummary):
    """Book with similarity score."""

    shared_topics: int
    similarity: float


class SearchResult(BaseModel):
    """Search result with score."""

    book: BookSummary
    score: float
    match_type: str  # "keyword" or "semantic"


class PaginatedResponse(BaseModel):
    """Generic paginated response."""

    items: list
    total: int
    page: int
    page_size: int
    total_pages: int
