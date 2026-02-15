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


class ThemeRef(BaseModel):
    """Cluster reference with label for display."""

    cluster_id: int
    label: str


class BookDetail(BookSummary):
    """Full book info with topics."""

    description: str | None = None
    gutenberg_url: str | None = None
    topics: list[TopicInfo] = []
    themes: list[ThemeRef] = []
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


class ClusterInfo(BaseModel):
    """Brief cluster info for domain listings."""

    cluster_id: int
    label: str
    size: int
    book_count: int | None = None


class DomainSummary(BaseModel):
    """Brief domain (super-cluster) info for lists."""

    domain_id: int
    label: str
    cluster_count: int
    book_count: int
    sample_books: list[BookSummary] = []
    top_clusters: list[dict] = []


class DomainDetail(BaseModel):
    """Full domain info with all clusters and books."""

    domain_id: int
    label: str
    cluster_count: int
    clusters: list[dict] = []
    books: list[BookSummary] = []


class UniverseCluster(BaseModel):
    """A cluster positioned in the 3D galaxy map."""

    cluster_id: int
    label: str
    size: int
    book_count: int
    book_ids: list[int] = []
    domain_id: int
    domain_label: str
    x: float
    y: float
    z: float
    top_topics: list[str] = []


class UniverseDomain(BaseModel):
    """Domain with its display color."""

    domain_id: int
    label: str
    color: str


class UniverseData(BaseModel):
    """Full galaxy visualization payload."""

    clusters: list[UniverseCluster]
    domains: list[UniverseDomain]
