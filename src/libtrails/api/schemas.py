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
    community_id: int | None = None


class ThemeRef(BaseModel):
    """Community reference with label for display."""

    cluster_id: int
    community_id: int | None = None
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

    shared_topics: int = 0
    similarity: float
    match_type: str = ""


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


class CommunityRef(BaseModel):
    """Community reference for domain listings."""

    community_id: int
    label: str
    topic_count: int
    book_count: int = 0


class DomainSummary(BaseModel):
    """Brief domain (super-cluster) info for lists."""

    domain_id: int
    label: str
    cluster_count: int
    book_count: int
    primary_book_count: int = 0
    community_count: int = 0
    sample_books: list[BookSummary] = []
    top_clusters: list[dict] = []
    top_communities: list[CommunityRef] = []


class DomainBook(BookSummary):
    """Book with domain-specific concentration and membership info."""

    concentration: float
    is_primary: bool


class DomainDetail(BaseModel):
    """Full domain info with all clusters and books."""

    domain_id: int
    label: str
    cluster_count: int
    clusters: list[dict] = []
    books: list[DomainBook] = []


class CommunitySummary(BaseModel):
    """Brief community info for lists."""

    community_id: int
    label: str
    topic_count: int
    cluster_count: int = 0
    book_count: int
    primary_book_count: int = 0
    domain_id: int | None = None
    domain_label: str = ""
    sample_books: list[BookSummary] = []


class CommunityBook(BookSummary):
    """Book with community-specific concentration and membership info."""

    concentration: float
    is_primary: bool


class CommunityDetail(BaseModel):
    """Full community info with clusters and books."""

    community_id: int
    label: str
    topic_count: int
    domain_id: int | None = None
    domain_label: str = ""
    clusters: list[ClusterInfo] = []
    books: list[CommunityBook] = []


class UniverseCommunity(BaseModel):
    """A community positioned in the 3D galaxy map."""

    community_id: int
    label: str
    size: int
    book_count: int
    book_ids: list[int] = []
    domain_id: int
    domain_label: str
    x: float
    y: float
    z: float
    cluster_count: int = 0
    top_clusters: list[str] = []
    top_topics: list[str] = []


class UniverseDomain(BaseModel):
    """Domain with its display color."""

    domain_id: int
    label: str
    color: str


class UniverseData(BaseModel):
    """Full galaxy visualization payload."""

    communities: list[UniverseCommunity]
    domains: list[UniverseDomain]


# ── Hybrid search schemas ──


class HybridBookResult(BaseModel):
    """Book result from hybrid search."""

    book_id: int
    title: str
    author: str
    calibre_id: int | None = None
    score: float
    match_type: str  # "keyword", "content", "semantic", "topic"


class HybridClusterResult(BaseModel):
    """Cluster result from hybrid search."""

    cluster_id: int
    label: str
    size: int
    book_count: int
    score: float
    sample_books: list[BookSummary] = []


class HybridCommunityResult(BaseModel):
    """Community result from hybrid search."""

    community_id: int
    label: str
    topic_count: int
    book_count: int
    score: float
    sample_books: list[BookSummary] = []


class HybridDomainResult(BaseModel):
    """Domain result from hybrid search."""

    domain_id: int
    label: str
    score: float
    matching_clusters: int


class UniverseSearchResult(BaseModel):
    """Minimal community hit for universe highlighting."""

    community_id: int
    score: float


class HybridSearchResponse(BaseModel):
    """Unified hybrid search response."""

    query: str
    scope: str
    total: int
    timing_ms: int
    books: list[HybridBookResult] = []
    clusters: list[HybridClusterResult] = []
    communities: list[HybridCommunityResult] = []
    domains: list[HybridDomainResult] = []
    universe: list[UniverseSearchResult] = []
