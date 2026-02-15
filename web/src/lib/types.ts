// TypeScript interfaces matching the API schemas

export interface BookSummary {
  id: number;
  title: string;
  author: string;
  calibre_id: number | null;
}

export interface TopicInfo {
  id: number;
  label: string;
  count: number;
  cluster_id: number | null;
}

export interface ThemeRef {
  cluster_id: number;
  label: string;
}

export interface BookDetail extends BookSummary {
  description: string | null;
  gutenberg_url: string | null;
  topics: TopicInfo[];
  themes: ThemeRef[];
  chunk_count: number;
}

export interface ThemeSummary {
  cluster_id: number;
  label: string;
  size: number;
  book_count: number;
  sample_books: BookSummary[];
}

export interface ThemeDetail {
  cluster_id: number;
  label: string;
  size: number;
  topics: TopicInfo[];
  books: BookSummary[];
}

export interface RelatedBook extends BookSummary {
  shared_topics: number;
  similarity: number;
}

export interface SearchResult {
  book: BookSummary;
  score: number;
  match_type: "keyword" | "semantic";
}

export interface ClusterInfo {
  cluster_id: number;
  label: string;
  size: number;
  book_count?: number;
}

export interface DomainSummary {
  domain_id: number;
  label: string;
  cluster_count: number;
  book_count: number;
  sample_books: BookSummary[];
  top_clusters: ClusterInfo[];
}

export interface DomainDetail {
  domain_id: number;
  label: string;
  cluster_count: number;
  clusters: ClusterInfo[];
  books: BookSummary[];
}

export interface UniverseCluster {
  cluster_id: number;
  label: string;
  size: number;
  book_count: number;
  book_ids: number[];
  domain_id: number;
  domain_label: string;
  x: number;
  y: number;
  z: number;
  top_topics: string[];
}

export interface UniverseDomain {
  domain_id: number;
  label: string;
  color: string;
}

export interface UniverseData {
  clusters: UniverseCluster[];
  domains: UniverseDomain[];
}

// Hybrid search types

export interface HybridBookResult {
  book_id: number;
  title: string;
  author: string;
  calibre_id: number | null;
  score: number;
  match_type: "keyword" | "content" | "semantic" | "topic";
}

export interface HybridClusterResult {
  cluster_id: number;
  label: string;
  size: number;
  book_count: number;
  score: number;
  sample_books: BookSummary[];
}

export interface HybridDomainResult {
  domain_id: number;
  label: string;
  score: number;
  matching_clusters: number;
}

export interface UniverseSearchResult {
  cluster_id: number;
  score: number;
}

export interface HybridSearchResponse {
  query: string;
  scope: string;
  total: number;
  timing_ms: number;
  books: HybridBookResult[];
  clusters: HybridClusterResult[];
  domains: HybridDomainResult[];
  universe: UniverseSearchResult[];
}
