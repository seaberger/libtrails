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
  community_id: number | null;
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
  match_type?: string;
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

export interface CommunityRef {
  community_id: number;
  label: string;
  topic_count: number;
  book_count: number;
}

export interface DomainSummary {
  domain_id: number;
  label: string;
  cluster_count: number;
  book_count: number;
  primary_book_count: number;
  community_count: number;
  sample_books: BookSummary[];
  top_clusters: ClusterInfo[];
  top_communities: CommunityRef[];
}

export interface DomainBook extends BookSummary {
  concentration: number;
  is_primary: boolean;
}

export interface DomainDetail {
  domain_id: number;
  label: string;
  cluster_count: number;
  clusters: ClusterInfo[];
  books: DomainBook[];
}

export interface CommunitySummary {
  community_id: number;
  label: string;
  topic_count: number;
  book_count: number;
  primary_book_count: number;
  domain_id: number | null;
  domain_label: string;
  sample_books: BookSummary[];
}

export interface CommunityBook extends BookSummary {
  concentration: number;
  is_primary: boolean;
}

export interface CommunityDetail {
  community_id: number;
  label: string;
  topic_count: number;
  domain_id: number | null;
  domain_label: string;
  clusters: ClusterInfo[];
  books: CommunityBook[];
}

export interface UniverseCommunity {
  community_id: number;
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
  communities: UniverseCommunity[];
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

export interface HybridCommunityResult {
  community_id: number;
  label: string;
  topic_count: number;
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
  community_id: number;
  score: number;
}

export interface HybridSearchResponse {
  query: string;
  scope: string;
  total: number;
  timing_ms: number;
  books: HybridBookResult[];
  clusters: HybridClusterResult[];
  communities: HybridCommunityResult[];
  domains: HybridDomainResult[];
  universe: UniverseSearchResult[];
}
