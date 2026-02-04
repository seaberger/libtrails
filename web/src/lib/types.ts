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

export interface BookDetail extends BookSummary {
  description: string | null;
  topics: TopicInfo[];
  theme_ids: number[];
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
