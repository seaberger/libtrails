// API client for LibTrails backend

import type {
  BookDetail,
  BookSummary,
  CommunityDetail,
  CommunitySummary,
  DomainDetail,
  DomainSummary,
  HybridSearchResponse,
  RelatedBook,
  SearchResult,
  UniverseData,
} from "./types";

// During SSR, fetch directly from the backend.
// On client, use the base path (e.g., /libtrails in production, / in dev).
const isServer = typeof window === "undefined";
const basePath = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");
const API_BASE_URL = isServer ? "http://localhost:8000" : basePath;
const API_BASE = `${API_BASE_URL}/api/v1`;

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export async function getBooks(
  page = 1,
  pageSize = 50,
  indexedOnly = true
): Promise<BookSummary[]> {
  return fetchJson(
    `/books?page=${page}&page_size=${pageSize}&indexed_only=${indexedOnly}`
  );
}

export async function getBook(bookId: number): Promise<BookDetail> {
  return fetchJson(`/books/${bookId}`);
}

export async function getBooksBatch(bookIds: number[]): Promise<BookSummary[]> {
  const response = await fetch(`${API_BASE}/books/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bookIds),
  });
  if (!response.ok) throw new Error(`API error: ${response.status}`);
  return response.json();
}

export async function getRelatedBooks(
  bookId: number,
  limit = 10
): Promise<RelatedBook[]> {
  return fetchJson(`/books/${bookId}/related?limit=${limit}`);
}

export async function searchBooks(
  query: string,
  limit = 20
): Promise<SearchResult[]> {
  return fetchJson(`/search?q=${encodeURIComponent(query)}&limit=${limit}`);
}

export function getCoverUrl(calibreId: number | null): string {
  if (!calibreId) return `${basePath}/placeholder-cover.svg`;
  return `${basePath}/api/v1/covers/${calibreId}`;
}

export function getBookCoverUrl(bookId: number): string {
  return `${basePath}/api/v1/covers/book/${bookId}`;
}

// Community (mid-tier) API
export async function getCommunities(
  domainId?: number
): Promise<CommunitySummary[]> {
  const params = domainId != null ? `?domain_id=${domainId}` : "";
  return fetchJson(`/communities${params}`);
}

export async function getCommunity(
  communityId: number
): Promise<CommunityDetail> {
  return fetchJson(`/communities/${communityId}`);
}

// Domain (super-cluster) API
export async function getDomains(): Promise<DomainSummary[]> {
  return fetchJson("/domains");
}

export async function getDomain(domainId: number): Promise<DomainDetail> {
  return fetchJson(`/domains/${domainId}`);
}

// Universe (galaxy visualization) API
export async function getUniverse(): Promise<UniverseData> {
  return fetchJson("/universe");
}

// Hybrid search API
export async function searchHybrid(
  query: string,
  scope: "books" | "clusters" | "communities" | "domains" | "universe",
  limit = 20
): Promise<HybridSearchResponse> {
  return fetchJson(
    `/search/hybrid?q=${encodeURIComponent(query)}&scope=${scope}&limit=${limit}`
  );
}
