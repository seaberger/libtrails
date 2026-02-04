// API client for LibTrails backend

import type {
  BookDetail,
  BookSummary,
  RelatedBook,
  SearchResult,
  ThemeDetail,
  ThemeSummary,
} from "./types";

// During SSR, we need the full URL. On client, relative works via proxy.
const isServer = typeof window === "undefined";
const API_BASE_URL = isServer ? "http://localhost:8000" : "";
const API_BASE = `${API_BASE_URL}/api/v1`;

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export async function getThemes(
  limit = 50,
  minBooks = 2
): Promise<ThemeSummary[]> {
  return fetchJson(`/themes?limit=${limit}&min_books=${minBooks}`);
}

export async function getTheme(clusterId: number): Promise<ThemeDetail> {
  return fetchJson(`/themes/${clusterId}`);
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
  if (!calibreId) return "/placeholder-cover.svg";
  return `/api/v1/covers/${calibreId}`;
}

export function getBookCoverUrl(bookId: number): string {
  return `/api/v1/covers/book/${bookId}`;
}
