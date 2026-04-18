// Match-badge vocabulary for hybrid-search results.
//
// The backend's hybrid_search pipeline fuses multiple retrieval signals via
// Reciprocal Rank Fusion. For each result, `match_type` names the signal where
// the result achieved its best rank — the "dominant signal" that explains why
// the result landed where it did. This module maps those raw signal names to
// user-facing labels and CSS class lists so the UX stays consistent across
// every surface that renders match badges (main book search, related-books,
// etc.).
//
// The paired colors live in global.css under `.match-badge--<signal>`.

export const MATCH_TYPE_LABELS: Record<string, string> = {
  keyword: "Title/Author",
  topic: "Topic",
  content: "Content",
  theme: "Theme",
  book: "Book",
  chunk: "Passage",
  semantic: "Semantic",
};

export function matchBadgeLabel(matchType: string | undefined | null): string | null {
  if (!matchType) return null;
  return MATCH_TYPE_LABELS[matchType] ?? "Match";
}

export function matchBadgeClasses(matchType: string | undefined | null): string[] {
  if (!matchType) return [];
  return ["match-badge", `match-badge--${matchType}`];
}
