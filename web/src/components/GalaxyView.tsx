import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import type { ThreeEvent } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { getBooksBatch, getCommunity, getCoverUrl, getUniverse, searchHybrid } from "../lib/api";
import type {
  BookSummary,
  CommunityDetail,
  UniverseCommunity,
  UniverseData,
  UniverseDomain,
  UniverseSearchResult,
} from "../lib/types";

const SPREAD = 40;
const HIGHLIGHT_COLOR = "#f59e0b"; // amber for selected

interface TooltipState {
  x: number;
  y: number;
  community: UniverseCommunity;
  domain: UniverseDomain | undefined;
}
const SIDEBAR_WIDTH = 340;
const SCENE_BG_DARK = "#0f0f1a";
const SCENE_BG_LIGHT = "#e8e8f0";

function useIsMobile(breakpoint = 767) {
  const [isMobile, setIsMobile] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.innerWidth <= breakpoint;
  });
  useEffect(() => {
    const mql = window.matchMedia(`(max-width: ${breakpoint}px)`);
    const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    setIsMobile(mql.matches);
    mql.addEventListener("change", handler);
    return () => mql.removeEventListener("change", handler);
  }, [breakpoint]);
  return isMobile;
}

function useThemeMode() {
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    if (typeof document === "undefined") return "dark";
    return (document.documentElement.getAttribute("data-theme") as "light") === "light" ? "light" : "dark";
  });
  useEffect(() => {
    const observer = new MutationObserver(() => {
      const t = document.documentElement.getAttribute("data-theme");
      setTheme(t === "light" ? "light" : "dark");
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);
  return theme;
}

// Sidebar color palette that responds to theme
interface SidebarColors {
  bg: string;
  border: string;
  text: string;
  textMuted: string;
  textFaint: string;
  accent: string;
  chipBg: string;
  chipBgHover: string;
  ruleBorder: string;
  coverFallbackBg: string;
}

const SIDEBAR_DARK: SidebarColors = {
  bg: "rgba(12, 12, 22, 0.92)",
  border: "rgba(255,255,255,0.08)",
  text: "#e0e0e0",
  textMuted: "#888",
  textFaint: "rgba(255,255,255,0.3)",
  accent: "#a0a0ff",
  chipBg: "rgba(255,255,255,0.06)",
  chipBgHover: "rgba(255,255,255,0.07)",
  ruleBorder: "rgba(255,255,255,0.06)",
  coverFallbackBg: "rgba(255,255,255,0.06)",
};

const SIDEBAR_LIGHT: SidebarColors = {
  bg: "rgba(255, 255, 255, 0.92)",
  border: "#e5e5e5",
  text: "#1c1917",
  textMuted: "#78716c",
  textFaint: "#a8a29e",
  accent: "#6366f1",
  chipBg: "rgba(0,0,0,0.05)",
  chipBgHover: "rgba(0,0,0,0.08)",
  ruleBorder: "rgba(0,0,0,0.08)",
  coverFallbackBg: "rgba(0,0,0,0.05)",
};

// ── Small book cover for React context ──

function BookCoverImg({
  calibreId,
  title,
  colors,
}: {
  calibreId: number | null;
  title: string;
  colors: SidebarColors;
}) {
  const [failed, setFailed] = useState(false);
  const src = getCoverUrl(calibreId);

  if (!calibreId || failed) {
    return (
      <div
        style={{
          width: 56,
          height: 80,
          borderRadius: 3,
          background: colors.coverFallbackBg,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "0.6rem",
          color: colors.textMuted,
          textAlign: "center",
          padding: 4,
          overflow: "hidden",
        }}
      >
        {title.slice(0, 30)}
      </div>
    );
  }

  return (
    <img
      src={src}
      alt={title}
      loading="lazy"
      onError={() => setFailed(true)}
      style={{
        width: 56,
        height: 80,
        objectFit: "cover",
        borderRadius: 3,
      }}
    />
  );
}

// ── Expose Three.js camera via ref for screen projection ──

function CameraRef({ cameraRef }: { cameraRef: React.MutableRefObject<THREE.Camera | null> }) {
  const { camera } = useThree();
  useEffect(() => {
    cameraRef.current = camera;
  }, [camera, cameraRef]);
  return null;
}

// ── Community spheres using InstancedMesh ──

const SEARCH_HIGHLIGHT_COLOR = "#f59e0b"; // amber for search hits

interface CommunitySpheresProps {
  communities: UniverseCommunity[];
  colorMap: Map<number, string>;
  selectedIds: Set<number>;
  searchHitIds: Set<number> | null;
  onClickSphere: (community: UniverseCommunity) => void;
  onHover: (community: UniverseCommunity | null, x: number, y: number) => void;
}

function CommunitySpheres({
  communities,
  colorMap,
  selectedIds,
  searchHitIds,
  onClickSphere,
  onHover,
}: CommunitySpheresProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const maxBooks = useMemo(
    () => Math.max(...communities.map((c) => c.book_count), 1),
    [communities]
  );

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    const col = new THREE.Color();
    const isSearching = searchHitIds !== null;

    for (let i = 0; i < communities.length; i++) {
      const c = communities[i];
      const isSelected = selectedIds.has(c.community_id);
      const isSearchHit = isSearching && searchHitIds.has(c.community_id);
      dummy.position.set(c.x * SPREAD, c.y * SPREAD, (c.z ?? 0) * SPREAD);
      const base = 0.2 + 0.8 * Math.sqrt(c.book_count / maxBooks);
      const scale = isSelected ? base * 1.6 : isSearchHit ? base * 1.3 : base;
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      if (isSelected) {
        col.set(HIGHLIGHT_COLOR);
      } else if (isSearchHit) {
        col.set(SEARCH_HIGHLIGHT_COLOR);
      } else if (isSearching) {
        // Dim non-matching communities during search
        col.set(colorMap.get(c.domain_id) || "#888888");
        col.multiplyScalar(0.15);
      } else {
        col.set(colorMap.get(c.domain_id) || "#888888");
      }
      mesh.setColorAt(i, col);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [communities, colorMap, maxBooks, selectedIds, searchHitIds]);

  const handlePointerMove = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (e.instanceId !== undefined && e.instanceId < communities.length) {
        document.body.style.cursor = "pointer";
        onHover(communities[e.instanceId], e.nativeEvent.clientX, e.nativeEvent.clientY);
      }
    },
    [communities, onHover]
  );

  const handlePointerOut = useCallback(() => {
    document.body.style.cursor = "default";
    onHover(null, 0, 0);
  }, [onHover]);

  const handleClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      const idx = e.instanceId;
      if (idx !== undefined && idx < communities.length) {
        onClickSphere(communities[idx]);
      }
    },
    [communities, onClickSphere]
  );

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, communities.length]}
      onPointerMove={handlePointerMove}
      onPointerOut={handlePointerOut}
      onClick={handleClick}
    >
      <sphereGeometry args={[1, 24, 24]} />
      <meshStandardMaterial
        transparent
        opacity={0.9}
        roughness={0.4}
        metalness={0.1}
      />
    </instancedMesh>
  );
}

// ── Selection rectangle overlay ──

interface SelectRect {
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
}

function SelectionOverlay({ rect }: { rect: SelectRect }) {
  const left = Math.min(rect.startX, rect.currentX);
  const top = Math.min(rect.startY, rect.currentY);
  const width = Math.abs(rect.currentX - rect.startX);
  const height = Math.abs(rect.currentY - rect.startY);

  if (width < 3 && height < 3) return null;

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        border: "1.5px dashed rgba(245, 158, 11, 0.7)",
        background: "rgba(245, 158, 11, 0.08)",
        pointerEvents: "none",
        zIndex: 30,
      }}
    />
  );
}

// ── Sidebar panel ──

interface SearchState {
  query: string;
  results: UniverseSearchResult[];
  loading: boolean;
}

interface SidebarProps {
  selectedCommunities: UniverseCommunity[];
  singleDetail: CommunityDetail | null;
  detailLoading: boolean;
  domains: UniverseDomain[];
  activeDomains: Set<number> | null;
  onToggleDomain: (id: number) => void;
  onClearDomainFilter: () => void;
  onSelectCommunity: (community: UniverseCommunity) => void;
  onClose: () => void;
  colors: SidebarColors;
  isMobile: boolean;
  search: SearchState;
  onSearchChange: (query: string) => void;
  allCommunities: UniverseCommunity[];
}

const SNAP_POINTS = [20, 40, 85]; // vh: peek, half, expanded
const DEFAULT_SHEET_HEIGHT = 40; // vh

function Sidebar({
  selectedCommunities,
  singleDetail,
  detailLoading,
  domains,
  activeDomains,
  onToggleDomain,
  onClearDomainFilter,
  onSelectCommunity,
  onClose,
  colors,
  isMobile,
  search,
  onSearchChange,
  allCommunities,
}: SidebarProps) {
  const basePath = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");
  const searchInputRef = useRef<HTMLInputElement>(null);

  const domainMap = useMemo(() => {
    const m = new Map<number, UniverseDomain>();
    for (const d of domains) m.set(d.domain_id, d);
    return m;
  }, [domains]);

  // Map search results to community objects for the sidebar list,
  // filtered by activeDomains so hidden domains don't appear in results
  const searchResultCommunities = useMemo(() => {
    if (!search.results.length) return [];
    const communityMap = new Map(allCommunities.map((c) => [c.community_id, c]));
    return search.results
      .map((r) => communityMap.get(r.community_id))
      .filter((c): c is UniverseCommunity =>
        c !== undefined && (!activeDomains || activeDomains.has(c.domain_id))
      );
  }, [search.results, allCommunities, activeDomains]);

  // ── Bottom sheet drag-to-resize (mobile only) ──
  // Uses ref-based DOM listeners with { passive: false } so preventDefault()
  // works in touchmove (React 18 attaches touch handlers as passive by default).
  const [sheetHeight, setSheetHeight] = useState(DEFAULT_SHEET_HEIGHT);
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef<{ startY: number; startHeight: number } | null>(null);
  const sheetHeightRef = useRef(sheetHeight);
  const dragHandleRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    sheetHeightRef.current = sheetHeight;
  }, [sheetHeight]);

  useEffect(() => {
    const el = dragHandleRef.current;
    if (!el || !isMobile) return;

    const onTouchStart = (e: TouchEvent) => {
      setIsDragging(true);
      dragRef.current = {
        startY: e.touches[0].clientY,
        startHeight: sheetHeightRef.current,
      };
    };

    const onTouchMove = (e: TouchEvent) => {
      if (!dragRef.current) return;
      e.preventDefault(); // works because listener is { passive: false }
      const deltaY = dragRef.current.startY - e.touches[0].clientY;
      const deltaVh = (deltaY / window.innerHeight) * 100;
      const newHeight = Math.max(12, Math.min(92, dragRef.current.startHeight + deltaVh));
      setSheetHeight(newHeight);
    };

    const onTouchEnd = () => {
      if (!dragRef.current) return;
      const current = sheetHeightRef.current;
      const closest = SNAP_POINTS.reduce((prev, curr) =>
        Math.abs(curr - current) < Math.abs(prev - current) ? curr : prev
      );
      setSheetHeight(closest);
      dragRef.current = null;
      setIsDragging(false);
    };

    el.addEventListener("touchstart", onTouchStart, { passive: true });
    el.addEventListener("touchmove", onTouchMove, { passive: false });
    el.addEventListener("touchend", onTouchEnd, { passive: true });
    el.addEventListener("touchcancel", onTouchEnd, { passive: true });
    return () => {
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove", onTouchMove);
      el.removeEventListener("touchend", onTouchEnd);
      el.removeEventListener("touchcancel", onTouchEnd);
    };
  }, [isMobile]);

  // Reset sheet height when switching between mobile/desktop (e.g. rotation)
  useEffect(() => {
    setSheetHeight(DEFAULT_SHEET_HEIGHT);
  }, [isMobile]);

  const containerStyle: React.CSSProperties = isMobile
    ? {
        position: "absolute",
        bottom: 0,
        left: 0,
        right: 0,
        height: `${sheetHeight}vh`,
        background: colors.bg,
        borderTop: `1px solid ${colors.border}`,
        backdropFilter: "blur(12px)",
        zIndex: 45,
        display: "flex",
        flexDirection: "column",
        fontFamily: "Inter, sans-serif",
        color: colors.text,
        overflowY: "auto",
        overflowX: "hidden",
        paddingTop: 0,
        borderRadius: "16px 16px 0 0",
        transition: isDragging
          ? "background 0.15s ease"
          : "height 0.25s ease, background 0.15s ease",
      }
    : {
        position: "absolute",
        top: 0,
        right: 0,
        bottom: 0,
        width: SIDEBAR_WIDTH,
        background: colors.bg,
        borderLeft: `1px solid ${colors.border}`,
        backdropFilter: "blur(12px)",
        zIndex: 45,
        display: "flex",
        flexDirection: "column",
        fontFamily: "Inter, sans-serif",
        color: colors.text,
        overflowY: "auto",
        overflowX: "hidden",
        paddingTop: 52,
        transition: "transform 0.2s ease, background 0.15s ease",
      };

  return (
    <div style={containerStyle}>
      {/* Drag handle for mobile — touch target for resizing */}
      {isMobile && (
        <div
          ref={dragHandleRef}
          data-testid="sheet-drag-handle"
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            minHeight: 28,
            padding: "10px 0 6px",
            flexShrink: 0,
            cursor: "grab",
            touchAction: "none",
          }}
        >
          <div
            style={{
              width: 36,
              height: 4,
              borderRadius: 2,
              background: colors.textFaint,
            }}
          />
        </div>
      )}
      {/* Search input (always visible at top when no community selected) */}
      {selectedCommunities.length === 0 && (
        <div style={{ padding: "12px 16px 0" }}>
          <div style={{ position: "relative" }}>
            <input
              ref={searchInputRef}
              type="text"
              value={search.query}
              onChange={(e) => onSearchChange(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Escape") { onSearchChange(""); searchInputRef.current?.blur(); } }}
              placeholder="Search universe..."
              style={{
                width: "100%",
                padding: "8px 12px 8px 32px",
                background: colors.chipBg,
                border: `1px solid ${colors.border}`,
                borderRadius: 8,
                color: colors.text,
                fontSize: "0.8rem",
                outline: "none",
              }}
            />
            <svg
              style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", width: 14, height: 14 }}
              fill="none"
              stroke={colors.textMuted}
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            {search.loading && (
              <div style={{ position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)", width: 14, height: 14, border: `2px solid ${colors.accent}`, borderTopColor: "transparent", borderRadius: "50%", animation: "spin 0.6s linear infinite" }} />
            )}
          </div>
        </div>
      )}

      {selectedCommunities.length === 1 ? (
        <CommunityPanel
          community={selectedCommunities[0]}
          detail={singleDetail}
          loading={detailLoading}
          domain={domainMap.get(selectedCommunities[0].domain_id)}
          basePath={basePath}
          onClose={onClose}
          colors={colors}
        />
      ) : selectedCommunities.length > 1 ? (
        <MultiSelectPanel
          communities={selectedCommunities}
          domainMap={domainMap}
          basePath={basePath}
          onSelectCommunity={onSelectCommunity}
          onClose={onClose}
          colors={colors}
        />
      ) : search.query && searchResultCommunities.length > 0 && !search.loading ? (
        /* Search results list */
        <div style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.05em", color: colors.textMuted, marginBottom: 8 }}>
            {searchResultCommunities.length} matching topics
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {searchResultCommunities.map((c) => {
              const domain = domainMap.get(c.domain_id);
              return (
                <div
                  key={c.community_id}
                  onClick={() => onSelectCommunity(c)}
                  style={{ padding: "6px 10px", borderRadius: 6, cursor: "pointer", background: colors.chipBg, transition: "background 0.1s" }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = colors.chipBgHover)}
                  onMouseLeave={(e) => (e.currentTarget.style.background = colors.chipBg)}
                >
                  {domain && (
                    <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: domain.color, marginRight: 6, verticalAlign: "middle" }} />
                  )}
                  <span style={{ fontSize: "0.78rem", fontWeight: 500 }}>{c.label}</span>
                  <span style={{ fontSize: "0.68rem", color: colors.textMuted, marginLeft: 8 }}>{c.book_count} books</span>
                </div>
              );
            })}
          </div>
        </div>
      ) : search.query && !search.loading ? (
        <div style={{ padding: "16px", fontSize: "0.8rem", color: colors.textMuted, textAlign: "center" }}>
          No topics found
        </div>
      ) : (
        <DomainLegend
          domains={domains}
          activeDomains={activeDomains}
          onToggle={onToggleDomain}
          onClear={onClearDomainFilter}
          colors={colors}
          isMobile={isMobile}
        />
      )}
    </div>
  );
}

// ── Multi-select panel (rich overview of selected communities) ──

function MultiSelectPanel({
  communities,
  domainMap,
  basePath,
  onSelectCommunity,
  onClose,
  colors,
}: {
  communities: UniverseCommunity[];
  domainMap: Map<number, UniverseDomain>;
  basePath: string;
  onSelectCommunity: (community: UniverseCommunity) => void;
  onClose: () => void;
  colors: SidebarColors;
}) {
  const [books, setBooks] = useState<BookSummary[]>([]);
  const [booksLoading, setBooksLoading] = useState(false);
  const [showAllBooks, setShowAllBooks] = useState(false);
  const [showAllCommunities, setShowAllCommunities] = useState(false);

  // ── Derived data ──

  const uniqueBookIds = useMemo(() => {
    const ids = new Set<number>();
    for (const c of communities) {
      if (c.book_ids) for (const id of c.book_ids) ids.add(id);
    }
    return ids;
  }, [communities]);

  // Domain breakdown: group communities by domain, sorted by count desc
  const domainBreakdown = useMemo(() => {
    const counts = new Map<number, number>();
    for (const c of communities) {
      counts.set(c.domain_id, (counts.get(c.domain_id) || 0) + 1);
    }
    return [...counts.entries()]
      .map(([domainId, count]) => ({
        domainId,
        count,
        domain: domainMap.get(domainId),
      }))
      .sort((a, b) => b.count - a.count);
  }, [communities, domainMap]);

  // Top books: ranked by how many selected communities they appear in
  const topBookIds = useMemo(() => {
    const freq = new Map<number, number>();
    for (const c of communities) {
      if (c.book_ids) {
        for (const id of c.book_ids) {
          freq.set(id, (freq.get(id) || 0) + 1);
        }
      }
    }
    return [...freq.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 50)
      .map(([id]) => id);
  }, [communities]);

  // Top clusters: aggregate top_clusters across communities, count frequency
  const topClusters = useMemo(() => {
    const freq = new Map<string, number>();
    for (const c of communities) {
      for (const t of c.top_clusters) {
        freq.set(t, (freq.get(t) || 0) + 1);
      }
    }
    return [...freq.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15);
  }, [communities]);

  // Fetch book details when selection changes
  useEffect(() => {
    if (topBookIds.length === 0) {
      setBooks([]);
      return;
    }
    setBooksLoading(true);
    getBooksBatch(topBookIds)
      .then(setBooks)
      .catch(() => setBooks([]))
      .finally(() => setBooksLoading(false));
  }, [topBookIds]);

  // Sort fetched books to match frequency order
  const sortedBooks = useMemo(() => {
    const order = new Map(topBookIds.map((id, i) => [id, i]));
    return [...books].sort(
      (a, b) => (order.get(a.id) ?? 99) - (order.get(b.id) ?? 99)
    );
  }, [books, topBookIds]);

  const MAX_DOMAINS_SHOWN = 6;
  const visibleDomains = domainBreakdown.slice(0, MAX_DOMAINS_SHOWN);
  const hiddenDomainCount = domainBreakdown.length - MAX_DOMAINS_SHOWN;

  return (
    <div style={{ padding: "16px" }}>
      {/* ── 1. Header + Stats ── */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: 6,
        }}
      >
        <h3 style={{ margin: 0, fontSize: "1rem", fontWeight: 600 }}>
          {communities.length} topics selected
        </h3>
        <button
          onClick={onClose}
          style={{
            background: "none",
            border: "none",
            color: colors.textMuted,
            cursor: "pointer",
            fontSize: "1.2rem",
            padding: "0 0 0 8px",
            lineHeight: 1,
          }}
          aria-label="Clear selection"
        >
          ×
        </button>
      </div>
      <div
        style={{
          display: "flex",
          gap: 6,
          flexWrap: "wrap",
          marginBottom: 16,
        }}
      >
        {[
          `${uniqueBookIds.size} books`,
          `${communities.reduce((sum, c) => sum + c.cluster_count, 0)} clusters`,
          `${domainBreakdown.length} themes`,
        ].map((label) => (
          <span
            key={label}
            style={{
              fontSize: "0.68rem",
              color: colors.textMuted,
              background: colors.chipBg,
              borderRadius: 4,
              padding: "2px 8px",
            }}
          >
            {label}
          </span>
        ))}
      </div>

      {/* ── 2. Domain Breakdown ── */}
      {domainBreakdown.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div
            style={{
              fontSize: "0.65rem",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              color: colors.textMuted,
              marginBottom: 6,
            }}
          >
            Theme Breakdown
          </div>
          {/* Stacked bar */}
          <div
            style={{
              display: "flex",
              height: 6,
              borderRadius: 3,
              overflow: "hidden",
              marginBottom: 8,
            }}
          >
            {domainBreakdown.map(({ domainId, count, domain }) => (
              <div
                key={domainId}
                style={{
                  flex: count,
                  background: domain?.color || "#555",
                  minWidth: 2,
                }}
                title={`${domain?.label || `Theme ${domainId}`}: ${count} communities`}
              />
            ))}
          </div>
          {/* Domain list */}
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            {visibleDomains.map(({ domainId, count, domain }) => (
              <div
                key={domainId}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  fontSize: "0.72rem",
                }}
              >
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background: domain?.color || "#555",
                    flexShrink: 0,
                  }}
                />
                <span style={{ color: colors.text, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {domain?.label || `Theme ${domainId}`}
                </span>
                <span style={{ color: colors.textMuted, fontSize: "0.65rem", flexShrink: 0 }}>
                  {count}
                </span>
              </div>
            ))}
            {hiddenDomainCount > 0 && (
              <div style={{ fontSize: "0.65rem", color: colors.textMuted, paddingLeft: 14 }}>
                +{hiddenDomainCount} more
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── 3. Top Books (cover grid) ── */}
      <div style={{ marginBottom: 16 }}>
        <div
          style={{
            fontSize: "0.65rem",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
            color: colors.textMuted,
            marginBottom: 8,
          }}
        >
          Top Books
        </div>
        {booksLoading ? (
          <div style={{ fontSize: "0.75rem", color: colors.textMuted, padding: "8px 0" }}>
            Loading books...
          </div>
        ) : sortedBooks.length > 0 ? (
          <>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)",
                gap: 8,
              }}
            >
              {(showAllBooks ? sortedBooks : sortedBooks.slice(0, 9)).map((book) => (
                <a
                  key={book.id}
                  href={`${basePath}/books/${book.id}`}
                  title={`${book.title} — ${book.author}`}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    textDecoration: "none",
                    gap: 4,
                  }}
                >
                  <BookCoverImg
                    calibreId={book.calibre_id}
                    title={book.title}
                    colors={colors}
                  />
                  <span
                    style={{
                      fontSize: "0.6rem",
                      color: colors.textMuted,
                      textAlign: "center",
                      lineHeight: 1.2,
                      maxWidth: 80,
                      overflow: "hidden",
                      display: "-webkit-box",
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: "vertical",
                    }}
                  >
                    {book.title}
                  </span>
                </a>
              ))}
            </div>
            {sortedBooks.length > 9 && (
              <button
                onClick={() => setShowAllBooks((v) => !v)}
                style={{
                  background: "none",
                  border: "none",
                  color: colors.accent,
                  cursor: "pointer",
                  fontSize: "0.72rem",
                  padding: "8px 0 0",
                }}
              >
                {showAllBooks
                  ? "Show less"
                  : `Show all ${sortedBooks.length} books →`}
              </button>
            )}
          </>
        ) : (
          <div style={{ fontSize: "0.7rem", color: colors.textMuted }}>No book data available</div>
        )}
      </div>

      {/* ── 4. Top Clusters ── */}
      {topClusters.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div
            style={{
              fontSize: "0.65rem",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              color: colors.textMuted,
              marginBottom: 6,
            }}
          >
            Top Clusters
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {topClusters.map(([cluster, count]) => (
              <span
                key={cluster}
                style={{
                  background: colors.chipBgHover,
                  borderRadius: 4,
                  padding: "3px 8px",
                  fontSize: "0.7rem",
                  color: colors.text,
                }}
              >
                {cluster}
                {count > 1 && (
                  <span style={{ color: colors.textMuted, marginLeft: 4, fontSize: "0.6rem" }}>
                    ×{count}
                  </span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* ── 5. Community List (collapsible) ── */}
      <div
        style={{
          borderTop: `1px solid ${colors.ruleBorder}`,
          paddingTop: 12,
        }}
      >
        <button
          onClick={() => setShowAllCommunities((v) => !v)}
          style={{
            background: "none",
            border: "none",
            color: colors.accent,
            cursor: "pointer",
            fontSize: "0.75rem",
            padding: 0,
            display: "flex",
            alignItems: "center",
            gap: 4,
          }}
        >
          <span
            style={{
              display: "inline-block",
              transition: "transform 0.15s",
              transform: showAllCommunities ? "rotate(90deg)" : "rotate(0deg)",
            }}
          >
            ▸
          </span>
          {showAllCommunities ? "Hide" : "View all"} {communities.length} topics
        </button>
        {showAllCommunities && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 2,
              marginTop: 8,
            }}
          >
            {communities.map((c) => {
              const domain = domainMap.get(c.domain_id);
              return (
                <div
                  key={c.community_id}
                  onClick={() => onSelectCommunity(c)}
                  style={{
                    padding: "6px 10px",
                    borderRadius: 6,
                    cursor: "pointer",
                    background: colors.chipBg,
                    transition: "background 0.1s",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.background = colors.chipBgHover)
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.background = colors.chipBg)
                  }
                >
                  {domain && (
                    <span
                      style={{
                        display: "inline-block",
                        width: 8,
                        height: 8,
                        borderRadius: "50%",
                        background: domain.color,
                        marginRight: 6,
                        verticalAlign: "middle",
                      }}
                    />
                  )}
                  <span style={{ fontSize: "0.78rem", fontWeight: 500 }}>
                    {c.label}
                  </span>
                  <span style={{ fontSize: "0.68rem", color: colors.textMuted, marginLeft: 8 }}>
                    {c.book_count} books
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Community detail panel (single selection) ──

function CommunityPanel({
  community,
  detail,
  loading,
  domain,
  basePath,
  onClose,
  colors,
}: {
  community: UniverseCommunity;
  detail: CommunityDetail | null;
  loading: boolean;
  domain: UniverseDomain | undefined;
  basePath: string;
  onClose: () => void;
  colors: SidebarColors;
}) {
  return (
    <div style={{ padding: "16px" }}>
      {/* Header with close button */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: 12,
        }}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          {domain && (
            <span
              style={{
                display: "inline-block",
                fontSize: "0.65rem",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                color: domain.color,
                background: `${domain.color}18`,
                border: `1px solid ${domain.color}40`,
                borderRadius: 4,
                padding: "2px 8px",
                marginBottom: 8,
              }}
            >
              {domain.label}
            </span>
          )}
          <h3
            style={{
              margin: 0,
              fontSize: "1rem",
              fontWeight: 600,
              lineHeight: 1.3,
            }}
          >
            {community.label}
          </h3>
        </div>
        <button
          onClick={onClose}
          style={{
            background: "none",
            border: "none",
            color: colors.textMuted,
            cursor: "pointer",
            fontSize: "1.2rem",
            padding: "0 0 0 8px",
            lineHeight: 1,
          }}
          aria-label="Close"
        >
          ×
        </button>
      </div>

      {/* Stats */}
      <div
        style={{
          fontSize: "0.75rem",
          color: colors.textMuted,
          marginBottom: 14,
          display: "flex",
          gap: 8,
        }}
      >
        <span>{community.book_count} books</span>
        <span>·</span>
        <span>{community.cluster_count} clusters</span>
      </div>

      {/* Cluster labels */}
      {community.top_clusters.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div
            style={{
              fontSize: "0.65rem",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              color: colors.textMuted,
              marginBottom: 6,
            }}
          >
            Clusters
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {community.top_clusters.slice(0, 8).map((t) => (
              <span
                key={t}
                style={{
                  background: colors.chipBgHover,
                  borderRadius: 4,
                  padding: "3px 8px",
                  fontSize: "0.7rem",
                  color: colors.text,
                }}
              >
                {t}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Books */}
      {loading && (
        <div style={{ fontSize: "0.75rem", color: colors.textMuted, padding: "8px 0" }}>
          Loading books...
        </div>
      )}

      {detail && detail.books.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div
            style={{
              fontSize: "0.65rem",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              color: colors.textMuted,
              marginBottom: 8,
            }}
          >
            Books ({detail.books.length})
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, 56px)",
              gap: 6,
            }}
          >
            {detail.books.slice(0, 18).map((book: BookSummary) => (
              <a
                key={book.id}
                href={`${basePath}/books/${book.id}`}
                title={`${book.title} — ${book.author}`}
                style={{ display: "block", textDecoration: "none" }}
              >
                <BookCoverImg
                  calibreId={book.calibre_id}
                  title={book.title}
                  colors={colors}
                />
              </a>
            ))}
          </div>
        </div>
      )}

      {/* Explore link */}
      <a
        href={`${basePath}/communities/${community.community_id}`}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          fontSize: "0.8rem",
          color: colors.accent,
          textDecoration: "none",
          padding: "8px 0",
        }}
      >
        View topic →
      </a>
    </div>
  );
}

// ── Domain legend (shown when no community selected) ──

function DomainLegend({
  domains,
  activeDomains,
  onToggle,
  onClear,
  colors,
  isMobile,
}: {
  domains: UniverseDomain[];
  activeDomains: Set<number> | null;
  onToggle: (id: number) => void;
  onClear: () => void;
  colors: SidebarColors;
  isMobile: boolean;
}) {
  const [expanded, setExpanded] = useState(!isMobile);
  const MOBILE_LIMIT = 10;
  const visibleDomains = expanded ? domains : domains.slice(0, MOBILE_LIMIT);
  const hiddenCount = domains.length - MOBILE_LIMIT;

  return (
    <div style={{ padding: "16px" }}>
      <div
        style={{
          fontSize: "0.7rem",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          color: colors.textMuted,
          marginBottom: 10,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span>Themes</span>
        {activeDomains && (
          <button
            onClick={onClear}
            style={{
              background: "none",
              border: "none",
              color: colors.accent,
              cursor: "pointer",
              fontSize: "0.7rem",
              padding: 0,
            }}
          >
            Show all
          </button>
        )}
      </div>
      {visibleDomains.map((d) => (
        <div
          key={d.domain_id}
          onClick={() => onToggle(d.domain_id)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "4px 0",
            cursor: "pointer",
            opacity:
              activeDomains && !activeDomains.has(d.domain_id) ? 0.3 : 1,
            transition: "opacity 0.15s",
          }}
        >
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: d.color,
              flexShrink: 0,
            }}
          />
          <span
            style={{
              color: colors.text,
              fontSize: "0.75rem",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {d.label}
          </span>
        </div>
      ))}

      {isMobile && !expanded && hiddenCount > 0 && (
        <button
          onClick={() => setExpanded(true)}
          style={{
            background: "none",
            border: "none",
            color: colors.accent,
            cursor: "pointer",
            fontSize: "0.72rem",
            padding: "6px 0",
          }}
        >
          +{hiddenCount} more themes
        </button>
      )}
      {isMobile && expanded && hiddenCount > 0 && (
        <button
          onClick={() => setExpanded(false)}
          style={{
            background: "none",
            border: "none",
            color: colors.accent,
            cursor: "pointer",
            fontSize: "0.72rem",
            padding: "6px 0",
          }}
        >
          Show less
        </button>
      )}

      <div
        style={{
          marginTop: 20,
          paddingTop: 14,
          borderTop: `1px solid ${colors.ruleBorder}`,
          fontSize: "0.7rem",
          color: colors.textFaint,
          lineHeight: 1.5,
        }}
      >
        {isMobile ? (
          <>
            Tap a sphere to explore
            <br />
            Pinch to zoom · Drag to rotate
          </>
        ) : (
          <>
            Click a sphere to explore
            <br />
            Shift+drag to select multiple
            <br />
            Drag to rotate · Scroll to zoom
          </>
        )}
      </div>
    </div>
  );
}

// ── Project 3D community position to 2D screen coordinates ──

function projectToScreen(
  community: UniverseCommunity,
  camera: THREE.Camera,
  canvasRect: DOMRect
): { x: number; y: number } {
  const vec = new THREE.Vector3(
    community.x * SPREAD,
    community.y * SPREAD,
    (community.z ?? 0) * SPREAD
  );
  vec.project(camera);
  return {
    x: ((vec.x + 1) / 2) * canvasRect.width + canvasRect.left,
    y: ((-vec.y + 1) / 2) * canvasRect.height + canvasRect.top,
  };
}

// ── Main component ──

const EMPTY_SET = new Set<number>();

export default function GalaxyView() {
  const themeMode = useThemeMode();
  const isMobile = useIsMobile();
  const [data, setData] = useState<UniverseData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeDomains, setActiveDomains] = useState<Set<number> | null>(null);

  // Selection state (supports single and multi-select)
  const [selectedIds, setSelectedIds] = useState<Set<number>>(EMPTY_SET);
  const [singleDetail, setSingleDetail] = useState<CommunityDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // Hover tooltip
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<UniverseSearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const searchAbortRef = useRef<AbortController | null>(null);

  const searchHitIds = useMemo(() => {
    if (!searchQuery || searchResults.length === 0) return null;
    return new Set(searchResults.map((r) => r.community_id));
  }, [searchQuery, searchResults]);

  const handleSearchChange = useCallback((query: string) => {
    setSearchQuery(query);
    clearTimeout(searchTimerRef.current);
    if (searchAbortRef.current) searchAbortRef.current.abort();
    if (!query.trim()) {
      setSearchResults([]);
      setSearchLoading(false);
      return;
    }
    setSearchLoading(true);
    const controller = new AbortController();
    searchAbortRef.current = controller;
    searchTimerRef.current = setTimeout(() => {
      searchHybrid(query, "universe", 50)
        .then((resp) => {
          if (controller.signal.aborted) return;
          setSearchResults(resp.universe);
          setSearchLoading(false);
        })
        .catch((err) => {
          if (err?.name === "AbortError") return;
          setSearchResults([]);
          setSearchLoading(false);
        });
    }, 300);
  }, []);

  const handleHover = useCallback(
    (community: UniverseCommunity | null, x: number, y: number) => {
      if (!community || !data) {
        setTooltip(null);
        return;
      }
      setTooltip({
        x,
        y,
        community,
        domain: data.domains.find((d) => d.domain_id === community.domain_id),
      });
    },
    [data]
  );

  // Shift+drag rectangle selection
  const [selectRect, setSelectRect] = useState<SelectRect | null>(null);
  const isDraggingRef = useRef(false);
  const cameraRef = useRef<THREE.Camera | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const orbitControlsRef = useRef<any>(null);

  useEffect(() => {
    getUniverse()
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);

  const colorMap = useMemo(() => {
    const map = new Map<number, string>();
    if (data) for (const d of data.domains) map.set(d.domain_id, d.color);
    return map;
  }, [data]);

  const visibleCommunities = useMemo(() => {
    if (!data) return [];
    return activeDomains
      ? data.communities.filter((c) => activeDomains.has(c.domain_id))
      : data.communities;
  }, [data, activeDomains]);

  const selectedCommunities = useMemo(() => {
    if (selectedIds.size === 0) return [];
    return visibleCommunities.filter((c) => selectedIds.has(c.community_id));
  }, [visibleCommunities, selectedIds]);

  // Fetch detail when exactly one community is selected
  useEffect(() => {
    if (selectedIds.size !== 1) {
      setSingleDetail(null);
      setDetailLoading(false);
      return;
    }
    const id = [...selectedIds][0];
    setSingleDetail(null);
    setDetailLoading(true);
    getCommunity(id)
      .then(setSingleDetail)
      .catch(() => {})
      .finally(() => setDetailLoading(false));
  }, [selectedIds]);

  // Click a sphere → single select (or toggle)
  const handleClickSphere = useCallback(
    (community: UniverseCommunity) => {
      setSelectedIds((prev) => {
        if (prev.size === 1 && prev.has(community.community_id)) {
          return EMPTY_SET;
        }
        return new Set([community.community_id]);
      });
    },
    []
  );

  // Click on a community in multi-select list → drill into single select
  const handleSelectSingleFromMulti = useCallback(
    (community: UniverseCommunity) => {
      setSelectedIds(new Set([community.community_id]));
    },
    []
  );

  // Click empty space → deselect all
  const handleClickEmpty = useCallback(() => {
    setSelectedIds(EMPTY_SET);
  }, []);

  // ── Shift+drag handlers ──

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!e.shiftKey) return;
      e.preventDefault();
      // Disable orbit controls during shift-drag
      if (orbitControlsRef.current) {
        orbitControlsRef.current.enabled = false;
      }
      isDraggingRef.current = true;
      setSelectRect({
        startX: e.clientX,
        startY: e.clientY,
        currentX: e.clientX,
        currentY: e.clientY,
      });
    },
    []
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDraggingRef.current || !selectRect) return;
      setSelectRect((prev) =>
        prev ? { ...prev, currentX: e.clientX, currentY: e.clientY } : null
      );
    },
    [selectRect]
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      // Re-enable orbit controls
      if (orbitControlsRef.current) {
        orbitControlsRef.current.enabled = true;
      }
      if (!isDraggingRef.current || !selectRect) return;
      isDraggingRef.current = false;

      const camera = cameraRef.current;
      const container = containerRef.current;
      if (!camera || !container) {
        setSelectRect(null);
        return;
      }

      const canvas = container.querySelector("canvas");
      if (!canvas) {
        setSelectRect(null);
        return;
      }
      const canvasRect = canvas.getBoundingClientRect();

      // Compute selection bounds
      const left = Math.min(selectRect.startX, e.clientX);
      const right = Math.max(selectRect.startX, e.clientX);
      const top = Math.min(selectRect.startY, e.clientY);
      const bottom = Math.max(selectRect.startY, e.clientY);

      // Skip if too small (was just a click, not a drag)
      if (right - left < 5 && bottom - top < 5) {
        setSelectRect(null);
        return;
      }

      // Find communities within the rectangle
      const hits = new Set<number>();
      for (const c of visibleCommunities) {
        const screen = projectToScreen(c, camera, canvasRect);
        if (
          screen.x >= left &&
          screen.x <= right &&
          screen.y >= top &&
          screen.y <= bottom
        ) {
          hits.add(c.community_id);
        }
      }

      if (hits.size > 0) {
        // Ctrl+Shift adds to existing selection
        if (e.ctrlKey || e.metaKey) {
          setSelectedIds((prev) => {
            const next = new Set(prev);
            for (const id of hits) next.add(id);
            return next;
          });
        } else {
          setSelectedIds(hits);
        }
      }

      setSelectRect(null);
    },
    [selectRect, visibleCommunities]
  );

  const toggleDomain = useCallback(
    (domainId: number) => {
      setActiveDomains((prev) => {
        if (!prev) return new Set([domainId]);
        const next = new Set(prev);
        if (next.has(domainId)) {
          next.delete(domainId);
          return next.size === 0 ? null : next;
        }
        next.add(domainId);
        if (data && next.size === data.domains.length) return null;
        return next;
      });
    },
    [data]
  );

  const sidebarColors = themeMode === "light" ? SIDEBAR_LIGHT : SIDEBAR_DARK;

  if (error) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          color: sidebarColors.text,
          fontFamily: "Inter, sans-serif",
          flexDirection: "column",
          gap: "1rem",
          position: "absolute",
          inset: 0,
        }}
      >
        <p style={{ fontSize: "1.1rem" }}>Could not load universe data</p>
        <p style={{ fontSize: "0.85rem", color: sidebarColors.textMuted }}>{error}</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          color: sidebarColors.text,
          fontFamily: "Inter, sans-serif",
          position: "absolute",
          inset: 0,
        }}
      >
        <p>Loading universe...</p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      style={{ position: "absolute", inset: 0, overflow: "hidden" }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      {/* Selection rectangle overlay */}
      {selectRect && <SelectionOverlay rect={selectRect} />}

      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [0, 20, 70], fov: 60, near: 0.1, far: 500 }}
        gl={{ antialias: true }}
        dpr={[1, 2]}
        onPointerMissed={handleClickEmpty}
        style={{
          position: "absolute",
          inset: 0,
        }}
      >
        <CameraRef cameraRef={cameraRef} />
        <color attach="background" args={[themeMode === "light" ? SCENE_BG_LIGHT : SCENE_BG_DARK]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[50, 80, 60]} intensity={0.8} />
        <directionalLight position={[-40, -20, -50]} intensity={0.3} />
        {visibleCommunities.length > 0 && (
          <CommunitySpheres
            communities={visibleCommunities}
            colorMap={colorMap}
            selectedIds={selectedIds}
            searchHitIds={searchHitIds}
            onClickSphere={handleClickSphere}
            onHover={handleHover}
          />
        )}
        <OrbitControls
          ref={orbitControlsRef}
          enableDamping
          dampingFactor={0.05}
          rotateSpeed={0.5}
          zoomSpeed={0.7}
          minDistance={5}
          maxDistance={150}
          autoRotate
          autoRotateSpeed={0.2}
        />
      </Canvas>

      {/* Hover tooltip */}
      {tooltip && (
        <div
          style={{
            position: "fixed",
            left: tooltip.x + 16,
            top: tooltip.y - 12,
            background: sidebarColors.bg,
            border: `1px solid ${tooltip.domain?.color || sidebarColors.border}`,
            borderRadius: "10px",
            padding: "12px 16px",
            pointerEvents: "none",
            zIndex: 100,
            maxWidth: "280px",
            fontFamily: "Inter, sans-serif",
            backdropFilter: "blur(8px)",
            boxShadow: "0 8px 32px rgba(0,0,0,0.25)",
          }}
        >
          <div
            style={{
              fontWeight: 600,
              fontSize: "0.9rem",
              color: sidebarColors.text,
              marginBottom: "4px",
              textTransform: "capitalize",
              lineHeight: 1.3,
            }}
          >
            {tooltip.community.label}
          </div>
          {tooltip.domain && (
            <div
              style={{
                display: "inline-block",
                fontSize: "0.68rem",
                fontWeight: 500,
                color: tooltip.domain.color,
                background: `${tooltip.domain.color}18`,
                border: `1px solid ${tooltip.domain.color}40`,
                borderRadius: "9999px",
                padding: "1px 8px",
                marginBottom: "8px",
              }}
            >
              {tooltip.domain.label}
            </div>
          )}
          <div
            style={{
              color: sidebarColors.textMuted,
              fontSize: "0.75rem",
              marginBottom: "6px",
            }}
          >
            {tooltip.community.book_count} books &middot; {tooltip.community.cluster_count} clusters
          </div>
          {tooltip.community.top_clusters.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
              {tooltip.community.top_clusters.slice(0, 4).map((t) => (
                <span
                  key={t}
                  style={{
                    background: sidebarColors.chipBg,
                    borderRadius: "4px",
                    padding: "2px 7px",
                    fontSize: "0.68rem",
                    color: sidebarColors.textMuted,
                  }}
                >
                  {t}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Sidebar */}
      <Sidebar
        selectedCommunities={selectedCommunities}
        singleDetail={singleDetail}
        detailLoading={detailLoading}
        domains={data.domains}
        activeDomains={activeDomains}
        onToggleDomain={toggleDomain}
        onClearDomainFilter={() => setActiveDomains(null)}
        onSelectCommunity={handleSelectSingleFromMulti}
        onClose={handleClickEmpty}
        colors={sidebarColors}
        isMobile={isMobile}
        search={{ query: searchQuery, results: searchResults, loading: searchLoading }}
        onSearchChange={handleSearchChange}
        allCommunities={data.communities}
      />
    </div>
  );
}
