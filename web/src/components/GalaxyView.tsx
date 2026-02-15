import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import type { ThreeEvent } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { getCoverUrl, getUniverse } from "../lib/api";
import type {
  BookSummary,
  ThemeDetail,
  UniverseCluster,
  UniverseData,
  UniverseDomain,
} from "../lib/types";

const SPREAD = 40;
const HIGHLIGHT_COLOR = "#f59e0b"; // amber for selected
const SIDEBAR_WIDTH = 340;

// ── Fetch theme detail (cluster info with books) ──

async function fetchThemeDetail(clusterId: number): Promise<ThemeDetail> {
  const isServer = typeof window === "undefined";
  const basePath = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");
  const base = isServer ? "http://localhost:8000" : basePath;
  const res = await fetch(`${base}/api/v1/themes/${clusterId}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

// ── Small book cover for React context ──

function BookCoverImg({
  calibreId,
  title,
}: {
  calibreId: number | null;
  title: string;
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
          background: "rgba(255,255,255,0.06)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "0.6rem",
          color: "#888",
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

// ── Cluster spheres using InstancedMesh ──

interface ClusterSpheresProps {
  clusters: UniverseCluster[];
  colorMap: Map<number, string>;
  selectedIds: Set<number>;
  onClickSphere: (cluster: UniverseCluster) => void;
}

function ClusterSpheres({
  clusters,
  colorMap,
  selectedIds,
  onClickSphere,
}: ClusterSpheresProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const maxBooks = useMemo(
    () => Math.max(...clusters.map((c) => c.book_count), 1),
    [clusters]
  );

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    const col = new THREE.Color();

    for (let i = 0; i < clusters.length; i++) {
      const c = clusters[i];
      const isSelected = selectedIds.has(c.cluster_id);
      dummy.position.set(c.x * SPREAD, c.y * SPREAD, (c.z ?? 0) * SPREAD);
      const base = 0.2 + 0.8 * Math.sqrt(c.book_count / maxBooks);
      dummy.scale.setScalar(isSelected ? base * 1.6 : base);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      col.set(isSelected ? HIGHLIGHT_COLOR : colorMap.get(c.domain_id) || "#888888");
      mesh.setColorAt(i, col);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [clusters, colorMap, maxBooks, selectedIds]);

  const handlePointerMove = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (e.instanceId !== undefined && e.instanceId < clusters.length) {
        document.body.style.cursor = "pointer";
      }
    },
    [clusters]
  );

  const handlePointerOut = useCallback(() => {
    document.body.style.cursor = "default";
  }, []);

  const handleClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      const idx = e.instanceId;
      if (idx !== undefined && idx < clusters.length) {
        onClickSphere(clusters[idx]);
      }
    },
    [clusters, onClickSphere]
  );

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, clusters.length]}
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

interface SidebarProps {
  selectedClusters: UniverseCluster[];
  singleDetail: ThemeDetail | null;
  detailLoading: boolean;
  domains: UniverseDomain[];
  activeDomains: Set<number> | null;
  onToggleDomain: (id: number) => void;
  onClearDomainFilter: () => void;
  onSelectCluster: (cluster: UniverseCluster) => void;
  onClose: () => void;
}

function Sidebar({
  selectedClusters,
  singleDetail,
  detailLoading,
  domains,
  activeDomains,
  onToggleDomain,
  onClearDomainFilter,
  onSelectCluster,
  onClose,
}: SidebarProps) {
  const basePath = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");

  const domainMap = useMemo(() => {
    const m = new Map<number, UniverseDomain>();
    for (const d of domains) m.set(d.domain_id, d);
    return m;
  }, [domains]);

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        right: 0,
        bottom: 0,
        width: SIDEBAR_WIDTH,
        background: "rgba(12, 12, 22, 0.92)",
        borderLeft: "1px solid rgba(255,255,255,0.08)",
        backdropFilter: "blur(12px)",
        zIndex: 45,
        display: "flex",
        flexDirection: "column",
        fontFamily: "Inter, sans-serif",
        color: "#e0e0e0",
        overflowY: "auto",
        overflowX: "hidden",
        paddingTop: 52,
        transition: "transform 0.2s ease",
      }}
    >
      {selectedClusters.length === 1 ? (
        <ClusterPanel
          cluster={selectedClusters[0]}
          detail={singleDetail}
          loading={detailLoading}
          domain={domainMap.get(selectedClusters[0].domain_id)}
          basePath={basePath}
          onClose={onClose}
        />
      ) : selectedClusters.length > 1 ? (
        <MultiSelectPanel
          clusters={selectedClusters}
          domainMap={domainMap}
          basePath={basePath}
          onSelectCluster={onSelectCluster}
          onClose={onClose}
        />
      ) : (
        <DomainLegend
          domains={domains}
          activeDomains={activeDomains}
          onToggle={onToggleDomain}
          onClear={onClearDomainFilter}
        />
      )}
    </div>
  );
}

// ── Multi-select panel (list of selected clusters) ──

function MultiSelectPanel({
  clusters,
  domainMap,
  basePath,
  onSelectCluster,
  onClose,
}: {
  clusters: UniverseCluster[];
  domainMap: Map<number, UniverseDomain>;
  basePath: string;
  onSelectCluster: (cluster: UniverseCluster) => void;
  onClose: () => void;
}) {
  const totalBooks = useMemo(
    () => clusters.reduce((sum, c) => sum + c.book_count, 0),
    [clusters]
  );

  return (
    <div style={{ padding: "16px" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: 12,
        }}
      >
        <div>
          <h3 style={{ margin: 0, fontSize: "1rem", fontWeight: 600 }}>
            {clusters.length} clusters selected
          </h3>
          <div style={{ fontSize: "0.75rem", color: "#999", marginTop: 4 }}>
            {totalBooks} books total
          </div>
        </div>
        <button
          onClick={onClose}
          style={{
            background: "none",
            border: "none",
            color: "#888",
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

      {/* Cluster list */}
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {clusters.map((c) => {
          const domain = domainMap.get(c.domain_id);
          return (
            <div
              key={c.cluster_id}
              onClick={() => onSelectCluster(c)}
              style={{
                padding: "8px 10px",
                borderRadius: 6,
                cursor: "pointer",
                background: "rgba(255,255,255,0.03)",
                transition: "background 0.1s",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background = "rgba(255,255,255,0.07)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.background = "rgba(255,255,255,0.03)")
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
              <span style={{ fontSize: "0.8rem", fontWeight: 500 }}>
                {c.label}
              </span>
              <span
                style={{ fontSize: "0.7rem", color: "#777", marginLeft: 8 }}
              >
                {c.book_count} books
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Cluster detail panel (single selection) ──

function ClusterPanel({
  cluster,
  detail,
  loading,
  domain,
  basePath,
  onClose,
}: {
  cluster: UniverseCluster;
  detail: ThemeDetail | null;
  loading: boolean;
  domain: UniverseDomain | undefined;
  basePath: string;
  onClose: () => void;
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
            {cluster.label}
          </h3>
        </div>
        <button
          onClick={onClose}
          style={{
            background: "none",
            border: "none",
            color: "#888",
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
          color: "#999",
          marginBottom: 14,
          display: "flex",
          gap: 8,
        }}
      >
        <span>{cluster.book_count} books</span>
        <span style={{ color: "#555" }}>·</span>
        <span>{cluster.size} topics</span>
      </div>

      {/* Top topics */}
      {cluster.top_topics.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div
            style={{
              fontSize: "0.65rem",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              color: "#777",
              marginBottom: 6,
            }}
          >
            Top Topics
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {cluster.top_topics.slice(0, 8).map((t) => (
              <span
                key={t}
                style={{
                  background: "rgba(255,255,255,0.07)",
                  borderRadius: 4,
                  padding: "3px 8px",
                  fontSize: "0.7rem",
                  color: "#ccc",
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
        <div style={{ fontSize: "0.75rem", color: "#777", padding: "8px 0" }}>
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
              color: "#777",
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
                />
              </a>
            ))}
          </div>
        </div>
      )}

      {/* More topics from detail */}
      {detail && detail.topics.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div
            style={{
              fontSize: "0.65rem",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              color: "#777",
              marginBottom: 6,
            }}
          >
            All Topics
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {detail.topics.slice(0, 20).map((t) => (
              <span
                key={t.id}
                style={{
                  background: "rgba(255,255,255,0.05)",
                  borderRadius: 4,
                  padding: "2px 7px",
                  fontSize: "0.65rem",
                  color: "#aaa",
                }}
              >
                {t.label}
                <span style={{ color: "#666", marginLeft: 4 }}>
                  {t.count}
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Explore link */}
      <a
        href={`${basePath}/clusters/${cluster.cluster_id}`}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          fontSize: "0.8rem",
          color: "#a0a0ff",
          textDecoration: "none",
          padding: "8px 0",
        }}
      >
        Explore cluster →
      </a>
    </div>
  );
}

// ── Domain legend (shown when no cluster selected) ──

function DomainLegend({
  domains,
  activeDomains,
  onToggle,
  onClear,
}: {
  domains: UniverseDomain[];
  activeDomains: Set<number> | null;
  onToggle: (id: number) => void;
  onClear: () => void;
}) {
  return (
    <div style={{ padding: "16px" }}>
      <div
        style={{
          fontSize: "0.7rem",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          color: "#888",
          marginBottom: 10,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span>Domains</span>
        {activeDomains && (
          <button
            onClick={onClear}
            style={{
              background: "none",
              border: "none",
              color: "#a0a0ff",
              cursor: "pointer",
              fontSize: "0.7rem",
              padding: 0,
            }}
          >
            Show all
          </button>
        )}
      </div>
      {domains.map((d) => (
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
              color: "#ccc",
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

      <div
        style={{
          marginTop: 20,
          paddingTop: 14,
          borderTop: "1px solid rgba(255,255,255,0.06)",
          fontSize: "0.7rem",
          color: "rgba(255,255,255,0.3)",
          lineHeight: 1.5,
        }}
      >
        Click a sphere to explore
        <br />
        Shift+drag to select multiple
        <br />
        Drag to rotate · Scroll to zoom
      </div>
    </div>
  );
}

// ── Project 3D cluster position to 2D screen coordinates ──

function projectToScreen(
  cluster: UniverseCluster,
  camera: THREE.Camera,
  canvasRect: DOMRect
): { x: number; y: number } {
  const vec = new THREE.Vector3(
    cluster.x * SPREAD,
    cluster.y * SPREAD,
    (cluster.z ?? 0) * SPREAD
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
  const [data, setData] = useState<UniverseData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeDomains, setActiveDomains] = useState<Set<number> | null>(null);

  // Selection state (supports single and multi-select)
  const [selectedIds, setSelectedIds] = useState<Set<number>>(EMPTY_SET);
  const [singleDetail, setSingleDetail] = useState<ThemeDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

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

  const visibleClusters = useMemo(() => {
    if (!data) return [];
    return activeDomains
      ? data.clusters.filter((c) => activeDomains.has(c.domain_id))
      : data.clusters;
  }, [data, activeDomains]);

  const selectedClusters = useMemo(() => {
    if (selectedIds.size === 0) return [];
    return visibleClusters.filter((c) => selectedIds.has(c.cluster_id));
  }, [visibleClusters, selectedIds]);

  // Fetch detail when exactly one cluster is selected
  useEffect(() => {
    if (selectedIds.size !== 1) {
      setSingleDetail(null);
      setDetailLoading(false);
      return;
    }
    const id = [...selectedIds][0];
    setSingleDetail(null);
    setDetailLoading(true);
    fetchThemeDetail(id)
      .then(setSingleDetail)
      .catch(() => {})
      .finally(() => setDetailLoading(false));
  }, [selectedIds]);

  // Click a sphere → single select (or toggle)
  const handleClickSphere = useCallback(
    (cluster: UniverseCluster) => {
      setSelectedIds((prev) => {
        if (prev.size === 1 && prev.has(cluster.cluster_id)) {
          return EMPTY_SET;
        }
        return new Set([cluster.cluster_id]);
      });
    },
    []
  );

  // Click on a cluster in multi-select list → drill into single select
  const handleSelectSingleFromMulti = useCallback(
    (cluster: UniverseCluster) => {
      setSelectedIds(new Set([cluster.cluster_id]));
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

      // Find clusters within the rectangle
      const hits = new Set<number>();
      for (const c of visibleClusters) {
        const screen = projectToScreen(c, camera, canvasRect);
        if (
          screen.x >= left &&
          screen.x <= right &&
          screen.y >= top &&
          screen.y <= bottom
        ) {
          hits.add(c.cluster_id);
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
    [selectRect, visibleClusters]
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

  if (error) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          color: "#e0e0e0",
          fontFamily: "Inter, sans-serif",
          flexDirection: "column",
          gap: "1rem",
          position: "absolute",
          inset: 0,
        }}
      >
        <p style={{ fontSize: "1.1rem" }}>Could not load universe data</p>
        <p style={{ fontSize: "0.85rem", color: "#888" }}>{error}</p>
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
          color: "#e0e0e0",
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
        <color attach="background" args={["#0f0f1a"]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[50, 80, 60]} intensity={0.8} />
        <directionalLight position={[-40, -20, -50]} intensity={0.3} />
        {visibleClusters.length > 0 && (
          <ClusterSpheres
            clusters={visibleClusters}
            colorMap={colorMap}
            selectedIds={selectedIds}
            onClickSphere={handleClickSphere}
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

      {/* Sidebar */}
      <Sidebar
        selectedClusters={selectedClusters}
        singleDetail={singleDetail}
        detailLoading={detailLoading}
        domains={data.domains}
        activeDomains={activeDomains}
        onToggleDomain={toggleDomain}
        onClearDomainFilter={() => setActiveDomains(null)}
        onSelectCluster={handleSelectSingleFromMulti}
        onClose={handleClickEmpty}
      />
    </div>
  );
}
