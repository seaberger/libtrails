import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
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

// ── Cluster spheres using InstancedMesh ──

interface ClusterSpheresProps {
  clusters: UniverseCluster[];
  colorMap: Map<number, string>;
  selectedId: number | null;
  onClickSphere: (cluster: UniverseCluster) => void;
}

function ClusterSpheres({
  clusters,
  colorMap,
  selectedId,
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
      const isSelected = c.cluster_id === selectedId;
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
  }, [clusters, colorMap, maxBooks, selectedId]);

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


// ── Sidebar panel ──

interface SidebarProps {
  selection: {
    cluster: UniverseCluster;
    detail: ThemeDetail | null;
    loading: boolean;
  } | null;
  domains: UniverseDomain[];
  activeDomains: Set<number> | null;
  onToggleDomain: (id: number) => void;
  onClearDomainFilter: () => void;
  onClose: () => void;
}

function Sidebar({
  selection,
  domains,
  activeDomains,
  onToggleDomain,
  onClearDomainFilter,
  onClose,
}: SidebarProps) {
  const basePath = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");

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
      {selection ? (
        <ClusterPanel
          cluster={selection.cluster}
          detail={selection.detail}
          loading={selection.loading}
          domain={domains.find((d) => d.domain_id === selection.cluster.domain_id)}
          basePath={basePath}
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

// ── Cluster detail panel (when a sphere is selected) ──

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
        Drag to rotate · Scroll to zoom
        <br />
        Right-drag to pan
      </div>
    </div>
  );
}

// ── Main component ──

export default function GalaxyView() {
  const [data, setData] = useState<UniverseData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeDomains, setActiveDomains] = useState<Set<number> | null>(null);

  // Selection state
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [selectedCluster, setSelectedCluster] =
    useState<UniverseCluster | null>(null);
  const [clusterDetail, setClusterDetail] = useState<ThemeDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

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

  // Click a sphere → select it and fetch detail
  const handleClickSphere = useCallback(
    (cluster: UniverseCluster) => {
      // Toggle off if clicking the already-selected cluster
      if (selectedId === cluster.cluster_id) {
        setSelectedId(null);
        setSelectedCluster(null);
        setClusterDetail(null);
        return;
      }
      setSelectedId(cluster.cluster_id);
      setSelectedCluster(cluster);
      setClusterDetail(null);
      setDetailLoading(true);
      fetchThemeDetail(cluster.cluster_id)
        .then(setClusterDetail)
        .catch(() => {}) // silently fail, cluster info from universe data is still shown
        .finally(() => setDetailLoading(false));
    },
    [selectedId]
  );

  // Click empty space → deselect
  const handleClickEmpty = useCallback(() => {
    setSelectedId(null);
    setSelectedCluster(null);
    setClusterDetail(null);
  }, []);

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

  const selection = selectedCluster
    ? { cluster: selectedCluster, detail: clusterDetail, loading: detailLoading }
    : null;

  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
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
        <color attach="background" args={["#0f0f1a"]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[50, 80, 60]} intensity={0.8} />
        <directionalLight position={[-40, -20, -50]} intensity={0.3} />
        {visibleClusters.length > 0 && (
          <ClusterSpheres
            clusters={visibleClusters}
            colorMap={colorMap}
            selectedId={selectedId}
            onClickSphere={handleClickSphere}
          />
        )}
        <OrbitControls
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
        selection={selection}
        domains={data.domains}
        activeDomains={activeDomains}
        onToggleDomain={toggleDomain}
        onClearDomainFilter={() => setActiveDomains(null)}
        onClose={handleClickEmpty}
      />
    </div>
  );
}
