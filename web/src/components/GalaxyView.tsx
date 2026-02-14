import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import type { ThreeEvent } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { getUniverse } from "../lib/api";
import type { UniverseCluster, UniverseData, UniverseDomain } from "../lib/types";

const SPREAD = 40;

interface TooltipState {
  x: number;
  y: number;
  cluster: UniverseCluster;
  domain: UniverseDomain | undefined;
}

// ── Cluster spheres using InstancedMesh for vibrant, colorful 3D spheres ──

interface ClusterSpheresProps {
  clusters: UniverseCluster[];
  colorMap: Map<number, string>;
  onHover: (cluster: UniverseCluster | null, x: number, y: number) => void;
  onClick: (cluster: UniverseCluster) => void;
}

function ClusterSpheres({ clusters, colorMap, onHover, onClick }: ClusterSpheresProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const maxBooks = useMemo(() => Math.max(...clusters.map((c) => c.book_count), 1), [clusters]);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    const col = new THREE.Color();

    for (let i = 0; i < clusters.length; i++) {
      const c = clusters[i];
      dummy.position.set(c.x * SPREAD, c.y * SPREAD, (c.z ?? 0) * SPREAD);
      const s = 0.3 + 1.2 * Math.sqrt(c.book_count / maxBooks);
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      col.set(colorMap.get(c.domain_id) || "#888888");
      mesh.setColorAt(i, col);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [clusters, colorMap, maxBooks]);

  const handlePointerMove = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      const idx = e.instanceId;
      if (idx !== undefined && idx < clusters.length) {
        document.body.style.cursor = "pointer";
        onHover(clusters[idx], e.nativeEvent.clientX, e.nativeEvent.clientY);
      }
    },
    [clusters, onHover]
  );

  const handlePointerOut = useCallback(() => {
    document.body.style.cursor = "default";
    onHover(null, 0, 0);
  }, [onHover]);

  const handleClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      const idx = e.instanceId;
      if (idx !== undefined && idx < clusters.length) {
        onClick(clusters[idx]);
      }
    },
    [clusters, onClick]
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
      <meshStandardMaterial transparent opacity={0.9} roughness={0.4} metalness={0.1} />
    </instancedMesh>
  );
}

// ── Main component ──

export default function GalaxyView() {
  const [data, setData] = useState<UniverseData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const [activeDomains, setActiveDomains] = useState<Set<number> | null>(null);

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

  const handleHover = useCallback(
    (cluster: UniverseCluster | null, x: number, y: number) => {
      if (!cluster || !data) { setTooltip(null); return; }
      setTooltip({
        x, y, cluster,
        domain: data.domains.find((d) => d.domain_id === cluster.domain_id),
      });
    },
    [data]
  );

  const handleClick = useCallback((cluster: UniverseCluster) => {
    const base = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");
    window.location.href = `${base}/clusters/${cluster.cluster_id}`;
  }, []);

  const toggleDomain = useCallback(
    (domainId: number) => {
      setActiveDomains((prev) => {
        if (!prev) return new Set([domainId]);
        const next = new Set(prev);
        if (next.has(domainId)) { next.delete(domainId); return next.size === 0 ? null : next; }
        next.add(domainId);
        if (data && next.size === data.domains.length) return null;
        return next;
      });
    },
    [data]
  );

  if (error) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh", color: "#e0e0e0", fontFamily: "Inter, sans-serif", flexDirection: "column", gap: "1rem" }}>
        <p style={{ fontSize: "1.1rem" }}>Could not load universe data</p>
        <p style={{ fontSize: "0.85rem", color: "#888" }}>{error}</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh", color: "#e0e0e0", fontFamily: "Inter, sans-serif" }}>
        <p>Loading universe...</p>
      </div>
    );
  }

  return (
    <div style={{ width: "100vw", height: "100vh", position: "relative" }}>
      <Canvas
        camera={{ position: [0, 20, 70], fov: 60, near: 0.1, far: 500 }}
        gl={{ antialias: true }}
        dpr={[1, 2]}
      >
        <color attach="background" args={["#0f0f1a"]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[50, 80, 60]} intensity={0.8} />
        <directionalLight position={[-40, -20, -50]} intensity={0.3} />
        {visibleClusters.length > 0 && (
          <ClusterSpheres
            clusters={visibleClusters}
            colorMap={colorMap}
            onHover={handleHover}
            onClick={handleClick}
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

      {/* Tooltip */}
      {tooltip && (
        <div style={{
          position: "fixed", left: tooltip.x + 14, top: tooltip.y - 10,
          background: "rgba(20, 20, 35, 0.95)",
          border: `1px solid ${tooltip.domain?.color || "#555"}`,
          borderRadius: "8px", padding: "10px 14px",
          pointerEvents: "none", zIndex: 100, maxWidth: "280px",
          fontFamily: "Inter, sans-serif", fontSize: "0.8rem",
          color: "#e0e0e0", backdropFilter: "blur(4px)",
        }}>
          <div style={{ color: tooltip.domain?.color, fontSize: "0.75rem", marginBottom: "4px" }}>
            {tooltip.domain?.label || "Unknown domain"}
          </div>
          <div style={{ fontWeight: 600, fontSize: "0.9rem", marginBottom: "6px" }}>
            {tooltip.cluster.label}
          </div>
          <div style={{ color: "#aaa", fontSize: "0.75rem", marginBottom: "4px" }}>
            {tooltip.cluster.book_count} books &middot; {tooltip.cluster.size} topics
          </div>
          {tooltip.cluster.top_topics.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "4px", marginTop: "4px" }}>
              {tooltip.cluster.top_topics.slice(0, 5).map((t) => (
                <span key={t} style={{
                  background: "rgba(255,255,255,0.08)", borderRadius: "4px",
                  padding: "1px 6px", fontSize: "0.7rem", color: "#ccc",
                }}>{t}</span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Domain legend */}
      <div style={{
        position: "absolute", bottom: "16px", right: "16px",
        background: "rgba(15, 15, 26, 0.85)",
        border: "1px solid rgba(255,255,255,0.1)", borderRadius: "10px",
        padding: "12px 14px", maxHeight: "calc(100vh - 100px)", overflowY: "auto",
        fontFamily: "Inter, sans-serif", fontSize: "0.75rem",
        backdropFilter: "blur(8px)", zIndex: 40,
      }}>
        <div style={{
          fontSize: "0.7rem", textTransform: "uppercase",
          letterSpacing: "0.05em", color: "#888", marginBottom: "8px",
          display: "flex", justifyContent: "space-between", alignItems: "center", gap: "12px",
        }}>
          <span>Domains</span>
          {activeDomains && (
            <button onClick={() => setActiveDomains(null)} style={{
              background: "none", border: "none", color: "#a0a0ff",
              cursor: "pointer", fontSize: "0.7rem", padding: 0,
            }}>Show all</button>
          )}
        </div>
        {data.domains.map((d) => (
          <div key={d.domain_id} onClick={() => toggleDomain(d.domain_id)} style={{
            display: "flex", alignItems: "center", gap: "8px",
            padding: "3px 0", cursor: "pointer",
            opacity: activeDomains && !activeDomains.has(d.domain_id) ? 0.3 : 1,
            transition: "opacity 0.15s",
          }}>
            <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: d.color, flexShrink: 0 }} />
            <span style={{ color: "#ccc" }}>{d.label}</span>
          </div>
        ))}
      </div>

      {/* Info hint */}
      <div style={{
        position: "absolute", bottom: "16px", left: "16px",
        color: "rgba(255,255,255,0.3)", fontSize: "0.7rem", fontFamily: "Inter, sans-serif",
      }}>
        Drag to rotate &middot; Scroll to zoom &middot; Right-drag to pan &middot; Click a sphere to explore
      </div>
    </div>
  );
}
