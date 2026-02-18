"""GPU-accelerated KNN via FAISS on remote RTX 3090.

Pipeline: export embeddings → upload to PC → SSH run faiss_knn.py → download results → cache.

Uses SSH pipe-based transfers (not SCP/SFTP) because WSL2's SSH
doesn't have the SFTP subsystem configured.

Configuration in ~/.libtrails/config.yaml:
    gpu:
      host: seanb@192.168.1.36
      port: 2222
      remote_dir: ~/projects/gpu-knn
"""

import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .config import DATA_DIR, get_user_config
from .database import get_db, get_topic_embeddings
from .embeddings import bytes_to_embedding

GRAPH_CACHE_DIR = DATA_DIR / "graph_cache"


def get_gpu_config() -> dict:
    """Read GPU SSH settings from ~/.libtrails/config.yaml."""
    cfg = get_user_config().get("gpu", {})
    if not cfg:
        raise RuntimeError(
            "GPU config not found. Add to ~/.libtrails/config.yaml:\n"
            "  gpu:\n"
            "    host: user@gpu-host\n"
            "    port: 22\n"
            "    remote_dir: ~/projects/gpu-knn"
        )
    return {
        "host": cfg["host"],
        "port": str(cfg.get("port", 22)),
        "remote_dir": cfg.get("remote_dir", "~/gpu-knn"),
    }


def knn_cache_path(embed_count: int, k: int) -> Path:
    """Return cache file path for GPU KNN results."""
    return GRAPH_CACHE_DIR / f"knn_gpu_{embed_count}_{k}.npz"


def find_cached_knn(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Check for cached GPU KNN results matching the current embedding count.

    Returns (topic_ids, distances, indices) or None if no valid cache exists.
    """
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) FROM topics WHERE embedding IS NOT NULL").fetchone()
        embed_count = row[0]

    cache = knn_cache_path(embed_count, k)
    if not cache.exists():
        return None

    with np.load(cache) as data:
        cached_ids = data["topic_ids"]
        if len(cached_ids) != embed_count:
            return None
        return cached_ids, data["distances"], data["indices"]


def export_embeddings(output_path: Path) -> int:
    """Export topic embeddings from SQLite to numpy NPZ.

    Returns the number of embeddings exported.
    """
    topic_data = get_topic_embeddings()
    if not topic_data:
        raise RuntimeError("No topic embeddings found. Run 'libtrails embed' first.")

    topic_ids = np.array([t[0] for t in topic_data], dtype=np.int64)
    embeddings = np.array([bytes_to_embedding(t[1]) for t in topic_data], dtype=np.float32)

    np.savez(output_path, embeddings=embeddings, topic_ids=topic_ids)
    return len(topic_ids)


def _ssh_base(gpu_cfg: dict) -> list[str]:
    """Build base SSH command with port."""
    return ["ssh", "-p", gpu_cfg["port"], gpu_cfg["host"]]


def _quote_remote_path(path: str) -> str:
    """Shell-quote a remote path, preserving tilde expansion."""
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def _ssh_upload(gpu_cfg: dict, local_path: Path, remote_path: str) -> None:
    """Upload a file via SSH pipe (cat | ssh 'cat > remote')."""
    with open(local_path, "rb") as f:
        result = subprocess.run(
            [*_ssh_base(gpu_cfg), f"cat > {_quote_remote_path(remote_path)}"],
            stdin=f,
            capture_output=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"SSH upload failed: {result.stderr.decode()}")


def _ssh_download(gpu_cfg: dict, remote_path: str, local_path: Path) -> None:
    """Download a file via SSH pipe (ssh 'cat remote' > local)."""
    result = subprocess.run(
        [*_ssh_base(gpu_cfg), f"cat {_quote_remote_path(remote_path)}"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"SSH download failed: {result.stderr.decode()}")
    with open(local_path, "wb") as f:
        f.write(result.stdout)


def run_gpu_knn(
    k: int = 11,
    force: bool = False,
    export_only: bool = False,
) -> dict:
    """Run the full GPU KNN pipeline.

    Args:
        k: Number of neighbors (including self).
        force: Recompute even if cached.
        export_only: Just export the NPZ for manual transfer.

    Returns:
        Dict with results: embed_count, cache_path, elapsed, etc.
    """
    gpu_cfg = get_gpu_config()
    remote_dir = gpu_cfg["remote_dir"]

    # Check cache first
    if not force and not export_only:
        cached = find_cached_knn(k)
        if cached is not None:
            cache = knn_cache_path(len(cached[0]), k)
            return {
                "status": "cached",
                "embed_count": len(cached[0]),
                "cache_path": str(cache),
            }

    # Export embeddings
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "embeddings.npz"
        t0 = time.time()
        embed_count = export_embeddings(export_path)
        export_time = time.time() - t0

        if export_only:
            final_path = GRAPH_CACHE_DIR / "embeddings.npz"
            shutil.copy2(export_path, final_path)
            return {
                "status": "exported",
                "embed_count": embed_count,
                "export_path": str(final_path),
                "export_time": export_time,
            }

        # Upload via SSH pipe
        t1 = time.time()
        remote_input = f"{remote_dir}/embeddings.npz"
        _ssh_upload(gpu_cfg, export_path, remote_input)
        upload_time = time.time() - t1

        # SSH: run FAISS KNN
        t2 = time.time()
        remote_cmd = (
            f"cd {remote_dir} && "
            f"~/.local/bin/uv run python faiss_knn.py embeddings.npz --k {k} --output knn_results.npz"
        )
        result = subprocess.run(
            [*_ssh_base(gpu_cfg), remote_cmd],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Remote FAISS failed: {result.stderr}\n{result.stdout}")
        gpu_time = time.time() - t2

        # Download results via SSH pipe
        t3 = time.time()
        local_result = Path(tmpdir) / "knn_results.npz"
        remote_output = f"{remote_dir}/knn_results.npz"
        _ssh_download(gpu_cfg, remote_output, local_result)
        download_time = time.time() - t3

        # Validate and cache
        data = np.load(local_result)
        if len(data["topic_ids"]) != embed_count:
            raise RuntimeError(
                f"Result mismatch: expected {embed_count}, got {len(data['topic_ids'])}"
            )

        cache = knn_cache_path(embed_count, k)
        shutil.copy2(local_result, cache)

        # Cleanup remote files
        cleanup_cmd = f"rm -f {remote_dir}/embeddings.npz {remote_dir}/knn_results.npz"
        subprocess.run(
            [*_ssh_base(gpu_cfg), cleanup_cmd],
            capture_output=True,
        )

    total_time = time.time() - t0
    return {
        "status": "completed",
        "embed_count": embed_count,
        "k": k,
        "cache_path": str(cache),
        "export_time": export_time,
        "upload_time": upload_time,
        "gpu_time": gpu_time,
        "download_time": download_time,
        "total_time": total_time,
    }
