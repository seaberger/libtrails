#!/usr/bin/env python3
"""Standalone FAISS-GPU KNN script for the RTX 3090 PC.

Runs on the remote Windows/WSL2 machine with faiss-gpu installed.
Zero libtrails dependencies — just numpy and faiss.

Usage:
    python faiss_knn.py embeddings.npz --k 11
    python faiss_knn.py embeddings.npz --k 11 --output knn_results.npz
"""

import argparse
import time

import faiss
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="FAISS-GPU k-NN search")
    parser.add_argument("input", help="Path to embeddings.npz (embeddings + topic_ids)")
    parser.add_argument("--k", type=int, default=11, help="Number of neighbors (default: 11)")
    parser.add_argument("--output", "-o", default="knn_results.npz", help="Output file path")
    args = parser.parse_args()

    # Load embeddings
    print(f"Loading embeddings from {args.input}...")
    data = np.load(args.input)
    embeddings = data["embeddings"].astype(np.float32)
    topic_ids = data["topic_ids"]
    n, dim = embeddings.shape
    print(f"  {n:,} vectors, {dim} dimensions")

    # L2-normalize for cosine similarity via inner product
    print("Normalizing vectors...")
    faiss.normalize_L2(embeddings)

    # Build flat inner-product index on GPU
    print("Building FAISS index on GPU...")
    cpu_index = faiss.IndexFlatIP(dim)
    gpu_res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
    gpu_index.add(embeddings)
    print(f"  Index contains {gpu_index.ntotal:,} vectors")

    # Search
    k = min(args.k, n)
    print(f"Searching {k} nearest neighbors...")
    t0 = time.time()
    similarities, indices = gpu_index.search(embeddings, k)
    elapsed = time.time() - t0
    print(f"  Search completed in {elapsed:.1f}s")

    # Convert similarity → cosine distance (1 - sim) to match sklearn convention
    distances = 1.0 - similarities

    # Save results
    np.savez_compressed(
        args.output,
        distances=distances,
        indices=indices,
        topic_ids=topic_ids,
        k=np.array([k]),
    )
    print(f"Saved results to {args.output}")
    print(f"  Shape: distances={distances.shape}, indices={indices.shape}")


if __name__ == "__main__":
    main()
