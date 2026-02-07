"""
Generate concise domain names for super-clusters using LLM.
"""

import json
import httpx
from pathlib import Path

SUPER_CLUSTERS_PATH = Path(__file__).parent / "super_clusters_robust.json"
OUTPUT_PATH = Path(__file__).parent / "domain_labels.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"


def generate_domain_name(topics: list[str], model: str = MODEL) -> str:
    """Use LLM to generate a concise domain name from topic list."""

    topics_str = ", ".join(topics[:15])

    prompt = f"""You are naming categories for a book library. Given these related topics:

{topics_str}

Generate a single concise category name (1-3 words, preferably 1-2) that captures the overall theme.

Rules:
- Use title case (e.g., "Machine Learning" not "machine learning")
- Be specific but not too narrow
- Avoid generic words like "Topics" or "Miscellaneous"
- Just output the category name, nothing else

Category name:"""

    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 20,
            }
        },
        timeout=30.0
    )

    if response.status_code == 200:
        result = response.json()
        name = result.get("response", "").strip()
        # Clean up: remove quotes, newlines, extra text
        name = name.split("\n")[0].strip('"\'')
        return name
    else:
        return "Unknown"


def main():
    # Load super-clusters
    with open(SUPER_CLUSTERS_PATH) as f:
        super_clusters = json.load(f)

    print(f"Generating names for {len(super_clusters)} domains...\n")

    results = []

    for sc in super_clusters:
        super_id = sc["super_cluster_id"]
        n_clusters = len(sc["leiden_clusters"])

        # Get top topics (already sorted by count in the JSON)
        top_topics = [t["label"] for t in sc["top_topics"]]
        auto_label = sc["auto_label"]

        # Generate LLM name
        llm_name = generate_domain_name(top_topics)

        results.append({
            "domain_id": super_id,
            "cluster_count": n_clusters,
            "auto_label": auto_label,
            "llm_label": llm_name,
            "top_topics": top_topics[:10]
        })

        print(f"[{super_id:2d}] {n_clusters:3d} clusters | {llm_name:25s} | was: {auto_label[:40]}")

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY: LLM-Generated Domain Names")
    print("="*60)
    print(f"| {'ID':>2} | {'#':>3} | {'LLM Label':<25} |")
    print(f"|{'-'*4}|{'-'*5}|{'-'*27}|")
    for r in sorted(results, key=lambda x: x["cluster_count"], reverse=True):
        print(f"| {r['domain_id']:2d} | {r['cluster_count']:3d} | {r['llm_label']:<25} |")


if __name__ == "__main__":
    main()
