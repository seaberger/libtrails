"""
Apply human-refined labels to super-clusters and prepare for database integration.
"""

import json
from pathlib import Path

SUPER_CLUSTERS_PATH = Path(__file__).parent / "super_clusters_robust.json"
OUTPUT_PATH = Path(__file__).parent / "domain_labels_final.json"

# Human-refined labels based on LLM suggestions + review
# Maps super_cluster_id -> final label
REFINED_LABELS = {
    8: "Literary Worlds",           # was: Literary & Historical Figures (fiction, fantasy)
    15: "Everyday Life",            # was: Urban Landscapes (domestic, weather, architecture)
    9: "Human Condition",           # keep LLM (philosophy, conversation, character)
    7: "Wild Earth",                # keep LLM (nature, survival, rural)
    14: "Digital Futures",          # keep LLM (technology, surveillance, engineering)
    23: "AI & Robotics",            # was: Computational Intelligence (speculative AI, robotics)
    17: "Global Narratives",        # keep LLM (travel, history, culture)
    22: "Organizational Dynamics",  # keep LLM (leadership, communication, class)
    1: "Financial Strategy",        # keep LLM (risk, finance, markets)
    4: "Dark Secrets",              # keep LLM (crime, suspense, violence)
    19: "Machine Learning",         # was: Artificial Intelligence (technical ML/DL)
    24: "Sacred Narratives",        # keep LLM (religion, magic, mythology)
    18: "Inner Landscapes",         # keep LLM (memory, death, dreams)
    21: "Family Wellbeing",         # keep LLM (relationships, family)
    2: "Global Affairs",            # keep LLM (politics, race, government)
    3: "Space & Beyond",            # keep LLM (space exploration, travel)
    5: "Culinary Arts",             # keep LLM (cooking, food, baking)
    10: "Arts & Perception",        # was: Visual Culture (music, appearance, aesthetics)
    6: "Creative Writing",          # keep LLM (literature, poetry)
    12: "Emergency Response",       # was: Medical Response (rescue, medical, security)
    11: "Warfare & Conflict",       # merge target: War & Revolution
    16: "Warfare & Conflict",       # merge: Military Operations → same as 11
    20: "Maritime Adventures",      # keep LLM (exploration, navigation)
    13: "Personal Journeys",        # keep LLM (reflection, time travel)
    0: "Identity Formation",        # keep LLM (identity, self)
}

def main():
    # Load super-clusters
    with open(SUPER_CLUSTERS_PATH) as f:
        super_clusters = json.load(f)

    # Build final domains, merging where labels match
    domains = {}

    for sc in super_clusters:
        old_id = sc["super_cluster_id"]
        label = REFINED_LABELS[old_id]

        if label not in domains:
            domains[label] = {
                "label": label,
                "original_ids": [],
                "leiden_clusters": [],
                "top_topics": {},
            }

        domains[label]["original_ids"].append(old_id)
        domains[label]["leiden_clusters"].extend(sc["leiden_clusters"])

        # Aggregate top topics
        for t in sc["top_topics"]:
            topic_label = t["label"]
            count = t["total_count"]
            if topic_label not in domains[label]["top_topics"]:
                domains[label]["top_topics"][topic_label] = 0
            domains[label]["top_topics"][topic_label] += count

    # Convert to list and assign new sequential IDs
    result = []
    for i, (label, data) in enumerate(sorted(domains.items(), key=lambda x: len(x[1]["leiden_clusters"]), reverse=True)):
        # Sort topics by count and take top 10
        sorted_topics = sorted(data["top_topics"].items(), key=lambda x: x[1], reverse=True)[:10]

        result.append({
            "domain_id": i,
            "label": label,
            "cluster_count": len(data["leiden_clusters"]),
            "original_super_ids": data["original_ids"],
            "leiden_cluster_ids": [lc["cluster_id"] for lc in data["leiden_clusters"]],
            "top_topics": [{"label": t[0], "count": t[1]} for t in sorted_topics],
        })

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print("=" * 70)
    print("FINAL DOMAIN LABELS (Human-Refined)")
    print("=" * 70)
    print(f"| {'ID':>2} | {'Clusters':>8} | {'Label':<30} |")
    print(f"|{'-'*4}|{'-'*10}|{'-'*32}|")

    total_clusters = 0
    for d in result:
        print(f"| {d['domain_id']:2d} | {d['cluster_count']:8d} | {d['label']:<30} |")
        total_clusters += d['cluster_count']

    print(f"|{'-'*4}|{'-'*10}|{'-'*32}|")
    print(f"| {'':2} | {total_clusters:8d} | {'TOTAL':<30} |")
    print()
    print(f"Domains: {len(result)} (was 25, merged military)")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
