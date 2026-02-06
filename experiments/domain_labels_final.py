"""
Apply human-refined labels to super-clusters and prepare for database integration.
"""

import json
from pathlib import Path

SUPER_CLUSTERS_PATH = Path(__file__).parent / "super_clusters_robust.json"
OUTPUT_PATH = Path(__file__).parent / "domain_labels_final.json"

# Human-refined labels based on split super-clusters (Feb 2026)
# Maps super_cluster_id -> final label
# 34 super-clusters → ~25 domains via strategic merges
REFINED_LABELS = {
    # === Core domains (no merge) ===
    0: "History & Archaeology",       # history, religious beliefs, archaeology
    1: "Logic & Mathematics",         # logic, risk assessment, mathematics
    2: "Politics & Power",            # politics, power, government
    3: "Inner Landscapes",            # memory, observation, personal reflection
    4: "Religion & Philosophy",       # religion, philosophy, human nature
    5: "AI & Machine Learning",       # artificial intelligence, ML, neural networks
    6: "Nature & Agriculture",        # animals, economics, agriculture
    7: "Leadership & Strategy",       # communication, leadership, decision making
    8: "Financial Strategy",          # investment strategies, risk management, wealth
    9: "Space & Science",             # space exploration, space travel, science
    10: "Architecture & Design",      # architecture, urban life, domesticity
    11: "Crime & Suspense",           # crime, suspense, violence
    12: "Family & Relationships",     # relationships, family, family dynamics
    15: "Identity & Dreams",          # identity, dreams, deception
    16: "Conflict & Emotion",         # conflict, fear, betrayal
    17: "Nature & Travel",            # travel, nature, weather
    18: "Historical Drama",           # shakespeare, royal court, american history
    19: "Technology & Data",          # technology, text, data structures
    21: "Arts & Society",             # music, marriage, social status
    22: "World Cultures",             # india, japanese cuisine, japanese culture
    23: "Survival & Mortality",       # death, rescue, survival
    25: "Literature & Poetry",        # literature, poetry, symbols
    26: "Espionage & Security",       # espionage, surveillance, security
    27: "Education & Class",          # social interactions, education, social class
    35: "Time & Communication",       # time, human interaction, typography
    36: "Engineering & Robotics",     # robotics, engineering, construction (+ holograms)

    # === Merged domains ===
    # Culinary Arts (merge 13 + 34)
    13: "Culinary Arts",              # cooking techniques, baking, food
    34: "Culinary Arts",              # fire, grilling → merge with cooking

    # Warfare & Military (merge 14 + 28)
    14: "Warfare & Military",         # american revolution, military strategy, WWII
    28: "Warfare & Military",         # combat, military, warfare

    # Fantasy & Speculative Fiction (merge 20 + 30 + 31 + 32)
    20: "Fantasy & Speculative",      # characters, political intrigue, magic
    30: "Fantasy & Speculative",      # glyphs, allomancy, nietzsche (Mistborn + philosophy)
    31: "Fantasy & Speculative",      # warriors, hell, home (dark fantasy)
    32: "Fantasy & Speculative",      # aes sedai, oasis, data collection (Wheel of Time)
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
