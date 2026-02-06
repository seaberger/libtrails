"""
Apply human-refined labels to super-clusters and prepare for database integration.
"""

import json
from pathlib import Path

SUPER_CLUSTERS_PATH = Path(__file__).parent / "super_clusters_robust.json"
OUTPUT_PATH = Path(__file__).parent / "domain_labels_final.json"

# Human-refined labels based on new super-cluster content (Feb 2026)
# Maps super_cluster_id -> final label
REFINED_LABELS = {
    24: "Literary Worlds",          # allomancy, aes sedai, maze (Mistborn, WoT fantasy)
    9: "Literary Worlds",           # harry potter, hogwarts, voldemort (merge fantasy fiction)
    2: "Games & Strategy",          # chess strategy, pawn structure (new domain)
    20: "Organizational Dynamics",  # natural selection, product development, business models
    21: "Human Condition",          # human identity, ethical considerations, ideals
    12: "Digital Futures",          # renewable energy, gate, oil and gas industry (tech/energy)
    18: "Sports & Athletics",       # cycling, doping, golf (new domain)
    14: "Wild Earth",               # orchids, pottery, knife (crafts/nature)
    3: "Financial Strategy",        # investment strategies, risk management, financial markets
    11: "Warfare & Conflict",       # isis, northern ireland conflict/troubles
    5: "Global Narratives",         # mediterranean trade, ancient mesopotamia, 19th century
    17: "Family Wellbeing",         # family tradition, rural communities, military camps
    10: "Human Condition",          # language and communication, scholarship, philosophers (merge)
    15: "Personal Journeys",        # relationships, conflict, travel
    19: "Wild Earth",               # surfing, paleontology, plate tectonics (merge nature/science)
    23: "Machine Learning",         # glyphs, symbols, machine learning
    22: "Culinary Arts",            # cooking techniques, baking, cooking
    8: "Sacred Narratives",         # cultural rituals, religious prophecy, jews
    4: "Dark Secrets",              # restoration, job loss, human survival (struggle/survival)
    1: "Warfare & Conflict",        # military regiments, raiding, veterans (merge)
    7: "Global Affairs",            # political events, political governance, personal interests
    0: "Arts & Perception",         # cinema, perception and reality, museum exhibits
    16: "Family Wellbeing",         # nutrition, alzheimer's disease, diabetes (merge health)
    13: "Inner Landscapes",         # human consciousness, dreams and imagination, sleep
    6: "Identity Formation",        # sexual relationships, misconceptions, personal desire
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
