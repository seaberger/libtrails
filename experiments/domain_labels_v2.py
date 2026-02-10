"""
Apply human-refined labels to V2 super-clusters and prepare for database integration.

V2 pipeline: 338 books, 305K topics, 1,182 Leiden clusters → 28 super-clusters.
"""

import json
from pathlib import Path

SUPER_CLUSTERS_PATH = Path(__file__).parent / "super_clusters_new.json"
OUTPUT_PATH = Path(__file__).parent / "domain_labels_v2.json"

# Human-refined labels for V2 super-clusters (Feb 2026)
# Maps super_cluster_id -> final label
# 28 super-clusters → ~25 domains via strategic merges
REFINED_LABELS = {
    # === Large narrative domains ===
    3: "Narrative Fiction",            # broad character-driven stories across genres
    20: "Epic & Historical Fiction",   # Dune, 100 Years of Solitude, Stormlight, Roman history
    25: "Character Psychology",        # internal conflicts, anxieties, social dynamics

    # === Literary & cultural domains ===
    6: "Language & Culture",           # linguistics, storytelling, literary traditions
    9: "Literary Arts",               # writing process, art, aesthetics, literary criticism
    0: "Arrivals & Departures",       # greetings, farewells, reunions, social gatherings

    # === Genre fiction ===
    21: "Science Fiction",             # aliens, invasions, post-human, space
    4: "Spirituality & Mysticism",    # religion, prophecy, faith, mysticism
    1: "Fear & the Unknown",          # anxiety, drugs, darkness, dread

    # === Human experience ===
    23: "Love & Relationships",        # romance, intimacy, belonging, sacrifice
    17: "Family & Childhood",          # parents, children, domestic conflict
    16: "Death & Grief",               # mortality, loss, violence, body disposal

    # === Society & power ===
    19: "Crime & Justice",             # investigations, law, bureaucracy, corruption
    14: "Colonialism & Culture",       # power dynamics, cultural identity, social class
    8: "Politics & Social Critique",   # ideology, class struggle, empire, anarchism
    22: "Leadership & Growth",         # self-development, mentorship, responsibility

    # === Action & conflict ===
    12: "War & Revolution",            # military history, resistance, revolution
    10: "Military & Combat",           # special ops, weapons, tactical operations

    # === Knowledge & technology ===
    15: "Science & Innovation",        # physics, startups, technology, research
    2: "Engineering & Infrastructure", # cables, space elevators, craft, automotive

    # === Nature & physical world ===
    18: "Nature & Survival",           # wilderness, rivers, islands, journeys
    11: "Animals & Wildlife",          # wildlife, hunting, confrontations with animals

    # === Places & things ===
    13: "Places & Architecture",       # cities, buildings, libraries, settings
    27: "Mystery & Intrigue",          # secrets, espionage, heists, locked doors

    # === Economics & trade ===
    24: "Money & Finance",             # speculation, gambling, markets
    5: "Resources & Trade",            # agriculture, commodities, economics

    # === Other ===
    7: "Food & Cooking",              # recipes, baking, preparation, ingredients
    26: "Travel & Transportation",     # journeys, vehicles, exploration
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
    print("V2 DOMAIN LABELS (Human-Refined)")
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
    print(f"Domains: {len(result)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
