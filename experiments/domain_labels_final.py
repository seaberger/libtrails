"""
Apply human-refined labels to super-clusters and prepare for database integration.

Usage:
    LIBTRAILS_DB=demo uv run python experiments/domain_labels_final.py
    LIBTRAILS_DB=demo uv run libtrails load-domains
"""

import json
from pathlib import Path

SUPER_CLUSTERS_PATH = Path(__file__).parent / "super_clusters_new.json"
OUTPUT_PATH = Path(__file__).parent / "domain_labels_final.json"

# Human-refined labels for demo library (100 Gutenberg classics)
# Generated from Leiden domain clustering at resolution 0.006 (26 domains)
# Maps super_cluster_id -> final label
REFINED_LABELS = {
    # === Literary Worlds ===
    0: "Chivalric & Heroic Romance",    # Don Quixote, musketeers, heroic code, Sancho Panza
    1: "Adventure & Human Folly",       # Candide, Voltaire, rapid-paced adventure, human weakness
    2: "Voyages & Exploration",         # Captain Nemo, submarine, Pacific Ocean, seafaring
    4: "Psychological Portraits",       # Lady Glyde, Dorian Gray, Nastasia, Hester Prynne, Scarlet Letter
    19: "Character Types",              # young man, wise man, coachman, provincial ladies

    # === Ideas & Belief ===
    5: "Religion & Spirituality",       # divine intervention, worship, sacrifice, convent life
    16: "Ethics & Morality",            # moral compromise, ethical conduct, slave morality, virtue
    18: "Death & Mortality",            # dying man, immortality, impending death, grief

    # === Society & Power ===
    3: "Economics & Industry",          # financial manipulation, industrial army, mercantile system, tax
    7: "Race & Social Justice",         # social inequality, Tuskegee, racial dynamics, suppressed desires
    10: "Revolution & Political Theory", # French Revolution, Napoleon, federal republic, Articles of Confederation
    11: "War & Military",              # civil war, military discipline, dueling, standing armies
    12: "Law & Governance",            # implied warranties, property, courts of justice, intellectual property

    # === Daily Life & Setting ===
    6: "Material Culture & Leisure",    # capital, beauty, chess, diamond, silver, Paris salons
    8: "Domestic & Social Life",        # district council, customs, hospitality, public execution, wine
    9: "Crime & Intrigue",             # moral decay, asylum, body identification, investigation, secrets
    14: "Family & Medicine",           # maternal abandonment, family honor, doctor-patient, infant care
    20: "Travel & Transport",          # hussar regiment, hotel, railway, carriage, homeward journey
    21: "Night & Atmosphere",          # moonlight, churchyard, darkness, lightning, midnight

    # === Expression & Knowledge ===
    13: "Letters & Communication",      # literary criticism, telegraph, love letters, speech, poetry
    15: "Love & Desire",               # erotic desire, personal fulfillment, romance, divorce, passion
    22: "Education & Intellect",        # education of children, university, classical literature, discipline
    25: "Theater & Performance",        # Grace Poole, Klondike, opera, theatrical innovation, nihilism

    # === Nature & Survival ===
    17: "Nature & Animals",            # hunting techniques, bird species, guanaco, horseback riding
    23: "Slavery & Desolation",        # freedmen's bureau, free labor, blight of slavery, ruins
    24: "Ships & the Sea",            # shipwreck, ship's crew, naval, maritime trade, Captain Smollett
}


def main():
    # Load super-clusters
    with open(SUPER_CLUSTERS_PATH) as f:
        super_clusters = json.load(f)

    # Verify all IDs are mapped
    unmapped = [sc["super_cluster_id"] for sc in super_clusters if sc["super_cluster_id"] not in REFINED_LABELS]
    if unmapped:
        print(f"WARNING: Unmapped super-cluster IDs: {unmapped}")
        print("Add these to REFINED_LABELS before proceeding.")
        return

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
    print(f"| {'ID':>2} | {'Clusters':>8} | {'Label':<35} |")
    print(f"|{'-'*4}|{'-'*10}|{'-'*37}|")

    total_clusters = 0
    for d in result:
        print(f"| {d['domain_id']:2d} | {d['cluster_count']:8d} | {d['label']:<35} |")
        total_clusters += d['cluster_count']

    print(f"|{'-'*4}|{'-'*10}|{'-'*37}|")
    print(f"| {'':2} | {total_clusters:8d} | {'TOTAL':<35} |")
    print()
    print(f"Domains: {len(result)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
