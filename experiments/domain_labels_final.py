"""
Apply human-refined labels to super-clusters and prepare for database integration.

Usage:
    LIBTRAILS_DB=demo uv run python experiments/domain_labels_final.py
    LIBTRAILS_DB=demo uv run libtrails load-domains
"""

import json
from pathlib import Path

SUPER_CLUSTERS_PATH = Path(__file__).parent / "domains_outliers_only.json"
OUTPUT_PATH = Path(__file__).parent / "domain_labels_final.json"

# Human-refined labels for demo library (100 Gutenberg classics)
# Generated from Leiden domain clustering at resolution 0.006 (27 domains)
# With occurrence_count bug fix + outlier detection (84 clusters reassigned)
# Maps super_cluster_id -> final label
REFINED_LABELS = {
    # === Literary Worlds ===
    0: "Literary Classics & Philosophy", # Montaigne, Plato, Nietzsche, Don Quixote, Emerson, Marcus Aurelius
    3: "Heroic & Military Epic",         # Musketeers, Achilles, swordsmanship, Bronze Age warfare, French Revolution
    1: "Voyages & Exploration",          # Captain Nemo, submarine, Pacific Ocean, Yorkshire moors, seafaring
    2: "Psychological Portraits",        # Lady Glyde, Nastasia, Jean Valjean, Scarlet Letter, Dorian Gray
    5: "Adventure & Human Folly",        # Candide, Voltaire, Nietzsche's higher men, cynicism, social disillusionment
    7: "Psychology & Personal Struggle", # Return to homeland, epilepsy, Anna Karenina, personal feelings, trauma

    # === Ideas & Belief ===
    8: "Religion & Spirituality",        # divine intervention, sacrifice, worship, convent life, church & state
    13: "Culture & Moral Inquiry",       # Grace Poole, Klondike, opera, moral compromise, general happiness
    19: "Archetypes & Aging",            # old man, young man, wise man, effects of aging, timocratical man
    20: "Death & Mortality",             # dying man, immortality, poisoning, impending death, grave digger

    # === Society & Power ===
    4: "Economics & Industry",           # financial manipulation, mercantile system, tax, stock market, political economy
    14: "Politics & Governance",         # political corruption, Articles of Confederation, federal republic, Dead Souls
    26: "Politics & Governance",         # plurality in executive, magistracy, Baskerville legacy — MERGED into 14
    9: "Race & Social Justice",          # post-convict rehabilitation, poverty, Tuskegee, social inequality
    12: "Law & Civil Society",           # implied warranties, intellectual property, courts of justice, filial duty

    # === Daily Life & Setting ===
    6: "Objects & Aesthetics",           # beauty, chess, diamonds, silver, Paris salons, human imperfection
    11: "Domestic & Social Life",        # hussar regiment, street urchin, customs, hospitality, public execution
    10: "Nature & Rural Life",           # hunting techniques, birds, district council, Spanish wine, guanaco
    15: "Crime & Intrigue",              # moral decay, body identification, Porfiry's investigation, clandestine meetings
    18: "Family & Kinship",              # maternal abandonment, family honor, childhood innocence, Countess Olenska
    25: "Sport & Physical Culture",      # horseback riding, athletic competition, gymnastics, physical strength

    # === Expression & Knowledge ===
    16: "Love & Desire",                 # erotic desire, secret rendezvous, sympotic rituals, passion
    17: "Letters & Communication",       # telegram, love letters, speech, letter delivery, chivalry orders
    22: "Education & Intellect",         # education of children, university, classical literature, Atlanta University
    24: "Medicine & Health",             # doctor-patient, medical treatment, surgery, sick man's bedside

    # === Nature & Survival ===
    23: "Slavery & Desolation",          # freedmen's bureau, free labor, blight of slavery, Indian ruins
    21: "Ships & the Sea",              # shipwreck, ship's crew, Captain Smollett, maritime trade
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
