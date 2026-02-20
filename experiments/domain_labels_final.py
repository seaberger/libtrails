"""
Apply human-refined labels to super-clusters and prepare for database integration.

Usage:
    LIBTRAILS_DB=v2 uv run python experiments/domain_labels_final.py
    LIBTRAILS_DB=v2 uv run libtrails load-domains
"""

import json
from pathlib import Path

SUPER_CLUSTERS_PATH = Path(__file__).parent / "super_clusters_k5_g0035.json"
OUTPUT_PATH = Path(__file__).parent / "domain_labels_final.json"

# Human-curated labels for V2 library (921 books)
# Generated from Leiden domain clustering at k=5, γ=0.0035 (66 domains)
# Curated by Claude Opus 4.6 from top-30 topic analysis of each domain
# Maps super_cluster_id -> final label
REFINED_LABELS = {
    # === Fiction & Narrative (large character/world domains) ===
    0: "Character Drama & Emotion",       # Character emotional arcs: Devon, Briony, Atticus, Doctor Manette, George & Lennie
    1: "Speculative Fiction",             # Worldbuilding fiction: Dune, Hyperion, His Dark Materials, 100 Years of Solitude, Murakami
    25: "European Literary Fiction",       # Mann, Hesse, Musil, Bellow: Buddenbrooks, Glass Bead Game, Man Without Qualities, Herzog
    58: "Royal Courts & Dynasties",        # ASOIAF political drama + historical monarchy: direwolves, Maester Aemon, Catherine de Medici

    # === Knowledge & Technical ===
    2: "Technical & Quantitative",         # ML/programming (HuggingFace, Python), physics (Einstein), trading signals, linear algebra
    8: "Writing Craft & Publishing",       # Rushdie fatwa, publishing industry, short story craft, screenwriting, New Yorker
    11: "Literary Figures & Criticism",    # Browning, Henry James, Hardy, Kipling, Swinburne, Marlowe — British literary heritage

    # === Business & Economics ===
    3: "Business & Investing",             # Applied: startups, marketing, Berkshire Hathaway, freelancing, niche markets, IPOs
    35: "Economic Theory & Systems",       # Theoretical: Keynesian, Marxism, capitalism critique, usury, Cantillon effect

    # === Mind & Self ===
    4: "Psychology & Self-Improvement",    # CBT, growth mindset, lifestyle design, second brain, deliberate practice, scout mindset
    15: "Fear & Anxiety",                  # Fear of the unknown, stage fright, foreboding, stress response, PTSD
    20: "Philosophy & Character Study",    # Cynic philosophy, Stoic indifferents, intellectual arrogance, Belbo, Aschenbach/Tadzio
    24: "Philosophy & Epistemology",       # Falsifiability, cargo cult science, Hume, Spinoza, Schrödinger's cat, nihilism
    42: "Identity & Fate",                 # Amor fati, character identity, doomsday, rash decisions, road trips
    49: "Vices & Indulgence",              # Smoking, alcohol, absinthe, drugs, food as comfort, drunkenness

    # === Religion & Philosophy ===
    5: "Religion & Classical Learning",    # Christianity, psalms, Augustine, mystery plays, liberal education, Archbishop Whately
    50: "Eastern Philosophy & Asian History", # Siddhartha, Daoism, Confucius, Bhagavad Gita, Qing dynasty, Zen Buddhism

    # === Culinary ===
    6: "Culinary Arts",                    # Pure cooking: sous vide, forcemeat, brioche, reverse sear, egg coagulation, polenta

    # === Politics & Governance ===
    7: "Politics & Governance",            # Elections, presidential norms, populism, New Deal, Weimar, political movements
    13: "Early American History",           # Mount Vernon, Valley Forge, Civil War, Emancipation Proclamation, Shays' Rebellion
    27: "Civil Rights & World History",    # Nation of Islam, MLK, civil rights, Du Bois + Mesopotamia, Anthropocene

    # === Science & Technology ===
    9: "Sci-Fi Technology & Space",        # Spaceships, propulsion, space elevators, brain-computer interfaces — NOT military
    61: "AI & Technological Progress",     # Automaton prototypes, advanced AI, cyborgization, robotic reconnaissance, innovation
    54: "Climate & Energy",                # Arctic exploitation, fracking, carbon emissions, solar panels, climate change

    # === Nature & Environment ===
    10: "Natural Environments & Wilderness", # Eels, Everglades, Sargasso Sea, volcanoes, weather, mountain passes
    39: "Land, Agriculture & Forestry",    # Crop rotation, cotton picking, land speculation, forests, farmers' markets

    # === Military & Conflict ===
    12: "Modern Warfare & Geopolitics",    # ISIS, SEAL teams, Vietnam, Syria, Israel-Palestine — real-world conflicts
    23: "Weapons & Authority",             # Pat Tillman, Steelheart, antimatter weaponry, chain of command, artillery

    # === Society & Institutions ===
    14: "Society & Social Order",          # Class, duty, aristocracy, philanthropy, population control, social commentary
    26: "Education & Training",            # Elodin's teaching, Harvard, Oxford exams, meritocracy critique, aptitude testing
    43: "Employment & Institutional Power", # Employee benefits, civil service reform, slavery, labor, Tenure of Office Act
    45: "Law & Justice",                   # Habeas corpus, legal profession, systemic injustice, eyewitness testimony

    # === Relationships & Family ===
    17: "Family & Parenthood",             # Pregnancy, child rearing, parental identity, maternal guilt, foster care
    32: "Love & Relationships",            # Romance, marriage, divorce, love at first sight, engagement, companionship
    52: "Sexuality & Gender",              # Bloom's monologue, decadence, sex work, gay literature, AIDS, female warriors
    28: "Coming of Age & Hidden Pasts",    # WWI childhood, generational trauma, concealed identity, grandmother's influence

    # === Health & Medicine ===
    22: "Health, Medicine & Biology",       # Pathogens, longevity medicine, mRNA vaccines, yoga, embodiment, sociobiology
    33: "Medicine & Drug Trade",           # Surgery, Alzheimer's, iatrogenics, drug trafficking, mental deterioration
    63: "Wounds & Battlefield Medicine",   # Gunshot wounds, burial, blood transfusions, casualties, battlefield care

    # === Crime & Security ===
    19: "Espionage & Secrecy",             # Spy revenge plots, surveillance, polygraph, whistleblowers, intelligence failures
    29: "Crime & Investigation",           # Barefoot Bandit, forensics, courtroom arrests, prison system, detective studies

    # === Places & Spaces ===
    36: "Cities & Urban Life",             # NYC, Harlem, rural China, Central Park, neighborhood dynamics
    34: "Buildings & Physical Spaces",     # House/palace architecture, heists, hotels, vaults, log cabins — NOT intrusion
    30: "Colonialism & the Tropics",       # Cholera, Panama Canal, Congo reform, García Márquez, African colonization
    21: "Ancient & Medieval History",       # Roman Empire, Crusades, Julian the Apostate, Germanic tribes, Renaissance, Plato

    # === Arts & Culture ===
    31: "Art, Treasure & Archaeology",     # Young artists, Caravaggio, Vermeer, Baroque, Flemish painting, treasure recovery
    44: "Music & Performance",             # Mozart, Bach, Beethoven, Vinteuil sonata, opera, jazz, dance
    55: "Film, Photography & Theatre",     # Aperture, film aesthetics, image processing, adjustment layers, Diderot, theatre
    47: "Crafts & Trades",                 # Baking, glove making, silk industry, woodworking, grooming, artisanal skills
    53: "Celebrations & Gatherings",       # Parties, fireworks, birthday, summer vacation, feasts, jubilees

    # === Survival & Danger ===
    16: "Survival & Animal Encounters",    # Richard Parker, lions, tigers, Morlocks, coyotes — predator-prey dynamics
    18: "Games, Sports & Competition",     # Baseball, card games, pull-ups, grip strength, tennis, chess — NOT conflict
    41: "Maritime & Naval",                # Shipwrecks, U-boats, Suez Canal, boat building, circumnavigation

    # === Death & Destruction ===
    37: "Catastrophe & Destruction",       # Cultural artifact destruction, plane crashes, Auschwitz, nuclear waste, wrecks
    38: "Death & Mortality",               # Body disposal, grief, funeral attendees, finality of death, scientific immortality
    65: "Fire & Flames",                   # Goblet of Fire, wildfire, campfire, arson, fire magic — all about fire

    # === Communication & Time ===
    40: "Communication & Messaging",       # Telegrams, Morse code, radio, email, spam classification, secret messages
    60: "Time & Temporality",              # Time travel, Python time module, clockwork, alien calendars, productivity

    # === Perception & Magic ===
    57: "Light, Darkness & Color",         # Night vision, chromaturgy, color wights, Zima Blue, moonlit scenes, candles
    59: "Psychic Powers & Perception",     # Telepathy, séances, spirit guides, perspective shifting, psychic trauma
    62: "Magic & Fairy Tales",             # Muggle-born prejudice, Faerie Queene, fairy stories, enchantment

    # === Travel & Movement ===
    46: "Travel & Transportation",         # Trains, cabs, coaches, Trans-Siberian, automobile, horse-drawn travel
    48: "Migration & Displacement",        # Reunions, shelter-seeking, housing crisis, La Bestia, refugees, immigration

    # === Media & Narrative ===
    56: "Media & Journalism",              # Political controversies, BBC, colonial newspapers, propaganda, TV scriptwriting
    64: "Historiography",                  # Death of the author, Walter Scott, historical fiction, role of the historian

    # === Language ===
    51: "Language & Linguistics",          # Houyhnhnm language, active/passive voice, Swift, Babel-17, translation, POS tagging
}


def main():
    with open(SUPER_CLUSTERS_PATH) as f:
        super_clusters = json.load(f)

    # Verify all IDs are mapped
    unmapped = [sc["super_cluster_id"] for sc in super_clusters if sc["super_cluster_id"] not in REFINED_LABELS]
    if unmapped:
        print(f"WARNING: Unmapped super-cluster IDs: {unmapped}")
        print("Add these to REFINED_LABELS before proceeding.")
        return

    # Check for duplicate labels
    label_counts: dict[str, list[int]] = {}
    for sc_id, label in REFINED_LABELS.items():
        label_counts.setdefault(label, []).append(sc_id)
    duplicates = {label: ids for label, ids in label_counts.items() if len(ids) > 1}
    if duplicates:
        print("WARNING: Duplicate labels found (domains will be merged):")
        for label, ids in duplicates.items():
            print(f"  {label}: IDs {ids}")
        print()

    # Build final domains (merging where labels match)
    domains: dict[str, dict] = {}

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

        # Aggregate top topics across all clusters
        for cluster in sc["leiden_clusters"]:
            for t in cluster["top_topics"]:
                topic_label = t["label"]
                count = t["count"]
                if topic_label not in domains[label]["top_topics"]:
                    domains[label]["top_topics"][topic_label] = 0
                domains[label]["top_topics"][topic_label] += count

    # Convert to list and assign new sequential IDs
    result = []
    for i, (label, data) in enumerate(
        sorted(domains.items(), key=lambda x: len(x[1]["leiden_clusters"]), reverse=True)
    ):
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
    print("FINAL DOMAIN LABELS (Human-Curated, Claude Opus 4.6)")
    print("=" * 70)
    print(f"| {'ID':>2} | {'Clusters':>8} | {'Label':<40} |")
    print(f"|{'-' * 4}|{'-' * 10}|{'-' * 42}|")

    total_clusters = 0
    for d in result:
        print(f"| {d['domain_id']:2d} | {d['cluster_count']:8d} | {d['label']:<40} |")
        total_clusters += d["cluster_count"]

    print(f"|{'-' * 4}|{'-' * 10}|{'-' * 42}|")
    print(f"| {'':2} | {total_clusters:8d} | {'TOTAL':<40} |")
    print()
    print(f"Domains: {len(result)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
