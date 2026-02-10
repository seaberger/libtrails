"""
DSPy experiment: Optimize topic extraction prompt using MIPROv2.

Uses gemma3:27b as the prompt proposer and gemma3:4b as the task model
(matching our production pipeline). Trains on 50 gold-standard examples
spanning 10 books across genres.

Quality metric uses BGE-small-en-v1.5 cosine similarity (same embedding
model as our pipeline) for semantic overlap scoring, plus structural
checks for word count, stoplist, and topic count compliance.

Usage:
    uv run python experiments/dspy_topic_optimization.py

After optimization, the best prompt is saved to experiments/optimized_topic_prompt.json
and can be inspected or integrated into the extraction pipeline.
"""

from functools import lru_cache
from pathlib import Path

import dspy
import numpy as np
from dotenv import load_dotenv

from libtrails.embeddings import embed_texts

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Configure DSPy with Ollama (task) + Gemini (prompt proposer)
# ---------------------------------------------------------------------------

# Task model: local gemma3:4b — our production model for chunk-level extraction.
# MIPROv2 evaluates candidates against this model, so the optimized prompt
# is guaranteed to work well with our actual pipeline.
task_lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
    temperature=0.7,
    num_ctx=8192,
)

# Prompt proposer: Gemini 3 Flash via API — generates candidate instructions
# and few-shot bootstrapping. Beats 2.5 Pro on benchmarks at 75% less cost.
# Reads GEMINI_API_KEY from .env automatically.
prompt_lm = dspy.LM(
    "gemini/gemini-3-flash-preview",
    temperature=1.0,
    max_tokens=8192,
)

dspy.configure(lm=task_lm)


# ---------------------------------------------------------------------------
# 2. Define the DSPy Signature and Module
# ---------------------------------------------------------------------------

class ExtractTopics(dspy.Signature):
    """Extract specific topic labels from a book passage.

    Given a passage from a book along with the book's title, author, and
    high-level themes, extract exactly 5 topic labels that capture the
    key concepts discussed in this specific passage.
    """

    passage: str = dspy.InputField(desc="A ~500 word passage from a book")
    book_context: str = dspy.InputField(
        desc="Book title, author, and high-level themes"
    )
    topics: list[str] = dspy.OutputField(
        desc="Exactly 5 specific topic labels as multi-word noun phrases (2-5 words each)"
    )


class TopicExtractor(dspy.Module):
    def __init__(self):
        self.extract = dspy.Predict(ExtractTopics)

    def forward(self, passage, book_context):
        return self.extract(passage=passage, book_context=book_context)


# ---------------------------------------------------------------------------
# 3. Quality Metric
# ---------------------------------------------------------------------------

# Stoplist from our pipeline
STOPLIST = {
    "manipulation", "relationships", "technology", "society", "nature", "life",
    "death", "love", "power", "time", "people", "world", "change", "future",
    "communication", "conflict", "loss", "survival", "identity", "freedom",
    "control", "trust", "fear", "growth", "knowledge", "science", "culture",
    "politics", "art", "history",
}


@lru_cache(maxsize=512)
def _embed_topic(topic: str) -> tuple:
    """Embed a single topic string, cached to avoid recomputation."""
    vec = embed_texts([topic])[0]
    return tuple(vec.tolist())


def _cosine_similarity_matrix(topics_a: list[str], topics_b: list[str]) -> np.ndarray:
    """Compute pairwise cosine similarities using BGE embeddings.

    Returns matrix of shape (len(topics_a), len(topics_b)).
    Embeddings are normalized by BGE, so dot product = cosine similarity.
    Raw similarities are rescaled: floor at 0.5, then stretch to [0, 1].
    This makes unrelated topics (~0.4 cosine) map to ~0 and related
    topics (~0.8+ cosine) map to ~0.6+, giving the optimizer a clearer signal.
    """
    vecs_a = np.array([_embed_topic(t) for t in topics_a])
    vecs_b = np.array([_embed_topic(t) for t in topics_b])
    raw = vecs_a @ vecs_b.T
    return np.clip((raw - 0.5) / 0.5, 0.0, 1.0)


def topic_quality_metric(example, prediction, trace=None):
    """
    Score predicted topics against gold standard on multiple criteria.

    Returns float 0.0-1.0 combining:
    - Semantic overlap with gold topics via BGE cosine similarity (40%)
    - Word count compliance: 2-5 words per topic (25%)
    - No single-word generics from stoplist (15%)
    - No overly long topics > 6 words (10%)
    - Correct count: exactly 5 topics (10%)
    """
    gold_topics = example.topics
    pred_topics = prediction.topics if hasattr(prediction, "topics") else []

    if not pred_topics:
        return 0.0

    # Normalize
    pred_normalized = [t.strip().lower() for t in pred_topics if t.strip()]
    gold_normalized = [t.strip().lower() for t in gold_topics if t.strip()]

    if not pred_normalized:
        return 0.0

    # --- Count compliance (10%) ---
    count_score = 1.0 if len(pred_normalized) == 5 else max(0, 1 - abs(len(pred_normalized) - 5) * 0.2)

    # --- Word count compliance (25%) ---
    word_count_scores = []
    for t in pred_normalized:
        wc = len(t.split())
        if 2 <= wc <= 5:
            word_count_scores.append(1.0)
        elif wc == 1:
            word_count_scores.append(0.0)
        elif wc == 6:
            word_count_scores.append(0.5)
        else:  # 7+
            word_count_scores.append(0.0)
    wc_score = sum(word_count_scores) / len(word_count_scores) if word_count_scores else 0

    # --- Stoplist compliance (15%) ---
    stoplist_violations = sum(1 for t in pred_normalized if t in STOPLIST)
    stoplist_score = 1.0 - (stoplist_violations / len(pred_normalized))

    # --- No overly long topics (10%) ---
    long_violations = sum(1 for t in pred_normalized if len(t.split()) > 6)
    length_score = 1.0 - (long_violations / len(pred_normalized))

    # --- Semantic overlap with gold via BGE embeddings (40%) ---
    sim_matrix = _cosine_similarity_matrix(gold_normalized, pred_normalized)

    # For each gold topic, find best matching predicted topic
    gold_matches = sim_matrix.max(axis=1).tolist()

    # For each predicted topic, find best matching gold topic
    pred_matches = sim_matrix.max(axis=0).tolist()

    # F1-style: harmonic mean of recall-analog and precision-analog
    recall_analog = sum(gold_matches) / len(gold_matches) if gold_matches else 0
    precision_analog = sum(pred_matches) / len(pred_matches) if pred_matches else 0
    semantic_score = (
        2 * recall_analog * precision_analog / (recall_analog + precision_analog)
        if (recall_analog + precision_analog) > 0
        else 0
    )

    # --- Weighted combination ---
    total = (
        0.40 * semantic_score
        + 0.25 * wc_score
        + 0.15 * stoplist_score
        + 0.10 * length_score
        + 0.10 * count_score
    )
    return total


# ---------------------------------------------------------------------------
# 4. Gold Standard Training Examples (50 examples across 10 books)
# ---------------------------------------------------------------------------

def build_training_set():
    """Build 50 gold-standard examples spanning diverse genres."""

    examples = []

    def ex(passage, context, topics):
        examples.append(
            dspy.Example(
                passage=passage,
                book_context=context,
                topics=topics,
            ).with_inputs("passage", "book_context")
        )

    # -----------------------------------------------------------------------
    # SIDDHARTHA by Hermann Hesse (spiritual fiction) — 8 examples
    # -----------------------------------------------------------------------
    sid_ctx = (
        "Book: Siddhartha by Hermann Hesse\n"
        "Book themes: indian religious philosophy, spiritual self-discovery, "
        "renunciation and worldly life, brahmanical social structure, "
        "the search for enlightenment, early buddhist influences"
    )

    ex(
        "Siddhartha had started to nurse discontent in himself, he had started to feel that the love of his father and the love of his mother, and also the love of his friend, Govinda, would not bring him joy for ever and ever, would not nurse him, feed him, satisfy him. He had started to suspect that his venerable father and his other teachers, that the wise Brahmans had already revealed to him the most and best of their wisdom, that they had already filled his expecting vessel with their richness, and the vessel was not full, the spirit was not content, the soul was not calm, the heart was not satisfied. The ablutions were good, but they were water, they did not wash off the sin, they did not heal the spirit's thirst, they did not relieve the fear in his heart.",
        sid_ctx,
        [
            "spiritual dissatisfaction",
            "brahmanical ritual limitations",
            "quest for deeper wisdom",
            "father-son expectations",
            "unfulfilled spiritual thirst",
        ],
    )

    ex(
        "mind to be void of all conceptions. These and other ways he learned to go, a thousand times he left his self, for hours and days he remained in the non-self. But though the ways led away from the self, their end nevertheless always led back to the self. Though Siddhartha fled from the self a thousand times, stayed in nothingness, stayed in the animal, in the stone, the return was inevitable, inescapable was the hour, when he found himself back in the sunshine or in the moonlight, in the shade or in the rain, and was once again his self and Siddhartha, and again felt the agony of the cycle which had been forced upon him.",
        sid_ctx,
        [
            "samana meditation practices",
            "escape from self",
            "inescapable return to selfhood",
            "agony of cyclical existence",
            "non-self experience",
        ],
    )

    ex(
        "What is meditation? What is leaving one's body? What is fasting? What is holding one's breath? It is fleeing from the self, it is a short escape of the agony of being a self, it is a short numbing of the senses against the pain and the pointlessness of life. The same escape, the same short numbing is what the driver of an ox-cart finds in the inn, drinking a few bowls of rice-wine or fermented coconut-milk. Then he won't feel his self any more, then he won't feel the pains of life any more, then he finds a short numbing of the senses.",
        sid_ctx,
        [
            "critique of ascetic practices",
            "meditation as escapism",
            "sensory numbing parallel",
            "pointlessness of self-denial",
            "samana-drunkard comparison",
        ],
    )

    ex(
        "At times he felt, deep in his chest, a dying, quiet voice, which admonished him quietly, lamented quietly; he hardly perceived it. And then, for an hour, he became aware of the strange life he was leading, of him doing lots of things which were only a game, of, though being happy and feeling joy at times, real life still passing him by and not touching him. As a ball-player plays with his balls, he played with his business-deals, with the people around him, watched them, found amusement in them; with his heart, with the source of his being, he was not with them.",
        sid_ctx,
        [
            "inner voice of warning",
            "worldly life as game",
            "emotional detachment from commerce",
            "hollow material pleasures",
            "spiritual disconnection",
        ],
    )

    ex(
        "Bring him into the city, bring him into his mother's house, there'll still be servants around, give him to them. And when there aren't any around any more, bring him to a teacher, not for the teachings' sake, but so that he shall be among other boys, and among girls, and in the world which is his own. Have you never thought of this?",
        sid_ctx,
        [
            "vasudeva's parenting counsel",
            "father-son conflict",
            "releasing parental control",
            "child's need for peers",
            "samsaric cycle of generations",
        ],
    )

    ex(
        "Which father, which teacher had been able to protect him from living his life for himself, from soiling himself with life, from burdening himself with guilt, from drinking the bitter drink for himself, from finding his path for himself? Would you think, my dear, anybody might perhaps be spared from taking this path?",
        sid_ctx,
        [
            "inevitability of personal experience",
            "limits of parental protection",
            "self-discovery through suffering",
            "vasudeva's river wisdom",
            "futility of shielding from life",
        ],
    )

    ex(
        "Not all people are smart. You are like me, you are different from most people. You are Kamala, nothing else, and inside of you, there is a peace and refuge, to which you can go at every hour of the day and be at home at yourself, as I can also do. Few people have this, and yet all could have it. Most people are like a falling leaf, which is blown and is turning around through the air, and wavers, and tumbles to the ground. But others, a few, are like stars, they go on a fixed course, no wind reaches them, in themselves they have their law and their course.",
        sid_ctx,
        [
            "inner refuge of self",
            "kamala's spiritual kinship",
            "leaf-and-star metaphor",
            "rare self-possession",
            "distinction from ordinary people",
        ],
    )

    ex(
        "This here, he said playing with it, is a stone, and will, after a certain time, perhaps turn into soil, and will turn from soil into a plant or animal or human being. I could think: this stone is just a stone, it is worthless. But perhaps it is also God and Buddha. I do not respect and love it because it was one thing and will become something else, but because it has already long been everything and always is everything.",
        sid_ctx,
        [
            "stone as teaching metaphor",
            "unity of all existence",
            "timelessness of matter",
            "sacred nature of objects",
            "rejection of linear progress",
        ],
    )

    # -----------------------------------------------------------------------
    # DUNE by Frank Herbert (sci-fi/politics) — 6 examples
    # -----------------------------------------------------------------------
    dune_ctx = (
        "Book: Dune by Frank Herbert\n"
        "Book themes: desert ecology and survival, political intrigue and feudalism, "
        "prescient abilities and fate, spice melange economy, "
        "fremen culture and resistance, messianic prophecy"
    )

    ex(
        "A beginning is the time for taking the most delicate care that the balances are correct. This every sister of the Bene Gesserit knows. To begin your study of the life of Muad'Dib, then, take care that you first place him in his time: born in the 57th year of the Padishah Emperor, Shaddam IV. And take the most special care that you locate Muad'Dib in his place: the planet Arrakis. Do not be deceived by the fact that he was born on Caladan and lived his first fifteen years there. Arrakis, the planet known as Dune, is forever his place.",
        dune_ctx,
        [
            "bene gesserit philosophy",
            "muad'dib's origin story",
            "padishah imperial timeline",
            "arrakis as destiny",
            "caladan to dune transition",
        ],
    )

    ex(
        "The Fremen were supreme in that quality the ancients called 'spannungsbogen' — the self-imposed delay between desire for a thing and the act of reaching out to grasp that thing. They could look at their barren land and see paradise. They had the discipline to use their water wisely, to plant wind traps and moisture collectors, to build underground reservoirs.",
        dune_ctx,
        [
            "fremen self-discipline",
            "desert water conservation",
            "spannungsbogen delayed gratification",
            "wind traps and collectors",
            "terraforming vision",
        ],
    )

    ex(
        "He who controls the spice controls the universe. The spice extends life. The spice expands consciousness. The spice is vital to space travel. The Spacing Guild and its navigators, who the spice has mutated over four thousand years, use the orange spice gas, which gives them the ability to fold space. Without the spice, interstellar travel would be impossible.",
        dune_ctx,
        [
            "spice melange properties",
            "spacing guild navigators",
            "interstellar travel dependency",
            "consciousness expansion",
            "galactic economic control",
        ],
    )

    ex(
        "I must not fear. Fear is the mind-killer. Fear is the little-death that brings total obliteration. I will face my fear. I will permit it to pass over me and through me. And when it has gone past, I will turn the inner eye to see its path. Where the fear has gone there will be nothing. Only I will remain.",
        dune_ctx,
        [
            "litany against fear",
            "bene gesserit mental training",
            "mind over emotion",
            "inner eye discipline",
            "fear as annihilation",
        ],
    )

    ex(
        "The Sardaukar were the Emperor's elite troops, trained on the harsh prison planet of Salusa Secundus. Their fighting ability was legendary, forged in conditions so brutal that six out of thirteen recruits died during training. They were the foundation of Imperial power, and their secret was the planet itself — only extreme hardship produced such fierce warriors.",
        dune_ctx,
        [
            "sardaukar military elite",
            "salusa secundus training",
            "imperial military power",
            "harsh environment warfare",
            "brutality breeds warriors",
        ],
    )

    ex(
        "Paul walked among the Fremen, learning their ways. He learned to walk without rhythm on the sand to avoid attracting the great sandworms. He learned the value of a man's water, the ritual of the stillsuit, and the deep respect for Shai-Hulud. Among these desert people, he found not the primitives the Imperium believed them to be, but a culture of profound depth and deadly competence.",
        dune_ctx,
        [
            "paul's fremen integration",
            "sandworm avoidance technique",
            "stillsuit water discipline",
            "shai-hulud reverence",
            "fremen cultural depth",
        ],
    )

    # -----------------------------------------------------------------------
    # 1984 by George Orwell (dystopia) — 5 examples
    # -----------------------------------------------------------------------
    orwell_ctx = (
        "Book: 1984 by George Orwell\n"
        "Book themes: totalitarian surveillance state, thought control and propaganda, "
        "language manipulation through newspeak, individual resistance and rebellion, "
        "historical revisionism"
    )

    ex(
        "It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him. The hallway smelt of boiled cabbage and old rag mats. At one end of it a coloured poster, too large for indoor display, had been tacked to the wall. It depicted simply an enormous face, more than a metre wide: the face of a man of about forty-five, with a heavy black moustache and ruggedly handsome features. BIG BROTHER IS WATCHING YOU, the caption beneath it ran.",
        orwell_ctx,
        [
            "big brother surveillance",
            "victory mansions decay",
            "oppressive urban atmosphere",
            "winston smith introduction",
            "omnipresent propaganda posters",
        ],
    )

    ex(
        "Don't you see that the whole aim of Newspeak is to narrow the range of thought? In the end we shall make thoughtcrime literally impossible, because there will be no words in which to express it. Every concept that can ever be needed will be expressed by exactly one word, with its meaning rigidly defined and all its subsidiary meanings rubbed out and forgotten.",
        orwell_ctx,
        [
            "newspeak linguistic reduction",
            "thought control through language",
            "elimination of conceptual range",
            "impossibility of thoughtcrime",
            "rigid meaning enforcement",
        ],
    )

    ex(
        "Who controls the past controls the future: who controls the present controls the past. The mutability of the past is the central tenet of Ingsoc. Past events, it is argued, have no objective existence, but survive only in written records and in human memories. The past is whatever the records and the memories agree upon.",
        orwell_ctx,
        [
            "historical revisionism doctrine",
            "ingsoc central tenet",
            "mutability of the past",
            "records and memory control",
            "power over historical narrative",
        ],
    )

    ex(
        "He gazed up at the enormous face. Forty years it had taken him to learn what kind of smile was hidden beneath the dark moustache. O cruel, needless misunderstanding! O stubborn, self-willed exile from the loving breast! Two gin-scented tears trickled down the sides of his nose. But it was all right, everything was all right, the struggle was finished. He loved Big Brother.",
        orwell_ctx,
        [
            "winston's final capitulation",
            "psychological breaking point",
            "love for big brother",
            "destroyed individual will",
            "totalitarian victory over spirit",
        ],
    )

    ex(
        "The Ministry of Truth — Minitrue, in Newspeak — was startlingly different from any other object in sight. It was an enormous pyramidal structure of glittering white concrete, soaring up, terrace after terrace, three hundred metres into the air. From where Winston stood it was just possible to read, picked out on its white face in elegant lettering, the three slogans of the Party: WAR IS PEACE, FREEDOM IS SLAVERY, IGNORANCE IS STRENGTH.",
        orwell_ctx,
        [
            "ministry of truth architecture",
            "party contradictory slogans",
            "doublethink institutionalized",
            "monumental state buildings",
            "newspeak naming conventions",
        ],
    )

    # -----------------------------------------------------------------------
    # THINKING FAST AND SLOW by Daniel Kahneman (psychology/behavioral econ) — 5 examples
    # -----------------------------------------------------------------------
    kahn_ctx = (
        "Book: Thinking, Fast and Slow by Daniel Kahneman\n"
        "Book themes: dual process theory of cognition, cognitive biases and heuristics, "
        "prospect theory and loss aversion, overconfidence in judgment, "
        "system 1 versus system 2 thinking"
    )

    ex(
        "System 1 operates automatically and quickly, with little or no effort and no sense of voluntary control. System 2 allocates attention to the effortful mental activities that demand it, including complex computations. The operations of System 2 are often associated with the subjective experience of agency, choice, and concentration.",
        kahn_ctx,
        [
            "dual process theory",
            "system 1 automatic thinking",
            "system 2 deliberate reasoning",
            "effortful mental computation",
            "cognitive agency and control",
        ],
    )

    ex(
        "A reliable way to make people believe in falsehoods is frequent repetition, because familiarity is not easily distinguished from truth. Authoritarian institutions and marketers have always known this fact. But it was psychologists who discovered that you do not have to repeat the entire statement of a fact or idea to make it appear true.",
        kahn_ctx,
        [
            "repetition creates belief",
            "familiarity-truth confusion",
            "cognitive ease and persuasion",
            "propaganda psychological mechanism",
            "illusory truth effect",
        ],
    )

    ex(
        "The concept of loss aversion is certainly the most significant contribution of psychology to behavioral economics. People are not loss-neutral: they are much more unhappy about losing $100 than they are happy about gaining $100. The loss aversion ratio has been estimated in several experiments and is usually in the range of 1.5 to 2.5.",
        kahn_ctx,
        [
            "loss aversion ratio",
            "asymmetric value perception",
            "behavioral economics foundation",
            "losing versus gaining psychology",
            "prospect theory empirical evidence",
        ],
    )

    ex(
        "The anchoring effect is one of the most robust findings of experimental psychology: estimates of all kinds are contaminated by prior numbers. When people are asked whether Gandhi was more or less than 144 years old when he died, they end up with much higher estimates of his age at death than people who were asked if he was more or less than 35.",
        kahn_ctx,
        [
            "anchoring effect demonstration",
            "prior number contamination",
            "cognitive bias in estimation",
            "gandhi age experiment",
            "judgment under uncertainty",
        ],
    )

    ex(
        "We are prone to overestimate how much we understand about the world and to underestimate the role of chance in events. Overconfidence is fed by the illusory certainty of hindsight. The confidence people have in their beliefs is not a measure of the quality of evidence but of the coherence of the story the mind has constructed.",
        kahn_ctx,
        [
            "overconfidence bias",
            "hindsight illusory certainty",
            "narrative coherence fallacy",
            "role of chance underestimated",
            "belief-evidence disconnect",
        ],
    )

    # -----------------------------------------------------------------------
    # SAPIENS by Yuval Noah Harari (history/anthropology) — 5 examples
    # -----------------------------------------------------------------------
    sap_ctx = (
        "Book: Sapiens by Yuval Noah Harari\n"
        "Book themes: cognitive revolution and human evolution, "
        "agricultural revolution consequences, imagined orders and social constructs, "
        "unification of humankind through empire and religion, "
        "scientific revolution and modernity"
    )

    ex(
        "The Cognitive Revolution is the point when history declared its independence from biology. Until the Cognitive Revolution, the doings of all human species belonged to the realm of biology. From the Cognitive Revolution onwards, historical narratives replace biological theories. To understand the rise of Christianity or the French Revolution, it is not enough to comprehend the interaction of genes, hormones, and organisms. It is necessary to take into account the interaction of ideas, images, and fantasies as well.",
        sap_ctx,
        [
            "cognitive revolution threshold",
            "history beyond biology",
            "ideas replacing genetic determinism",
            "cultural versus biological evolution",
            "narrative-driven human development",
        ],
    )

    ex(
        "The Agricultural Revolution was history's biggest fraud. Who was responsible? Neither kings, nor priests, nor merchants. The culprits were a handful of plant species, including wheat, rice, and potatoes. These plants domesticated Homo sapiens, rather than vice versa. Wheat didn't like rocks and pebbles, so Sapiens broke their backs clearing fields. Wheat didn't like sharing water with other plants, so men and women spent long hours weeding.",
        sap_ctx,
        [
            "agricultural revolution critique",
            "wheat domesticating humans",
            "farming as historical fraud",
            "labor burden of agriculture",
            "plant-human power reversal",
        ],
    )

    ex(
        "How did Homo sapiens manage to cross this critical threshold, eventually founding cities comprising tens of thousands of inhabitants and empires ruling hundreds of millions? The secret was probably the appearance of fiction. Large numbers of strangers can cooperate successfully by believing in common myths. Any large-scale human cooperation — whether a modern state, a medieval church, an ancient city, or an archaic tribe — is rooted in common myths that exist only in people's collective imagination.",
        sap_ctx,
        [
            "fiction enables cooperation",
            "imagined orders and myths",
            "collective belief systems",
            "large-scale human coordination",
            "social constructs as reality",
        ],
    )

    ex(
        "Money is the most universal and most efficient system of mutual trust ever devised. What created this trust was a very complex and long-term network of political, social, and economic relations. Why do I believe in the gold coin or the dollar bill? Because my neighbours believe in them. And my neighbours believe in them because I believe in them.",
        sap_ctx,
        [
            "money as mutual trust",
            "intersubjective belief networks",
            "economic faith systems",
            "circular validation of currency",
            "universal exchange medium",
        ],
    )

    ex(
        "The Scientific Revolution has not been a revolution of knowledge. It has been above all a revolution of ignorance. The great discovery that launched the Scientific Revolution was the discovery that humans do not know the answers to their most important questions. Premodern traditions of knowledge such as Islam, Christianity, Buddhism and Confucianism asserted that everything that is important to know about the world was already known.",
        sap_ctx,
        [
            "revolution of ignorance",
            "admitting knowledge gaps",
            "premodern certainty versus inquiry",
            "scientific humility principle",
            "departure from dogmatic knowledge",
        ],
    )

    # -----------------------------------------------------------------------
    # MISTBORN: THE FINAL EMPIRE by Brandon Sanderson (fantasy) — 5 examples
    # -----------------------------------------------------------------------
    mist_ctx = (
        "Book: Mistborn: The Final Empire by Brandon Sanderson\n"
        "Book themes: allomantic magic system, class rebellion against tyranny, "
        "the lord ruler's immortal reign, skaa oppression and resistance, "
        "heist narrative structure, prophecy and the hero of ages"
    )

    ex(
        "Kelsier burned tin, flaring his senses. The mists swirled around him, and he could hear the faintest sounds from the streets below — a guard's footstep, the distant clank of a forge. Tin was one of the more subtle metals, but no less useful. With tin burning, he could see in near-darkness, hear whispers across a room, and feel the slightest brush of wind against his skin.",
        mist_ctx,
        [
            "tin allomantic enhancement",
            "heightened sensory perception",
            "kelsier's mistborn abilities",
            "mist-shrouded night operations",
            "subtle metal applications",
        ],
    )

    ex(
        "The skaa had been beaten down for a thousand years. They worked in the lord ruler's fields, labored in his mines, and served in the noble houses. They were forbidden to own property, forbidden to read, forbidden to travel without permission. And yet, underground, there was always resistance — the skaa underground, a network of rebels who dreamed of overthrowing the Final Empire.",
        mist_ctx,
        [
            "skaa systematic oppression",
            "thousand-year subjugation",
            "forbidden literacy and property",
            "underground resistance network",
            "dreams of imperial overthrow",
        ],
    )

    ex(
        "Vin pulled on her copper, hiding her allomantic pulses from any seekers who might be watching. Then she burned steel, pushing against the metal studs in the wall. She lurched into the air, coins trailing behind her like a metallic tail, and soared over the rooftops of Luthadel, the mists embracing her like a cloak.",
        mist_ctx,
        [
            "steel push aerial travel",
            "copper cloud concealment",
            "coin-based allomantic combat",
            "vin's growing mistborn skill",
            "luthadel rooftop navigation",
        ],
    )

    ex(
        "The plan was simple in concept, audacious in scope. Kelsier's crew would simultaneously undermine the Lord Ruler's economy, sow distrust among the noble houses, build a skaa army in secret, and ultimately assault Kredik Shaw, the Lord Ruler's palace. It was, as Breeze noted with characteristic understatement, 'the most ambitious heist in the history of the Final Empire.'",
        mist_ctx,
        [
            "kelsier's rebellion plan",
            "economic destabilization strategy",
            "noble house manipulation",
            "secret skaa army recruitment",
            "kredik shaw assault plan",
        ],
    )

    ex(
        "The Lord Ruler had reigned for a thousand years, his power seemingly absolute and eternal. He had reshaped the world itself — moving the planet closer to the sun, raising the ashmounts that perpetually darkened the sky, and somehow granting himself immortality. To the skaa, he was God. To the nobles, he was their ultimate authority. To Kelsier, he was the target.",
        mist_ctx,
        [
            "lord ruler's millennium reign",
            "planetary reshaping power",
            "ashmount environmental devastation",
            "immortality and divine authority",
            "kelsier's defiant opposition",
        ],
    )

    # -----------------------------------------------------------------------
    # A BRIEF HISTORY OF TIME by Stephen Hawking (popular science) — 4 examples
    # -----------------------------------------------------------------------
    hawk_ctx = (
        "Book: A Brief History of Time by Stephen Hawking\n"
        "Book themes: origins of the universe, black holes and singularities, "
        "quantum mechanics and general relativity, the nature of time, "
        "unification of physics"
    )

    ex(
        "A well-known scientist once gave a public lecture on astronomy. He described how the earth orbits around the sun and how the sun, in turn, orbits around the center of a vast collection of stars called our galaxy. At the end of the lecture, a little old lady at the back of the room got up and said: What you have told us is rubbish. The world is really a flat plate supported on the back of a giant tortoise. It's turtles all the way down.",
        hawk_ctx,
        [
            "turtles all the way down",
            "cosmological common sense",
            "scientific versus folk models",
            "infinite regression problem",
            "public science communication",
        ],
    )

    ex(
        "If the rate of expansion one second after the Big Bang had been smaller by even one part in a hundred thousand million million, the universe would have recollapsed before it ever reached its present size. On the other hand, if it had been greater by a part in a million, the universe would have expanded too rapidly for stars and planets to form.",
        hawk_ctx,
        [
            "fine-tuning of expansion rate",
            "big bang initial conditions",
            "cosmological precision problem",
            "star formation requirements",
            "anthropic principle evidence",
        ],
    )

    ex(
        "The event horizon, the boundary of the region of space-time from which it is not possible to escape, acts rather like a one-way membrane around the black hole: objects, such as unwary astronauts, can fall through the event horizon into the black hole, but nothing can ever get out of the black hole through the event horizon.",
        hawk_ctx,
        [
            "event horizon boundary",
            "one-way membrane analogy",
            "black hole escape impossibility",
            "space-time boundary region",
            "gravitational point of no return",
        ],
    )

    ex(
        "Einstein's general relativity predicts that time should appear to run slower near a massive body like the earth. This is because there is a relation between the energy of light and its frequency: the greater the energy, the higher the frequency. As light travels upward in the earth's gravitational field, it loses energy, and so its frequency goes down.",
        hawk_ctx,
        [
            "gravitational time dilation",
            "general relativity prediction",
            "light energy and frequency",
            "gravitational field effects",
            "time near massive bodies",
        ],
    )

    # -----------------------------------------------------------------------
    # THE ART OF WAR by Sun Tzu (strategy/philosophy) — 3 examples
    # -----------------------------------------------------------------------
    war_ctx = (
        "Book: The Art of War by Sun Tzu\n"
        "Book themes: military strategy and tactics, deception in warfare, "
        "terrain and positioning advantage, knowing enemy and self, "
        "winning without fighting"
    )

    ex(
        "All warfare is based on deception. Hence, when able to attack, we must seem unable; when using our forces, we must seem inactive; when we are near, we must make the enemy believe we are far away; when far away, we must make him believe we are near. Hold out baits to entice the enemy. Feign disorder, and crush him.",
        war_ctx,
        [
            "deception as warfare foundation",
            "feigned weakness strategy",
            "enemy perception manipulation",
            "baiting tactical approach",
            "appearance versus reality",
        ],
    )

    ex(
        "If you know the enemy and know yourself, you need not fear the result of a hundred battles. If you know yourself but not the enemy, for every victory gained you will also suffer a defeat. If you know neither the enemy nor yourself, you will succumb in every battle.",
        war_ctx,
        [
            "self-knowledge in strategy",
            "enemy intelligence imperative",
            "symmetry of understanding",
            "battle outcome prediction",
            "strategic self-assessment",
        ],
    )

    ex(
        "The supreme art of war is to subdue the enemy without fighting. Thus the highest form of generalship is to balk the enemy's plans; the next best is to prevent the junction of the enemy's forces; the next in order is to attack the enemy's army in the field; and the worst policy of all is to besiege walled cities.",
        war_ctx,
        [
            "subduing without combat",
            "hierarchy of strategic options",
            "enemy plan disruption",
            "force junction prevention",
            "siege as worst strategy",
        ],
    )

    # -----------------------------------------------------------------------
    # THE LEAN STARTUP by Eric Ries (business/entrepreneurship) — 4 examples
    # -----------------------------------------------------------------------
    lean_ctx = (
        "Book: The Lean Startup by Eric Ries\n"
        "Book themes: build-measure-learn feedback loop, minimum viable product, "
        "validated learning through experimentation, pivot versus persevere decisions, "
        "innovation accounting metrics"
    )

    ex(
        "The fundamental activity of a startup is to turn ideas into products, measure how customers respond, and then learn whether to pivot or persevere. All successful startup processes should be geared to accelerate that feedback loop. The Build-Measure-Learn feedback loop is at the core of the Lean Startup model.",
        lean_ctx,
        [
            "build-measure-learn loop",
            "customer response measurement",
            "pivot or persevere decision",
            "feedback loop acceleration",
            "lean startup core model",
        ],
    )

    ex(
        "A minimum viable product (MVP) is that version of a new product which allows a team to collect the maximum amount of validated learning about customers with the least effort. It is not necessarily the smallest product imaginable, but rather the fastest way to get through the Build-Measure-Learn feedback loop with the minimum amount of effort.",
        lean_ctx,
        [
            "minimum viable product definition",
            "validated learning maximization",
            "least effort experimentation",
            "rapid feedback iteration",
            "mvp strategic purpose",
        ],
    )

    ex(
        "Vanity metrics are the numbers that look good on paper but don't actually help you understand how your business is performing. Total registered users, total downloads, and raw page views are classic vanity metrics. What matters are actionable metrics — ones that demonstrate clear cause and effect and are tied to specific actions you can take.",
        lean_ctx,
        [
            "vanity metrics critique",
            "actionable metrics distinction",
            "cause-and-effect measurement",
            "misleading growth numbers",
            "innovation accounting rigor",
        ],
    )

    ex(
        "A pivot is a structured course correction designed to test a new fundamental hypothesis about the product, strategy, and engine of growth. The decision to pivot requires a combination of data-driven analysis and entrepreneurial intuition. Some pivots involve changing only one element of the business model, while others may require a complete rethinking of the product.",
        lean_ctx,
        [
            "structured pivot definition",
            "hypothesis testing correction",
            "data and intuition balance",
            "business model element changes",
            "engine of growth adjustment",
        ],
    )

    # -----------------------------------------------------------------------
    # THE GREAT GATSBY by F. Scott Fitzgerald (American literature) — 5 examples
    # -----------------------------------------------------------------------
    gats_ctx = (
        "Book: The Great Gatsby by F. Scott Fitzgerald\n"
        "Book themes: american dream disillusionment, jazz age excess and decadence, "
        "old money versus new money, unrequited obsessive love, "
        "moral decay of the wealthy"
    )

    ex(
        "In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since. Whenever you feel like criticizing any one, he told me, just remember that all the people in this world haven't had the advantages that you've had. He didn't say any more, but we've always been unusually communicative in a reserved way, and I understood that he meant a great deal more than that.",
        gats_ctx,
        [
            "nick carraway's moral education",
            "withholding judgment philosophy",
            "privilege and perspective",
            "reserved family communication",
            "narrator's ethical foundation",
        ],
    )

    ex(
        "Gatsby believed in the green light, the orgastic future that year by year recedes before us. It eluded us then, but that's no matter — tomorrow we will run faster, stretch out our arms farther. And one fine morning — So we beat on, boats against the current, borne back ceaselessly into the past.",
        gats_ctx,
        [
            "green light symbolism",
            "unattainable american dream",
            "boats against the current",
            "ceaselessly into the past",
            "futile pursuit of future",
        ],
    )

    ex(
        "He had one of those rare smiles with a quality of eternal reassurance in it, that you may come across four or five times in life. It faced — or seemed to face — the whole external world for an instant, and then concentrated on you with an irresistible prejudice in your favor. It understood you just as far as you wanted to be understood, believed in you as you would like to believe in yourself.",
        gats_ctx,
        [
            "gatsby's magnetic charisma",
            "performative social warmth",
            "selective understanding appeal",
            "irresistible personal charm",
            "crafted public persona",
        ],
    )

    ex(
        "They were careless people, Tom and Daisy — they smashed up things and creatures and then retreated back into their money or their vast carelessness, or whatever it was that kept them together, and let other people clean up the mess they had made.",
        gats_ctx,
        [
            "old money moral carelessness",
            "tom and daisy's destruction",
            "wealth as insulation",
            "class privilege consequences",
            "emotional wreckage abandoned",
        ],
    )

    ex(
        "The valley of ashes is a fantastic farm where ashes grow like wheat into ridges and hills and grotesque gardens; where ashes take the forms of houses and chimneys and rising smoke and, finally, with a transcendent effort, of ash-grey men, who move dimly and already crumbling through the powdery air. Above the grey land and the spasms of bleak dust, the eyes of Doctor T. J. Eckleburg keep their vigil.",
        gats_ctx,
        [
            "valley of ashes landscape",
            "industrial waste dystopia",
            "eckleburg's watchful eyes",
            "class divide geography",
            "moral wasteland symbolism",
        ],
    )

    return examples


# ---------------------------------------------------------------------------
# 5. Run Optimization
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DSPy Topic Extraction Optimization")
    print("=" * 60)
    print("Task model:   gemma3:4b (Ollama local)")
    print("Prompt model: Gemini 3 Flash (Google API)")
    print()

    # Build training set
    all_examples = build_training_set()
    print(f"Total gold examples: {len(all_examples)}")

    # Split into train / val (DSPy recommends 20% train / 80% val)
    train_set = all_examples[:10]
    val_set = all_examples[10:]
    print(f"Training: {len(train_set)}, Validation: {len(val_set)}")
    print()

    # Warm up BGE model so first metric call doesn't skew timing
    print("Loading BGE embedding model for metric...")
    _embed_topic("warmup")
    print("  BGE model ready.")
    print()

    # First evaluate baseline (unoptimized) on full val set
    print("--- Evaluating baseline (unoptimized) ---")
    baseline = TopicExtractor()
    baseline_scores = []
    for i, ex_item in enumerate(val_set):
        try:
            pred = baseline(passage=ex_item.passage, book_context=ex_item.book_context)
            score = topic_quality_metric(ex_item, pred)
            baseline_scores.append(score)
            if i < 5:
                print(f"  Score: {score:.3f} | Topics: {pred.topics[:3]}...")
        except Exception as e:
            print(f"  Error: {e}")
            baseline_scores.append(0.0)

    if baseline_scores:
        print(f"  ... ({len(baseline_scores)} examples evaluated)")
        print(f"  Baseline avg: {sum(baseline_scores)/len(baseline_scores):.3f}")
    print()

    # Run MIPROv2 optimization (heavy = more Bayesian trials for better results)
    print("--- Running MIPROv2 optimization (heavy) ---")
    print("  (this may take 2-4 hours with local models)")
    print()

    optimizer = dspy.MIPROv2(
        metric=topic_quality_metric,
        prompt_model=prompt_lm,
        task_model=task_lm,
        auto="heavy",
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        verbose=True,
        num_threads=1,  # Ollama handles one request at a time
    )

    optimized = optimizer.compile(
        student=TopicExtractor(),
        trainset=train_set,
        valset=val_set,
        minibatch=True,
        minibatch_size=15,
    )

    # Evaluate optimized
    print()
    print("--- Evaluating optimized ---")
    optimized_scores = []
    for ex_item in val_set:
        try:
            pred = optimized(passage=ex_item.passage, book_context=ex_item.book_context)
            score = topic_quality_metric(ex_item, pred)
            optimized_scores.append(score)
            print(f"  Score: {score:.3f} | Topics: {pred.topics[:3]}...")
        except Exception as e:
            print(f"  Error: {e}")
            optimized_scores.append(0.0)

    if optimized_scores:
        print(f"  Optimized avg: {sum(optimized_scores)/len(optimized_scores):.3f}")
    print()

    # Save results
    output_dir = Path(__file__).parent
    optimized.save(str(output_dir / "optimized_topic_extractor.json"))
    print(f"Saved optimized program to {output_dir / 'optimized_topic_extractor.json'}")

    # Print the optimized prompt
    print()
    print("=" * 60)
    print("OPTIMIZED PROMPT:")
    print("=" * 60)
    for name, param in optimized.named_predictors():
        if hasattr(param, "extended_signature"):
            print(f"\n[{name}]")
            sig = param.extended_signature
            if hasattr(sig, "instructions"):
                print(f"Instructions: {sig.instructions}")
            for field_name in sig.output_fields:
                field = sig.output_fields[field_name]
                if hasattr(field, "json_schema_extra") and field.json_schema_extra:
                    prefix = field.json_schema_extra.get("prefix", "")
                    if prefix:
                        print(f"Output prefix [{field_name}]: {prefix}")
            if hasattr(param, "demos") and param.demos:
                print(f"Few-shot demos: {len(param.demos)}")

    # Summary
    print()
    print("=" * 60)
    if baseline_scores and optimized_scores:
        baseline_avg = sum(baseline_scores) / len(baseline_scores)
        optimized_avg = sum(optimized_scores) / len(optimized_scores)
        improvement = optimized_avg - baseline_avg
        print(f"Baseline avg:   {baseline_avg:.3f}")
        print(f"Optimized avg:  {optimized_avg:.3f}")
        print(f"Improvement:    {improvement:+.3f} ({improvement/max(baseline_avg, 0.001)*100:+.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
