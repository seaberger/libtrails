"""
Compare topic extraction quality across models using the DSPy-optimized prompt.

Loads the optimized prompt + few-shot demos from the DSPy run, then evaluates
the same 40 validation examples on multiple models to determine if a cheaper
API model can match local gemma3:4b quality.

Usage:
    uv run python experiments/compare_models.py
"""

import sys
import time
from pathlib import Path

import dspy
from dotenv import load_dotenv

# Import from the DSPy experiment (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from dspy_topic_optimization import (
    TopicExtractor,
    build_training_set,
    topic_quality_metric,
    _embed_topic,
)

load_dotenv()


def evaluate_model(program, lm, val_set, model_name):
    """Evaluate an optimized program with a specific LM on the validation set."""
    scores = []
    times = []
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")

    with dspy.context(lm=lm):
        for i, ex in enumerate(val_set):
            try:
                t0 = time.time()
                pred = program(passage=ex.passage, book_context=ex.book_context)
                elapsed = time.time() - t0
                score = topic_quality_metric(ex, pred)
                scores.append(score)
                times.append(elapsed)

                # Show first few and last few
                if i < 3 or i >= len(val_set) - 2:
                    topics_preview = pred.topics[:3] if hasattr(pred, "topics") else []
                    print(f"  [{i+1:2d}] {score:.3f} ({elapsed:.1f}s) | {topics_preview}...")
                elif i == 3:
                    print(f"  ... ({len(val_set) - 5} more examples) ...")
            except Exception as e:
                print(f"  [{i+1:2d}] ERROR: {e}")
                scores.append(0.0)
                times.append(0.0)

    avg = sum(scores) / len(scores) if scores else 0
    avg_time = sum(times) / len(times) if times else 0
    total_time = sum(times)
    print(f"\n  Avg score:   {avg:.3f}")
    print(f"  Avg time:    {avg_time:.1f}s per example")
    print(f"  Total time:  {total_time:.0f}s")
    return scores, times


def main():
    print("=" * 60)
    print("Model Comparison: Optimized Topic Extraction Prompt")
    print("=" * 60)

    # Warm up BGE model
    print("\nLoading BGE embedding model...")
    _embed_topic("warmup")
    print("  Ready.")

    # Build validation set (same split as optimization)
    all_examples = build_training_set()
    val_set = all_examples[10:]
    print(f"\nValidation examples: {len(val_set)}")

    # Load optimized program
    optimized_path = Path(__file__).parent / "optimized_topic_extractor.json"
    if not optimized_path.exists():
        print(f"ERROR: {optimized_path} not found. Run dspy_topic_optimization.py first.")
        sys.exit(1)

    # Define models to compare
    models = {
        "gemma3:4b (local Ollama)": dspy.LM(
            "ollama_chat/gemma3:4b",
            api_base="http://localhost:11434",
            api_key="",
            temperature=0.7,
            num_ctx=8192,
        ),
        "Gemini 2.5 Flash-Lite (API)": dspy.LM(
            "gemini/gemini-2.5-flash-lite",
            temperature=0.7,
            max_tokens=4096,
        ),
        "Gemini 3 Flash (API)": dspy.LM(
            "gemini/gemini-3-flash-preview",
            temperature=0.7,
            max_tokens=4096,
        ),
    }

    # Evaluate each model with the optimized prompt
    results = {}
    for model_name, lm in models.items():
        # Load fresh copy of optimized program for each model
        program = TopicExtractor()
        program.load(str(optimized_path))

        scores, times = evaluate_model(program, lm, val_set, model_name)
        results[model_name] = {
            "scores": scores,
            "times": times,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "avg_time": sum(times) / len(times) if times else 0,
        }

    # Summary comparison
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"{'Model':<35} {'Avg Score':>10} {'Avg Time':>10} {'Total':>8}")
    print("-" * 65)

    baseline_name = "gemma3:4b (local Ollama)"
    baseline_avg = results[baseline_name]["avg_score"]

    for name, r in results.items():
        delta = r["avg_score"] - baseline_avg
        delta_str = f"({delta:+.3f})" if name != baseline_name else "(baseline)"
        total = sum(r["times"])
        print(
            f"{name:<35} {r['avg_score']:>8.3f}  {r['avg_time']:>8.1f}s {total:>6.0f}s  {delta_str}"
        )

    # Per-example comparison
    print(f"\n{'=' * 60}")
    print("  PER-EXAMPLE BREAKDOWN (worst 5 gaps)")
    print("=" * 60)
    model_names = list(results.keys())
    gaps = []
    for i in range(len(val_set)):
        local_score = results[baseline_name]["scores"][i]
        for name in model_names:
            if name == baseline_name:
                continue
            api_score = results[name]["scores"][i]
            gaps.append((local_score - api_score, i, name, local_score, api_score))

    gaps.sort(reverse=True)
    for gap, idx, name, local, api in gaps[:5]:
        short_name = name.split("(")[0].strip()
        print(f"  Example {idx+1:2d}: {baseline_name.split('(')[0].strip()} {local:.3f} vs {short_name} {api:.3f}  (gap: {gap:+.3f})")


if __name__ == "__main__":
    main()
