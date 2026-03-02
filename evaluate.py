"""
Evaluation script for personality induction results.

Reads inference output JSON files and computes:
- Per-trait accuracy (utils_traits_eval)
- Full 5D profile exact match (utils_full_eval)
- Variance/standard deviation analysis across prompt sets
- Censored vs uncensored comparison
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def load_results(results_path):
    """Load inference results JSON file."""
    with open(results_path) as f:
        return json.load(f)


def compute_variance_analysis(results_files):
    """
    Compute standard deviation of trait scores across 32 profiles.
    Corresponds to Table 4 / Figure 4 in the paper.
    """
    print("\n" + "=" * 60)
    print("VARIANCE ANALYSIS (Std Dev of trait scores across profiles)")
    print("=" * 60)

    for path in results_files:
        data = load_results(path)
        results = data.get("results", [])
        name = os.path.basename(path)

        trait_scores_by_trait = defaultdict(list)
        for r in results:
            for trait, score in r.get("trait_scores", {}).items():
                trait_scores_by_trait[trait].append(score)

        print(f"\n  {name}:")
        all_stds = []
        for trait, scores in trait_scores_by_trait.items():
            std = np.std(scores)
            all_stds.append(std)
            print(f"    {trait}: mean={np.mean(scores):.3f}, std={std:.3f}")

        if all_stds:
            print(f"    Overall avg std: {np.mean(all_stds):.3f}")


def compute_exact_match(results_files):
    """
    Compute exact match for full 5D personality profile.
    Corresponds to Table 5 in the paper.
    """
    print("\n" + "=" * 60)
    print("EXACT MATCH ANALYSIS (Full 5D profile accuracy)")
    print("=" * 60)

    for path in results_files:
        data = load_results(path)
        results = data.get("results", [])
        metrics = data.get("metrics", {})
        name = os.path.basename(path)

        exact = metrics.get("exact_match", 0)
        total = metrics.get("total_profiles", 32)
        pct = metrics.get("exact_match_pct", 0)

        print(f"\n  {name}:")
        print(f"    Exact match: {exact}/{total} ({pct:.2f}%)")
        print(f"    Random baseline: {100/32:.2f}%")

        trait_acc = metrics.get("per_trait_accuracy", {})
        for trait, acc in trait_acc.items():
            print(f"    {trait}: {acc:.2f}%")


def compute_per_trait_accuracy(results_files):
    """
    Compute per-trait accuracy (positive/negative).
    """
    print("\n" + "=" * 60)
    print("PER-TRAIT ACCURACY")
    print("=" * 60)

    for path in results_files:
        data = load_results(path)
        results = data.get("results", [])
        name = os.path.basename(path)

        trait_correct = defaultdict(int)
        trait_total = defaultdict(int)

        for r in results:
            target = r.get("target", {})
            predicted = r.get("predicted", {})
            for trait in target:
                trait_total[trait] += 1
                if target[trait] == predicted.get(trait):
                    trait_correct[trait] += 1

        print(f"\n  {name}:")
        for trait in sorted(trait_total.keys()):
            acc = trait_correct[trait] / trait_total[trait] * 100 if trait_total[trait] > 0 else 0
            print(f"    {trait}: {trait_correct[trait]}/{trait_total[trait]} ({acc:.2f}%)")


def compute_nan_analysis(results_files):
    """Analyze NaN rates across models and prompt sets."""
    print("\n" + "=" * 60)
    print("NaN RATE ANALYSIS")
    print("=" * 60)

    for path in results_files:
        data = load_results(path)
        results = data.get("results", [])
        name = os.path.basename(path)

        total_nan = sum(r.get("nan_count", 0) for r in results)
        total_qs = sum(r.get("total_questions", 0) for r in results)
        rate = total_nan / total_qs * 100 if total_qs > 0 else 0

        print(f"  {name}: {total_nan}/{total_qs} ({rate:.1f}%)")


def compare_prompt_sets(results_dir, model_name):
    """Compare S1, S2, S3 results for the same model."""
    print("\n" + "=" * 60)
    print(f"PROMPT SET COMPARISON: {model_name}")
    print("=" * 60)

    for prompt_set in ["S1", "S2", "S3"]:
        pattern = f"results_{model_name}_{prompt_set}.json"
        path = os.path.join(results_dir, pattern)
        if os.path.exists(path):
            data = load_results(path)
            results = data.get("results", [])

            trait_scores = defaultdict(list)
            for r in results:
                for trait, score in r.get("trait_scores", {}).items():
                    trait_scores[trait].append(score)

            avg_std = np.mean([np.std(scores) for scores in trait_scores.values()])
            exact = data.get("metrics", {}).get("exact_match_pct", 0)

            print(f"  {prompt_set}: avg_std={avg_std:.3f}, exact_match={exact:.2f}%")
        else:
            print(f"  {prompt_set}: not found")


def main():
    parser = argparse.ArgumentParser(description="Evaluate personality induction results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./eval_results",
        help="Directory containing inference result JSON files",
    )
    parser.add_argument(
        "--results_files",
        nargs="+",
        default=None,
        help="Specific result files to evaluate (overrides --results_dir)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name for prompt set comparison",
    )
    parser.add_argument(
        "--analysis",
        nargs="+",
        choices=["variance", "exact_match", "per_trait", "nan", "prompt_comparison", "all"],
        default=["all"],
        help="Which analyses to run",
    )
    args = parser.parse_args()

    if args.results_files:
        files = args.results_files
    else:
        if not os.path.exists(args.results_dir):
            print(f"Results directory not found: {args.results_dir}")
            return
        files = sorted([
            os.path.join(args.results_dir, f)
            for f in os.listdir(args.results_dir)
            if f.endswith(".json")
        ])

    if not files:
        print("No result files found.")
        return

    print(f"Found {len(files)} result file(s)")
    analyses = args.analysis if "all" not in args.analysis else [
        "variance", "exact_match", "per_trait", "nan"
    ]

    if "variance" in analyses:
        compute_variance_analysis(files)
    if "exact_match" in analyses:
        compute_exact_match(files)
    if "per_trait" in analyses:
        compute_per_trait_accuracy(files)
    if "nan" in analyses:
        compute_nan_analysis(files)
    if "prompt_comparison" in analyses and args.model_name:
        compare_prompt_sets(args.results_dir, args.model_name)


if __name__ == "__main__":
    main()
