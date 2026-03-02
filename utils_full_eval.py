"""
Utility functions for full 5D personality profile evaluation.

Evaluates whether the model recovers the complete Big Five vector,
using a relaxed criterion (>= 4 out of 5 traits correct) or strict
exact match (all 5 correct).
"""

from utils_traits_eval import (
    split_responses_by_trait,
    parse_responses,
    calculate_personality_scores,
)


def split_dict_in_half(data):
    """Split each value list in a dict into two halves (for train/test splits)."""
    first, second = {}, {}
    for key, vals in data.items():
        mid = (len(vals) + 1) // 2
        first[key] = vals[:mid]
        second[key] = vals[mid:]
    return first, second


def evaluate_success_rate(questions_dict, responses_dict, expected_combinations,
                          total_samples=32, min_correct=4):
    """
    Evaluate success rate: a sample passes if >= min_correct traits match.

    Args:
        questions_dict: Trait -> question list
        responses_dict: Trait -> flat response list
        expected_combinations: List of expected trait polarity lists
        total_samples: Number of profiles (default 32)
        min_correct: Minimum traits correct to count as success (default 4)

    Returns:
        Dict with success_percentage, successful_samples, total_samples
    """
    split_samples = split_responses_by_trait(questions_dict, responses_dict, total_samples)
    parsed = parse_responses(questions_dict, split_samples)

    successful = 0
    for idx, response_dict in enumerate(parsed):
        trait_scores = calculate_personality_scores(questions_dict, response_dict)
        calculated = [
            "positive" if score >= 3 else "negative"
            for score in trait_scores.values()
        ]
        matches = sum(1 for a, b in zip(calculated, expected_combinations[idx]) if a == b)
        if matches >= min_correct:
            successful += 1

    return {
        "success_percentage": (successful / total_samples * 100) if total_samples > 0 else 0,
        "successful_samples": successful,
        "total_samples": total_samples,
    }


def evaluate_exact_match(questions_dict, responses_dict, expected_combinations, total_samples=32):
    """Strict exact match: all 5 traits must be correct."""
    return evaluate_success_rate(
        questions_dict, responses_dict, expected_combinations,
        total_samples=total_samples, min_correct=5,
    )
