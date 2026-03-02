"""
Utility functions for per-trait personality evaluation.

Evaluates individual Big Five trait accuracy by comparing calculated trait
scores from questionnaire responses against expected personality profiles.
"""


def split_responses_by_trait(questions_dict, responses_dict, total_samples):
    """
    Split flat responses into per-sample chunks based on question counts per trait.

    Args:
        questions_dict: {trait_name: [{"question": str, "math": int}, ...]}
        responses_dict: {trait_name: [all responses concatenated across samples]}
        total_samples: Number of personality profiles evaluated

    Returns:
        List of dicts, one per sample: [{trait_name: [responses for this sample]}, ...]
    """
    question_counts = {trait: len(qs) for trait, qs in questions_dict.items()}
    split_samples = [{} for _ in range(total_samples)]

    for trait, responses in responses_dict.items():
        chunk_size = question_counts[trait]
        assert len(responses) == total_samples * chunk_size, (
            f"Mismatch for {trait}: expected {total_samples * chunk_size}, got {len(responses)}"
        )
        for i in range(total_samples):
            split_samples[i][trait] = responses[i * chunk_size : (i + 1) * chunk_size]

    return split_samples


def parse_responses(questions_dict, split_samples):
    """
    Convert split samples into {question_id: int_score} dicts for scoring.

    Args:
        questions_dict: Trait -> question list mapping
        split_samples: Output of split_responses_by_trait

    Returns:
        List of {question_id: score} dicts, one per sample
    """
    parsed = []
    for sample in split_samples:
        responses = {}
        qid = 1
        for trait, vals in sample.items():
            for val in vals:
                try:
                    responses[qid] = int(val)
                except ValueError:
                    pass
                qid += 1
        parsed.append(responses)
    return parsed


def calculate_personality_scores(questions_dict, response_dict):
    """
    Compute average score per trait, applying reverse scoring where math == -1.

    Args:
        questions_dict: {trait: [{"question": str, "math": 1|-1}, ...]}
        response_dict: {question_id: score}

    Returns:
        {trait: average_score}
    """
    scores = {}
    qid = 1
    for trait, questions in questions_dict.items():
        total, count = 0, 0
        for q in questions:
            if qid in response_dict:
                try:
                    s = int(response_dict[qid])
                    if q["math"] == -1:
                        s = 6 - s
                    total += s
                    count += 1
                except ValueError:
                    pass
            qid += 1
        scores[trait] = total / count if count > 0 else 0
    return scores


def evaluate_trait_accuracy(questions_dict, responses_dict, expected_combinations, total_samples=32):
    """
    Evaluate per-trait accuracy: what fraction of samples have each trait correct.

    Args:
        questions_dict: Trait -> question list
        responses_dict: Trait -> flat response list
        expected_combinations: List of expected trait lists, e.g.
            [["positive", "negative", ...], ...] one per sample
        total_samples: Number of profiles evaluated (default 32)

    Returns:
        {trait: accuracy_percentage}
    """
    split_samples = split_responses_by_trait(questions_dict, responses_dict, total_samples)
    parsed = parse_responses(questions_dict, split_samples)

    trait_correct = {trait: 0 for trait in questions_dict}
    trait_names = list(questions_dict.keys())

    for idx, response_dict in enumerate(parsed):
        trait_scores = calculate_personality_scores(questions_dict, response_dict)
        calculated = {
            trait: "positive" if score >= 3 else "negative"
            for trait, score in trait_scores.items()
        }
        for i, trait in enumerate(trait_names):
            if calculated[trait] == expected_combinations[idx][i]:
                trait_correct[trait] += 1

    return {trait: (c / total_samples) * 100 for trait, c in trait_correct.items()}


def extract_scores(input_dict):
    """
    Extract numeric scores from responses prefixed with 'My score for the statement is: '.

    Args:
        input_dict: {trait: [response_strings]}

    Returns:
        {trait: [score_strings or "NaN"]}
    """
    result = {}
    prefix = "My score for the statement is: "
    for key, strings in input_dict.items():
        result[key] = []
        for s in strings:
            if s.startswith(prefix):
                try:
                    num = s[len(prefix):].split()[0].rstrip(".")
                    if num.replace(".", "", 1).isdigit():
                        result[key].append(num)
                    else:
                        result[key].append("NaN")
                except (IndexError, ValueError):
                    result[key].append("NaN")
            else:
                result[key].append("NaN")
    return result
