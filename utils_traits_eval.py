def split_responses_by_trait(
    questions_dict: dict, responses_dict: dict, total_samples: int
) -> list:
    """
    Split the responses based on the number of questions in each trait for total number of samples.
    """
    question_counts = {
        trait: len(questions) for trait, questions in questions_dict.items()
    }
    total_samples = total_samples
    split_samples = [{} for _ in range(total_samples)]

    for trait, responses in responses_dict.items():
        chunk_size = question_counts[trait]
        assert (
            len(responses) == total_samples * chunk_size
        ), f"Mismatch in response length for {trait}. Expected: {total_samples * chunk_size}, Got: {len(responses)}"

        for i in range(total_samples):
            start = i * chunk_size
            end = start + chunk_size
            split_samples[i][trait] = responses[start:end]

    return split_samples


def parse_responses(questions_dict: dict, split_samples: list) -> list:
    """
    Parse the split samples into a format compatible with the scoring function.
    """
    parsed_samples = []

    for sample in split_samples:
        sample_responses = {}
        question_id = 1

        for trait, responses in sample.items():
            questions = questions_dict[trait]
            for idx, response in enumerate(responses):
                try:
                    sample_responses[question_id] = int(response)
                except ValueError:
                    continue
                question_id += 1

        parsed_samples.append(sample_responses)

    return parsed_samples


def calculate_personality_scores(questions_dict: dict, response_dict: dict) -> dict:
    """
    Calculate the average scores for each personality trait.
    """
    trait_scores = {}
    question_id = 1

    for trait, questions in questions_dict.items():
        total_score = 0
        count = 0

        for question in questions:
            if question_id in response_dict:
                try:
                    score = int(response_dict[question_id])
                    if question["math"] == -1:
                        score = 6 - score

                    total_score += score
                    count += 1
                except ValueError:
                    continue
            question_id += 1

        trait_scores[trait] = total_score / count if count > 0 else 0

    return trait_scores


def evaluate_trait_accuracy(
    questions_dict: dict, responses_dict: dict, expected_combinations: list
) -> dict:
    """
    Evaluate individual trait accuracy based on comparison of calculated traits to expected traits.
    """
    split_samples = split_responses_by_trait(questions_dict, responses_dict)
    response_samples = parse_responses(questions_dict, split_samples)

    trait_correct_counts = {trait: 0 for trait in questions_dict.keys()}
    total_counts = len(response_samples)

    for idx, response_dict in enumerate(response_samples):
        trait_scores = calculate_personality_scores(questions_dict, response_dict)
        calculated_traits = {
            trait: "positive" if score >= 3 else "negative"
            for trait, score in trait_scores.items()
        }

        for trait, calculated in calculated_traits.items():
            expected = expected_combinations[idx][
                list(questions_dict.keys()).index(trait)
            ]
            if calculated == expected:
                trait_correct_counts[trait] += 1

    accuracy_percentages = {
        trait: (correct / total_counts) * 100
        for trait, correct in trait_correct_counts.items()
    }

    return accuracy_percentages
