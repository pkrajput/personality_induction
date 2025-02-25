def split_dict_in_half(data):
    first_half = {}
    second_half = {}

    for key, value_list in data.items():
        mid = (len(value_list) + 1) // 2
        first_half[key] = value_list[:mid]
        second_half[key] = value_list[mid:]

    return first_half, second_half


def split_responses_by_trait(questions_dict: dict, responses_dict: dict) -> list:
    """
    Split the responses based on the number of questions in each trait for 32 samples.

    Args:
        questions_dict (dict): Dictionary containing the questions for each trait.
                               Format: {trait: [{"question": str, "math": int}, ...]}.
        responses_dict (dict): Dictionary containing responses for each trait.
                               Format: {'trait': [responses]}.

    Returns:
        list: A list of response dictionaries split into 32 samples.
              Format: [{trait: [responses]}, ...].
    """
    # Calculate the number of questions per trait
    question_counts = {
        trait: len(questions) for trait, questions in questions_dict.items()
    }

    # Initialize a list to store the 32 samples
    total_samples = 32
    split_samples = [{} for _ in range(total_samples)]

    # For each trait, split responses into 32 chunks
    for trait, responses in responses_dict.items():
        chunk_size = question_counts[
            trait
        ]  # Determine the number of responses needed for each sample
        assert (
            len(responses) == total_samples * chunk_size
        ), f"Mismatch in response length for {trait}. Expected: {total_samples * chunk_size}, Got: {len(responses)}"

        # Split the responses into chunks for each sample
        for i in range(total_samples):
            start = i * chunk_size
            end = start + chunk_size
            split_samples[i][trait] = responses[start:end]

    return split_samples


def parse_responses(questions_dict: dict, split_samples: list) -> list:
    """
    Parse the split samples into a format compatible with the scoring function.

    Args:
        questions_dict (dict): Dictionary containing questions and multipliers for each trait.
                               Format: {trait: [{"question": str, "math": int}, ...]}.
        split_samples (list): List of split response samples.
                              Format: [{trait: [responses]}, ...].

    Returns:
        list: A list of response dictionaries in the format:
              [{question_id: response}, ...].
    """
    parsed_samples = []

    for sample in split_samples:
        sample_responses = {}
        question_id = 1

        # Map the responses to the corresponding questions in each trait
        for trait, responses in sample.items():
            questions = questions_dict[trait]
            for idx, response in enumerate(responses):
                try:
                    sample_responses[question_id] = int(response)
                except ValueError:
                    # Skip invalid responses that cannot be converted to int
                    continue
                question_id += 1

        parsed_samples.append(sample_responses)

    return parsed_samples


def evaluate_success_rate(
    questions_dict: dict, responses_dict: dict, expected_combinations: list
) -> dict:
    """
    Evaluate success rate based on comparison of calculated traits to expected traits.

    Args:
        questions_dict (dict): Contains trait mapping for questions.
                               Format: {trait_name: [{"question": str, "math": int}]}.
        responses_dict (dict): Dictionary containing responses for each trait.
                               Format: {'trait_name': [score1, score2, ...]}.
        expected_combinations (list): List of expected combinations of positive/negative traits.
                                      Format: [[trait1, trait2, trait3, ...], ...].

    Returns:
        dict: Dictionary with success percentage and number of successful samples.
    """
    # Split responses into 32 separate samples based on traits
    split_samples = split_responses_by_trait(questions_dict, responses_dict)

    # Parse responses into a format compatible with scoring
    response_samples = parse_responses(questions_dict, split_samples)

    # Calculate the success rate using parsed samples.
    successful_samples = 0

    for idx, response_dict in enumerate(response_samples):
        # Calculate the average scores for each trait.
        trait_scores = calculate_personality_scores(questions_dict, response_dict)

        # Determine if each trait is positive or negative.
        calculated_traits = [
            "positive" if score >= 3 else "negative" for score in trait_scores.values()
        ]

        # Compare with the expected combination for this sample.
        if (
            sum(
                1
                for x, y in zip(calculated_traits, expected_combinations[idx])
                if x == y
            )
            >= 4
        ):
            successful_samples += 1

    # Calculate success percentage.
    total_samples = len(response_samples)
    success_percentage = (
        (successful_samples / total_samples) * 100 if total_samples > 0 else 0
    )

    return {
        "success_percentage": success_percentage,
        "successful_samples": successful_samples,
        "total_samples": total_samples,
    }


# Supporting function: Calculates personality scores for individual responses.
def calculate_personality_scores(questions_dict: dict, response_dict: dict) -> dict:
    """
    Calculate the average scores for each personality trait.

    Args:
        questions_dict (dict): Dictionary with question mappings for each trait.
                               Format: {trait_name: [{"question": str, "math": int}]}.
        response_dict (dict): Dictionary with the recorded responses for each question.
                              Format: {question_id: score}.

    Returns:
        dict: Average scores for each personality trait.
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
                    # Reverse the score based on math value (if applicable)
                    if question["math"] == -1:
                        score = 6 - score

                    total_score += score
                    count += 1
                except ValueError:
                    # Skip invalid scores that cannot be converted to int
                    continue
            question_id += 1

        # Calculate average if count is not zero.
        if count > 0:
            trait_scores[trait] = total_score / count
        else:
            trait_scores[trait] = 0  # If no valid questions, set score to 0.

    return trait_scores
