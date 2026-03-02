"""
Create SFT dataset variant that includes questionnaire fragments in the prompt.

The paper describes two SFT variants:
1. Essays only (handled by moderation.py)
2. Essays + questionnaire fragments (this script)

For variant (2), we split the questionnaire 50/50 into train/test, then for each
personality essay we generate:
  (a) An ideal questionnaire response for that personality type
  (b) A randomized response where the mean matches the personality type

The train-split questionnaire + responses are appended to the input prompt.
This doubles the dataset (~4.2k samples).
"""

import argparse
import json
import os
import random

import pandas as pd


TRAIT_MAP = {
    "cAGR": "Agreeableness",
    "cEXT": "Extraversion",
    "cCON": "Conscientiousness",
    "cNEU": "Neuroticism",
    "cOPN": "Openness To Experience",
}


def convert_label(label):
    return "positive" if label == "y" else "negative"


def generate_ideal_responses(questions, trait_positive):
    """Generate ideal (deterministic) questionnaire responses for a given trait polarity."""
    responses = []
    for q in questions:
        if trait_positive:
            score = 5 if q["math"] == 1 else 1
        else:
            score = 1 if q["math"] == 1 else 5
        responses.append({"question": q["question"], "score": score})
    return responses


def generate_randomized_responses(questions, trait_positive):
    """
    Generate randomized responses where the mean matches the personality type.
    For positive traits: mean ~4.0, for negative traits: mean ~2.0.
    """
    target_mean = 4.0 if trait_positive else 2.0
    n = len(questions)
    responses = []
    scores = []

    for q in questions:
        if trait_positive:
            base = random.choice([3, 4, 5])
        else:
            base = random.choice([1, 2, 3])
        if q["math"] == -1:
            base = 6 - base
        scores.append(base)

    current_mean = sum(scores) / n
    adjustment = target_mean - current_mean
    for i in range(n):
        adjusted = max(1, min(5, round(scores[i] + adjustment)))
        scores[i] = adjusted

    for i, q in enumerate(questions):
        responses.append({"question": q["question"], "score": scores[i]})
    return responses


def format_questionnaire_context(questions_dict_train, row):
    """Format train-split questionnaire + responses as context string."""
    lines = []
    for trait_col, trait_name in TRAIT_MAP.items():
        if trait_name not in questions_dict_train:
            continue
        trait_positive = row[trait_col] == "y"
        train_qs = questions_dict_train[trait_name]
        responses = generate_ideal_responses(train_qs, trait_positive)
        lines.append(f"\n[{trait_name} Questionnaire Responses]")
        for r in responses:
            lines.append(f"  Q: {r['question']} -> Score: {r['score']}")
    return "\n".join(lines)


def create_json_structure_with_questions(row, questions_dict_train):
    """Create a chat-format JSON entry with questionnaire context in the user prompt."""
    traits = {
        "openness": convert_label(row["cOPN"]),
        "conscientiousness": convert_label(row["cCON"]),
        "extroversion": convert_label(row["cEXT"]),
        "agreeableness": convert_label(row["cAGR"]),
        "neuroticism": convert_label(row["cNEU"]),
    }

    questionnaire_context = format_questionnaire_context(questions_dict_train, row)

    user_content = (
        f"Write an essay as a person {traits['openness']} in openness, "
        f"{traits['conscientiousness']} in conscientiousness, "
        f"{traits['extroversion']} in extroversion, "
        f"{traits['agreeableness']} in agreeableness, "
        f"and {traits['neuroticism']} in neuroticism.\n\n"
        f"The following are the person's responses to personality questionnaire items:"
        f"{questionnaire_context}"
    )

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are writing an essay that mimics the personality of real humans. "
                    "You will be given a binary requirement for their Big Five traits "
                    "along with their questionnaire responses."
                ),
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row["TEXT"]},
        ]
    }


def create_json_structure_with_random_questions(row, questions_dict_train):
    """Same as above but with randomized (mean-matching) questionnaire responses."""
    traits = {
        "openness": convert_label(row["cOPN"]),
        "conscientiousness": convert_label(row["cCON"]),
        "extroversion": convert_label(row["cEXT"]),
        "agreeableness": convert_label(row["cAGR"]),
        "neuroticism": convert_label(row["cNEU"]),
    }

    lines = []
    for trait_col, trait_name in TRAIT_MAP.items():
        if trait_name not in questions_dict_train:
            continue
        trait_positive = row[trait_col] == "y"
        train_qs = questions_dict_train[trait_name]
        responses = generate_randomized_responses(train_qs, trait_positive)
        lines.append(f"\n[{trait_name} Questionnaire Responses]")
        for r in responses:
            lines.append(f"  Q: {r['question']} -> Score: {r['score']}")
    questionnaire_context = "\n".join(lines)

    user_content = (
        f"Write an essay as a person {traits['openness']} in openness, "
        f"{traits['conscientiousness']} in conscientiousness, "
        f"{traits['extroversion']} in extroversion, "
        f"{traits['agreeableness']} in agreeableness, "
        f"and {traits['neuroticism']} in neuroticism.\n\n"
        f"The following are the person's responses to personality questionnaire items:"
        f"{questionnaire_context}"
    )

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are writing an essay that mimics the personality of real humans. "
                    "You will be given a binary requirement for their Big Five traits "
                    "along with their questionnaire responses."
                ),
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row["TEXT"]},
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create SFT dataset with questionnaire fragments"
    )
    parser.add_argument(
        "--essays_path",
        type=str,
        required=True,
        help="Path to moderated essays CSV",
    )
    parser.add_argument(
        "--questions_path",
        type=str,
        required=True,
        help="Path to questions.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save output JSONL files",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.essays_path, encoding="ISO-8859-1")

    moderation_cols = [
        "harassment", "harassment_threatening", "hate", "hate_threatening",
        "self_harm", "self_harm_instructions", "self_harm_intent",
        "sexual", "sexual_minors", "violence", "violence_graphic",
    ]
    existing_mod_cols = [c for c in moderation_cols if c in df.columns]
    if existing_mod_cols:
        df = df[(df[existing_mod_cols] == False).all(axis=1)]
        print(f"After moderation filtering: {len(df)} essays remain")

    with open(args.questions_path) as f:
        questions_dict = json.load(f)

    questions_dict_train = {
        trait: data["train"] for trait, data in questions_dict.items()
    }

    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)

    all_samples = []

    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        samples = []
        for _, row in split_df.iterrows():
            samples.append(create_json_structure_with_questions(row, questions_dict_train))
            samples.append(create_json_structure_with_random_questions(row, questions_dict_train))

        out_path = os.path.join(args.output_dir, f"{split_name}_with_questions.jsonl")
        with open(out_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"Wrote {len(samples)} samples to {out_path}")
        all_samples.extend(samples)

    print(f"\nTotal samples (doubled from questionnaire variants): {len(all_samples)}")


if __name__ == "__main__":
    main()
