import argparse
import json
import random

import pandas as pd


def create_prompt(row):
    traits = {
        "cEXT": "positive" if row["cEXT"] == "y" else "negative",
        "cNEU": "positive" if row["cNEU"] == "y" else "negative",
        "cAGR": "positive" if row["cAGR"] == "y" else "negative",
        "cCON": "positive" if row["cCON"] == "y" else "negative",
        "cOPN": "positive" if row["cOPN"] == "y" else "negative",
    }
    return (
        f"Write a free-hand essay about your feelings in the moment as if you are a person "
        f"who is {traits['cEXT']} in extraversion, {traits['cNEU']} in neuroticism, "
        f"{traits['cAGR']} in agreeableness, {traits['cCON']} in conscientiousness, "
        f"and {traits['cOPN']} in openness."
    )


def get_non_matching_text(df, current_row):
    trait_cols = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
    while True:
        random_row = df.sample().iloc[0]
        if not all(current_row[t] == random_row[t] for t in trait_cols):
            return random_row["TEXT"]


def create_preference_examples(dataframe, num_rejected=3):
    """For each essay, create `num_rejected` preference pairs with non-matching profiles."""
    examples = []
    for _, row in dataframe.iterrows():
        prompt = create_prompt(row)
        chosen_text = row["TEXT"]
        for _ in range(num_rejected):
            rejected_text = get_non_matching_text(dataframe, row)
            examples.append({
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
            })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Create DPO/ORPO preference dataset from essays")
    parser.add_argument(
        "--essays_path",
        type=str,
        required=True,
        help="Path to the moderated essays CSV (essays_post_openai_moderation.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save the output JSON files",
    )
    parser.add_argument(
        "--num_rejected",
        type=int,
        default=3,
        help="Number of rejected samples per chosen essay (default: 3)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Fraction of data to use for test set (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

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

    split_idx = int(len(df) * (1 - args.test_size))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print(f"Train: {len(train_df)} essays -> ~{len(train_df) * args.num_rejected} preference pairs")
    print(f"Test:  {len(test_df)} essays -> ~{len(test_df) * args.num_rejected} preference pairs")

    train_examples = create_preference_examples(train_df, args.num_rejected)
    test_examples = create_preference_examples(test_df, args.num_rejected)

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "dpo_train_dataset.json")
    test_path = os.path.join(args.output_dir, "dpo_test_dataset.json")

    with open(train_path, "w") as f:
        json.dump(train_examples, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_examples, f, indent=2)

    print(f"Saved: {train_path} ({len(train_examples)} pairs)")
    print(f"Saved: {test_path} ({len(test_examples)} pairs)")


if __name__ == "__main__":
    main()
