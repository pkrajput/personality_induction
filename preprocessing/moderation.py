import argparse
import pandas as pd
import json
import time
from tqdm import tqdm
from openai import OpenAI, RateLimitError
from sklearn.model_selection import train_test_split
import yaml
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--essays_path",
    default=None,
    help="pick the path of the essays csv file",
)
parser.add_argument(
    "--config_path",
    default="./config.yaml",
    help="pick the path of the config yaml file",
)
parser.add_argument(
    "--essays_with_moderation_save_path",
    default=None,
    help="pick the path to save the essays with moderation csv file, for checking manually",
)
parser.add_argument(
    "--path_to_folder_to_save_jsonl_files",
    default=None,
    help="folder path to save the jsonl files",
)


# Define a function to convert 'y' or 'n' to 'positive' or 'negative'
def convert_label_to_trait(label):
    return "positive" if label == "y" else "negative"


# Define a function to create the JSON structure for each row
def create_json_structure(row):
    traits = {
        "openness": convert_label_to_trait(row["cOPN"]),
        "conscientiousness": convert_label_to_trait(row["cCON"]),
        "extroversion": convert_label_to_trait(row["cEXT"]),
        "agreeableness": convert_label_to_trait(row["cAGR"]),
        "neuroticism": convert_label_to_trait(row["cNEU"]),
    }

    user_content = (
        f"Write an essay as a person {traits['openness']} in openness, "
        f"{traits['conscientiousness']} in conscientiousness, "
        f"{traits['extroversion']} in extroversion, "
        f"{traits['agreeableness']} in agreeableness, "
        f"and {traits['neuroticism']} in neuroticism."
    )

    assistant_content = row["TEXT"]

    message = {
        "messages": [
            {
                "role": "system",
                "content": "You are writing an essay that mimics the personality of real humans. You will be given a binary requirement for their Big Five traits.",
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }

    return message


# Function to write a DataFrame to a JSONL file
def write_to_jsonl(df, filename):
    json_list = [create_json_structure(row) for _, row in df.iterrows()]
    with open(filename, "w") as f:
        for item in json_list:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        api_key = config["openai"]["api_key"]

    client = OpenAI(api_key=api_key)
    df = pd.read_csv(args.essays_path, sep=",", encoding="ISO-8859-1")

    # List of categories to add as columns
    categories = [
        "harassment",
        "harassment_threatening",
        "hate",
        "hate_threatening",
        "self_harm",
        "self_harm_instructions",
        "self_harm_intent",
        "sexual",
        "sexual_minors",
        "violence",
        "violence_graphic",
    ]

    # Add new columns for each category with default value as False
    for category in categories:
        df[category] = False

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = row["TEXT"]

        time.sleep(0.5)

        while True:
            try:
                response = client.moderations.create(input=text)
                moderation_result = response.results[0].categories
                df.at[index, "harassment"] = moderation_result.harassment
                df.at[index, "harassment_threatening"] = (
                    moderation_result.harassment_threatening
                )
                df.at[index, "hate"] = moderation_result.hate
                df.at[index, "hate_threatening"] = moderation_result.hate_threatening
                df.at[index, "self_harm"] = moderation_result.self_harm
                df.at[index, "self_harm_instructions"] = (
                    moderation_result.self_harm_instructions
                )
                df.at[index, "self_harm_intent"] = moderation_result.self_harm_intent
                df.at[index, "sexual"] = moderation_result.sexual
                df.at[index, "sexual_minors"] = moderation_result.sexual_minors
                df.at[index, "violence"] = moderation_result.violence
                df.at[index, "violence_graphic"] = moderation_result.violence_graphic

                break
            except RateLimitError:
                print("Rate limit exceeded, waiting for 25 seconds before retrying...")
                time.sleep(25)

    # Save the DataFrame with the moderation categories back to a CSV file
    df.to_csv(args.essays_with_moderation_save_path, index=False, encoding="ISO-8859-1")

    # Filter the DataFrame to keep only rows where all category values are False
    filtered_df = df[(df[categories] == False).all(axis=1)]

    # Split the filtered DataFrame into train (80%), validation (10%), and test (10%) sets
    train_df, temp_df = train_test_split(filtered_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Write the train, validation, and test sets to separate JSONL files
    write_to_jsonl(
        train_df, os.path.join(args.path_to_folder_to_save_jsonl_files, "train.jsonl")
    )
    write_to_jsonl(
        val_df,
        os.path.join(args.path_to_folder_to_save_jsonl_files, "validation.jsonl"),
    )
    write_to_jsonl(
        test_df, os.path.join(args.path_to_folder_to_save_jsonl_files, "test.jsonl")
    )

    print("CSV with moderation categories and JSONL files created successfully.")
