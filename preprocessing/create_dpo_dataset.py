import pandas as pd
import random
import json

# Load the dataset
df = pd.read_csv("/Users/prateek.rajput/Documents/personality_essays/essays_with_moderation.csv", encoding="ISO-8859-1")

# Function to construct the prompt
def create_prompt(row):
    traits = {
        "cEXT": "positive" if row["cEXT"] == "y" else "negative",
        "cNEU": "positive" if row["cNEU"] == "y" else "negative",
        "cAGR": "positive" if row["cAGR"] == "y" else "negative",
        "cCON": "positive" if row["cCON"] == "y" else "negative",
        "cOPN": "positive" if row["cOPN"] == "y" else "negative",
    }
    
    prompt = (f"Write a free-hand essay about your feelings in the moment as if you are a person "
              f"who is {traits['cEXT']} in extraversion, {traits['cNEU']} in neuroticism, "
              f"{traits['cAGR']} in agreeableness, {traits['cCON']} in conscientiousness, "
              f"and {traits['cOPN']} in openness.")
    
    return prompt

# Function to find a non-matching random essay for the "rejected" text
def get_non_matching_text(df, current_row):
    while True:
        random_row = df.sample().iloc[0]
        if not all(current_row[trait] == random_row[trait] for trait in ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]):
            return random_row["TEXT"]

# Split into training and test sets (last 5 rows for test)
train_df = df[:-5]  # All except last 5 rows
test_df = df[-5:]   # Last 5 rows

# Function to generate preference examples
def create_preference_examples(dataframe):
    preference_examples = []
    for _, row in dataframe.iterrows():
        prompt = create_prompt(row)
        chosen_text = row["TEXT"]

        for _ in range(3):  # Create 3 data points per essay
            rejected_text = get_non_matching_text(dataframe, row)
            preference_example = {
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text
            }
            preference_examples.append(preference_example)
    return preference_examples

# Create training and test examples
train_examples = create_preference_examples(train_df)
test_examples = create_preference_examples(test_df)

# Save the training and test examples as separate JSON files
with open("./dpo_train_dataset.json", "w") as train_file:
    json.dump(train_examples, train_file, indent=4)

with open("./dpo_test_dataset.json", "w") as test_file:
    json.dump(test_examples, test_file, indent=4)

print("Training and test datasets created and saved as 'dpo_train_dataset.json' and 'dpo_test_dataset.json'")
