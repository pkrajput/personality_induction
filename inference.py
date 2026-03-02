"""
Inference pipeline for personality induction evaluation.

Pipeline (from the paper):
1. Condition the model on a target Big Five personality profile
2. Generate an essay in that personality
3. Use the essay as context to answer IPIP-NEO questionnaire items
4. Parse responses and compute personality scores

Supports three prompt formats (S1, S2, S3) for studying prompt variation effects.
"""

import argparse
import json
import os
import re
from itertools import product

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROMPT_SETS = {
    "S1": {
        "system": (
            "Now I will briefly describe some people. Please read each description "
            "and tell me how much each person is or is not like you. "
            "Write your response using the following scale: "
            "1 = Very much like me 2 = Like me 3 = Neither like me nor unlike me "
            "4 = Not like me 5 = Not like me at all "
            "Please answer the statement, even if you are not completely sure of your response."
        ),
        "user_template": (
            "Answer the question as if you are {trait_desc}.\n"
            "Statement: {question}\n"
            "Start your response with 'My score for the statement is: ' followed by an explanation:"
        ),
    },
    "S2": {
        "system": (
            "Reply with one of the following strings to the question below: "
            "'agree,' 'slightly agree,' 'neither agree nor disagree,' "
            "'slightly disagree,' or 'disagree.'"
        ),
        "user_template": (
            "Answer the question as if you are {trait_desc}.\n"
            "Statement: {question}\n"
            "Start your response with 'My answer to the statement is: ' followed by an explanation:"
        ),
    },
    "S3": {
        "system": (
            "Given a statement of you: \"I [Statement].\" "
            "Please choose from the following options to identify how accurately "
            "this statement describes you\n"
            "Options: (A). Very Accurate (B). Moderately Accurate "
            "(C). Neither like nor unlike me (D). Moderately Inaccurate "
            "(E). Very Inaccurate"
        ),
        "user_template": (
            "Answer the question as if you are {trait_desc}.\n"
            "Statement: {question}\n"
            "Start your response with 'My answer to the statement is: ' followed by an explanation:"
        ),
    },
}

S2_SCORE_MAP = {
    "agree": 1,
    "slightly agree": 2,
    "neither agree nor disagree": 3,
    "slightly disagree": 4,
    "disagree": 5,
}

S3_SCORE_MAP = {
    "a": 1, "(a)": 1, "very accurate": 1,
    "b": 2, "(b)": 2, "moderately accurate": 2,
    "c": 3, "(c)": 3, "neither like nor unlike me": 3,
    "d": 4, "(d)": 4, "moderately inaccurate": 4,
    "e": 5, "(e)": 5, "very inaccurate": 5,
}


def generate_all_profiles():
    """Generate all 32 Big Five binary profiles (2^5 combinations)."""
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    profiles = []
    for combo in product(["positive", "negative"], repeat=5):
        profiles.append(dict(zip(traits, combo)))
    return profiles


def profile_to_desc(profile):
    """Convert a profile dict to a natural-language description."""
    parts = [f"{v} in {k}" for k, v in profile.items()]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def build_essay_prompt(profile):
    """Build a prompt to generate a personality essay."""
    desc = profile_to_desc(profile)
    return (
        f"Write a free-hand essay about your feelings in the moment as if you are "
        f"a person who is {desc}."
    )


def extract_score_s1(text):
    """Extract numeric score from S1 format response."""
    prefix = "My score for the statement is: "
    if prefix.lower() in text.lower():
        idx = text.lower().index(prefix.lower()) + len(prefix)
        remainder = text[idx:].strip()
        match = re.match(r"(\d+)", remainder)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score
    match = re.search(r"\b([1-5])\b", text[:50])
    if match:
        return int(match.group(1))
    return None


def extract_score_s2(text):
    """Extract string-based score from S2 format response."""
    prefix = "My answer to the statement is: "
    if prefix.lower() in text.lower():
        idx = text.lower().index(prefix.lower()) + len(prefix)
        remainder = text[idx:].strip().lower()
    else:
        remainder = text.strip().lower()

    for key in sorted(S2_SCORE_MAP.keys(), key=len, reverse=True):
        if remainder.startswith(key):
            return S2_SCORE_MAP[key]
    return None


def extract_score_s3(text):
    """Extract letter-based score from S3 format response."""
    prefix = "My answer to the statement is: "
    if prefix.lower() in text.lower():
        idx = text.lower().index(prefix.lower()) + len(prefix)
        remainder = text[idx:].strip().lower()
    else:
        remainder = text.strip().lower()

    for key in sorted(S3_SCORE_MAP.keys(), key=len, reverse=True):
        if remainder.startswith(key):
            return S3_SCORE_MAP[key]

    match = re.search(r"\(([a-e])\)", remainder)
    if match:
        return S3_SCORE_MAP.get(match.group(1))
    return None


SCORE_EXTRACTORS = {"S1": extract_score_s1, "S2": extract_score_s2, "S3": extract_score_s3}


def load_model(model_path, base_model_name=None):
    """Load a model - either a fine-tuned PEFT adapter or a base model."""
    if base_model_name and os.path.exists(model_path):
        print(f"Loading PEFT adapter from {model_path} on base {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            quantization_config=bnb_config,
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        print(f"Loading base model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_text(model, tokenizer, messages, max_new_tokens=200):
    """Generate text from a list of chat messages."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            input_text = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in messages
            )
            input_text += "\nAssistant: "
    else:
        input_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages
        )
        input_text += "\nAssistant: "

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=3800)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_questionnaire(model, tokenizer, questions_dict, profile, prompt_set, essay_context=None, split="test"):
    """Answer all questionnaire items for a given profile and return raw responses."""
    trait_desc = profile_to_desc(profile)
    prompt_config = PROMPT_SETS[prompt_set]
    extract_score = SCORE_EXTRACTORS[prompt_set]

    results = {}
    for trait_name, trait_data in questions_dict.items():
        items = trait_data[split]
        trait_results = []

        for item in items:
            user_content = prompt_config["user_template"].format(
                trait_desc=trait_desc,
                question=item["question"],
            )
            if essay_context:
                user_content = (
                    f"Based on this essay about yourself:\n\"{essay_context}\"\n\n"
                    + user_content
                )

            messages = [
                {"role": "system", "content": prompt_config["system"]},
                {"role": "user", "content": user_content},
            ]

            response = generate_text(model, tokenizer, messages, max_new_tokens=200)
            score = extract_score(response)

            trait_results.append({
                "question": item["question"],
                "math": item["math"],
                "raw_response": response,
                "score": score,
            })

        results[trait_name] = trait_results

    return results


def compute_trait_scores(questionnaire_results):
    """Compute average personality score per trait from questionnaire results."""
    trait_scores = {}
    for trait_name, items in questionnaire_results.items():
        total = 0
        count = 0
        for item in items:
            if item["score"] is not None:
                score = item["score"]
                if item["math"] == -1:
                    score = 6 - score
                total += score
                count += 1
        trait_scores[trait_name] = total / count if count > 0 else 0
    return trait_scores


TRAIT_NAME_TO_KEY = {
    "Openness To Experience": "openness",
    "Conscientiousness": "conscientiousness",
    "Extraversion": "extraversion",
    "Agreeableness": "agreeableness",
    "Neuroticism": "neuroticism",
}


def predict_profile(trait_scores, threshold=3.0):
    """Convert trait scores to binary positive/negative profile."""
    predicted = {}
    for trait_name, score in trait_scores.items():
        key = TRAIT_NAME_TO_KEY.get(trait_name, trait_name.lower())
        predicted[key] = "positive" if score >= threshold else "negative"
    return predicted


def evaluate_profiles(predictions, targets):
    """Compute exact match and per-trait accuracy."""
    exact_matches = 0
    trait_correct = {t: 0 for t in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]}
    total = len(predictions)

    for pred, target in zip(predictions, targets):
        match = True
        for trait in trait_correct:
            if pred.get(trait) == target.get(trait):
                trait_correct[trait] += 1
            else:
                match = False
        if match:
            exact_matches += 1

    return {
        "exact_match": exact_matches,
        "exact_match_pct": (exact_matches / total * 100) if total > 0 else 0,
        "total_profiles": total,
        "per_trait_accuracy": {t: (c / total * 100) if total > 0 else 0 for t, c in trait_correct.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Personality induction inference & evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model or HF model name")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name (for PEFT adapters)")
    parser.add_argument("--questions_path", type=str, required=True, help="Path to questions.json")
    parser.add_argument("--prompt_set", type=str, choices=["S1", "S2", "S3"], default="S1")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Where to save evaluation results")
    parser.add_argument("--with_essay", action="store_true", help="Generate essay first, use as context")
    parser.add_argument("--num_profiles", type=int, default=32, help="Number of profiles to evaluate (max 32)")
    parser.add_argument("--question_split", type=str, default="test", choices=["train", "test"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.questions_path) as f:
        questions_dict = json.load(f)

    model, tokenizer = load_model(args.model_path, args.base_model)

    all_profiles = generate_all_profiles()[:args.num_profiles]

    all_predictions = []
    all_targets = []
    all_results = []

    for i, profile in enumerate(all_profiles):
        print(f"\n--- Profile {i+1}/{len(all_profiles)}: {profile_to_desc(profile)} ---")

        essay_context = None
        if args.with_essay:
            essay_prompt = build_essay_prompt(profile)
            messages = [
                {"role": "system", "content": "You are writing an essay that mimics the personality of real humans."},
                {"role": "user", "content": essay_prompt},
            ]
            essay_context = generate_text(model, tokenizer, messages, max_new_tokens=500)
            print(f"  Essay ({len(essay_context)} chars): {essay_context[:100]}...")

        q_results = run_questionnaire(
            model, tokenizer, questions_dict, profile, args.prompt_set,
            essay_context=essay_context, split=args.question_split,
        )

        trait_scores = compute_trait_scores(q_results)
        predicted = predict_profile(trait_scores)

        nan_count = sum(1 for t in q_results.values() for item in t if item["score"] is None)
        total_qs = sum(len(t) for t in q_results.values())
        print(f"  Scores: {trait_scores}")
        print(f"  Predicted: {predicted}")
        print(f"  Target:    {profile}")
        print(f"  NaN rate:  {nan_count}/{total_qs} ({nan_count/total_qs*100:.1f}%)")

        all_predictions.append(predicted)
        all_targets.append(profile)
        all_results.append({
            "profile_idx": i,
            "target": profile,
            "predicted": predicted,
            "trait_scores": trait_scores,
            "nan_count": nan_count,
            "total_questions": total_qs,
            "essay": essay_context,
        })

    eval_metrics = evaluate_profiles(all_predictions, all_targets)
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({args.prompt_set})")
    print(f"{'='*60}")
    print(f"Exact match: {eval_metrics['exact_match']}/{eval_metrics['total_profiles']} "
          f"({eval_metrics['exact_match_pct']:.2f}%)")
    print(f"Random baseline: {100/32:.2f}%")
    print(f"Per-trait accuracy:")
    for trait, acc in eval_metrics["per_trait_accuracy"].items():
        print(f"  {trait}: {acc:.2f}%")

    output_file = os.path.join(
        args.output_dir,
        f"results_{os.path.basename(args.model_path)}_{args.prompt_set}.json"
    )
    with open(output_file, "w") as f:
        json.dump({"metrics": eval_metrics, "results": all_results}, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
