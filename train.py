import argparse
import gc
import os
import sys
import warnings

import torch
import yaml
from datasets import load_dataset
from huggingface_hub import login as hf_login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import (
    DPOConfig,
    DPOTrainer,
    ORPOConfig,
    ORPOTrainer,
    SFTConfig,
    SFTTrainer,
    setup_chat_format,
)

MODEL_REGISTRY = {
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-7b": "google/gemma-7b",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
}

METHOD_PREFIX = {"ORPO": "orpo", "SFT": "sft", "DPO": "dpo"}


def get_output_name(model_key, method):
    return f"{METHOD_PREFIX[method]}_{model_key}"


def parse_args():
    parser = argparse.ArgumentParser(description="Train personality induction models")
    parser.add_argument(
        "--training_method",
        type=str,
        choices=["ORPO", "SFT", "DPO"],
        required=True,
        help="Training method: ORPO, SFT, or DPO",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--dpo_train_file",
        type=str,
        default="./data/dpo_train_dataset.json",
        help="Path to DPO/ORPO preference dataset",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="./data/train.jsonl",
        help="Path to SFT training file (JSONL with chat messages)",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default="./data/validation.jsonl",
        help="Path to SFT validation file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Base output directory for trained models",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML with API keys",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Max training steps (-1 for full epochs)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    return parser.parse_args()


def train_orpo(base_model, new_model, args, output_path, torch_dtype):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=bnb_config,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["up_proj", "down_proj", "gate_proj", "k_proj", "q_proj", "v_proj", "o_proj"],
    )

    dataset = load_dataset("json", data_files=args.dpo_train_file, split="train")
    dataset = dataset.train_test_split(test_size=0.01)

    orpo_args = ORPOConfig(
        learning_rate=8e-6,
        lr_scheduler_type="linear",
        max_length=4000,
        max_prompt_length=1024,
        beta=0.1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_hf",
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        warmup_steps=500,
        report_to="none" if args.no_wandb else "wandb",
        output_dir=output_path,
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    return model, tokenizer


def train_sft(base_model, new_model, args, output_path, torch_dtype):
    dataset = load_dataset("json", data_files=args.train_file, split="train")
    val_dataset = load_dataset("json", data_files=args.validation_file, split="train")

    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model, tokenizer = setup_chat_format(model, tokenizer)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    training_args = SFTConfig(
        output_dir=output_path,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_dir=os.path.join(output_path, "logs"),
        max_seq_length=4000,
        packing=False,
    )

    def formatting_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )

    trainer.train()
    final_path = os.path.join(output_path, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    return model, tokenizer


def train_dpo(base_model, new_model, args, output_path, torch_dtype):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=bnb_config,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["up_proj", "down_proj", "gate_proj", "k_proj", "q_proj", "v_proj", "o_proj"],
    )

    dataset = load_dataset("json", data_files=args.dpo_train_file, split="train")
    dataset = dataset.train_test_split(test_size=0.01)

    dpo_args = DPOConfig(
        learning_rate=8e-6,
        lr_scheduler_type="linear",
        max_length=4000,
        max_prompt_length=1024,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_hf",
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        warmup_steps=500,
        report_to="none" if args.no_wandb else "wandb",
        output_dir=output_path,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    return model, tokenizer


def main():
    args = parse_args()

    if not args.no_wandb:
        import wandb
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        wandb_key = config.get("wandb", {}).get("wandb_key")
        hf_token = config.get("huggingface", {}).get("token")
        if wandb_key and wandb_key != "[PLACE WANDB KEY HERE]":
            wandb.login(key=wandb_key)
        if hf_token and hf_token != "[PLACE TOKEN HERE]":
            hf_login(token=hf_token)
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        hf_token = config.get("huggingface", {}).get("token")
        if hf_token and hf_token != "[PLACE TOKEN HERE]":
            hf_login(token=hf_token)

    os.makedirs(args.output_dir, exist_ok=True)
    torch_dtype = torch.float16

    if args.model == "all":
        models_to_train = list(MODEL_REGISTRY.items())
    else:
        models_to_train = [(args.model, MODEL_REGISTRY[args.model])]

    train_fn = {"ORPO": train_orpo, "SFT": train_sft, "DPO": train_dpo}[args.training_method]

    for model_key, base_model in models_to_train:
        new_model = get_output_name(model_key, args.training_method)
        output_path = os.path.join(args.output_dir, new_model)
        os.makedirs(output_path, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Training: {base_model} -> {new_model} [{args.training_method}]")
        print(f"Output:   {output_path}")
        print(f"{'='*60}\n")

        warnings.filterwarnings("ignore")
        model, tokenizer = train_fn(base_model, new_model, args, output_path, torch_dtype)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\nCompleted: {new_model}\n")


if __name__ == "__main__":
    main()
