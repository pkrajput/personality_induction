import argparse
import os
import sys
import gc
import warnings
import yaml

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    ORPOConfig,
    ORPOTrainer,
    SFTConfig,
    SFTTrainer,
    setup_chat_format,
    DPOConfig,
    DPOTrainer,
)
from huggingface_hub import login as hf_login

# ------------------------------------------------------------------------------
# Define the four possible base models and corresponding new model names
# ------------------------------------------------------------------------------
base_models = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/Gemma-2-2B",
    "google/Gemma-7B",
    "meta-llama/Llama-3.1-8B",
]

new_models_orpo = [
    "orpollama-3.2-3B",
    "orpogemma-2-2B",
    "orpogemma-7B",
    "orpollama-3.1-8B",
]

new_models_sft = ["sftllama-3.2-3B", "sftgemma-2-2B", "sftgemma-7B", "sftllama-3.1-8B"]

new_models_dpo = [
    "dpollama-3.2-3B",
    "dpopgemma-2-2B",
    "dpopgemma-7B",
    "dpollama-3.1-8B",
]

# ------------------------------------------------------------------------------
# Argument parser: choose training method and provide paths for data and config
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--training_method",
    type=str,
    choices=["ORPO", "SFT", "DPO"],
    default="ORPO",
    help="Training method to use: ORPO, SFT, or DPO",
)
parser.add_argument(
    "--dpo_train_file",
    type=str,
    default="dpo_train_dataset.json",
    help="Path to the DPO training dataset (used for ORPO and DPO)",
)
parser.add_argument(
    "--train_file",
    type=str,
    default="./train.jsonl",
    help="Path to the SFT training file.",
)
parser.add_argument(
    "--validation_file",
    type=str,
    default="./validation.jsonl",
    help="Path to the SFT validation file.",
)
parser.add_argument(
    "--config",
    type=str,
    default="config.yaml",
    help="Path to the config file containing WandB and Hugging Face tokens.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # ------------------------------------------------------------------------------
    # Load configuration from YAML and log in to WandB and Hugging Face
    # ------------------------------------------------------------------------------
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    wandb_key = config["wandb"]["wandb_key"]
    hf_token = config["huggingface"]["token"]

    wandb.login(key=wandb_key)
    hf_login(token=hf_token)

    # ------------------------------------------------------------------------------
    # Create the output directory if it does not exist (for ORPO, SFT, and DPO)
    # ------------------------------------------------------------------------------
    os.makedirs("./results", exist_ok=True)

    # Set global torch settings
    torch_dtype = torch.float16
    attn_implementation = "eager"

    # ------------------------------------------------------------------------------
    # Loop over all 4 models and perform training according to the chosen method
    # ------------------------------------------------------------------------------
    for i in range(4):
        base_model = base_models[i]
        if args.training_method == "ORPO":
            new_model = new_models_orpo[i]
        elif args.training_method == "SFT":
            new_model = new_models_sft[i]
        elif args.training_method == "DPO":
            new_model = new_models_dpo[i]

        print(
            f"\nTraining model: {base_model} -> {new_model} using {args.training_method}"
        )

        if args.training_method == "ORPO":
            # ---------------------------
            # ORPO Training Setup
            # ---------------------------
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                base_model, device_map="auto", attn_implementation=attn_implementation
            )

            model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "up_proj",
                    "down_proj",
                    "gate_proj",
                    "k_proj",
                    "q_proj",
                    "v_proj",
                    "o_proj",
                ],
            )

            dataset = load_dataset(
                "json", data_files=args.dpo_train_file, split="train"
            )
            dataset = dataset.train_test_split(test_size=0.01)

            warnings.filterwarnings("ignore")
            sys.stderr = open(os.devnull, "w")

            orpo_args = ORPOConfig(
                learning_rate=8e-6,
                lr_scheduler_type="linear",
                max_length=4000,
                max_prompt_length=1024,
                beta=0.1,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,
                optim="adamw_hf",
                num_train_epochs=3,
                evaluation_strategy="steps",
                eval_steps=200,
                logging_steps=600,
                warmup_steps=500,
                report_to="wandb",
                output_dir=f"./results/{new_model}/",
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
            trainer.save_model(new_model)

            sys.stderr = sys.__stderr__

        elif args.training_method == "SFT":
            # ---------------------------
            # SFT Training Setup
            # ---------------------------
            dataset = load_dataset("json", data_files=args.train_file, split="train")
            val_dataset = load_dataset(
                "json", data_files=args.validation_file, split="train"
            )

            model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(base_model)

            model, tokenizer = setup_chat_format(model, tokenizer)

            training_args = SFTConfig(
                output_dir=f"./results/{new_model}/",
                num_train_epochs=3,
                learning_rate=1e-5,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_steps=50,
                evaluation_strategy="steps",
                save_steps=1000,
                max_steps=20000,
                logging_dir=f"./results/{new_model}/logs",
                max_seq_length=2048,
                packing=True,
            )

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
            )

            trainer.train()

            model.save_pretrained(f"./results/{new_model}/final_sft")
            tokenizer.save_pretrained(f"./results/{new_model}/final_sft")

        elif args.training_method == "DPO":
            # ---------------------------
            # DPO Training Setup
            # ---------------------------
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                base_model, device_map="auto", attn_implementation=attn_implementation
            )

            model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "up_proj",
                    "down_proj",
                    "gate_proj",
                    "k_proj",
                    "q_proj",
                    "v_proj",
                    "o_proj",
                ],
            )

            dataset = load_dataset(
                "json", data_files=args.dpo_train_file, split="train"
            )
            dataset = dataset.train_test_split(test_size=0.01)

            dpo_args = DPOConfig(
                learning_rate=8e-6,
                lr_scheduler_type="linear",
                max_length=4000,
                max_prompt_length=1024,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,
                optim="adamw_hf",
                num_train_epochs=3,
                evaluation_strategy="steps",
                eval_steps=200,
                logging_steps=600,
                warmup_steps=500,
                report_to="wandb",
                output_dir=f"./results/{new_model}/",
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
            trainer.save_model(new_model)

        gc.collect()
        torch.cuda.empty_cache()
