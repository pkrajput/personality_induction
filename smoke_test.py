"""
Smoke test: verify that SFT, DPO, and ORPO training pipelines
can load models, initialize trainers, and run a few steps.
Tests with Gemma-2-2B and Llama-3.2-3B.
"""

import gc
import json
import os
import tempfile

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import (
    DPOConfig, DPOTrainer,
    ORPOConfig, ORPOTrainer,
    SFTConfig, SFTTrainer,
    setup_chat_format,
)


def make_sft_dataset(n=16):
    """Create a tiny SFT dataset for smoke testing."""
    samples = []
    for i in range(n):
        samples.append({
            "messages": [
                {"role": "system", "content": "You are writing an essay that mimics human personality."},
                {"role": "user", "content": f"Write an essay as a person positive in openness. Sample {i}."},
                {"role": "assistant", "content": f"I feel creative and open today. Test essay {i}."},
            ]
        })
    return Dataset.from_list(samples)


def make_dpo_dataset(n=16):
    """Create a tiny DPO/ORPO dataset for smoke testing."""
    samples = []
    for i in range(n):
        samples.append({
            "prompt": f"Write an essay as a person positive in openness. Sample {i}.",
            "chosen": f"I feel creative and open today. Chosen {i}.",
            "rejected": f"I don't like anything. Rejected {i}.",
        })
    return Dataset.from_list(samples)


def cleanup(model=None, tokenizer=None):
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def test_sft(model_name, device_map="auto"):
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: SFT with {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=torch.float16
    )

    try:
        model, tokenizer = setup_chat_format(model, tokenizer)
    except Exception as e:
        print(f"  setup_chat_format skipped: {e}")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"], bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    dataset = make_sft_dataset()

    def formatting_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        args = SFTConfig(
            output_dir=tmpdir,
            max_steps=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
            max_seq_length=512,
            packing=False,
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model, args=args,
            train_dataset=dataset,
            formatting_func=formatting_func,
            processing_class=tokenizer,
        )
        trainer.train()
        print(f"  SFT OK - loss: {trainer.state.log_history[-1].get('loss', 'N/A')}")

    cleanup(model, tokenizer)
    return True


def test_dpo(model_name, device_map="auto"):
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: DPO with {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map,
        quantization_config=bnb_config, attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    dataset = make_dpo_dataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        dpo_args = DPOConfig(
            output_dir=tmpdir,
            max_steps=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
            max_length=256,
            max_prompt_length=128,
            report_to="none",
        )
        trainer = DPOTrainer(
            model=model, args=dpo_args,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        trainer.train()
        print(f"  DPO OK - final log: {trainer.state.log_history[-1]}")

    cleanup(model, tokenizer)
    return True


def test_orpo(model_name, device_map="auto"):
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: ORPO with {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map,
        quantization_config=bnb_config, attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    dataset = make_dpo_dataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        orpo_args = ORPOConfig(
            output_dir=tmpdir,
            max_steps=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
            max_length=256,
            max_prompt_length=128,
            beta=0.1,
            report_to="none",
        )
        trainer = ORPOTrainer(
            model=model, args=orpo_args,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        trainer.train()
        print(f"  ORPO OK - final log: {trainer.state.log_history[-1]}")

    cleanup(model, tokenizer)
    return True


if __name__ == "__main__":
    models = [
        "google/gemma-2-2b",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]
    methods = [
        ("SFT", test_sft),
        ("DPO", test_dpo),
        ("ORPO", test_orpo),
    ]

    results = {}
    for model_name in models:
        for method_name, test_fn in methods:
            key = f"{model_name}_{method_name}"
            try:
                ok = test_fn(model_name)
                results[key] = "PASS"
            except Exception as e:
                print(f"  FAILED: {e}")
                results[key] = f"FAIL: {e}"
            cleanup()

    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    for key, status in results.items():
        print(f"  {key}: {status}")
