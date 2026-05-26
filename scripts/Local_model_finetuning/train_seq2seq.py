"""
Generic seq2seq fine-tuning batch script for encoder-decoder models.
Supports any HuggingFace AutoModelForSeq2SeqLM (T5, BART, LongT5, CodeT5, etc.).

Usage:
    python train_seq2seq.py --model_name <hf-model-id> --output_dir <path> [--args]
"""

import argparse
import gc
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pmc_links",    default="scripts/exp_input/REV.txt")
    p.add_argument("--ground_truth", default="scripts/Local_model_finetuning/ground_truth/gt_dataset_info_no_dspage_extraction_from_snippet.xlsx")
    p.add_argument("--output_dir",   default="scripts/Local_model_finetuning/seq2seq-models")
    p.add_argument("--model_name",   default="google/flan-t5-base")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--train_batch",  type=int,   default=2)
    p.add_argument("--eval_batch",   type=int,   default=2)
    p.add_argument("--grad_accum",   type=int,   default=8)
    p.add_argument("--max_input",    type=int,   default=512)
    p.add_argument("--max_output",   type=int,   default=256)
    p.add_argument("--warmup_steps", type=int,   default=100)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--test_size",    type=float, default=0.2)
    p.add_argument("--resume",       action="store_true", help="Resume from latest checkpoint if one exists")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_device():
    if torch.cuda.is_available():
        # bf16 is numerically stable and supported on Ampere+ (A100, A10, 3090, etc.)
        # fp16 causes loss underflow / NaN gradients with seq2seq models
        has_bf16 = torch.cuda.is_bf16_supported()
        return "cuda", False, has_bf16
    if torch.backends.mps.is_available():
        return "mps", False, False
    return "cpu", False, False


def get_last_checkpoint(dir_path):
    if not os.path.isdir(dir_path):
        return None
    entries = [d for d in os.listdir(dir_path) if d.startswith("checkpoint-")]
    if not entries:
        return None
    def step_num(name):
        m = re.search(r"checkpoint-(\d+)", name)
        return int(m.group(1)) if m else -1
    return os.path.join(dir_path, sorted(entries, key=step_num)[-1])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(pmc_links_path, ground_truth_path, test_size, seed):
    with open(pmc_links_path) as f:
        pmc_links = f.read().splitlines()

    train_links, test_links = train_test_split(pmc_links, test_size=test_size, random_state=seed)
    print(f"PMC links — train: {len(train_links)}, test: {len(test_links)}")

    df = pd.read_excel(ground_truth_path)
    train_df = df[df["url"].isin(train_links)]
    test_df  = df[df["url"].isin(test_links)]
    print(f"Ground truth rows — train: {len(train_df)}, test: {len(test_df)}")
    return train_df, test_df


def to_hf_dataset(df):
    from datasets import Dataset
    df = df.dropna(subset=["input_text", "output_text"]).copy()
    df["input_text"]  = df["input_text"].astype(str)
    df["output_text"] = df["output_text"].astype(str)
    return Dataset.from_dict({"input": df["input_text"].tolist(), "output": df["output_text"].tolist()})


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def make_preprocess(tokenizer, max_input, max_output):
    def preprocess(examples):
        inputs = ["Extract dataset information: " + doc for doc in examples["input"]]
        model_inputs = tokenizer(inputs, max_length=max_input, truncation=True, padding=False)
        labels = tokenizer(text_target=examples["output"], max_length=max_output, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(tokenizer):
    import evaluate
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.clip(labels, 0, tokenizer.vocab_size - 1)
        preds  = np.clip(preds,  0, tokenizer.vocab_size - 1)
        try:
            decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        except (OverflowError, ValueError) as e:
            print(f"Warning: decoding error: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "exact_match": 0.0}

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 2) for k, v in result.items()}
        exact  = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels)) / len(decoded_preds)
        result["exact_match"] = round(exact * 100, 2)
        return result

    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device_str, use_fp16, use_bf16 = detect_device()
    print(f"Device: {device_str}, fp16: {use_fp16}, bf16: {use_bf16}")

    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        torch.mps.empty_cache()
    gc.collect()

    # Data
    train_df, test_df = load_split(args.pmc_links, args.ground_truth, args.test_size, args.seed)
    train_dataset = to_hf_dataset(train_df)
    test_dataset  = to_hf_dataset(test_df)

    # Model & tokenizer
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    print(f"Parameters: {model.num_parameters():,}")

    # Tokenize
    preprocess = make_preprocess(tokenizer, args.max_input, args.max_output)
    tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    tokenized_test  = test_dataset.map(preprocess,  batched=True, remove_columns=test_dataset.column_names)

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=args.max_output,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=make_compute_metrics(tokenizer),
    )

    checkpoint = get_last_checkpoint(args.output_dir) if args.resume else None
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
    else:
        print("Starting training from scratch.")

    result = trainer.train(resume_from_checkpoint=checkpoint)

    print(f"\nTraining complete — loss: {result.training_loss:.4f}, time: {result.metrics['train_runtime']:.0f}s")

    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
