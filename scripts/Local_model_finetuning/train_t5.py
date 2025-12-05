#!/usr/bin/env python3
"""
T5 Model Fine-tuning Script for Dataset Information Extraction
Converts notebook Train_Validate_Test.ipynb to a standalone Python script for HPC execution
"""

import os
import sys
import argparse
from pathlib import Path

# Setup environment - remove MPS for HPC (use CUDA instead)
# os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # MPS is Mac-specific

import sklearn
import numpy as np
import pandas as pd
import time
import re
import json
from sklearn.model_selection import train_test_split

import torch
import gc
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fine-tune T5 for dataset extraction')
    parser.add_argument('--input-pmcids', type=str, 
                        default='scripts/exp_input/REV.txt',
                        help='Path to PMC IDs file')
    parser.add_argument('--ground-truth', type=str,
                        default='scripts/Local_model_finetuning/ground_truth/gt_dataset_info_no_dspage_extraction_from_snippet.xlsx',
                        help='Path to ground truth Excel file')
    parser.add_argument('--output-dir', type=str,
                        default='scripts/Local_model_finetuning/flan-t5-models',
                        help='Output directory for trained model')
    parser.add_argument('--model-name', type=str,
                        default='google/flan-t5-base',
                        help='Pretrained model name')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training batch size per device')
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training (fp16)')
    return parser.parse_args()


def prepare_dataset(df):
    """Convert DataFrame to HuggingFace Dataset format, filtering out NaN values"""
    # Remove rows where input_text or output_text is NaN
    df_clean = df.dropna(subset=['input_text', 'output_text']).copy()
    
    # Convert to string to ensure all values are strings
    df_clean['input_text'] = df_clean['input_text'].astype(str)
    df_clean['output_text'] = df_clean['output_text'].astype(str)
    
    data = {
        'input': df_clean['input_text'].tolist(),
        'output': df_clean['output_text'].tolist()
    }
    
    print(f"  Filtered out {len(df) - len(df_clean)} rows with missing values")
    return Dataset.from_dict(data)


def main():
    args = parse_args()
    
    print("="*80)
    print("T5 Fine-tuning for Dataset Information Extraction")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  FP16: {args.fp16}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {args.output_dir}\n")
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}\n")
    
    # Load PMC IDs and split into train/test
    print("Loading PMC IDs...")
    with open(args.input_pmcids, 'r') as f:
        pmc_links = f.read().splitlines()
    
    print(f"Total number of PMCIDs: {len(pmc_links)}")
    
    train_pmc_links, test_pmc_links = train_test_split(
        pmc_links, test_size=0.2, random_state=args.seed
    )
    
    print(f"Training set: {len(train_pmc_links)}")
    print(f"Test set: {len(test_pmc_links)}\n")
    
    # Load ground truth data
    print("Loading ground truth data...")
    train_test_df = pd.read_excel(args.ground_truth)
    
    # Split into train/test based on PMC links
    train_df = train_test_df[train_test_df['url'].isin(train_pmc_links)]
    test_df = train_test_df[train_test_df['url'].isin(test_pmc_links)]
    
    print(f"Original DataFrame: {len(train_test_df)} rows")
    print(f"Train DataFrame: {len(train_df)} rows")
    print(f"Test DataFrame: {len(test_df)} rows")
    print(f"Total matched: {len(train_df) + len(test_df)} rows\n")
    
    # Show sample data
    print("Sample input_text:")
    print(train_df['input_text'].iloc[0][:300])
    print("\n" + "="*80 + "\n")
    print("Sample output_text:")
    print(train_df['output_text'].iloc[0][:300])
    print("\n" + "="*80 + "\n")
    print(f"Average input length: {train_df['input_text'].str.len().mean():.0f} chars")
    print(f"Average output length: {train_df['output_text'].str.len().mean():.0f} chars\n")
    
    # Prepare datasets
    print("Preparing train dataset...")
    train_dataset = prepare_dataset(train_df)
    
    print("\nPreparing test dataset...")
    test_dataset = prepare_dataset(test_df)
    
    print(f"\nTrain dataset: {len(train_dataset)} examples")
    print(f"Test dataset: {len(test_dataset)} examples\n")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    print(f"Model loaded: {model.num_parameters():,} parameters\n")
    
    # Tokenization function
    def preprocess_function(examples):
        """Tokenize inputs and outputs"""
        # Add task prefix to help the model understand the task
        inputs = ["Extract dataset information: " + doc for doc in examples['input']]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=False)
        
        # Tokenize targets
        labels = tokenizer(text_target=examples['output'], max_length=256, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        preprocess_function, batched=True, remove_columns=train_dataset.column_names
    )
    tokenized_test = test_dataset.map(
        preprocess_function, batched=True, remove_columns=test_dataset.column_names
    )
    
    print(f"Tokenized train dataset: {len(tokenized_train)} examples")
    print(f"Tokenized test dataset: {len(tokenized_test)} examples\n")
    
    # Setup evaluation metrics
    rouge = evaluate.load("rouge")
    
    def compute_metrics(eval_preds):
        """Compute ROUGE scores for evaluation"""
        preds, labels = eval_preds
        
        # Decode predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Replace -100 in labels (padding token)
        # Clip values to valid range to prevent overflow
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.clip(labels, 0, tokenizer.vocab_size - 1)
        preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
        
        try:
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        except (OverflowError, ValueError) as e:
            print(f"Warning: Decoding error: {e}")
            # Return default metrics on error
            return {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "exact_match": 0.0
            }
        
        # Compute ROUGE scores
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        # Extract F1 scores
        result = {k: round(v * 100, 2) for k, v in result.items()}
        
        # Compute exact match (for structured output)
        exact_match = sum([p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels)]) / len(decoded_preds)
        result["exact_match"] = round(exact_match * 100, 2)
        
        return result
    
    # Setup training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        
        # Training hyperparameters
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        
        # Evaluation and logging
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        
        # Generation settings for evaluation
        predict_with_generate=True,
        generation_max_length=256,
        
        # Optimizer settings
        weight_decay=0.01,
        warmup_steps=100,
        
        # Save settings
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        
        # Performance
        fp16=args.fp16,
        
        # Reproducibility
        seed=args.seed,
    )
    
    print(f"Training configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Output directory: {args.output_dir}\n")
    
    # Initialize data collator and trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Trainer initialized successfully!")
    print(f"Training samples: {len(tokenized_train)}")
    print(f"Evaluation samples: {len(tokenized_test)}\n")
    
    # Clear memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleared. Ready to train.\n")
    
    # Train the model
    print("="*80)
    print("Starting training...")
    print("="*80)
    train_result = trainer.train()
    
    # Print training summary
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    
    # Save the final model
    final_model_path = f"{args.output_dir}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\nModel saved to {final_model_path}\n")
    
    # Evaluate on test set
    print("="*80)
    print("Evaluating on test set...")
    print("="*80)
    eval_results = trainer.evaluate()
    
    print("\nTest Set Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")
    
    # Test inference on a few examples
    print("\n" + "="*80)
    print("Sample Predictions:")
    print("="*80)
    
    # Ensure model is on correct device
    model.to(device)
    
    for i in range(min(3, len(test_dataset))):
        input_text = test_dataset[i]['input']
        expected_output = test_dataset[i]['output']
        
        # Prepare input and move to device
        input_ids = tokenizer(
            "Extract dataset information: " + input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).input_ids.to(device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nExample {i+1}:")
        print(f"Input: {input_text[:200]}...")
        print(f"Expected: {expected_output}")
        print(f"Predicted: {prediction}")
        print(f"Match: {'✓' if prediction.strip() == expected_output.strip() else '✗'}")
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()
