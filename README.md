# Qwen2.5-0.5B SFT Pipeline Repo

## Introduction

This repository documents the complete Supervised Fine-Tuning (SFT) pipeline for the Qwen2.5-0.5B base model (https://huggingface.co/Qwen/Qwen2.5-0.5B), encompassing three core modules: data cleaning, model training, and evaluation verification.

The experiment utilized the 1M-GPT4-Augmented subset of the OpenOrca dataset (https://huggingface.co/datasets/Open-Orca/OpenOrca), which was processed through a four-stage filtering pipeline to retain 300,000 high-quality instruction-response pairs.

SFT employed LlaMa-Factory framework (https://github.com/hiyouga/LLaMAFactory) and evaluation employed the lm_eval framework (https://github.com/EleutherAI/lm-evaluation-harness) to test the model on 8 standard benchmark datasets (MMLU, ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, TruthfulQA-MC2, PIQA, and BoolQ).

The subset after data cleaning is at https://huggingface.co/datasets/qqz03/OpenOrca-300k and the model after SFT is at https://huggingface.co/qqz03/Qwen2.5-0.5B-SFT-OpenOrca-300k.

## File Descriptions

| File Name | Function Description |
|-----------|---------------------|
| `filter_openorca.py` | **Data Cleaning Script**: Implements a four-stage pipeline filtering (format checking, quality filtering, consistency checking, information density filtering), cleaning 300,000 high-quality training samples from 994,896 raw samples |
| `train_full.yaml` | **Training Configuration File**: Defines hyperparameter settings (learning rate 2e-5, batch size 64, 2 epochs, sequence length 1024, etc.), to be read by LLaMA-Factory |
| `run_sft_training.py` | **All-in-One Training Script**: Executes pre-training checks, launches SFT training, monitors progress in real-time, performs post-training verification, automatically generates Loss and learning rate curves, and outputs complete training reports |
| `generate_curves.py` | **Curve Generation Script** (Backup): When `run_sft_training.py` automatic plotting fails, manually extracts Loss values and learning rates from training logs and generates visualization curves |
| `evaluate_models.py` | **Evaluation Script**: Calls the lm_eval framework to evaluate both Base and SFT models on 8 benchmark tests, automatically extracts accuracy scores and generates comparison reports |
| `summarize_results.py` | **Result Summarization Script** (Backup): When `evaluate_models.py` log generation fails, manually extracts test scores from each dataset's JSON result files and generates summary reports |
