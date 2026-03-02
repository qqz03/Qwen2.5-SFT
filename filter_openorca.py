# filter_openorca.py
# ==============================================================================
# Data Cleaning and Subsetting Script for Qwen2.5-0.5B SFT
# ==============================================================================
# Final optimized version with detailed statistics and configurable thresholds
# ==============================================================================

import os
import json
import random
import re
from datasets import load_dataset, Dataset
from typing import Dict, Any, Tuple
from tqdm import tqdm
import pandas as pd

# ------------------------------------------------------------------------------
# Configuration Constants
# ------------------------------------------------------------------------------
LOCAL_PARQUET_PATH = "/home/qianqz/Dataset/OpenOrca/1M-GPT4-Augmented.parquet"
OUTPUT_JSONL_PATH = "/home/qianqz/Dataset/OpenOrca/OpenOrca_filtered_300k.jsonl"

RANDOM_SEED = 42
TARGET_SIZE = 300_000

# Quality Filtering Thresholds
MIN_RESPONSE_LENGTH = 20
MAX_RESPONSE_LENGTH = 4096
MIN_QUESTION_LENGTH = 10
MAX_QUESTION_LENGTH = 2048

# Information Density Thresholds
MIN_UNIQUE_WORD_RATIO = 0.25
MIN_PUNCTUATION_RATIO = 0.01


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def load_local_parquet(file_path: str) -> Dataset:
    """Load the OpenOrca 1M subset from local parquet file."""
    print(f"Loading dataset from local parquet: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
    print(f"File size: {file_size_gb:.2f} GB")

    dataset = load_dataset("parquet", data_files=file_path, split="train")
    print(f"Successfully loaded {len(dataset)} samples from local file.")
    return dataset


def filter_formatting_issues(sample: Dict[str, Any]) -> bool:
    """Stage 1: Check for formatting issues."""
    required_fields = ["question", "response"]
    for field in required_fields:
        if field not in sample or sample[field] is None:
            return False
    try:
        str(sample["question"])
        str(sample["response"])
    except (TypeError, ValueError):
        return False
    return True


def filter_quality_issues(sample: Dict[str, Any]) -> bool:
    """Stage 2: Check for quality issues."""
    question = str(sample["question"]).strip()
    response = str(sample["response"]).strip()

    if not question or not response:
        return False
    if len(response) < MIN_RESPONSE_LENGTH or len(response) > MAX_RESPONSE_LENGTH:
        return False
    if len(question) < MIN_QUESTION_LENGTH or len(question) > MAX_QUESTION_LENGTH:
        return False

    garbage_keywords = ["[INSERT]", "TODO", "placeholder", "xxx", "YOUR_TEXT_HERE"]
    if any(keyword.lower() in response.lower() for keyword in garbage_keywords):
        return False
    if re.search(r'(.)\1{10,}', response):
        return False

    url_count = len(re.findall(r'http[s]?://', response))
    if url_count > 5:
        return False

    return True


def filter_consistency_issues(dataset: Dataset) -> Tuple[Dataset, Dict[str, int]]:
    """Stage 3: Check for consistency issues with progress bar."""
    print("Stage 3: Checking consistency issues...")

    df = dataset.to_pandas()
    stats = {
        "total_before": len(df),
        "unique_questions": 0,
        "duplicate_questions": 0,
        "conflicting_groups": 0,
        "removed_samples": 0
    }

    question_groups = df.groupby("question")
    indices_to_remove = []

    for question, group in tqdm(question_groups, total=len(question_groups),
                                desc="Stage 3: Consistency", ncols=100):
        stats["unique_questions"] += 1

        if len(group) > 1:
            stats["duplicate_questions"] += len(group) - 1
            unique_responses = group["response"].nunique()
            if unique_responses > 1:
                stats["conflicting_groups"] += 1
                indices_to_remove.extend(group.index[1:].tolist())

    if indices_to_remove:
        df_filtered = df.drop(index=indices_to_remove).reset_index(drop=True)
        stats["removed_samples"] = len(indices_to_remove)
    else:
        df_filtered = df.reset_index(drop=True)

    stats["total_after"] = len(df_filtered)
    filtered_dataset = Dataset.from_pandas(df_filtered, preserve_index=False)

    print(f"  Unique questions: {stats['unique_questions']:,}")
    print(f"  Duplicate question occurrences: {stats['duplicate_questions']:,}")
    print(f"  Conflicting answer groups: {stats['conflicting_groups']:,}")
    print(f"  Avg samples per conflicting group: {stats['removed_samples'] / max(stats['conflicting_groups'], 1):.2f}")
    print(f"  Samples removed: {stats['removed_samples']:,}")
    print(f"  Samples after Stage 3: {stats['total_after']:,}")

    return filtered_dataset, stats


def filter_information_density(sample: Dict[str, Any]) -> bool:
    """Stage 4: Filter samples with low information density."""
    response = str(sample["response"])
    words = re.findall(r'\b\w+\b', response.lower())

    if len(words) == 0:
        return False

    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < MIN_UNIQUE_WORD_RATIO:
        return False

    punctuation_count = len(re.findall(r'[.!?,:;]', response))
    punctuation_density = punctuation_count / max(len(words), 1)
    if punctuation_density < MIN_PUNCTUATION_RATIO:
        return False

    return True


def save_to_jsonl(dataset: Dataset, output_path: str):
    """Save the processed dataset to a JSONL file."""
    print(f"Saving dataset to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            clean_sample = {
                "system_prompt": sample.get("system_prompt", ""),
                "question": sample["question"],
                "response": sample["response"]
            }
            f.write(json.dumps(clean_sample, ensure_ascii=False) + "\n")

    print(f"Successfully saved {len(dataset)} samples.")


def main():
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("DATA FILTERING PIPELINE - 4 STAGE PROCESS (FINAL)")
    print("=" * 80)

    raw_dataset = load_local_parquet(LOCAL_PARQUET_PATH)
    initial_count = len(raw_dataset)

    stage_stats = {}
    current_dataset = raw_dataset

    # STAGE 1
    print("\n" + "-" * 80)
    print("STAGE 1: Formatting Issues Filter")
    print("-" * 80)
    stage1_before = len(current_dataset)
    current_dataset = current_dataset.filter(filter_formatting_issues, num_proc=4, desc="Stage 1")
    stage1_after = len(current_dataset)
    stage_stats["stage1"] = {"before": stage1_before, "after": stage1_after,
                             "removed": stage1_before - stage1_after}
    print(f"Before: {stage1_before:,} | After: {stage1_after:,} | Removed: {stage_stats['stage1']['removed']:,}")

    # STAGE 2
    print("\n" + "-" * 80)
    print("STAGE 2: Quality Issues Filter")
    print("-" * 80)
    stage2_before = len(current_dataset)
    current_dataset = current_dataset.filter(filter_quality_issues, num_proc=4, desc="Stage 2")
    stage2_after = len(current_dataset)
    stage_stats["stage2"] = {"before": stage2_before, "after": stage2_after,
                             "removed": stage2_before - stage2_after}
    print(f"Before: {stage2_before:,} | After: {stage2_after:,} | Removed: {stage_stats['stage2']['removed']:,}")

    # STAGE 3
    print("\n" + "-" * 80)
    print("STAGE 3: Consistency Issues Filter")
    print("-" * 80)
    stage3_before = len(current_dataset)
    current_dataset, stage3_details = filter_consistency_issues(current_dataset)
    stage3_after = len(current_dataset)
    stage_stats["stage3"] = {"before": stage3_before, "after": stage3_after,
                             "removed": stage3_before - stage3_after, "details": stage3_details}
    print(f"Before: {stage3_before:,} | After: {stage3_after:,} | Removed: {stage_stats['stage3']['removed']:,}")

    # STAGE 4
    print("\n" + "-" * 80)
    print("STAGE 4: Information Density Filter (Advanced)")
    print("-" * 80)
    stage4_before = len(current_dataset)
    current_dataset = current_dataset.filter(filter_information_density, num_proc=4, desc="Stage 4")
    stage4_after = len(current_dataset)
    stage_stats["stage4"] = {"before": stage4_before, "after": stage4_after,
                             "removed": stage4_before - stage4_after}
    print(f"Before: {stage4_before:,} | After: {stage4_after:,} | Removed: {stage_stats['stage4']['removed']:,}")

    # SUBSAMPLE
    print("\n" + "-" * 80)
    print("SUBSAMPLING TO TARGET SIZE")
    print("-" * 80)
    final_count = len(current_dataset)
    if final_count > TARGET_SIZE:
        print(f"Subsampling from {final_count:,} to {TARGET_SIZE:,}...")
        current_dataset = current_dataset.shuffle(seed=RANDOM_SEED).select(range(TARGET_SIZE))
        final_count = TARGET_SIZE

    # SAVE
    print("\n" + "-" * 80)
    print("SAVING OUTPUT")
    print("-" * 80)
    save_to_jsonl(current_dataset, OUTPUT_JSONL_PATH)

    # SUMMARY
    print("\n" + "=" * 80)
    print("FILTERING PIPELINE SUMMARY")
    print("=" * 80)
    print(f"{'Stage':<45} {'Before':>12} {'After':>12} {'Removed':>12} {'Rate':>10}")
    print("-" * 80)
    print(f"{'Initial Load':<45} {initial_count:>12,} {initial_count:>12,} {0:>12,} {'0.00%':>10}")
    print(
        f"{'Stage 1 - Formatting':<45} {stage_stats['stage1']['before']:>12,} {stage_stats['stage1']['after']:>12,} {stage_stats['stage1']['removed']:>12,} {stage_stats['stage1']['removed'] / max(stage_stats['stage1']['before'], 1) * 100:>9.2f}%")
    print(
        f"{'Stage 2 - Quality':<45} {stage_stats['stage2']['before']:>12,} {stage_stats['stage2']['after']:>12,} {stage_stats['stage2']['removed']:>12,} {stage_stats['stage2']['removed'] / max(stage_stats['stage2']['before'], 1) * 100:>9.2f}%")
    print(
        f"{'Stage 3 - Consistency':<45} {stage_stats['stage3']['before']:>12,} {stage_stats['stage3']['after']:>12,} {stage_stats['stage3']['removed']:>12,} {stage_stats['stage3']['removed'] / max(stage_stats['stage3']['before'], 1) * 100:>9.2f}%")
    print(
        f"{'Stage 4 - Info Density':<45} {stage_stats['stage4']['before']:>12,} {stage_stats['stage4']['after']:>12,} {stage_stats['stage4']['removed']:>12,} {stage_stats['stage4']['removed'] / max(stage_stats['stage4']['before'], 1) * 100:>9.2f}%")
    print("-" * 80)
    print(
        f"{'Final (before subsample)':<45} {stage_stats['stage4']['after']:>12,} {stage_stats['stage4']['after']:>12,} {0:>12,} {'0.00%':>10}")
    print(
        f"{'Final (after subsample)':<45} {stage_stats['stage4']['after']:>12,} {final_count:>12,} {stage_stats['stage4']['after'] - final_count:>12,} {'-':>10}")
    print("=" * 80)

    # VALIDATION
    print("\n" + "=" * 80)
    print("DATA VALIDATION CHECKS")
    print("=" * 80)
    checks_passed = 0
    total_checks = 5

    # Check 1: Stage 1 math
    if stage_stats['stage1']['before'] - stage_stats['stage1']['removed'] == stage_stats['stage1']['after']:
        print("✓ Stage 1 math verification: PASSED")
        checks_passed += 1
    else:
        print("✗ Stage 1 math verification: FAILED")

    # Check 2: Stage 2 math
    if stage_stats['stage2']['before'] - stage_stats['stage2']['removed'] == stage_stats['stage2']['after']:
        print("✓ Stage 2 math verification: PASSED")
        checks_passed += 1
    else:
        print("✗ Stage 2 math verification: FAILED")

    # Check 3: Stage 3 math
    if stage_stats['stage3']['before'] - stage_stats['stage3']['removed'] == stage_stats['stage3']['after']:
        print("✓ Stage 3 math verification: PASSED")
        checks_passed += 1
    else:
        print("✗ Stage 3 math verification: FAILED")

    # Check 4: Stage 4 math
    if stage_stats['stage4']['before'] - stage_stats['stage4']['removed'] == stage_stats['stage4']['after']:
        print("✓ Stage 4 math verification: PASSED")
        checks_passed += 1
    else:
        print("✗ Stage 4 math verification: FAILED")

    # Check 5: Final count
    if final_count == TARGET_SIZE:
        print(f"✓ Final dataset size: PASSED ({final_count:,} samples)")
        checks_passed += 1
    else:
        print(f"✗ Final dataset size: FAILED ({final_count:,} samples)")

    print("-" * 80)
    print(f"Validation Result: {checks_passed}/{total_checks} checks passed")
    print("=" * 80)

    print("\nData preparation pipeline completed successfully.")


if __name__ == "__main__":
    main()