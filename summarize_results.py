#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
# Evaluation Results Summary Script
# Version: 1.3 (Fixed: Better debugging for missing files)
# ==============================================================================

import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# Configuration Paths
# ==============================================================================
PROJECT_ROOT = "/home/qianqz/Qwen-SFT"
EVAL_RESULTS_DIR = "/home/qianqz/Qwen-SFT/eval_results"

# ==============================================================================
# Evaluation Tasks Configuration
# ==============================================================================
EVALUATION_TASKS = {
    "mmlu": 5,
    "arc_easy": 0,
    "arc_challenge": 25,
    "hellaswag": 10,
    "winogrande": 5,
    "truthfulqa_mc2": 0,
    "piqa": 0,
    "boolq": 0,
}

TASK_DISPLAY_NAMES = {
    "mmlu": "MMLU (Test)",
    "arc_easy": "ARC-Easy (Test)",
    "arc_challenge": "ARC-Challenge (Test)",
    "hellaswag": "HellaSwag (Test)",
    "winogrande": "WinoGrande (Test)",
    "truthfulqa_mc2": "TruthfulQA-MC2 (Test)",
    "piqa": "PIQA (Test)",
    "boolq": "BoolQ (Test)",
}


# ==============================================================================
# Color Codes
# ==============================================================================
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text):
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")


def print_header(text):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text.center(80)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 80}{Colors.END}\n")


def print_section(text):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'-' * 80}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'-' * 80}{Colors.END}\n")


# ==============================================================================
# File Discovery (FIXED)
# ==============================================================================
def find_model_directories(base_dir: str) -> Tuple[Optional[str], Optional[str]]:
    base_model_dir = None
    sft_model_dir = None

    print_info(f"Searching for model directories in: {base_dir}")

    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            if item.startswith('base_model_'):
                base_model_dir = item_path
                print_success(f"Found base_model directory: {item}")
            elif item.startswith('sft_model_'):
                sft_model_dir = item_path
                print_success(f"Found sft_model directory: {item}")

    return base_model_dir, sft_model_dir


def find_json_files(directory: str) -> Dict[str, str]:
    """Find all evaluation JSON files in a directory (FIXED with better debugging)."""
    json_files = {}

    # Pattern: {task}_{fewshot}shot_{timestamp}.json
    # Updated to handle task names with numbers (like truthfulqa_mc2)
    pattern = re.compile(r'^([a-z0-9_]+)_(\d+)shot_.*\.json$')

    print_info(f"  Scanning directory: {directory}")
    print_info(f"  All files in directory:")

    for file in sorted(os.listdir(directory)):
        if file.endswith('.json'):
            file_path = os.path.join(directory, file)
            match = pattern.match(file)

            if match:
                task_name = match.group(1)
                json_files[task_name] = file_path
                print_info(f"    ✓ MATCHED: {file} → task={task_name}")
            else:
                print_warning(f"    ✗ NOT MATCHED: {file}")

    print_info(f"  Total JSON files found: {len(json_files)}")
    print_info(f"  Tasks found: {list(json_files.keys())}")

    return json_files


# ==============================================================================
# Score Extraction
# ==============================================================================
def extract_accuracy_from_json(json_file: str) -> Optional[Dict]:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = None
        task_name = None

        if 'results' in data:
            results = data['results']
            task_name = list(results.keys())[0]
        elif len(data) == 1:
            task_name = list(data.keys())[0]
            results = {task_name: data[task_name]}

        if results and task_name:
            task_results = results[task_name]

            accuracy = None
            acc_norm = None

            for key in ['acc,none', 'acc', 'accuracy,none', 'accuracy']:
                if key in task_results:
                    accuracy = task_results[key]
                    break

            for key in ['acc_norm,none', 'acc_norm', 'accuracy_norm,none', 'accuracy_norm']:
                if key in task_results:
                    acc_norm = task_results[key]
                    break

            return {
                "task": task_name,
                "accuracy": accuracy,
                "acc_norm": acc_norm,
                "file": json_file
            }

        return None

    except Exception as e:
        print_warning(f"Failed to extract from {json_file}: {e}")
        return None


def extract_all_scores(directory: str) -> Dict[str, Dict]:
    print_info(f"\nExtracting scores from: {directory}")

    json_files = find_json_files(directory)

    # Check for missing tasks
    missing_tasks = set(EVALUATION_TASKS.keys()) - set(json_files.keys())
    if missing_tasks:
        print_warning(f"  Missing tasks: {missing_tasks}")
        print_info(f"  Searching for alternative files...")

        # Try to find files with alternative naming
        for file in os.listdir(directory):
            if file.endswith('.json') and 'truthfulqa' in file.lower():
                print_info(f"    Found truthfulqa file: {file}")

    results = {}

    for task_name, file_path in json_files.items():
        score_data = extract_accuracy_from_json(file_path)
        if score_data:
            results[task_name] = score_data
            if task_name in EVALUATION_TASKS:
                results[task_name]['fewshot'] = EVALUATION_TASKS[task_name]
            if score_data['accuracy']:
                print_success(f"  {task_name}: acc={score_data['accuracy']:.4f}")
            else:
                print_warning(f"  {task_name}: accuracy is None")
        else:
            results[task_name] = {
                "task": task_name,
                "accuracy": None,
                "acc_norm": None,
                "file": file_path,
                "fewshot": EVALUATION_TASKS.get(task_name, 'N/A')
            }
            print_warning(f"  {task_name}: extraction failed")

    return results


# ==============================================================================
# Report Generation
# ==============================================================================
def generate_model_report(results: Dict[str, Dict], model_name: str, output_file: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    valid_accuracies = [r['accuracy'] for r in results.values() if r['accuracy'] is not None]
    avg_accuracy = sum(valid_accuracies) / len(valid_accuracies) if valid_accuracies else 0

    valid_acc_norms = [r['acc_norm'] for r in results.values() if r['acc_norm'] is not None]
    avg_acc_norm = sum(valid_acc_norms) / len(valid_acc_norms) if valid_acc_norms else 0

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"EVALUATION RESULTS SUMMARY: {model_name.upper()}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {timestamp}\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Note: All evaluations use TEST split (lm_eval default)\n\n")

        f.write("-" * 80 + "\n")
        f.write("INDIVIDUAL DATASET RESULTS (8 Benchmark Test Sets)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Dataset':<30} {'Fewshot':<10} {'Accuracy':<15} {'Acc Norm':<15}\n")
        f.write("-" * 80 + "\n")

        for task in EVALUATION_TASKS.keys():
            if task in results:
                result = results[task]
                display_name = TASK_DISPLAY_NAMES.get(task, task)
                fewshot = result.get('fewshot', 'N/A')
                accuracy = f"{result['accuracy']:.4f}" if result['accuracy'] else "N/A"
                acc_norm = f"{result['acc_norm']:.4f}" if result['acc_norm'] else "N/A"
                f.write(f"{display_name:<30} {fewshot:<10} {accuracy:<15} {acc_norm:<15}\n")
            else:
                display_name = TASK_DISPLAY_NAMES.get(task, task)
                fewshot = EVALUATION_TASKS[task]
                f.write(f"{display_name:<30} {fewshot:<10} {'N/A':<15} {'N/A':<15}\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'AVERAGE':<30} {'':<10} {avg_accuracy:.4f}{'':<11} {avg_acc_norm:.4f}\n")
        f.write("=" * 80 + "\n\n")

        f.write("-" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Datasets: {len(EVALUATION_TASKS)}\n")
        f.write(f"Successful Extractions: {len(valid_accuracies)}\n")
        f.write(f"Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy * 100:.2f}%)\n")
        f.write(f"Average Acc Norm: {avg_acc_norm:.4f} ({avg_acc_norm * 100:.2f}%)\n")
        f.write("=" * 80 + "\n")

    print_success(f"Model report saved: {output_file}")


def generate_comparison_report(base_results: Dict[str, Dict], sft_results: Dict[str, Dict], output_file: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    base_accuracies = [r['accuracy'] for r in base_results.values() if r['accuracy'] is not None]
    sft_accuracies = [r['accuracy'] for r in sft_results.values() if r['accuracy'] is not None]

    base_avg = sum(base_accuracies) / len(base_accuracies) if base_accuracies else 0
    sft_avg = sum(sft_accuracies) / len(sft_accuracies) if sft_accuracies else 0
    improvement = sft_avg - base_avg

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SFT EVALUATION COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {timestamp}\n")
        f.write(f"Note: All evaluations use TEST split (lm_eval default)\n\n")

        f.write("-" * 80 + "\n")
        f.write("INDIVIDUAL DATASET COMPARISON (8 Benchmark Test Sets)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Dataset':<25} {'Fewshot':<8} {'Base':<12} {'SFT':<12} {'Improvement':<15}\n")
        f.write("-" * 80 + "\n")

        for task in EVALUATION_TASKS.keys():
            display_name = TASK_DISPLAY_NAMES.get(task, task)
            fewshot = EVALUATION_TASKS[task]

            base_acc = base_results.get(task, {}).get('accuracy')
            sft_acc = sft_results.get(task, {}).get('accuracy')

            base_str = f"{base_acc:.4f}" if base_acc else "N/A"
            sft_str = f"{sft_acc:.4f}" if sft_acc else "N/A"

            if base_acc and sft_acc:
                imp = sft_acc - base_acc
                if imp > 0:
                    imp_str = f"+{imp:.4f} ↑"
                elif imp < 0:
                    imp_str = f"{imp:.4f} ↓"
                else:
                    imp_str = f"{imp:.4f} ="
            else:
                imp_str = "N/A"

            f.write(f"{display_name:<25} {fewshot:<8} {base_str:<12} {sft_str:<12} {imp_str:<15}\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'AVERAGE':<25} {'':<8} {base_avg:.4f}{'':<8} {sft_avg:.4f}{'':<8} {improvement:+.4f}\n")
        f.write("=" * 80 + "\n\n")

        f.write("-" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Base Model Average Accuracy: {base_avg:.4f} ({base_avg * 100:.2f}%)\n")
        f.write(f"SFT Model Average Accuracy: {sft_avg:.4f} ({sft_avg * 100:.2f}%)\n")
        f.write(f"Overall Improvement: {improvement:+.4f} ({improvement * 100:+.2f}%)\n")

        if improvement > 0:
            f.write(f"\nConclusion: SFT improved model performance by {improvement * 100:.2f}%\n")
        elif improvement < 0:
            f.write(f"\nConclusion: SFT decreased model performance by {abs(improvement) * 100:.2f}%\n")
        else:
            f.write(f"\nConclusion: SFT had no significant effect on model performance\n")

        f.write("=" * 80 + "\n")

    print_success(f"Comparison report saved: {output_file}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    print_header("EVALUATION RESULTS SUMMARY GENERATOR")

    base_model_dir, sft_model_dir = find_model_directories(EVAL_RESULTS_DIR)

    if not base_model_dir:
        print_error("base_model directory not found!")
        print_info("Expected format: base_model_YYYYMMDD_HHMMSS")
        sys.exit(1)

    if not sft_model_dir:
        print_error("sft_model directory not found!")
        print_info("Expected format: sft_model_YYYYMMDD_HHMMSS")
        sys.exit(1)

    print()

    print_section("EXTRACTING BASE MODEL SCORES")
    base_results = extract_all_scores(base_model_dir)

    print_section("EXTRACTING SFT MODEL SCORES")
    sft_results = extract_all_scores(sft_model_dir)

    print_section("GENERATING REPORTS")

    base_report_file = os.path.join(EVAL_RESULTS_DIR, "base_model_summary.txt")
    generate_model_report(base_results, "base_model", base_report_file)

    sft_report_file = os.path.join(EVAL_RESULTS_DIR, "sft_model_summary.txt")
    generate_model_report(sft_results, "sft_model", sft_report_file)

    comparison_report_file = os.path.join(EVAL_RESULTS_DIR, "comparison_summary.txt")
    generate_comparison_report(base_results, sft_results, comparison_report_file)

    print_section("SUMMARY")
    print(f"{'Dataset':<25} {'Base':<12} {'SFT':<12} {'Improvement':<15}")
    print(f"{'-' * 64}")

    for task in EVALUATION_TASKS.keys():
        display_name = TASK_DISPLAY_NAMES.get(task, task)
        base_acc = base_results.get(task, {}).get('accuracy')
        sft_acc = sft_results.get(task, {}).get('accuracy')

        base_str = f"{base_acc:.4f}" if base_acc else "N/A"
        sft_str = f"{sft_acc:.4f}" if sft_acc else "N/A"

        if base_acc and sft_acc:
            imp = sft_acc - base_acc
            imp_str = f"{imp:+.4f}"
        else:
            imp_str = "N/A"

        print(f"{display_name:<25} {base_str:<12} {sft_str:<12} {imp_str:<15}")

    print(f"{'-' * 64}")

    base_avgs = [r['accuracy'] for r in base_results.values() if r['accuracy']]
    sft_avgs = [r['accuracy'] for r in sft_results.values() if r['accuracy']]

    if base_avgs and sft_avgs:
        base_avg = sum(base_avgs) / len(base_avgs)
        sft_avg = sum(sft_avgs) / len(sft_avgs)
        print(f"{'AVERAGE':<25} {base_avg:.4f}{'':<8} {sft_avg:.4f}{'':<8} {sft_avg - base_avg:+.4f}")

    print()
    print_header("SUMMARY GENERATION COMPLETED")
    print(f"Results Files:")
    print(f"  - Base Model: {base_report_file}")
    print(f"  - SFT Model: {sft_report_file}")
    print(f"  - Comparison: {comparison_report_file}")


if __name__ == "__main__":
    main()