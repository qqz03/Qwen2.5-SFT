#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
# SFT Model Evaluation Script using lm_eval
# Version: 5.0 (Fixed: Correct lm_eval task names)
# ==============================================================================
# Note: lm_eval automatically uses test split for these tasks
# Task names should NOT have _test suffix
# ==============================================================================

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

# ==============================================================================
# Configuration Paths
# ==============================================================================
PROJECT_ROOT = "/home/qianqz/Qwen-SFT"
BASE_MODEL_PATH = "/home/qianqz/Model/Qwen2.5-0.5B"
SFT_MODEL_PATH = "/home/qianqz/Qwen-SFT/outputs/qwen2.5-0.5b-sft-full_20260302_020509/checkpoint-9376"
EVAL_RESULTS_DIR = "/home/qianqz/Qwen-SFT/eval_results"

# ==============================================================================
# Evaluation Tasks Configuration (FIXED: Correct lm_eval task names)
# ==============================================================================
# lm_eval task names (without _test suffix)
# lm_eval automatically uses test split for these benchmark tasks
EVALUATION_TASKS = {
    "mmlu": 5,  # Knowledge + Reasoning (uses test split)
    "arc_easy": 0,  # Scientific common sense (uses test split)
    "arc_challenge": 25,  # Difficult science questions (uses test split)
    "hellaswag": 10,  # Common sense completion (uses test split)
    "winogrande": 5,  # Reference resolution (uses test split)
    "truthfulqa_mc2": 0,  # Fact-checking reliability (uses test split)
    "piqa": 0,  # Physics general knowledge (uses test split)
    "boolq": 0,  # Reading comprehension (uses test split)
}

# Display names for reports (with Test notation)
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
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.END}\n")


def print_section(text: str):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'-' * 80}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'-' * 80}{Colors.END}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text: str):
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")


# ==============================================================================
# Utility Functions
# ==============================================================================
def generate_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_results_directory(model_name: str, timestamp: str) -> str:
    results_dir = os.path.join(EVAL_RESULTS_DIR, f"{model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print_success(f"Created results directory: {results_dir}")
    return results_dir


def check_gpu_available(gpu_id: int) -> bool:
    """Check if specified GPU is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            print_error("CUDA is not available!")
            return False

        gpu_count = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= gpu_count:
            print_error(f"Invalid GPU ID: {gpu_id}. Available GPUs: 0-{gpu_count - 1}")
            return False

        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)
        print_success(f"GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    except ImportError:
        print_warning("PyTorch not available, skipping GPU check")
        return True
    except Exception as e:
        print_warning(f"GPU check failed: {e}")
        return True


def check_lm_eval_installed() -> bool:
    """Check if lm_eval is properly installed."""
    try:
        import lm_eval
        version = getattr(lm_eval, '__version__', '0.4.x')
        print_success(f"lm_eval: v{version}")
        return True
    except:
        print_warning("lm_eval import warning, continuing...")
        return True


def check_model_exists(model_path: str) -> bool:
    """Check if model files exist."""
    required_files = ["config.json", "model.safetensors"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            return False
    print_success(f"Model verified: {model_path}")
    return True


# ==============================================================================
# Evaluation Functions
# ==============================================================================
def run_lm_eval(
        model_path: str,
        task_name: str,
        fewshot: int,
        output_dir: str,
        gpu_id: int = 0,
        batch_size: int = 4
) -> bool:
    """Run lm_eval for a specific task."""
    output_file = os.path.join(output_dir, f"{task_name}_{fewshot}shot.json")

    model_args = f"pretrained={model_path},trust_remote_code=True,dtype=float16"

    command = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"python -m lm_eval "
        f"--model hf "
        f"--model_args '{model_args}' "
        f"--tasks {task_name} "
        f"--num_fewshot {fewshot} "
        f"--device cuda:0 "
        f"--batch_size {batch_size} "
        f"--apply_chat_template "
        f"--output_path '{output_file}'"
    )

    display_name = TASK_DISPLAY_NAMES.get(task_name, task_name)
    print_info(f"Running: {display_name} ({fewshot}-shot) on GPU {gpu_id}")
    print_info(f"Output: {output_file}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            executable='/bin/bash'
        )

        if result.returncode == 0:
            print_success(f"{task_name} evaluation completed")
            return True
        else:
            print_error(f"{task_name} evaluation failed with code {result.returncode}")
            return False

    except Exception as e:
        print_error(f"{task_name} evaluation error: {str(e)}")
        return False


def evaluate_model(
        model_path: str,
        model_name: str,
        tasks: Dict[str, int] = None,
        gpu_id: int = 0,
        batch_size: int = 4
) -> str:
    """Evaluate a model on all tasks."""
    if tasks is None:
        tasks = EVALUATION_TASKS

    timestamp = generate_timestamp()
    results_dir = create_results_directory(model_name, timestamp)

    results = {}
    total_tasks = len(tasks)
    completed_tasks = 0

    print_section(f"EVALUATING: {model_name}")
    print(f"Model Path: {model_path}")
    print(f"Results Directory: {results_dir}")
    print(f"GPU ID: {gpu_id}")
    print(f"Total Tasks: {total_tasks}")
    print(f"Batch Size: {batch_size}")
    print(f"Note: All tasks use TEST split (lm_eval default)\n")

    for i, (task_name, fewshot) in enumerate(tasks.items(), 1):
        display_name = TASK_DISPLAY_NAMES.get(task_name, task_name)
        print(f"\n[{i}/{total_tasks}] Evaluating {display_name} ({fewshot}-shot)...")

        success = run_lm_eval(
            model_path=model_path,
            task_name=task_name,
            fewshot=fewshot,
            output_dir=results_dir,
            gpu_id=gpu_id,
            batch_size=batch_size
        )

        if success:
            completed_tasks += 1
            results[task_name] = {"status": "completed", "fewshot": fewshot}
        else:
            results[task_name] = {"status": "failed", "fewshot": fewshot}

    # Save evaluation summary
    summary_file = os.path.join(results_dir, "evaluation_summary.json")
    summary = {
        "model_name": model_name,
        "model_path": model_path,
        "timestamp": timestamp,
        "gpu_id": gpu_id,
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "results": results,
        "note": "All evaluations use TEST split (lm_eval default for benchmark tasks)"
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print_section("EVALUATION SUMMARY")
    print(f"Model: {model_name}")
    print(f"GPU: {gpu_id}")
    print(f"Completed: {completed_tasks}/{total_tasks} tasks")
    print(f"Results Directory: {results_dir}")

    return results_dir


# ==============================================================================
# Results Extraction and Recording
# ==============================================================================
def extract_scores_from_json(json_file: str) -> Optional[Dict]:
    """Extract accuracy scores from lm_eval result JSON."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'results' in data:
            results = data['results']
            task_name = list(results.keys())[0]
            task_results = results[task_name]

            accuracy = None
            acc_norm = None

            if 'acc,none' in task_results:
                accuracy = task_results['acc,none']
            elif 'acc' in task_results:
                accuracy = task_results['acc']

            if 'acc_norm,none' in task_results:
                acc_norm = task_results['acc_norm,none']
            elif 'acc_norm' in task_results:
                acc_norm = task_results['acc_norm']

            return {
                "task": task_name,
                "accuracy": accuracy,
                "acc_norm": acc_norm,
                "alias": task_results.get('alias', task_name)
            }
    except Exception as e:
        print_warning(f"Failed to extract scores from {json_file}: {e}")

    return None


def extract_all_results(results_dir: str) -> Dict[str, Dict]:
    """Extract all evaluation results from a results directory."""
    all_results = {}

    for task, fewshot in EVALUATION_TASKS.items():
        result_files = list(Path(results_dir).glob(f"{task}_{fewshot}shot.json"))

        if result_files:
            result = extract_scores_from_json(str(result_files[0]))
            if result:
                all_results[task] = result
                all_results[task]['fewshot'] = fewshot
        else:
            all_results[task] = {
                "task": task,
                "accuracy": None,
                "acc_norm": None,
                "fewshot": fewshot,
                "status": "failed"
            }

    return all_results


def save_model_results_to_txt(
        results: Dict[str, Dict],
        model_name: str,
        model_path: str,
        output_file: str,
        gpu_id: int
):
    """Save individual model results to a detailed txt file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    valid_accuracies = [r['accuracy'] for r in results.values() if r['accuracy'] is not None]
    avg_accuracy = sum(valid_accuracies) / len(valid_accuracies) if valid_accuracies else 0

    valid_acc_norms = [r['acc_norm'] for r in results.values() if r['acc_norm'] is not None]
    avg_acc_norm = sum(valid_acc_norms) / len(valid_acc_norms) if valid_acc_norms else 0

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"EVALUATION RESULTS: {model_name.upper()}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {timestamp}\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"GPU ID: {gpu_id}\n")
        f.write(f"Note: All evaluations use TEST split (lm_eval default for benchmark tasks)\n\n")

        f.write("-" * 80 + "\n")
        f.write("INDIVIDUAL DATASET RESULTS (8 Benchmark Test Sets)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Dataset':<30} {'Fewshot':<10} {'Accuracy':<15} {'Acc Norm':<15}\n")
        f.write("-" * 80 + "\n")

        for task, result in results.items():
            display_name = TASK_DISPLAY_NAMES.get(task, task)
            fewshot = result.get('fewshot', 'N/A')
            accuracy = f"{result['accuracy']:.4f}" if result['accuracy'] else "N/A"
            acc_norm = f"{result['acc_norm']:.4f}" if result['acc_norm'] else "N/A"
            f.write(f"{display_name:<30} {fewshot:<10} {accuracy:<15} {acc_norm:<15}\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'AVERAGE':<30} {'':<10} {avg_accuracy:.4f}{'':<11} {avg_acc_norm:.4f}\n")
        f.write("=" * 80 + "\n\n")

        f.write("-" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Datasets: {len(results)}\n")
        f.write(f"Successful Evaluations: {len([r for r in results.values() if r['accuracy'] is not None])}\n")
        f.write(f"Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy * 100:.2f}%)\n")
        f.write(f"Average Acc Norm: {avg_acc_norm:.4f} ({avg_acc_norm * 100:.2f}%)\n")
        f.write("=" * 80 + "\n")

    print_success(f"Model results saved: {output_file}")


def save_comparison_report_to_txt(
        base_results: Dict[str, Dict],
        sft_results: Dict[str, Dict],
        base_model_path: str,
        sft_model_path: str,
        output_file: str,
        gpu_id: int
):
    """Save comparison report to a detailed txt file."""
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
        f.write(f"GPU ID: {gpu_id}\n")
        f.write(f"Note: All evaluations use TEST split (lm_eval default)\n\n")

        f.write("-" * 80 + "\n")
        f.write("MODEL INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Base Model: {base_model_path}\n")
        f.write(f"SFT Model: {sft_model_path}\n\n")

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
                imp_str = f"{imp:+.4f}"
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
            f.write(f"Conclusion: SFT improved model performance by {improvement * 100:.2f}%\n")
        elif improvement < 0:
            f.write(f"Conclusion: SFT decreased model performance by {abs(improvement) * 100:.2f}%\n")
        else:
            f.write(f"Conclusion: SFT had no significant effect on model performance\n")

        f.write("=" * 80 + "\n")

    print_success(f"Comparison report saved: {output_file}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='SFT Model Evaluation Script')
    parser.add_argument('--base-model', type=str, default=BASE_MODEL_PATH,
                        help='Path to base model')
    parser.add_argument('--sft-model', type=str, default=SFT_MODEL_PATH,
                        help='Path to SFT model')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use (default: 0)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--eval-base', action='store_true',
                        help='Evaluate base model only')
    parser.add_argument('--eval-sft', action='store_true',
                        help='Evaluate SFT model only')
    parser.add_argument('--eval-both', action='store_true',
                        help='Evaluate both models (default)')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='Specific tasks to evaluate (default: all 8)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare results from both models')
    args = parser.parse_args()

    print_header("SFT MODEL EVALUATION")

    # Check prerequisites
    print_section("PRE-EVALUATION CHECKS")

    if not check_gpu_available(args.gpu_id):
        print_error("GPU check failed!")
        sys.exit(1)

    check_lm_eval_installed()

    # Display evaluation tasks
    print_section("EVALUATION TASKS (8 Benchmark Test Sets)")
    print(f"{'Task':<25} {'Fewshot':<10} {'Split':<10}")
    print(f"{'-' * 45}")
    for task, fewshot in EVALUATION_TASKS.items():
        display_name = TASK_DISPLAY_NAMES.get(task, task)
        print(f"{display_name:<25} {fewshot}-shot     {'test':<10}")
    print(f"{'-' * 45}")
    print(f"Note: lm_eval automatically uses test split for these tasks\n")

    # Determine which models to evaluate
    eval_base = args.eval_base or (not args.eval_sft and not args.eval_base)
    eval_sft = args.eval_sft or (not args.eval_base and not args.eval_sft)

    if args.eval_both:
        eval_base = True
        eval_sft = True

    # Prepare tasks
    tasks = EVALUATION_TASKS
    if args.tasks:
        tasks = {}
        for arg_task in args.tasks:
            # Remove _test suffix if present
            if arg_task.endswith('_test'):
                arg_task = arg_task[:-5]
            if arg_task in EVALUATION_TASKS:
                tasks[arg_task] = EVALUATION_TASKS[arg_task]
            else:
                print_warning(f"Unknown task: {arg_task}, skipping")

    base_results_dir = None
    sft_results_dir = None
    base_results = {}
    sft_results = {}

    # Evaluate base model
    if eval_base:
        print(f"\n{Colors.CYAN}Evaluating Base Model...{Colors.END}")
        if check_model_exists(args.base_model):
            base_results_dir = evaluate_model(
                model_path=args.base_model,
                model_name="base_model",
                tasks=tasks,
                gpu_id=args.gpu_id,
                batch_size=args.batch_size
            )
            print_success(f"Base model results: {base_results_dir}")

            # Extract and save results to txt
            base_results = extract_all_results(base_results_dir)
            base_results_txt = os.path.join(EVAL_RESULTS_DIR, "base_model_results.txt")
            save_model_results_to_txt(
                base_results, "base_model", args.base_model,
                base_results_txt, args.gpu_id
            )
        else:
            print_error(f"Base model not found: {args.base_model}")
            eval_base = False

    # Evaluate SFT model
    if eval_sft:
        print(f"\n{Colors.CYAN}Evaluating SFT Model...{Colors.END}")
        if check_model_exists(args.sft_model):
            sft_results_dir = evaluate_model(
                model_path=args.sft_model,
                model_name="sft_model",
                tasks=tasks,
                gpu_id=args.gpu_id,
                batch_size=args.batch_size
            )
            print_success(f"SFT model results: {sft_results_dir}")

            # Extract and save results to txt
            sft_results = extract_all_results(sft_results_dir)
            sft_results_txt = os.path.join(EVAL_RESULTS_DIR, "sft_model_results.txt")
            save_model_results_to_txt(
                sft_results, "sft_model", args.sft_model,
                sft_results_txt, args.gpu_id
            )
        else:
            print_error(f"SFT model not found: {args.sft_model}")
            eval_sft = False

    # Compare results and save comparison report
    if args.compare and base_results_dir and sft_results_dir:
        print_section("MODEL COMPARISON")
        print(f"{'Task':<25} {'Fewshot':<8} {'Base':<12} {'SFT':<12} {'Improvement':<12}")
        print(f"{'-' * 65}")
        for task in EVALUATION_TASKS.keys():
            display_name = TASK_DISPLAY_NAMES.get(task, task)
            fewshot = EVALUATION_TASKS[task]
            base_acc = base_results.get(task, {}).get('accuracy')
            sft_acc = sft_results.get(task, {}).get('accuracy')
            base_str = f"{base_acc:.4f}" if base_acc else "N/A"
            sft_str = f"{sft_acc:.4f}" if sft_acc else "N/A"
            if base_acc and sft_acc:
                imp = sft_acc - base_acc
                imp_str = f"{imp:+.4f}"
            else:
                imp_str = "N/A"
            print(f"{display_name:<25} {fewshot:<8} {base_str:<12} {sft_str:<12} {imp_str:<12}")
        print(f"{'-' * 65}\n")

        # Save comparison report to txt
        comparison_txt = os.path.join(EVAL_RESULTS_DIR, "comparison_report.txt")
        save_comparison_report_to_txt(
            base_results, sft_results,
            args.base_model, args.sft_model,
            comparison_txt, args.gpu_id
        )

    print_header("EVALUATION COMPLETED")
    print(f"GPU Used: {args.gpu_id}")
    print(f"Base Model Results: {base_results_dir}")
    print(f"SFT Model Results: {sft_results_dir}")
    print(f"\nResults Files:")
    print(f"  - Base Model: {EVAL_RESULTS_DIR}/base_model_results.txt")
    print(f"  - SFT Model: {EVAL_RESULTS_DIR}/sft_model_results.txt")
    print(f"  - Comparison: {EVAL_RESULTS_DIR}/comparison_report.txt")
    print(f"\nNote: All 8 benchmark evaluations use TEST split (lm_eval default)")


if __name__ == "__main__":
    main()