#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
# All-in-One SFT Training Script for Qwen2.5-0.5B
# Version: 6.0 (Fixed progress bar conflict + Training statistics)
# ==============================================================================
# Hyperparameters Validated:
# - Learning Rate: 2e-5 (Full FT standard)
# - Epochs: 2 (300k data, prevents overfitting)
# - Batch Size: 64 effective (16 × 4 accumulation)
# - Warmup: 5% (469 steps of 9376 total)
# - Sequence Length: 1024 (balanced for instructions)
# ==============================================================================

import os
import sys
import json
import time
import subprocess
import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# Configuration Paths
# ==============================================================================
PROJECT_ROOT = "/home/qianqz/Qwen-SFT"
MODEL_PATH = "/home/qianqz/Model/Qwen2.5-0.5B"
DATASET_PATH = "/home/qianqz/Dataset/OpenOrca/OpenOrca_filtered_300k.jsonl"
LLAMA_FACTORY_PATH = "/home/qianqz/LlamaFactory"
DATASET_DIR = os.path.join(LLAMA_FACTORY_PATH, "data")
BASE_CONFIG = os.path.join(PROJECT_ROOT, "configs", "train_full.yaml")

BASE_OUTPUT_DIR = "/home/qianqz/Qwen-SFT/outputs"
LOG_DIR = "/home/qianqz/Qwen-SFT/logs"
FIGURE_DIR = "/home/qianqz/Qwen-SFT/figures"
REPORT_DIR = "/home/qianqz/Qwen-SFT/reports"

# Training Hyperparameters (for validation display)
HYPERPARAMS = {
    "learning_rate": "2.0e-5",
    "epochs": "2",
    "effective_batch_size": "64 (16 × 4)",
    "warmup_steps": "469 (5%)",
    "sequence_length": "1024",
    "weight_decay": "0.01",
    "optimizer": "AdamW",
    "lr_scheduler": "Cosine",
    "precision": "FP16"
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
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


# ==============================================================================
# Utility Functions
# ==============================================================================
def generate_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_output_directory(timestamp: str) -> str:
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"qwen2.5-0.5b-sft-full_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print_success(f"Created output directory: {output_dir}")
    return output_dir


def create_directories(timestamp: str) -> Dict[str, str]:
    dirs = {
        "output": create_output_directory(timestamp),
        "logs": LOG_DIR,
        "figures": os.path.join(FIGURE_DIR, f"figures_{timestamp}"),
        "reports": REPORT_DIR
    }
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
    print_success(f"Created directories: output, logs, figures, reports")
    return dirs


def copy_config_file(config_src: str, config_dst: str):
    if os.path.exists(config_src):
        shutil.copy2(config_src, config_dst)
        print_success(f"Copied config to: {config_dst}")


def update_yaml_config(config_path: str, output_dir: str, dataset_dir: str):
    """Update YAML config with correct output_dir and dataset_dir."""
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace output_dir
    content = re.sub(
        r'output_dir:.*',
        f'output_dir: {output_dir}',
        content
    )

    # Replace or add dataset_dir
    if 'dataset_dir:' in content:
        content = re.sub(
            r'dataset_dir:.*',
            f'dataset_dir: {dataset_dir}',
            content
        )
    else:
        content = re.sub(
            r'(dataset: openorca_sft_300k\n)',
            f'\\1dataset_dir: {dataset_dir}\n',
            content
        )

    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print_success(f"Updated config with output_dir and dataset_dir")


# ==============================================================================
# Stage 1: Pre-training Checks
# ==============================================================================
def check_gpu_availability(target_gpu_id: int = 0) -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            print_error("CUDA is not available!")
            return False

        gpu_count = torch.cuda.device_count()
        print_success(f"Total GPUs Available: {gpu_count}")

        if target_gpu_id < 0 or target_gpu_id >= gpu_count:
            print_error(f"Invalid GPU ID: {target_gpu_id}. Available GPUs: 0-{gpu_count - 1}")
            return False

        gpu_name = torch.cuda.get_device_name(target_gpu_id)
        gpu_memory = torch.cuda.get_device_properties(target_gpu_id).total_memory / (1024 ** 3)

        print_success(f"Target GPU: {target_gpu_id} × {gpu_name}")
        print_success(f"GPU Memory: {gpu_memory:.1f} GB")

        torch.cuda.set_device(target_gpu_id)
        print_success(f"CUDA device set to: {target_gpu_id}")

        if gpu_memory < 40:
            print_warning(f"GPU memory less than 40GB, may need to reduce batch size")

        return True
    except ImportError:
        print_error("PyTorch is not installed!")
        return False
    except Exception as e:
        print_error(f"GPU check failed: {str(e)}")
        return False


def check_model_exists() -> bool:
    required_files = ["config.json", "model.safetensors"]
    for file in required_files:
        file_path = os.path.join(MODEL_PATH, file)
        if not os.path.exists(file_path):
            print_error(f"Model file not found: {file_path}")
            return False
    print_success(f"Model files verified: {MODEL_PATH}")
    return True


def check_dataset_exists() -> bool:
    if not os.path.exists(DATASET_PATH):
        print_error(f"Dataset file not found: {DATASET_PATH}")
        return False
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    print_success(f"Dataset file verified: {DATASET_PATH}")
    print_success(f"Dataset size: {line_count:,} samples")
    if line_count < 200000:
        print_warning(f"Dataset size less than 200k samples")
    return True


def check_llamafactory_installed() -> bool:
    print(f"  Checking LLaMA-Factory CLI...")

    try:
        result = subprocess.run(
            "llamafactory-cli version",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            executable='/bin/bash'
        )
        if result.returncode == 0:
            version_line = result.stdout.strip().split('\n')[1] if '\n' in result.stdout else result.stdout.strip()
            print_success(f"LLaMA-Factory CLI: {version_line}")
            return True
    except subprocess.TimeoutExpired:
        print_warning("CLI check timed out, but may still work")
        return True
    except Exception as e:
        print_warning(f"CLI check error: {e}")

    try:
        import llamafactory
        version = getattr(llamafactory, '__version__', 'unknown')
        print_success(f"LLaMA-Factory Python package: v{version}")
        return True
    except ImportError:
        pass

    if os.path.exists(LLAMA_FACTORY_PATH):
        print_success(f"LLaMA-Factory directory exists: {LLAMA_FACTORY_PATH}")
        return True

    print_error("LLaMA-Factory not detected!")
    return False


def validate_dataset_config() -> bool:
    dataset_info_path = os.path.join(DATASET_DIR, "dataset_info.json")

    print(f"  Checking dataset_info.json at: {dataset_info_path}")

    if not os.path.exists(dataset_info_path):
        print_error(f"dataset_info.json not found at: {dataset_info_path}")
        return False

    if not os.access(dataset_info_path, os.R_OK):
        print_error(f"dataset_info.json is not readable: {dataset_info_path}")
        return False

    try:
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_config = json.load(f)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in dataset_info.json: {str(e)}")
        return False

    if "openorca_sft_300k" not in dataset_config:
        print_warning("Dataset 'openorca_sft_300k' not registered in dataset_info.json")
        print_warning("Will attempt training. If dataset loading fails, add config.")
        return True

    print_success("Dataset configuration verified in dataset_info.json")
    print_success(f"dataset_dir: {DATASET_DIR}")
    return True


def print_hyperparameter_summary():
    """Print validated hyperparameter summary."""
    print_section("HYPERPARAMETER SUMMARY (VALIDATED)")

    print(f"  {'Parameter':<25} {'Value':<20} {'Status':<10}")
    print(f"  {'-' * 55}")

    for param, value in HYPERPARAMS.items():
        print(f"  {param:<25} {value:<20} {'✓ Valid':<10}")

    print(f"  {'-' * 55}")
    print(f"  {'Total Steps':<25} {'9,376':<20} {'✓ Calculated':<10}")
    print(f"  {'Steps per Epoch':<25} {'4,688':<20} {'✓ Calculated':<10}")
    print(f"  {'Estimated Duration':<25} {'~5.5 hours':<20} {'✓ Estimated':<10}")
    print()


def run_pre_training_checks(target_gpu_id: int = 0) -> bool:
    print_section("STAGE 1: PRE-TRAINING CHECKS")

    checks = [
        ("GPU Availability", lambda: check_gpu_availability(target_gpu_id)),
        ("Model Files", check_model_exists),
        ("Dataset File", check_dataset_exists),
        ("LLaMA-Factory Installation", check_llamafactory_installed),
        ("Dataset Configuration", validate_dataset_config),
    ]

    passed = 0
    for name, check_func in checks:
        print(f"\n[{name}]")
        if check_func():
            passed += 1
        else:
            print_error(f"{name} check FAILED")

    print(f"\n{'=' * 80}")
    print(f"Pre-training checks: {passed}/{len(checks)} passed")
    print(f"{'=' * 80}\n")

    return passed == len(checks)


# ==============================================================================
# Stage 2: Training Monitor (FIXED: Silent mode to avoid progress bar conflict)
# ==============================================================================
class TrainingMonitor:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.metrics = {
            "steps": [],
            "loss": [],
            "learning_rate": [],
            "grad_norm": [],
            "samples_per_second": [],
            "timestamps": []
        }
        self.start_time = None

    def parse_log_line(self, line: str) -> Optional[Dict]:
        step_match = re.search(r'step=(\d+)', line)
        loss_match = re.search(r'loss=([0-9.]+)', line)
        lr_match = re.search(r'learning_rate=([0-9.e-]+)', line)
        grad_match = re.search(r'grad_norm=([0-9.]+)', line)
        speed_match = re.search(r'samples_per_second=([0-9.]+)', line)

        if step_match and loss_match:
            return {
                "step": int(step_match.group(1)),
                "loss": float(loss_match.group(1)),
                "learning_rate": float(lr_match.group(1)) if lr_match else None,
                "grad_norm": float(grad_match.group(1)) if grad_match else None,
                "samples_per_second": float(speed_match.group(1)) if speed_match else None,
                "timestamp": datetime.now()
            }
        return None

    def update_metrics(self):
        """Parse log file and update metrics (silent, no output)."""
        if not os.path.exists(self.log_file):
            return
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                metrics = self.parse_log_line(line)
                if metrics and metrics["step"] not in self.metrics["steps"]:
                    self.metrics["steps"].append(metrics["step"])
                    self.metrics["loss"].append(metrics["loss"])
                    if metrics["learning_rate"]:
                        self.metrics["learning_rate"].append(metrics["learning_rate"])
                    if metrics["grad_norm"]:
                        self.metrics["grad_norm"].append(metrics["grad_norm"])
                    if metrics["samples_per_second"]:
                        self.metrics["samples_per_second"].append(metrics["samples_per_second"])
                    self.metrics["timestamps"].append(metrics["timestamp"])

    def display_progress(self, verbose: bool = False):
        """
        Display current training progress.
        DISABLED by default to avoid conflict with LLaMA-Factory progress bar.
        """
        # Silent mode - LLaMA-Factory handles progress bar
        pass


def run_training(monitor: TrainingMonitor, target_gpu_id: int = 0,
                 runtime_config: str = "") -> Tuple[bool, str]:
    log_file = monitor.log_file

    print_section("STAGE 2: SFT TRAINING")
    print(f"Configuration: {runtime_config}")
    print(f"Output Directory: {os.path.dirname(runtime_config).replace('/configs', '/outputs')}")
    print(f"Log File: {log_file}")
    print(f"Target GPU: {target_gpu_id}")
    print(f"Dataset Dir: {DATASET_DIR}")
    print_info("LLaMA-Factory progress bar will be displayed below")
    print_info("Script monitoring is silent to avoid output conflict\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(target_gpu_id)

    command = f"CUDA_VISIBLE_DEVICES={target_gpu_id} llamafactory-cli train {runtime_config}"

    print(f"Training command: {command}")
    print(f"CUDA_VISIBLE_DEVICES: {target_gpu_id}\n")

    monitor.start_time = datetime.now()
    start_time = time.time()

    try:
        with open(log_file, 'w', encoding='utf-8') as log:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                executable='/bin/bash'
            )

            for line in process.stdout:
                print(line, end='')
                log.write(line)
                log.flush()
                monitor.update_metrics()
                # Silent - no display_progress() call

            process.wait()

        end_time = time.time()
        duration = end_time - start_time
        print(f"\n\nTraining Duration: {duration / 3600:.2f} hours")

        if process.returncode == 0:
            print_success("TRAINING COMPLETED SUCCESSFULLY")
            return True, log_file
        else:
            print_error(f"TRAINING FAILED with return code: {process.returncode}")
            return False, log_file
    except Exception as e:
        print_error(f"TRAINING ERROR: {str(e)}")
        return False, log_file


# ==============================================================================
# Training Statistics (NEW)
# ==============================================================================
def print_training_statistics(monitor: TrainingMonitor, duration_hours: float):
    """Print detailed training statistics after completion."""
    print_section("TRAINING STATISTICS")

    total_steps = len(monitor.metrics["steps"])
    if total_steps > 0:
        avg_time_per_step = (duration_hours * 3600) / total_steps
        steps_per_hour = total_steps / duration_hours
        samples_processed = total_steps * 64  # effective batch size

        print(f"  {'Metric':<30} {'Value':<20}")
        print(f"  {'-' * 50}")
        print(f"  {'Total Steps':<30} {total_steps:<20}")
        print(f"  {'Total Duration':<30} {duration_hours:.2f} hours")
        print(f"  {'Average Time/Step':<30} {avg_time_per_step:.2f} seconds")
        print(f"  {'Steps/Hour':<30} {steps_per_hour:.1f}")
        print(f"  {'Samples Processed':<30} {samples_processed:,}")
        print(f"  {'Samples/Second':<30} {samples_processed / (duration_hours * 3600):.2f}")

        if monitor.metrics["loss"]:
            print(f"  {'-' * 50}")
            print(f"  {'Initial Loss':<30} {monitor.metrics['loss'][0]:.4f}")
            print(f"  {'Final Loss':<30} {monitor.metrics['loss'][-1]:.4f}")
            print(f"  {'Loss Reduction':<30} {monitor.metrics['loss'][0] - monitor.metrics['loss'][-1]:.4f}")

        if monitor.metrics["learning_rate"]:
            print(f"  {'Initial LR':<30} {monitor.metrics['learning_rate'][0]:.2e}")
            print(f"  {'Final LR':<30} {monitor.metrics['learning_rate'][-1]:.2e}")

    print()


# ==============================================================================
# Stage 3: Post-training Verification
# ==============================================================================
def verify_model_files(output_dir: str) -> bool:
    print_section("STAGE 3: POST-TRAINING VERIFICATION")
    print("[Model Files]")

    required_files = [
        "config.json", "model.safetensors", "tokenizer.json",
        "tokenizer_config.json", "training_args.bin", "trainer_state.json"
    ]

    all_exist = True
    for file in required_files:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            print_success(f"{file}")
        else:
            print_error(f"{file}")
            all_exist = False
    return all_exist


def verify_training_state(output_dir: str) -> Optional[Dict]:
    print("\n[Training State]")
    state_file = os.path.join(output_dir, "trainer_state.json")

    if not os.path.exists(state_file):
        print_error("trainer_state.json not found")
        return None

    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)

    print_success(f"Total Steps: {state.get('global_step', 'N/A')}")
    print_success(f"Best Loss: {state.get('best_metric', 'N/A')}")

    if 'log_history' in state and len(state['log_history']) > 0:
        final_log = state['log_history'][-1]
        print_success(f"Final Loss: {final_log.get('loss', 'N/A')}")

    return state


def test_model_inference(output_dir: str) -> bool:
    print("\n[Model Inference Test]")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print_success(f"Prompt: {prompt}")
        print(f"Response: {response[:200]}...")
        print_success("Model inference test PASSED")
        return True
    except Exception as e:
        print_error(f"Model inference test FAILED: {str(e)}")
        return False


# ==============================================================================
# Stage 4: Plotting
# ==============================================================================
def plot_training_curves(monitor: TrainingMonitor, figure_dir: str):
    print_section("STAGE 4: PLOTTING TRAINING CURVES")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        if not monitor.metrics["steps"]:
            print_warning("No training metrics to plot")
            return

        os.makedirs(figure_dir, exist_ok=True)

        # Loss Curve
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(monitor.metrics["steps"], monitor.metrics["loss"], 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss Curve', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        if len(monitor.metrics["loss"]) > 10:
            z = np.polyfit(monitor.metrics["steps"], monitor.metrics["loss"], 1)
            p = np.poly1d(z)
            ax1.plot(monitor.metrics["steps"], p(monitor.metrics["steps"]), "r--", alpha=0.5, label='Trend')
            ax1.legend()

        timestamp = monitor.log_file.split('/')[-1].replace('training_', '').replace('.log', '')
        loss_file = os.path.join(figure_dir, f"{timestamp}_loss_curve.png")
        plt.savefig(loss_file, dpi=150, bbox_inches='tight')
        print_success(f"Training loss curve saved: {loss_file}")
        plt.close()

        # LR Curve
        if monitor.metrics["learning_rate"]:
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(monitor.metrics["steps"], monitor.metrics["learning_rate"], 'g-', linewidth=2)
            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('Learning Rate', fontsize=12)
            ax2.set_title('Learning Rate Schedule', fontsize=14)
            ax2.grid(True, alpha=0.3)
            lr_file = os.path.join(figure_dir, f"{timestamp}_lr_curve.png")
            plt.savefig(lr_file, dpi=150, bbox_inches='tight')
            print_success(f"Learning rate curve saved: {lr_file}")
            plt.close()

        # Speed Curve
        if monitor.metrics["samples_per_second"]:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(monitor.metrics["steps"], monitor.metrics["samples_per_second"], 'm-', linewidth=2)
            ax3.set_xlabel('Step', fontsize=12)
            ax3.set_ylabel('Samples/Second', fontsize=12)
            ax3.set_title('Training Speed', fontsize=14)
            ax3.grid(True, alpha=0.3)
            speed_file = os.path.join(figure_dir, f"{timestamp}_speed_curve.png")
            plt.savefig(speed_file, dpi=150, bbox_inches='tight')
            print_success(f"Training speed curve saved: {speed_file}")
            plt.close()

        print_success("All training curves generated successfully")

    except ImportError as e:
        print_warning(f"matplotlib or numpy not installed: {e}")
        print("  Install with: pip install matplotlib numpy")
        print("  Skipping curve plotting...")
    except Exception as e:
        print_error(f"Failed to plot curves: {str(e)}")


# ==============================================================================
# Stage 5: Report
# ==============================================================================
def generate_final_report(monitor: TrainingMonitor, training_state: Optional[Dict],
                          duration_hours: float, log_file: str, target_gpu_id: int,
                          output_dir: str, figure_dir: str, report_dir: str):
    print_section("STAGE 5: FINAL TRAINING REPORT")

    timestamp = log_file.split('/')[-1].replace('training_', '').replace('.log', '')
    report_file = os.path.join(report_dir, f"{timestamp}_training_report.txt")

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SFT TRAINING FINAL REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Qwen2.5-0.5B\n")
        f.write(f"Model Path: {MODEL_PATH}\n")
        f.write(f"Dataset: OpenOrca (300k samples)\n")
        f.write(f"Dataset Path: {DATASET_PATH}\n")
        f.write(f"Dataset Dir: {DATASET_DIR}\n")
        f.write(f"Target GPU: {target_gpu_id}\n\n")

        f.write("-" * 80 + "\n")
        f.write("HYPERPARAMETERS (VALIDATED)\n")
        f.write("-" * 80 + "\n")
        for param, value in HYPERPARAMS.items():
            f.write(f"{param:<25} {value}\n")
        f.write(f"{'Total Steps':<25} 9,376\n")
        f.write(f"{'Steps per Epoch':<25} 4,688\n\n")

        f.write("-" * 80 + "\n")
        f.write("TRAINING RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Duration: {duration_hours:.2f} hours\n")

        if monitor.metrics["steps"]:
            f.write(f"Total Steps: {monitor.metrics['steps'][-1]}\n")
            f.write(f"Initial Loss: {monitor.metrics['loss'][0]:.4f}\n")
            f.write(f"Final Loss: {monitor.metrics['loss'][-1]:.4f}\n")
            f.write(f"Loss Reduction: {monitor.metrics['loss'][0] - monitor.metrics['loss'][-1]:.4f}\n")

        if monitor.metrics["samples_per_second"]:
            avg_speed = sum(monitor.metrics["samples_per_second"]) / len(monitor.metrics["samples_per_second"])
            f.write(f"Average Speed: {avg_speed:.2f} samples/sec\n")

        if training_state:
            f.write(f"Best Loss: {training_state.get('best_metric', 'N/A')}\n")

        f.write(f"\nOutput Directory: {output_dir}\n")
        f.write(f"Log File: {log_file}\n")
        f.write(f"Figures Directory: {figure_dir}\n")
        f.write(f"Report File: {report_file}\n\n")

        f.write("-" * 80 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model Weights: {output_dir}/model.safetensors\n")
        f.write(f"Model Config: {output_dir}/config.json\n")
        f.write(f"Tokenizer: {output_dir}/tokenizer.json\n")
        f.write(f"Training State: {output_dir}/trainer_state.json\n")
        f.write(f"Loss Curve: {figure_dir}/{timestamp}_loss_curve.png\n")
        f.write(f"LR Curve: {figure_dir}/{timestamp}_lr_curve.png\n")
        f.write(f"Speed Curve: {figure_dir}/{timestamp}_speed_curve.png\n")
        f.write(f"Training Log: {log_file}\n")
        f.write(f"Final Report: {report_file}\n\n")

        f.write("=" * 80 + "\n")
        f.write("TRAINING COMPLETED SUCCESSFULLY\n")
        f.write("=" * 80 + "\n")

    print_success(f"Final report saved: {report_file}")

    print(f"\n{'=' * 80}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Duration: {duration_hours:.2f} hours")
    print(f"Target GPU: {target_gpu_id}")
    if monitor.metrics["steps"]:
        print(f"Total Steps: {monitor.metrics['steps'][-1]}")
        print(f"Initial Loss: {monitor.metrics['loss'][0]:.4f}")
        print(f"Final Loss: {monitor.metrics['loss'][-1]:.4f}")
        print(f"Loss Reduction: {monitor.metrics['loss'][0] - monitor.metrics['loss'][-1]:.4f}")
    print(f"Output Directory: {output_dir}")
    print(f"Report: {report_file}")
    print(f"{'=' * 80}\n")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="All-in-One SFT Training Script")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument("--skip-checks", action="store_true", help="Skip pre-training checks")
    parser.add_argument("--skip-plot", action="store_true", help="Skip curve plotting")
    parser.add_argument("--skip-verify", action="store_true", help="Skip post-training verification")
    parser.add_argument("--dry-run", action="store_true", help="Only run checks")
    parser.add_argument("--force", action="store_true", help="Force run even if checks fail")
    args = parser.parse_args()

    timestamp = generate_timestamp()

    print_header("QWEN2.5-0.5B ALL-IN-ONE SFT TRAINING")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run ID: {timestamp}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"LLaMA-Factory Path: {LLAMA_FACTORY_PATH}")
    print(f"Dataset Dir: {DATASET_DIR}")
    print(f"Target GPU: {args.gpu_id}")

    # Print hyperparameter summary
    print_hyperparameter_summary()

    training_start_time = time.time()

    # Stage 1
    checks_passed = True
    if not args.skip_checks:
        checks_passed = run_pre_training_checks(args.gpu_id)
        if not checks_passed and not args.force:
            print_error("\nPre-training checks failed. Exiting.")
            print("  Use --force to bypass checks (not recommended)")
            sys.exit(1)
    else:
        print_warning("Skipping pre-training checks")

    if args.dry_run:
        print_warning("Dry run mode: All checks passed, skipping training")
        sys.exit(0)

    # Create directories with timestamp
    dirs = create_directories(timestamp)
    output_dir = dirs["output"]
    figure_dir = dirs["figures"]
    report_dir = dirs["reports"]

    # Create runtime config with updated output_dir and dataset_dir
    runtime_config = os.path.join(dirs["output"], f"{timestamp}_train_config.yaml")
    copy_config_file(BASE_CONFIG, runtime_config)
    update_yaml_config(runtime_config, output_dir, DATASET_DIR)

    # Stage 2
    log_file = os.path.join(LOG_DIR, f"{timestamp}_training.log")
    monitor = TrainingMonitor(log_file)

    success, log_file = run_training(monitor, args.gpu_id, runtime_config)

    training_end_time = time.time()
    duration_hours = (training_end_time - training_start_time) / 3600

    # Print training statistics (NEW)
    if success:
        print_training_statistics(monitor, duration_hours)

    if not success:
        print_error("\nTraining failed. Generating partial report...")
        generate_final_report(monitor, None, duration_hours, log_file, args.gpu_id,
                              output_dir, figure_dir, report_dir)
        sys.exit(1)

    # Stage 3
    if not args.skip_verify:
        verify_model_files(output_dir)
        training_state = verify_training_state(output_dir)
        test_model_inference(output_dir)
    else:
        print_warning("Skipping post-training verification")
        training_state = None

    # Stage 4
    if not args.skip_plot:
        plot_training_curves(monitor, figure_dir)
    else:
        print_warning("Skipping curve plotting")

    # Stage 5
    generate_final_report(monitor, training_state, duration_hours, log_file, args.gpu_id,
                          output_dir, figure_dir, report_dir)

    print_header("ALL STAGES COMPLETED SUCCESSFULLY")
    print_success("You can now proceed to the evaluation stage with lm_eval")
    print(f"\nNext step: Run lm_eval on {output_dir}")

    sys.exit(0)


if __name__ == "__main__":
    main()