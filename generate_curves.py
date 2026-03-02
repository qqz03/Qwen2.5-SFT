#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
# Manual Loss Curve Generation Script (Version 3.0)
# Features:
# - Support specifying timestamp via command line
# - Auto-detect available log files
# - Output to matching figures folder with same timestamp
# - File names include timestamp (but NOT in images)
# - Clean plots without annotations
# ==============================================================================

import os
import re
import sys
import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ==============================================================================
# Configuration Paths
# ==============================================================================
PROJECT_ROOT = "/home/qianqz/Qwen-SFT"
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")


# ==============================================================================
# Color Codes
# ==============================================================================
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
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


# ==============================================================================
# Log Parsing
# ==============================================================================
def parse_log_file(log_file):
    """Parse training log and extract loss/learning rate data."""
    steps = []
    losses = []
    learning_rates = []
    grad_norms = []
    step_counter = 0

    print_info(f"Parsing log file: {log_file}")

    with open(log_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Pattern: {'loss': '1.031', 'grad_norm': '0.6468', 'learning_rate': '6.685e-08', 'epoch': '1.931'}
            json_match = re.search(r"\{'loss':\s*'([0-9.]+)'.*?'learning_rate':\s*'([0-9.e+-]+)'", line)

            if json_match:
                try:
                    loss = float(json_match.group(1))
                    lr = float(json_match.group(2))
                    step_counter += 1
                    # Calculate approximate step (logging_steps=50)
                    step = step_counter * 50
                    steps.append(step)
                    losses.append(loss)
                    learning_rates.append(lr)

                    # Show first 5 and every 20th data point
                    if step_counter <= 5 or step_counter % 20 == 0:
                        print(f"  Step {step:>5}: Loss={loss:.4f}, LR={lr:.2e}")
                except Exception as e:
                    print_warning(f"Failed to parse line {line_num}: {e}")
                    continue

            # Alternative pattern (more flexible)
            if not json_match:
                alt_match = re.search(r"'loss':\s*['\"]?([0-9.]+)['\"]?.*?'learning_rate':\s*['\"]?([0-9.e+-]+)['\"]?",
                                      line)
                if alt_match:
                    try:
                        loss = float(alt_match.group(1))
                        lr = float(alt_match.group(2))
                        step_counter += 1
                        step = step_counter * 50
                        steps.append(step)
                        losses.append(loss)
                        learning_rates.append(lr)
                    except:
                        continue

    print(f"\n{Colors.CYAN}Total data points parsed: {len(steps)}{Colors.END}")
    return steps, losses, learning_rates, grad_norms


# ==============================================================================
# Curve Generation
# ==============================================================================
def generate_curves(log_file, output_dir, timestamp):
    """Generate loss and learning rate curves."""
    steps, losses, learning_rates, grad_norms = parse_log_file(log_file)

    if not steps:
        print_error("No data found in log file!")
        print("  Possible reasons:")
        print("  1. logging_steps not set in train_full.yaml")
        print("  2. Log file doesn't contain loss information")
        print("  3. Wrong log file selected")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # ========== Loss Curve ==========
    print(f"\n{Colors.CYAN}Generating loss curve...{Colors.END}")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    loss_file = os.path.join(output_dir, f"{timestamp}_loss_curve.png")
    plt.savefig(loss_file, dpi=150, bbox_inches='tight')
    plt.close()
    print_success(f"Loss curve saved: {loss_file}")

    # ========== Learning Rate Curve ==========
    if learning_rates:
        print(f"\n{Colors.CYAN}Generating learning rate curve...{Colors.END}")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(steps, learning_rates, 'g-', linewidth=2, label='Learning Rate')
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        lr_file = os.path.join(output_dir, f"{timestamp}_lr_curve.png")
        plt.savefig(lr_file, dpi=150, bbox_inches='tight')
        plt.close()
        print_success(f"LR curve saved: {lr_file}")

    # ========== Combined Curve ==========
    if learning_rates:
        print(f"\n{Colors.CYAN}Generating combined curve...{Colors.END}")
        fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Loss subplot
        ax3a.plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
        ax3a.set_ylabel('Loss', fontsize=12)
        ax3a.set_title('Training Metrics', fontsize=14)
        ax3a.grid(True, alpha=0.3)
        ax3a.legend(loc='upper right')

        # LR subplot
        ax3b.plot(steps, learning_rates, 'g-', linewidth=2, label='Learning Rate')
        ax3b.set_xlabel('Step', fontsize=12)
        ax3b.set_ylabel('Learning Rate', fontsize=12)
        ax3b.grid(True, alpha=0.3)
        ax3b.legend(loc='upper right')

        combined_file = os.path.join(output_dir, f"{timestamp}_combined_curve.png")
        plt.savefig(combined_file, dpi=150, bbox_inches='tight')
        plt.close()
        print_success(f"Combined curve saved: {combined_file}")

    # ========== Print Summary ==========
    print(f"\n{Colors.GREEN}{'=' * 70}{Colors.END}")
    print(f"{Colors.GREEN}CURVE GENERATION SUMMARY{Colors.END}")
    print(f"{Colors.GREEN}{'=' * 70}{Colors.END}")
    print(f"  Timestamp:         {timestamp}")
    print(f"  Data points:       {len(steps)}")
    print(f"  Initial Loss:      {losses[0]:.4f}")
    print(f"  Final Loss:        {losses[-1]:.4f}")
    print(f"  Loss Reduction:    {losses[0] - losses[-1]:.4f} ({(1 - losses[-1] / losses[0]) * 100:.1f}%)")
    print(f"  Initial LR:        {learning_rates[0]:.2e}")
    print(f"  Final LR:          {learning_rates[-1]:.2e}")
    print(f"  Output Directory:  {output_dir}")
    print(f"{Colors.GREEN}{'=' * 70}{Colors.END}")
    print(f"\n{Colors.CYAN}Generated files:{Colors.END}")
    print(f"  - {loss_file}")
    if learning_rates:
        print(f"  - {lr_file}")
        print(f"  - {combined_file}")
    print()

    return True


# ==============================================================================
# List Available Logs
# ==============================================================================
def list_available_logs():
    """List all available training log files."""
    if not os.path.exists(LOG_DIR):
        print_error(f"Log directory not found: {LOG_DIR}")
        return []

    log_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('_training.log')])

    if not log_files:
        print_warning(f"No training log files found in {LOG_DIR}")
        return []

    print(f"\n{Colors.CYAN}{'=' * 70}{Colors.END}")
    print(f"{Colors.CYAN}AVAILABLE TRAINING LOGS{Colors.END}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.END}")
    print(f"  {'Index':<8} {'Timestamp':<20} {'File Size':<15}")
    print(f"  {'-' * 68}")

    for i, log_file in enumerate(log_files, 1):
        file_path = os.path.join(LOG_DIR, log_file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        timestamp = log_file.replace('_training.log', '')
        print(f"  {i:<8} {timestamp:<20} {file_size:.2f} MB")

    print(f"{Colors.CYAN}{'=' * 70}{Colors.END}\n")

    return log_files


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate training loss/learning rate curves from log file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Auto-select latest log file
  python generate_curves.py

  # Specify timestamp
  python generate_curves.py --timestamp 20260302_020509

  # List available logs first
  python generate_curves.py --list

  # Specify custom output directory
  python generate_curves.py --timestamp 20260302_020509 --output-dir /path/to/figures
        '''
    )

    parser.add_argument('--timestamp', type=str, default=None,
                        help='Timestamp of the training run (e.g., 20260302_020509)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory for figures')
    parser.add_argument('--list', action='store_true',
                        help='List all available training log files')

    args = parser.parse_args()

    print_header("TRAINING CURVE GENERATOR")

    # List available logs if requested
    if args.list:
        log_files = list_available_logs()
        if log_files:
            print_info("Select a log file with: python generate_curves.py --timestamp <timestamp>")
        sys.exit(0)

    # Find log file
    if args.timestamp:
        # Use specified timestamp
        timestamp = args.timestamp
        log_file = os.path.join(LOG_DIR, f"{timestamp}_training.log")

        if not os.path.exists(log_file):
            print_error(f"Log file not found: {log_file}")
            print_info("Use --list to see available log files")
            sys.exit(1)
    else:
        # Auto-select latest log file
        log_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('_training.log')])

        if not log_files:
            print_error(f"No training log files found in {LOG_DIR}")
            sys.exit(1)

        latest_log = log_files[-1]
        timestamp = latest_log.replace('_training.log', '')
        log_file = os.path.join(LOG_DIR, latest_log)

        print_info(f"Auto-selected latest log file: {latest_log}")

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(FIGURE_DIR, f"figures_{timestamp}")

    print(f"\n{Colors.CYAN}Configuration:{Colors.END}")
    print(f"  Log file:     {log_file}")
    print(f"  Timestamp:    {timestamp}")
    print(f"  Output dir:   {output_dir}")
    print()

    # Generate curves
    success = generate_curves(log_file, output_dir, timestamp)

    if success:
        print_header("CURVE GENERATION COMPLETED")
        sys.exit(0)
    else:
        print_header("CURVE GENERATION FAILED")
        sys.exit(1)


def print_header(text):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 70}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text.center(70)}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 70}{Colors.END}\n")


if __name__ == "__main__":
    main()