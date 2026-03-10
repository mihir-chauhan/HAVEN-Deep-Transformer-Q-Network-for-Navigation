#!/usr/bin/env python3
"""
End-to-end pipeline: collect data → train CQL → evaluate.

Usage
-----
    python -m encomp.code.run_pipeline [--collect-episodes 200] [--train-epochs 500] [--eval-runs 100]
"""

import argparse
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOV_DIR = os.path.join(SCRIPT_DIR, '..', '..')


def run(cmd, cwd=FOV_DIR):
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\nCommand failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description='EnCoMP full pipeline')
    parser.add_argument('--collect-episodes', type=int, default=200)
    parser.add_argument('--train-epochs', type=int, default=500)
    parser.add_argument('--eval-runs', type=int, default=100)
    parser.add_argument('--skip-collect', action='store_true', help='Skip data collection (use existing dataset)')
    parser.add_argument('--skip-train', action='store_true', help='Skip training (use existing checkpoint)')
    args = parser.parse_args()

    dataset_path = 'encomp/code/dataset.npz'
    ckpt_dir = 'encomp/code/checkpoints'
    best_ckpt = os.path.join(ckpt_dir, 'encomp_cql_best.pt')

    # Step 1: Collect offline data
    if not args.skip_collect:
        run([
            sys.executable, '-m', 'encomp.code.collect_data',
            '--episodes', str(args.collect_episodes),
            '--out', dataset_path,
        ])

    # Step 2: Train CQL
    if not args.skip_train:
        run([
            sys.executable, '-m', 'encomp.code.train',
            '--dataset', dataset_path,
            '--epochs', str(args.train_epochs),
            '--out-dir', ckpt_dir,
        ])

    # Step 3: Evaluate
    run([
        sys.executable, 'run_eval_encomp.py',
        '--runs', str(args.eval_runs),
        '--checkpoint', best_ckpt,
    ])


if __name__ == '__main__':
    main()
