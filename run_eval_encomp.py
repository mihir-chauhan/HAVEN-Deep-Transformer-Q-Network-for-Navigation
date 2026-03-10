#!/usr/bin/env python3
"""
Evaluate the EnCoMP CQL baseline.

Usage
-----
    python run_eval_encomp.py --runs 100 --checkpoint encomp/code/checkpoints/encomp_cql_best.pt
"""

import argparse
import os
import json
from sim_core import SimulationConfig, run_evaluation, summarize_results, save_results_csv


def main():
    parser = argparse.ArgumentParser(description='Evaluate EnCoMP CQL baseline')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='encomp/code/checkpoints/encomp_cql_best.pt')
    parser.add_argument('--outdir', type=str, default='eval')
    parser.add_argument('--video-dir', type=str, default=None,
                        help='Directory to save episode videos (omit to skip)')
    args = parser.parse_args()

    cfg = SimulationConfig()
    results = run_evaluation(
        method='encomp_cql',
        num_runs=args.runs,
        cfg=cfg,
        dtqn_checkpoint=args.checkpoint,
        base_seed=args.seed,
        video_dir=args.video_dir,
    )
    summary = summarize_results(results)

    os.makedirs(os.path.join(os.path.dirname(__file__), args.outdir), exist_ok=True)
    csv_path = os.path.join(os.path.dirname(__file__), args.outdir, 'encomp_cql_results.csv')
    save_results_csv(results, summary, csv_path)

    print("\n" + "=" * 60)
    print("EnCoMP CQL Evaluation Results")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nCSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
