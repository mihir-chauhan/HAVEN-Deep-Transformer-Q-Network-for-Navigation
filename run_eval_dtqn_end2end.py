import argparse
import os
import json
from sim_core import SimulationConfig, run_evaluation, summarize_results, save_results_csv


def main():
    parser = argparse.ArgumentParser(description='Evaluate DTQN End-to-End Navigation (no low-level)')
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='eval')
    parser.add_argument('--checkpoint', type=str, default='dtqn_checkpoint.pt')
    args = parser.parse_args()

    cfg = SimulationConfig()
    results = run_evaluation(method='dtqn_end2end', num_runs=args.runs, cfg=cfg, dtqn_checkpoint=os.path.join(os.path.dirname(__file__), args.checkpoint), base_seed=args.seed)
    summary = summarize_results(results)

    os.makedirs(os.path.join(os.path.dirname(__file__), args.outdir), exist_ok=True)
    csv_path = os.path.join(os.path.dirname(__file__), args.outdir, 'dtqn_end2end_results.csv')
    save_results_csv(results, summary, csv_path)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()


