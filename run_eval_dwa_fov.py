import argparse
import os
import json
from sim_core import SimulationConfig, run_evaluation, summarize_results, save_results_csv


def main():
    parser = argparse.ArgumentParser(description='Evaluate DWA (FoV-limited)')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='eval')
    args = parser.parse_args()

    cfg = SimulationConfig()
    results = run_evaluation(method='dwa_fov', num_runs=args.runs, cfg=cfg, base_seed=args.seed)
    summary = summarize_results(results)

    os.makedirs(os.path.join(os.path.dirname(__file__), args.outdir), exist_ok=True)
    csv_path = os.path.join(os.path.dirname(__file__), args.outdir, 'dwa_fov_results.csv')
    save_results_csv(results, summary, csv_path)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()


