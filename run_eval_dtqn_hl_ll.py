import argparse
import os
import json
from sim_core import SimulationConfig, run_evaluation, summarize_results, save_results_csv


def main():
    parser = argparse.ArgumentParser(description='Evaluate DTQN high-level + low-level controller')
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='eval')
    parser.add_argument('--checkpoint', type=str, default='dtqn_checkpoint.pt')
    parser.add_argument('--video', type=int, nargs='?', const=5, default=0,
                        help='Number of episodes to record as video (default: 5 if flag given)')
    parser.add_argument('--video-episodes', type=int, nargs='*', default=None,
                        help='Specific episode indices to record (overrides --video count)')
    args = parser.parse_args()

    cfg = SimulationConfig()
    base_dir = os.path.dirname(__file__) or '.'
    video_dir = None
    video_episodes = None
    if args.video > 0 or args.video_episodes:
        video_dir = os.path.join(base_dir, args.outdir, 'videos')
        if args.video_episodes:
            video_episodes = args.video_episodes
        else:
            video_episodes = list(range(min(args.video, args.runs)))

    results = run_evaluation(
        method='dtqn_hl_ll', num_runs=args.runs, cfg=cfg,
        dtqn_checkpoint=os.path.join(base_dir, args.checkpoint),
        base_seed=args.seed, video_dir=video_dir, video_episodes=video_episodes,
    )
    summary = summarize_results(results)

    os.makedirs(os.path.join(base_dir, args.outdir), exist_ok=True)
    csv_path = os.path.join(base_dir, args.outdir, 'dtqn_hl_ll_results.csv')
    save_results_csv(results, summary, csv_path)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()


