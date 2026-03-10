#!/usr/bin/env python3
"""Single-process, multi-episode DTQN training.

Usage:
    python train_dtqn.py <num_episodes> [--eval-every N] [--checkpoint PATH]

The policy, replay buffer, target network, and normalizer persist across episodes.
"""
import os
import sys
import time
import csv

sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault('USE_DTQN', '1')
os.environ.setdefault('TRAIN_HIGH_LEVEL', '1')


from train_env import (
    DTQNPolicy,
    run_training_episode,
    _init_train_metrics,
    DTQN_CHECKPOINT_PATH,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train DTQN for adversarial FOV navigation")
    parser.add_argument('num_episodes', type=int, help='Number of training episodes')
    parser.add_argument('--eval-every', type=int, default=25, help='Run eval summary every N episodes')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (default: auto)')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N episodes')
    parser.add_argument('--video-every', type=int, default=25, help='Save episode video every N episodes (0 to disable)')
    args = parser.parse_args()

    K = args.num_episodes

    policy = DTQNPolicy(train_mode=True)
    if args.checkpoint:
        policy.checkpoint_path = args.checkpoint

    _init_train_metrics()

    summary_path = os.path.join(os.path.dirname(__file__), 'train', 'episode_summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    write_header = not os.path.isfile(summary_path)
    summary_file = open(summary_path, 'a', newline='')
    summary_writer = csv.writer(summary_file)
    if write_header:
        summary_writer.writerow([
            'episode', 'success', 'decisions', 'total_reward', 'termination',
            'goal_dist', 'eps', 'buffer_size', 'update_count', 'wall_time_s',
        ])

    successes = []
    rewards = []
    t_start_total = time.time()

    video_out_dir = os.path.join(os.path.dirname(__file__), 'train', 'videos')

    for ep in range(1, K + 1):
        t0 = time.time()
        save_video = (args.video_every > 0) and (ep % args.video_every == 0)
        result = run_training_episode(policy, episode_idx=ep, save_video=save_video, video_out_dir=video_out_dir)
        wall = time.time() - t0

        successes.append(result['success'])
        rewards.append(result['total_reward'])

        tag = 'OK' if result['success'] else 'FAIL'
        vid_note = " [video saved]" if save_video else ""
        print(
            f"[ep {ep:4d}/{K}] {tag:4s} | "
            f"dec={result['decisions']:3d} R={result['total_reward']:+7.2f} "
            f"gdist={result['final_goal_dist']:5.1f} | "
            f"eps={policy.eps:.4f} buf={len(policy.replay_buffer):6d} "
            f"upd={policy.update_count:6d} | {wall:.1f}s{vid_note}"
        )

        summary_writer.writerow([
            ep, int(result['success']), result['decisions'],
            f"{result['total_reward']:.4f}", result['termination'],
            f"{result['final_goal_dist']:.3f}", f"{policy.eps:.5f}",
            len(policy.replay_buffer), policy.update_count, f"{wall:.2f}",
        ])
        summary_file.flush()

        if ep % args.save_every == 0:
            policy.save_checkpoint()

        if ep % args.eval_every == 0:
            recent = successes[-args.eval_every:]
            recent_r = rewards[-args.eval_every:]
            sr = sum(recent) / len(recent)
            avg_r = sum(recent_r) / len(recent_r)
            elapsed = time.time() - t_start_total
            print(f"  >>> Last {args.eval_every} eps: success={sr:.2%}  avg_R={avg_r:+.2f}  total_time={elapsed:.0f}s")

    policy.save_checkpoint()
    summary_file.close()

    total_sr = sum(successes) / len(successes) if successes else 0.0
    print(f"\n[train] Done. {K} episodes, overall success rate: {total_sr:.2%}")
    print(f"  Checkpoint: {policy.checkpoint_path}")
    print(f"  Summary:    {summary_path}")
    if args.video_every > 0:
        print(f"  Videos:     {video_out_dir} (every {args.video_every} eps)")

    # Auto-generate training plots
    train_dir = os.path.join(os.path.dirname(__file__), 'train')
    try:
        from plot_episode_metrics import plot_episode_metrics
        png, pdf = plot_episode_metrics(summary_path, train_dir)
        print(f"  Episode plots:  {png}")
    except Exception as e:
        print(f"  [warn] Could not generate episode plots: {e}")
    try:
        from plot_training_metrics import plot_training_metrics
        metrics_csv = os.path.join(train_dir, 'training_metrics.csv')
        if os.path.exists(metrics_csv):
            png, pdf = plot_training_metrics(metrics_csv, train_dir)
            print(f"  Training plots: {png}")
    except Exception as e:
        print(f"  [warn] Could not generate training plots: {e}")


if __name__ == '__main__':
    main()
