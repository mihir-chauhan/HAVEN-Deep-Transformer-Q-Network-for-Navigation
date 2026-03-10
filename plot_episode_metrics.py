#!/usr/bin/env python3
"""Plot episode-level training metrics from episode_summary.csv.

Generates a multi-panel figure:
  1. Rolling success rate (25-ep and 50-ep windows)
  2. Episode reward (raw + smoothed)
  3. Decisions per episode (fewer = more efficient)
  4. Epsilon schedule
  5. Goal distance at episode end (lower = closer to goal)

Usage:
    python plot_episode_metrics.py [--csv PATH] [--out DIR] [--window N]
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _rolling_mean(arr, window):
    if window <= 1 or len(arr) == 0:
        return arr.copy()
    kernel = np.ones(window) / window
    # Pad beginning with NaN so partial windows show as NaN
    padded = np.concatenate([np.full(window - 1, np.nan), arr])
    result = np.convolve(padded, kernel, mode='valid')
    return result[:len(arr)]


def read_episode_csv(path):
    episodes, successes, decisions, rewards = [], [], [], []
    goal_dists, epsilons, buf_sizes, update_counts, wall_times = [], [], [], [], []

    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                episodes.append(int(row['episode']))
                successes.append(int(row['success']))
                decisions.append(int(row['decisions']))
                rewards.append(float(row['total_reward']))
                goal_dists.append(float(row['goal_dist']))
                epsilons.append(float(row['eps']))
                buf_sizes.append(int(row['buffer_size']))
                update_counts.append(int(row['update_count']))
                wall_times.append(float(row['wall_time_s']))
            except (KeyError, ValueError):
                continue

    return {
        'episode': np.array(episodes),
        'success': np.array(successes, dtype=float),
        'decisions': np.array(decisions),
        'reward': np.array(rewards),
        'goal_dist': np.array(goal_dists),
        'eps': np.array(epsilons),
        'buffer_size': np.array(buf_sizes),
        'update_count': np.array(update_counts),
        'wall_time': np.array(wall_times),
    }


def plot_episode_metrics(csv_path, out_dir, window=25, dpi=180):
    data = read_episode_csv(csv_path)
    if len(data['episode']) == 0:
        raise RuntimeError(f"No episodes parsed from {csv_path}")

    ep = data['episode']
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True, dpi=dpi)

    # --- 1. Rolling success rate ---
    ax = axes[0]
    sr_25 = _rolling_mean(data['success'], 25) * 100
    sr_50 = _rolling_mean(data['success'], 50) * 100
    ax.plot(ep, sr_25, color='#2a6f97', linewidth=1.5, label='25-ep rolling')
    ax.plot(ep, sr_50, color='#e63946', linewidth=1.5, label='50-ep rolling')
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% target')
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(-5, 105)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)
    ax.set_title('Episode Success Rate', fontsize=11)

    # --- 2. Episode reward ---
    ax = axes[1]
    ax.plot(ep, data['reward'], color='#d1c4e9', linewidth=0.6, alpha=0.7, label='raw')
    r_smooth = _rolling_mean(data['reward'], window)
    ax.plot(ep, r_smooth, color='#5b34ad', linewidth=1.5, label=f'{window}-ep smooth')
    # Color success/fail
    ok_mask = data['success'] == 1
    fail_mask = ~ok_mask
    ax.scatter(ep[ok_mask], data['reward'][ok_mask], s=6, c='green', alpha=0.3, zorder=2)
    ax.scatter(ep[fail_mask], data['reward'][fail_mask], s=6, c='red', alpha=0.3, zorder=2)
    ax.set_ylabel('Episode Reward')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_title('Total Reward per Episode (green=success, red=fail)', fontsize=11)

    # --- 3. Decisions per episode ---
    ax = axes[2]
    ax.plot(ep, data['decisions'], color='#bbbbbb', linewidth=0.6, alpha=0.7)
    dec_smooth = _rolling_mean(data['decisions'].astype(float), window)
    ax.plot(ep, dec_smooth, color='#ff6700', linewidth=1.5, label=f'{window}-ep smooth')
    ax.set_ylabel('Decisions')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_title('Decisions per Episode (lower = more efficient)', fontsize=11)

    # --- 4. Epsilon ---
    ax = axes[3]
    ax.plot(ep, data['eps'], color='#2a6f97', linewidth=1.5)
    ax.set_ylabel('Epsilon')
    ax.grid(True, alpha=0.2)
    ax.set_title('Exploration Rate (epsilon)', fontsize=11)

    # --- 5. Goal distance at episode end ---
    ax = axes[4]
    ax.plot(ep, data['goal_dist'], color='#cccccc', linewidth=0.6, alpha=0.7)
    gd_smooth = _rolling_mean(data['goal_dist'], window)
    ax.plot(ep, gd_smooth, color='#e63946', linewidth=1.5, label=f'{window}-ep smooth')
    ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='goal threshold')
    ax.set_ylabel('Goal Distance')
    ax.set_xlabel('Episode')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_title('Final Distance to Goal (< 2.0 = success)', fontsize=11)

    fig.suptitle('DTQN Training — Episode Metrics', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])

    png_path = os.path.join(out_dir, 'episode_metrics_plot.png')
    pdf_path = os.path.join(out_dir, 'episode_metrics_plot.pdf')
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description='Plot episode-level training metrics.')
    parser.add_argument('--csv', default=os.path.join(os.path.dirname(__file__), 'train', 'episode_summary.csv'),
                        help='Path to episode_summary.csv')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'train'),
                        help='Output directory for plots')
    parser.add_argument('--window', type=int, default=25, help='Rolling window size')
    parser.add_argument('--dpi', type=int, default=180, help='Figure DPI')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Episode CSV not found: {args.csv}")

    png, pdf = plot_episode_metrics(args.csv, args.out, window=args.window, dpi=args.dpi)
    print(f"Saved plots to:\n  {png}\n  {pdf}")


if __name__ == '__main__':
    main()
