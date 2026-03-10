#!/usr/bin/env python3
import argparse
import csv
import os
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')  # headless save
import matplotlib.pyplot as plt
import numpy as np


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    window = min(window, values.size)
    kernel = np.ones(window, dtype=float) / float(window)
    # Use convolution with reflect padding for better edges
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode='reflect')
    smoothed = np.convolve(padded, kernel, mode='valid')
    # Ensure same length as input
    if smoothed.size > values.size:
        start = (smoothed.size - values.size) // 2
        smoothed = smoothed[start:start + values.size]
    return smoothed


def read_metrics_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
    """Read training metrics CSV.

    Supports two formats:
    1. Headered CSV with columns: episode, loss, q_pred, target, R, termination (others ignored)
    2. Headerless numeric CSV rows with fixed column indices:
       idx0=episode, idx1=global_step, idx3=loss, idx4=q_pred, idx5=target, idx6=R, idx8=termination
    """
    steps: List[int] = []
    loss_list: List[float] = []
    q_pred_list: List[float] = []
    target_list: List[float] = []
    R_list: List[float] = []
    episode_indices: List[int] = []  # global step values where a new episode begins
    goal_steps: List[int] = []       # global step values where termination == GOAL_REACHED

    # First try headered format via DictReader
    parsed_any = False
    current_episode = None

    with open(csv_path, 'r', newline='') as f:
        # Peek first line to decide
        first_line = f.readline()
        f.seek(0)
        looks_like_header = any(h in first_line.lower() for h in ['episode', 'loss', 'q_pred', 'target', 'termination'])
        if looks_like_header:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ep = int(float(row['episode']))
                    loss_val = float(row['loss'])
                    q_pred_val = float(row['q_pred'])
                    target_val = float(row['target'])
                    R_val = float(row['R'])
                    term = (row.get('termination') or '').strip()
                except (ValueError, KeyError, TypeError):
                    continue
                # Use running index as global step if explicit column absent
                gstep = int(row.get('global_step') or row.get('step') or (steps[-1] + 1 if steps else 1))
                steps.append(gstep)
                loss_list.append(loss_val)
                q_pred_list.append(q_pred_val)
                target_list.append(target_val)
                R_list.append(R_val)
                if current_episode is None or ep != current_episode:
                    episode_indices.append(gstep)
                    current_episode = ep
                if term == 'GOAL_REACHED':
                    goal_steps.append(gstep)
                parsed_any = True
        else:
            # Fallback: raw numeric rows
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 9:
                    continue
                try:
                    ep = int(float(row[0]))
                    gstep = int(float(row[1]))  # use provided global step
                    loss_val = float(row[3])
                    q_pred_val = float(row[4])
                    target_val = float(row[5])
                    R_val = float(row[6])
                    term = str(row[8]).strip()
                except (ValueError, IndexError):
                    continue
                steps.append(gstep)
                loss_list.append(loss_val)
                q_pred_list.append(q_pred_val)
                target_list.append(target_val)
                R_list.append(R_val)
                if current_episode is None or ep != current_episode:
                    episode_indices.append(gstep)
                    current_episode = ep
                if term == 'GOAL_REACHED':
                    goal_steps.append(gstep)
                parsed_any = True

    if not parsed_any:
        # Return empty arrays so caller can raise
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            [],
            [],
        )

    # Ensure arrays sorted by step (in case input not already)
    order = np.argsort(np.asarray(steps))
    steps_arr = np.asarray(steps, dtype=int)[order]
    loss_arr = np.asarray(loss_list, dtype=float)[order]
    q_pred_arr = np.asarray(q_pred_list, dtype=float)[order]
    target_arr = np.asarray(target_list, dtype=float)[order]
    R_arr = np.asarray(R_list, dtype=float)[order]

    # Recompute episode start indices mapping to sorted steps if needed
    # (they already store step values, so just leave as-is)

    return (
        steps_arr,
        loss_arr,
        q_pred_arr,
        target_arr,
        R_arr,
        episode_indices,
        goal_steps,
    )


def plot_training_metrics(
    csv_path: str,
    out_dir: str,
    smooth_window: int = 25,
    dpi: int = 180,
    max_step: Optional[int] = None,
) -> Tuple[str, str]:
    (
        steps,
        loss_arr,
        q_pred_arr,
        target_arr,
        R_arr,
        episode_starts,
        goal_steps,
    ) = read_metrics_csv(csv_path)

    if steps.size == 0:
        raise RuntimeError(f"No rows parsed from metrics CSV: {csv_path}")

    # Optional cutoff by global step
    if max_step is not None:
        mask = steps <= int(max_step)
        steps = steps[mask]
        loss_arr = loss_arr[mask]
        q_pred_arr = q_pred_arr[mask]
        target_arr = target_arr[mask]
        R_arr = R_arr[mask]
        episode_starts = [s for s in episode_starts if s <= int(max_step)]
        goal_steps = [s for s in goal_steps if s <= int(max_step)]

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, 'training_metrics_plot')

    # Smoothing
    loss_s = _smooth(loss_arr, smooth_window)
    q_pred_s = _smooth(q_pred_arr, smooth_window)
    target_s = _smooth(target_arr, smooth_window)
    R_s = _smooth(R_arr, smooth_window)

    # Build figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=dpi)

    ax = axes[0]
    ax.plot(steps, loss_arr, color='#c9d6ea', linewidth=0.8, label='loss (raw)')
    ax.plot(steps, loss_s, color='#2a6f97', linewidth=1.6, label=f'loss (smooth={smooth_window})')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(steps, q_pred_arr, color='#e4b1ab', linewidth=0.8, label='Q_pred (raw)')
    ax.plot(steps, target_arr, color='#b1e4b7', linewidth=0.8, label='Target (raw)')
    ax.plot(steps, q_pred_s, color='#9d2b2b', linewidth=1.6, label='Q_pred (smooth)')
    ax.plot(steps, target_s, color='#2f6f3e', linewidth=1.6, label='Target (smooth)')
    ax.set_ylabel('Q values')
    ax.legend(loc='upper right', ncol=2)  # fixed parameter name
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    ax.plot(steps, R_arr, color='#d1c4e9', linewidth=0.8, label='R (raw)')
    ax.plot(steps, R_s, color='#5b34ad', linewidth=1.6, label='R (smooth)')
    ax.set_xlabel('Global step')
    ax.set_ylabel('R / TD target')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    # Episode boundaries and goal markers across all subplots
    for ax in axes:
        for s in episode_starts:
            ax.axvline(x=s, color='#eeeeee', linewidth=0.6)
        if goal_steps:
            ax.scatter(goal_steps, _interp_y(ax, steps, goal_steps),
                       s=16, color='#ffa600', edgecolor='k', linewidths=0.3,
                       zorder=5, label=None)

    fig.suptitle('DTQN Training Metrics', fontsize=14)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])

    png_path = f"{base}.png"
    pdf_path = f"{base}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _interp_y(ax: plt.Axes, x_all: np.ndarray, x_points: List[int]) -> np.ndarray:
    # For scatter y positions on a given axis: place markers at current y-limits midpoint
    y_min, y_max = ax.get_ylim()
    y_mid = (y_min + y_max) / 2.0
    return np.full(len(x_points), y_mid)


def main() -> int:
    parser = argparse.ArgumentParser(description='Plot training metrics from CSV and save PNG/PDF.')
    parser.add_argument('--csv', default=os.path.join(os.path.dirname(__file__), 'train', 'training_metrics.csv'),
                        help='Path to training_metrics.csv')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'train'),
                        help='Output directory for plots')
    parser.add_argument('--smooth', type=int, default=25,
                        help='SMA window size for smoothing (>=1)')
    parser.add_argument('--dpi', type=int, default=180, help='Figure DPI')
    parser.add_argument('--max_step', type=int, default=None,
                        help='Only plot data up to this global step (inclusive)')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Metrics CSV not found: {args.csv}")

    png_path, pdf_path = plot_training_metrics(
        args.csv,
        args.out,
        smooth_window=args.smooth,
        dpi=args.dpi,
        max_step=args.max_step,
    )
    print(f"Saved plots to:\n  {png_path}\n  {pdf_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


