#!/usr/bin/env python3
import os
import sys

import matplotlib
matplotlib.use('Agg')  # headless backend for CI/servers
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import patheffects as pe
import numpy as np


FIG_OUT_DIR = os.path.join(os.path.dirname(__file__), 'figs')


def _ensure_out_dir() -> None:
    os.makedirs(FIG_OUT_DIR, exist_ok=True)


def _add_box(ax, xy, width, height, text, fc="#e8f1fb", ec="#2a6f97", lw=1.5, fontsize=10, roundness=0.12):
    x, y = xy
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle=f"round,pad=0.02,rounding_size={roundness}",
                         linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    tx = ax.text(x + width/2, y + height/2, text,
                 ha='center', va='center', fontsize=fontsize, color="#0b2239")
    tx.set_path_effects([pe.withStroke(linewidth=2, foreground='white', alpha=0.8)])
    return {"patch": box, "x": x, "y": y, "w": width, "h": height}


def _arrow(ax, xy_from, xy_to, text=None, color="#1f4e79", style='-|>', lw=1.6, mutation_scale=12, ls='solid', fontsize=9, text_offset=(0, 0)):
    arr = FancyArrowPatch(xy_from, xy_to, arrowstyle=style, mutation_scale=mutation_scale,
                          linewidth=lw, color=color, linestyle=ls, shrinkA=0.0, shrinkB=0.0)
    ax.add_patch(arr)
    if text:
        mid = ((xy_from[0] + xy_to[0]) / 2.0 + text_offset[0], (xy_from[1] + xy_to[1]) / 2.0 + text_offset[1])
        ax.text(mid[0], mid[1], text, fontsize=fontsize, color=color, ha='center', va='center')
    return arr


def generate_framework_figure(out_base: str) -> None:
    """Figure 1: Hierarchical framework overview (square-ish for double column)."""
    fig = plt.figure(figsize=(6.0, 6.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Layout parameters (normalized coordinates)
    box_w = 0.26
    box_h = 0.14

    # Rows y positions (top to bottom)
    y_top = 0.72
    y_mid = 0.46
    y_low = 0.20

    # Column x positions (left to right)
    x_left = 0.06
    x_mid = 0.37
    x_right = 0.68

    # Boxes
    inputs_text = "Perception + Odometry\n(2D map / PC → features)\nEnemies (FoVs)"
    b_inputs = _add_box(ax, (x_left, y_top), box_w, box_h, inputs_text)

    cand_text = "Candidate Generation\n+ Visibility Masking"
    b_cands = _add_box(ax, (x_mid, y_top), box_w, box_h, cand_text)

    dtqn_text = "DTQN (Transformer Q)\nlength-k memory, 16-D features"
    b_dtqn = _add_box(ax, (x_mid, y_mid), box_w, box_h, dtqn_text, fc="#eef9f0", ec="#2f6f3e")

    subgoal_text = "Selected Subgoal\n(ε-greedy train, greedy test)"
    b_sub = _add_box(ax, (x_right, y_mid + 0.01), box_w, box_h * 0.85, subgoal_text, fc="#fff3cd", ec="#ad8b00")

    low_text = "Low-Level Controller\n(force-based: goal, obs, enemy, ant., escape)"
    b_low = _add_box(ax, (x_mid, y_low), box_w, box_h, low_text, fc="#f2ecff", ec="#5b34ad")

    cmd_text = "Velocity Commands\n(v, ω)"
    b_cmd = _add_box(ax, (x_right, y_low + 0.01), box_w, box_h * 0.85, cmd_text, fc="#fde2e2", ec="#9d2b2b")

    reward_text = "Reward Function\n(progress, exposure, collision, time)"
    b_rew = _add_box(ax, (x_left, y_low + 0.01), box_w, box_h * 0.85, reward_text, fc="#e9f7fb", ec="#1e88a8")

    # Arrow helpers (center points)
    def center_of(node):
        return (node["x"] + node["w"]/2.0, node["y"] + node["h"]/2.0)

    def east_of(node):
        return (node["x"] + node["w"], node["y"] + node["h"]/2.0)

    def west_of(node):
        return (node["x"], node["y"] + node["h"]/2.0)

    def south_of(node):
        return (node["x"] + node["w"]/2.0, node["y"])

    def north_of(node):
        return (node["x"] + node["w"]/2.0, node["y"] + node["h"]) 

    # Main flow arrows
    _arrow(ax, east_of(b_inputs), west_of(b_cands))
    _arrow(ax, south_of(b_cands), north_of(b_dtqn))
    _arrow(ax, east_of(b_dtqn), west_of(b_sub), text="Q-values → argmax")
    _arrow(ax, south_of(b_sub), north_of(b_low))
    _arrow(ax, east_of(b_low), west_of(b_cmd))

    # Feedback/training arrows (dashed)
    _arrow(ax, west_of(b_low), east_of(b_rew), ls='dashed', color="#356c8c")
    _arrow(ax, north_of(b_rew), (x_left + box_w/2.0, y_mid), ls='dashed', color="#356c8c", text="TD targets (γ^k)")
    _arrow(ax, (x_left + box_w/2.0, y_mid), west_of(b_dtqn), ls='dashed', color="#356c8c")

    # Callouts
    ax.text(x_mid + box_w/2.0, y_mid + box_h + 0.025, "k-step memory", fontsize=9, color="#2f6f3e", ha='center')
    ax.text(x_mid + box_w/2.0, y_top - 0.02, "Mask by line-of-sight to enemies", fontsize=9, color="#0b2239", ha='center')

    # Title (kept small; caption will describe in LaTeX)
    ax.text(0.5, 0.97, "Hierarchical enemy-aware navigation framework", fontsize=11, ha='center', va='top', color="#0b2239")

    # Save
    png_path = f"{out_base}.png"
    pdf_path = f"{out_base}.pdf"
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def generate_pc_pipeline_figure(out_base: str) -> None:
    """Figure 8: 3D Unity–ROS pipeline collage (square-ish), using existing images."""
    # Resolve input images shipped in repo
    base_dir = os.path.dirname(__file__)
    img_paths = {
        'raw': os.path.join(base_dir, 'pc.png'),
        'proj': os.path.join(base_dir, 'pointcloud_processing.png'),
        'cands': os.path.join(base_dir, 'test_fixed_pc.png'),
    }
    for key, p in img_paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required image for pipeline panel '{key}': {p}")

    imgs = {k: plt.imread(v) for k, v in img_paths.items()}

    fig = plt.figure(figsize=(6.0, 6.0), dpi=300)
    # Three horizontal panels inside a square canvas
    ax_raw = fig.add_axes([0.05, 0.15, 0.27, 0.7])
    ax_proj = fig.add_axes([0.365, 0.15, 0.27, 0.7])
    ax_cand = fig.add_axes([0.68, 0.15, 0.27, 0.7])

    for ax, key, title in [
        (ax_raw, 'raw', '(a) Raw point cloud'),
        (ax_proj, 'proj', '(b) 2D projection + polygons'),
        (ax_cand, 'cands', '(c) Candidates overlay'),
    ]:
        ax.imshow(imgs[key])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#999')
            spine.set_linewidth(0.6)

    # Overlay axes for connecting arrows
    overlay = fig.add_axes([0, 0, 1, 1])
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis('off')
    # Connect panels with arrows
    y_mid = 0.90
    _arrow(overlay, (0.19, y_mid), (0.365, y_mid), text="Projection + polygonization", fontsize=9, color="#1f4e79")
    _arrow(overlay, (0.635, y_mid), (0.81, y_mid), text="Candidate\nconstruction", fontsize=9, color="#1f4e79")

    # Overall title (caption in LaTeX will carry the explanation)
    fig.text(0.5, 0.97, "3D→2D perception pipeline for hierarchical control", ha='center', va='top', fontsize=11, color="#0b2239")

    png_path = f"{out_base}.png"
    pdf_path = f"{out_base}.pdf"
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    _ensure_out_dir()
    framework_base = os.path.join(FIG_OUT_DIR, 'framework')
    pipeline_base = os.path.join(FIG_OUT_DIR, 'pc_pipeline')
    sys2d_base = os.path.join(FIG_OUT_DIR, 'system_2d')
    sys3d_base = os.path.join(FIG_OUT_DIR, 'system_3d')

    print(f"[fig_gen] Generating Figure 1 → {framework_base}.(png|pdf)")
    generate_framework_figure(framework_base)
    print(f"[fig_gen] Done.")

    print(f"[fig_gen] Generating Figure 8 → {pipeline_base}.(png|pdf)")
    try:
        generate_pc_pipeline_figure(pipeline_base)
        print(f"[fig_gen] Done.")
    except FileNotFoundError as e:
        print(f"[fig_gen] WARNING: {e}")
        print("[fig_gen] Skipped Figure 8 due to missing source images. Place pc.png, pointcloud_processing.png, test_fixed_pc.png under fov/ and re-run.")

    # System-level diagrams
    print(f"[fig_gen] Generating 2D system diagram → {sys2d_base}.(png|pdf)")
    generate_system_2d_figure(sys2d_base)
    print(f"[fig_gen] Done.")

    print(f"[fig_gen] Generating 3D system diagram → {sys3d_base}.(png|pdf)")
    generate_system_3d_figure(sys3d_base)
    print(f"[fig_gen] Done.")
    return 0


 


# ---------------- Additional Figures: Full System 2D and 3D ---------------- #

def generate_system_2d_figure(out_base: str) -> None:
    """System diagram for 2D hierarchical DTQN + low-level controller (square)."""
    fig = plt.figure(figsize=(6.0, 6.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Dimensions
    bw = 0.26
    bh = 0.13

    # Columns
    x0, x1, x2, x3 = 0.05, 0.36, 0.67, 0.67
    # Rows
    y_top, y_mid1, y_mid2, y_low = 0.75, 0.55, 0.35, 0.15

    # Environment and sensing
    env = _add_box(ax, (x0, y_top), bw, bh, "2D Environment\nObstacles (polygons)\nEnemies (FoVs)")
    odo = _add_box(ax, (x0, y_mid1), bw, bh, "Agent State\n(odometry)")

    # Features & candidates
    feat = _add_box(ax, (x1, y_top), bw, bh, "Feature Builder\n(16-D per candidate)")
    cand = _add_box(ax, (x1, y_mid1), bw, bh, "Candidate Gen + Masking\n(centroids, goal; LoS mask)")

    # DTQN and selection
    dtqn = _add_box(ax, (x1, y_mid2), bw, bh, "DTQN (k-step memory)\nQ(g|h) → argmax", fc="#eef9f0", ec="#2f6f3e")
    sgoal = _add_box(ax, (x2, y_mid2), bw, bh, "Selected Subgoal")

    # Low-level and dynamics
    ll = _add_box(ax, (x1, y_low), bw, bh, "Low-Level Controller\nforces: goal, obs, enemy, ant., escape", fc="#f2ecff", ec="#5b34ad")
    dyn = _add_box(ax, (x2, y_low), bw, bh, "Agent Dynamics\nstate update")

    # Reward / training
    rew = _add_box(ax, (x0, y_low), bw, bh, "Reward\nprogress, exposure, collision, time", fc="#e9f7fb", ec="#1e88a8")

    # Helpers
    def E(n): return (n["x"] + n["w"], n["y"] + n["h"]/2)
    def W(n): return (n["x"], n["y"] + n["h"]/2)
    def N(n): return (n["x"] + n["w"]/2, n["y"] + n["h"]) 
    def S(n): return (n["x"] + n["w"]/2, n["y"]) 

    # Flows
    _arrow(ax, E(env), W(feat))
    _arrow(ax, E(odo), W(feat))
    _arrow(ax, E(feat), W(cand), text="per decision")
    _arrow(ax, S(cand), N(dtqn))
    _arrow(ax, E(dtqn), W(sgoal))
    _arrow(ax, E(sgoal), (x3 + 0.02, y_mid2 + bh/2), text=None)
    _arrow(ax, N(ll), (x1 + bw/2, y_mid2), text="execute H steps", ls='dotted')
    _arrow(ax, E(ll), W(dyn))
    _arrow(ax, W(rew), (x0 - 0.02, y_low + bh/2))

    # Feedback
    _arrow(ax, E(dyn), (x0 + bw/2, y_mid1), ls='dashed', color="#356c8c", text="state → features")
    _arrow(ax, N(rew), (x1 - 0.04, y_mid2 + bh/2), ls='dashed', color="#356c8c", text="TD targets (γ^k)")
    _arrow(ax, (x1 - 0.04, y_mid2 + bh/2), W(dtqn), ls='dashed', color="#356c8c")

    fig.text(0.5, 0.97, "Full 2D system: hierarchical DTQN + low-level controller", ha='center', va='top', fontsize=11, color="#0b2239")
    fig.savefig(f"{out_base}.png", bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f"{out_base}.pdf", bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def generate_system_3d_figure(out_base: str) -> None:
    """System diagram for 3D Unity–ROS pipeline + hierarchical stack (square)."""
    fig = plt.figure(figsize=(6.0, 6.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    bw = 0.26
    bh = 0.13

    x0, x1, x2 = 0.05, 0.36, 0.67
    y_top, y_mid1, y_mid2, y_low = 0.78, 0.58, 0.38, 0.18

    # ROS inputs
    ros_in = _add_box(ax, (x0, y_top), bw, bh, "ROS Topics\nPointCloud, Odometry, Enemies")

    # Mapping
    proj = _add_box(ax, (x1, y_top), bw, bh, "Projection + Polygonization\n(ground-plane)")
    persist = _add_box(ax, (x1, y_mid1), bw, bh, "Persistence Filtering\n(temporal stability)")

    # Features & candidates
    feat = _add_box(ax, (x1, y_mid2), bw, bh, "Feature Builder\n(16-D per candidate)")
    cand = _add_box(ax, (x1, y_low), bw, bh, "Candidate Gen + Masking\n(centroids, goal; LoS mask)")

    # DTQN and low-level
    dtqn = _add_box(ax, (x2, y_mid2), bw, bh, "DTQN (k-step memory)\nQ(g|h) → argmax", fc="#eef9f0", ec="#2f6f3e")
    sgoal = _add_box(ax, (x2, y_low), bw, bh, "Selected Subgoal")
    ll = _add_box(ax, (x2, y_top), bw, bh, "Low-Level Controller\nforce-based", fc="#f2ecff", ec="#5b34ad")
    cmd = _add_box(ax, (x2, y_mid1), bw, bh, "ROS cmd_vel\n(v, ω)", fc="#fde2e2", ec="#9d2b2b")

    # Reward
    rew = _add_box(ax, (x0, y_low), bw, bh, "Reward\nprogress, exposure, collision, time", fc="#e9f7fb", ec="#1e88a8")

    # Helpers
    def E(n): return (n["x"] + n["w"], n["y"] + n["h"]/2)
    def W(n): return (n["x"], n["y"] + n["h"]/2)
    def N(n): return (n["x"] + n["w"]/2, n["y"] + n["h"]) 
    def S(n): return (n["x"] + n["w"]/2, n["y"]) 

    # Flows
    _arrow(ax, E(ros_in), W(proj))
    _arrow(ax, S(proj), N(persist))
    _arrow(ax, S(persist), N(feat))
    _arrow(ax, S(feat), N(cand))
    _arrow(ax, E(feat), W(dtqn), ls='dotted', text="k-step history")
    _arrow(ax, E(cand), (x2 - 0.02, y_low + bh/2))
    _arrow(ax, E(dtqn), (x2 + 0.02, y_mid2 + bh/2), text="argmax")
    _arrow(ax, (x2 + 0.02, y_mid2 + bh/2), W(sgoal))
    _arrow(ax, N(sgoal), (x2 + bw/2, y_mid1), text="re-evaluate ≤ H", ls='dotted')
    _arrow(ax, N(cmd), (x2 + bw/2, y_top), ls='dotted')
    _arrow(ax, E(ll), (x2 + bw + 0.02, y_top + bh/2))
    _arrow(ax, W(cmd), E(ll))

    # Feedback
    _arrow(ax, (x0 + bw/2, y_mid1), W(feat), ls='dashed', color="#356c8c", text="map → features")
    _arrow(ax, N(rew), (x2 - 0.04, y_mid2 + bh/2), ls='dashed', color="#356c8c", text="TD targets (γ^k)")
    _arrow(ax, (x2 - 0.04, y_mid2 + bh/2), W(dtqn), ls='dashed', color="#356c8c")

    fig.text(0.5, 0.97, "Full 3D system: Unity/ROS perception → hierarchical control", ha='center', va='top', fontsize=11, color="#0b2239")
    fig.savefig(f"{out_base}.png", bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f"{out_base}.pdf", bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())


