#!/usr/bin/env python3
"""
Train the EnCoMP CQL agent from an offline dataset.

Hyperparameters follow the paper:
  lr=1e-4, batch_size=256, gamma=0.95, alpha(CQL)=0.2, 500 epochs

Usage
-----
    python -m encomp.code.train --dataset encomp/code/dataset.npz --epochs 500
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from encomp.code.model import EnCompCQLAgent


class OfflineDataset(Dataset):
    def __init__(self, path: str):
        data = np.load(path)
        self.maps = data['maps']
        self.pos = data['pos']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.next_maps = data['next_maps']
        self.next_pos = data['next_pos']
        self.dones = data['dones']
        print(f"Loaded dataset: {len(self)} transitions from {path}")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return {
            'maps': torch.from_numpy(self.maps[idx]),
            'pos': torch.from_numpy(self.pos[idx]),
            'action': torch.tensor(self.actions[idx], dtype=torch.long),
            'reward': torch.tensor(self.rewards[idx], dtype=torch.float32),
            'next_maps': torch.from_numpy(self.next_maps[idx]),
            'next_pos': torch.from_numpy(self.next_pos[idx]),
            'done': torch.tensor(self.dones[idx], dtype=torch.float32),
        }


def main():
    parser = argparse.ArgumentParser(description='Train EnCoMP CQL')
    parser.add_argument('--dataset', type=str, default='encomp/code/dataset.npz')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--out-dir', type=str, default='encomp/code/checkpoints')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = OfflineDataset(args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    agent = EnCompCQLAgent(lr=args.lr, gamma=args.gamma, alpha=args.alpha)

    print(f"\nTraining CQL: {args.epochs} epochs, {len(dataset)} transitions, batch={args.batch_size}")
    print(f"  gamma={args.gamma}  alpha(CQL)={args.alpha}  lr={args.lr}")
    print(f"  Device: {agent.device}\n")

    best_loss = float('inf')
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_loss, epoch_bellman, epoch_cql, epoch_q, n_batches = 0, 0, 0, 0, 0
        agent.q_net.train()

        for batch in loader:
            stats = agent.train_step(batch)
            epoch_loss += stats['loss']
            epoch_bellman += stats['bellman']
            epoch_cql += stats['cql_penalty']
            epoch_q += stats['q_mean']
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_bell = epoch_bellman / max(n_batches, 1)
        avg_cql = epoch_cql / max(n_batches, 1)
        avg_q = epoch_q / max(n_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}  bellman={avg_bell:.4f}  "
                  f"cql={avg_cql:.4f}  Q_mean={avg_q:.3f}  [{elapsed:.0f}s]")

        if avg_loss < best_loss:
            best_loss = avg_loss
            agent.save(os.path.join(args.out_dir, 'encomp_cql_best.pt'))

        if epoch % args.save_every == 0:
            agent.save(os.path.join(args.out_dir, f'encomp_cql_ep{epoch}.pt'))

    agent.save(os.path.join(args.out_dir, 'encomp_cql_final.pt'))
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
