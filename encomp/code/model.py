"""
Conservative Q-Learning (CQL) agent for the EnCoMP baseline.

Architecture (follows the paper's description):
  - CNN encoder processes the 3-channel (cover, threat, goal) grid maps
  - MLP merges the CNN features with the 4-D position vector
  - Outputs Q-values for each of the 9 discrete actions

CQL loss adds the conservative regularization from Kumar et al. (NeurIPS 2020):
  L_CQL = alpha * (E_s[logsumexp Q(s,·)] - E_{(s,a)~D}[Q(s,a)])
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .maps import NUM_ACTIONS, GRID_SIZE


class EnCompCQLNetwork(nn.Module):
    def __init__(
        self,
        map_channels: int = 3,
        map_size: int = GRID_SIZE,
        pos_dim: int = 4,
        num_actions: int = NUM_ACTIONS,
        cnn_channels: tuple = (32, 64, 64),
    ):
        super().__init__()
        layers = []
        in_ch = map_channels
        for out_ch in cnn_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, map_channels, map_size, map_size)
            cnn_out_flat = self.cnn(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_flat + pos_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, maps: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        maps : (B, C, H, W)
        pos  : (B, pos_dim)

        Returns
        -------
        q_values : (B, num_actions)
        """
        x = self.cnn(maps)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, pos], dim=1)
        return self.fc(x)


class EnCompCQLAgent:
    """
    Full CQL agent: wraps Q-network, target network, optimiser, and the
    conservative training loop.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        gamma: float = 0.95,
        alpha: float = 0.2,
        tau: float = 0.005,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.alpha = alpha  # CQL regularisation weight
        self.tau = tau

        self.q_net = EnCompCQLNetwork().to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        if checkpoint_path:
            self.load(checkpoint_path)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(
        self,
        maps: np.ndarray,
        pos: np.ndarray,
        epsilon: float = 0.0,
    ) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(NUM_ACTIONS))
        maps_t = torch.from_numpy(maps).float().unsqueeze(0).to(self.device)
        pos_t = torch.from_numpy(pos).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(maps_t, pos_t)
        return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # CQL training step
    # ------------------------------------------------------------------
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        One gradient step of CQL.

        batch keys: maps, pos, action, reward, next_maps, next_pos, done
        """
        maps = batch['maps'].to(self.device)
        pos = batch['pos'].to(self.device)
        action = batch['action'].to(self.device).long()
        reward = batch['reward'].to(self.device)
        next_maps = batch['next_maps'].to(self.device)
        next_pos = batch['next_pos'].to(self.device)
        done = batch['done'].to(self.device)

        # Current Q
        q_all = self.q_net(maps, pos)
        q_sa = q_all.gather(1, action.unsqueeze(1)).squeeze(1)

        # Target Q (Double-DQN style: use online net to select, target net to evaluate)
        with torch.no_grad():
            next_q_online = self.q_net(next_maps, next_pos)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_maps, next_pos)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)
            target = reward + self.gamma * (1.0 - done) * next_q

        # Standard Bellman loss
        bellman_loss = F.mse_loss(q_sa, target)

        # CQL regularisation: logsumexp(Q(s,·)) - Q(s,a_data)
        logsumexp_q = torch.logsumexp(q_all, dim=1).mean()
        data_q = q_sa.mean()
        cql_penalty = logsumexp_q - data_q

        loss = bellman_loss + self.alpha * cql_penalty

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()

        # Soft target update
        self._soft_update()

        return {
            'loss': loss.item(),
            'bellman': bellman_loss.item(),
            'cql_penalty': cql_penalty.item(),
            'q_mean': q_sa.mean().item(),
        }

    def _soft_update(self):
        for tp, op in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        import os
        if not os.path.isfile(path):
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and 'q_net' in ckpt:
            self.q_net.load_state_dict(ckpt['q_net'])
            self.target_net.load_state_dict(ckpt['target_net'])
            if 'optimizer' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            self.q_net.load_state_dict(ckpt)
            self.target_net = copy.deepcopy(self.q_net)
        self.q_net.eval()
        self.target_net.eval()
