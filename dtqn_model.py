import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np


class DTQN(nn.Module):
    def __init__(self, input_dim, output_dim, k=3, n_heads=4, n_layers=3, d_model=64, dropout=0.1):
        super(DTQN, self).__init__()
        self.k = k
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.position_encoding = nn.Parameter(torch.randn(k, d_model))

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])

        self.output = nn.Linear(d_model, output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.position_encoding, mean=0.0, std=0.02)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batch_size, seq_len, _ = x.size()
        x = self.embedding(x)
        x = x + self.position_encoding[:seq_len].unsqueeze(0)

        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        for layer in self.transformer_layers:
            x = layer(x, mask)

        x = self.output(x)
        return x

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2, _ = self.self_attn(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1), attn_mask=mask)
        x2 = x2.transpose(0, 1)
        x = self.norm1(x + self.dropout(x2))
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay buffer."""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0
        self.size = 0

    def push(self, *data, td_error=None):
        """Store a transition. Data format is determined by the caller."""
        max_prio = self.priorities[:self.size].max() if self.size > 0 else 1.0
        prio = max_prio if td_error is None else (abs(td_error) + 1e-6)

        if self.size < self.capacity:
            self.buffer.append(data)
            self.size += 1
        else:
            self.buffer[self.pos] = data
        self.priorities[self.pos] = prio ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if self.size == 0:
            return [], [], np.array([])
        prios = self.priorities[:self.size]
        probs = prios / prios.sum()
        n = min(batch_size, self.size)
        indices = np.random.choice(self.size, size=n, p=probs, replace=False)
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        samples = [self.buffer[i] for i in indices]
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = (abs(err) + 1e-6) ** self.alpha

    def __len__(self):
        return self.size


class ObservationHistory:
    """Rolling window of past observations for real temporal context in the DTQN."""
    def __init__(self, k, dim=16):
        self.k = k
        self.dim = dim
        self.buffer = deque(maxlen=k - 1)

    def get_sequence(self, current_obs):
        seq = list(self.buffer)
        while len(seq) < self.k - 1:
            seq.insert(0, np.zeros(self.dim, dtype=np.float32))
        seq.append(current_obs)
        return np.array(seq[-self.k:], dtype=np.float32)

    def push(self, obs):
        self.buffer.append(obs.copy())

    def reset(self):
        self.buffer.clear()


class RunningNormalizer:
    """Welford online running mean/std for feature normalization."""
    def __init__(self, dim, clip=5.0):
        self.dim = dim
        self.clip = clip
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def normalize(self, x):
        if self.count < 2:
            return np.asarray(x, dtype=np.float32)
        std = np.sqrt(self.M2 / (self.count - 1) + 1e-8)
        normed = (np.asarray(x, dtype=np.float64) - self.mean) / std
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {'count': self.count, 'mean': self.mean.copy(), 'M2': self.M2.copy()}

    def load_state_dict(self, d):
        self.count = d['count']
        self.mean = d['mean'].copy()
        self.M2 = d['M2'].copy()


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_sequence, action, reward, next_state_sequence, done):
        self.buffer.append((state_sequence, action, reward, next_state_sequence, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
