#!/usr/bin/env python3
"""Quick validation that training loop runs without errors after fixes."""

import sys
import os
# Add parent (fov/) to path for train_env import when run from archive/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['USE_DTQN'] = '1'
os.environ['TRAIN_HIGH_LEVEL'] = '1'

from train_env import DTQNPolicy, run_training_episode, _init_train_metrics
import numpy as np

print("Testing DTQN training pipeline...")
print("-" * 50)

# Initialize policy
policy = DTQNPolicy(train_mode=True)
_init_train_metrics()

print(f"✓ Policy initialized successfully")
print(f"  - Device: {policy.device}")
print(f"  - Replay buffer capacity: {policy.replay_buffer.capacity}")
print(f"  - Reward normalizer: {policy.reward_normalizer}")
print(f"  - Tau (target update rate): {policy.tau}")
print(f"  - Grad clip: {policy.grad_clip}")
print()

# Run 3 quick episodes to verify training works
print("Running 3 test episodes...")
try:
    for ep in range(1, 4):
        result = run_training_episode(policy, episode_idx=ep, save_video=False)
        status = "✓ OK" if result['success'] else "✗ FAIL"
        print(f"  Episode {ep}: {status} | Success={result['success']}, Reward={result['total_reward']:.2f}, Updates={policy.update_count}")
    print()
    print("✓ Training pipeline validation PASSED")
    print("-" * 50)
    print("Key improvements:")
    print("  1. Reward normalization enabled")
    print("  2. Faster target network updates (tau=0.01)")
    print("  3. Better gradient stability (clip=5.0)")
    print("  4. Checkpoint persistence for reward stats")
    print()
    print("Ready to run full training!")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
