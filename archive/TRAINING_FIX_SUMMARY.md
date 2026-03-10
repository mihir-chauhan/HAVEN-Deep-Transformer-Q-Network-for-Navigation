# DTQN Training Degradation - Root Cause & Fix

## Problem
Training success rate collapsed from **68% (ep 25) → 92% (ep 50) → 56% (ep 100) → 8% (ep 150) → 28% (ep 200)** during episodes 100-103 when epsilon exploration dropped to minimum (0.05).

## Root Cause Analysis

### Scale Mismatch Between Features and Rewards

1. **Normalized Input Features**: The `RunningNormalizer` normalizes all 16 input features to mean≈0, std≈1, clipped to [-5, 5] range
   - Features are in a *unit scale* due to feature normalization

2. **Non-Normalized Rewards**: Rewards accumulate without normalization
   - Reward clipping: [-10, 10] per step
   - Max steps per decision: 10
   - Max decisions per episode: 100
   - **Potential max accumulated reward: 100 × 50 = 5,000+**
   - Individual episode returns observed: 0-230+ (see log)

3. **Training Instability**:
   - Q-network must map unit-scale normalized features → thousand-scale accumulated returns
   - Requires massive weight magnitudes in output layer
   - Causes exploding/vanishing gradients, numerical instability
   - Model learns unstable Q-functions that work with random exploration (high ε)
   - When forced to exploit (low ε), learned policy is worse than random → catastrophic failure

### Why It Failed at Episode 102-103
- Epsilon decay: ε = 1.0 × 0.9995^(~6000 decisions) ≈ 0.05
- At ε=0.05 (low exploration), model must exploit its learned policy
- Learned policy was unstable due to reward scale mismatch → failure cascade

## Solutions Implemented

### 1. **Reward Normalization** (Primary Fix)
```python
# NEW: Added reward normalizer alongside feature normalizer
self.reward_normalizer = RunningNormalizer(dim=1)

# Normalize rewards before storing in replay buffer
normalized_td_target = float(self.reward_normalizer.normalize(
    np.array([td_target], dtype=np.float64))[0])
self.replay_buffer.push(state_seq, normalized_td_target)

# Use normalized target in training
target_t = torch.tensor([normalized_td_target], ...)
loss = F.smooth_l1_loss(q_pred, target_t)
```

**Effect**: Q-network now learns to predict unit-scale values (like features), keeping weight magnitudes reasonable and gradients stable.

### 2. **Increased Target Network Update Rate**
```python
self.tau = 0.01  # Increased from 0.005
```
- Previous: tau=0.005 meant target network updated very slowly (only ~5% per step)
- After 1000 updates: only 50% of online network transferred to target
- **New**: tau=0.01 doubles update rate, reducing stale target issue

### 3. **Reduced Gradient Clipping**
```python
self.grad_clip = 5.0  # Reduced from 10.0
```
- Tighter clipping prevents weight explosion while stabilizing training
- Works better with normalized reward scale

### 4. **Improved Checkpoint Serialization**
- Added `reward_normalizer` state to checkpoint saving/loading
- Ensures reward statistics persist across training sessions
- Prevents distribution shift when resuming training

## Expected Improvements

✓ **Stable Q-value predictions** - Now learning unit-scale Q-values  
✓ **Better gradient flow** - Numerical stability during backprop  
✓ **Smoother transition** - From exploration to exploitation  
✓ **Reproducible training** - Reward normalization statistics persist  

## How to Verify

1. Run training from ep 200 onwards - should show improved success rate (not degradation)
2. Monitor `training_metrics.csv` for lower loss values at higher epsilon steps
3. Check Q-value ranges in logs - should stay in reasonable bounds (±10-20 range)

## Testing
Run training with saved checkpoint:
```bash
python train_dtqn.py 500 --checkpoint dtqn_checkpoint.pt --eval-every 25 --save-every 10
```

Monitor for:
- **Success rate stabilization** around episode 125-150
- **Smooth performance curves** - no more cliffs at ε-min transition
- **Better overall convergence** by episode 200+

## Files Modified
- `older_main.py`: DTQNPolicy class - added reward normalization, improved hyperparameters
  - Updated `__init__` to add reward_normalizer
  - Updated `update()` method to normalize TD targets
  - Updated checkpoint save/load for reward_normalizer state
  - Updated `_train_from_buffer()` to work with normalized targets
