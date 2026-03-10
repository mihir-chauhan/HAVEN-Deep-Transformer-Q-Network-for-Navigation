import os
import sys
import json
import subprocess
import argparse


def run(cmd: list):
    print("[cmd]", " ".join(cmd))
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    return out.stdout


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate LSTM baselines (HL+MPC and End2End).')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--steps-hl', type=int, default=30)
    parser.add_argument('--steps-e2e', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--teacher-dtqn', type=str, default='')
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    hl_ckpt = os.path.join(base_dir, 'lstm_hl_mpc_checkpoint.pt')
    e2e_ckpt = os.path.join(base_dir, 'lstm_end2end_checkpoint.pt')

    # Train HL + MPC
    train_hl_cmd = [
        sys.executable, os.path.join(base_dir, 'train_lstm_hl_mpc.py'),
        '--episodes', str(args.episodes),
        '--steps-per-episode', str(args.steps_hl),
        '--epochs', str(args.epochs),
        '--out', os.path.basename(hl_ckpt),
        '--seed', str(args.seed),
    ]
    if args.teacher_dtqn:
        train_hl_cmd += ['--teacher-dtqn', args.teacher_dtqn]
    print('[stage] Training LSTM HL + MPC')
    run(train_hl_cmd)

    # Train End2End action
    train_e2e_cmd = [
        sys.executable, os.path.join(base_dir, 'train_lstm_end2end.py'),
        '--episodes', str(args.episodes),
        '--steps-per-episode', str(args.steps_e2e),
        '--epochs', str(args.epochs),
        '--out', os.path.basename(e2e_ckpt),
        '--seed', str(args.seed),
    ]
    print('[stage] Training LSTM End-to-End Action')
    run(train_e2e_cmd)

    # Evaluate HL + MPC
    print('[stage] Evaluating LSTM HL + MPC')
    eval_hl_cmd = [
        sys.executable, os.path.join(base_dir, 'run_eval_lstm_hl_mpc.py'),
        '--runs', str(args.runs),
        '--seed', str(args.seed),
        '--checkpoint', os.path.basename(hl_ckpt),
    ]
    out_hl = run(eval_hl_cmd)
    # Last JSON printed by script
    try:
        summary_hl = json.loads(out_hl.strip().splitlines()[-1])
    except Exception:
        summary_hl = { 'error': 'failed to parse HL+MPC summary', 'raw_tail': out_hl.splitlines()[-10:] }

    # Evaluate End2End action
    print('[stage] Evaluating LSTM End-to-End Action')
    eval_e2e_cmd = [
        sys.executable, os.path.join(base_dir, 'run_eval_lstm_end2end.py'),
        '--runs', str(args.runs),
        '--seed', str(args.seed),
        '--checkpoint', os.path.basename(e2e_ckpt),
    ]
    out_e2e = run(eval_e2e_cmd)
    try:
        summary_e2e = json.loads(out_e2e.strip().splitlines()[-1])
    except Exception:
        summary_e2e = { 'error': 'failed to parse E2E summary', 'raw_tail': out_e2e.splitlines()[-10:] }

    print('\n===== Labeled Results =====')
    print('LSTM HL + MPC:', json.dumps(summary_hl, indent=2))
    print('LSTM End-to-End (direct action):', json.dumps(summary_e2e, indent=2))


if __name__ == '__main__':
    main()


