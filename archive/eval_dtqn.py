import os
import subprocess
import sys

def main():
    # Optional: number of episodes for evaluation
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    env = os.environ.copy()
    env['USE_DTQN'] = '1'
    env['TRAIN_HIGH_LEVEL'] = '0'
    # Enable video and write under eval/
    env['DISABLE_VIDEO'] = '0'
    env['VIDEO_SUBDIR'] = 'eval'

    script_path = os.path.join(os.path.dirname(__file__), 'older_main.py')
    for ep in range(K):
        print(f"[eval_dtqn] Running evaluation episode {ep+1}/{K}")
        env['EPISODE_INDEX'] = str(ep+1)
        subprocess.run([sys.executable, script_path], env=env, check=True)
    print("[eval_dtqn] Completed evaluation.")

if __name__ == '__main__':
    main()


