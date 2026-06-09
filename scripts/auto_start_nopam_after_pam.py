#!/usr/bin/env python3
"""
Auto-start NoPAM training after PAM training completes.
Runs as a cron job or standalone check.
"""
import subprocess
import sys
from pathlib import Path

TMUX_NAME = "bl5_pam_holdout_cgg"
NOPAM_CMD = [
    "bash", "run/run_bl5_v4_holdout_cgg.sh", "nopam"
]
CHECKPOINT = Path("results/bl5_v4_pam_holdout_cgg/checkpoints/best.pt")

def is_pam_running():
    """Check if PAM tmux session still exists."""
    result = subprocess.run(
        ["tmux", "has-session", "-t", TMUX_NAME],
        capture_output=True,
    )
    return result.returncode == 0

def main():
    if is_pam_running():
        print(f"[{__file__}] PAM session '{TMUX_NAME}' still running. Waiting...")
        return 0

    if not CHECKPOINT.exists():
        print(f"[{__file__}] WARNING: PAM session ended but best.pt NOT found at {CHECKPOINT}")
        return 1

    # Check if NoPAM already running
    result = subprocess.run(
        ["tmux", "has-session", "-t", "bl5_nopam_holdout_cgg"],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"[{__file__}] NoPAM session already exists. Nothing to do.")
        return 0

    print(f"[{__file__}] PAM training complete (best.pt found). Starting NoPAM training...")
    subprocess.Popen(NOPAM_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[{__file__}] NoPAM training launched.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
