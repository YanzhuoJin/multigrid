"""Single-agent MultiGrid PPO entry point using the same tested IPPO code.

Usage: python ppo.py --total-episodes 2000 --wandb-mode offline
The supplied earlier CleanRL/CartPole script is preserved in legacy/ppo_original.py.
"""
import sys
from main import main

if __name__ == "__main__":
    defaults = [] if "--resume" in sys.argv else [
        "--env-id", "MultiGrid-Cluttered-Fixed-Single-6x6-v0",
        "--total-episodes", "2000", "--run-dir", "results/single_agent",
    ]
    main(defaults + sys.argv[1:])
