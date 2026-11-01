"""Command line entry point for independent PPO training."""
import argparse
import json
from dataclasses import fields
from pathlib import Path

import yaml

from multiagent_metacontroller import MultiAgent, TrainingConfig


def parse_args(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=Path("config/mode/ppo.yaml"))
    preliminary.add_argument("--resume", type=Path)
    known, _ = preliminary.parse_known_args(argv)
    defaults = {}
    if known.config.exists():
        defaults.update(yaml.safe_load(known.config.read_text()) or {})
    if known.resume:
        import torch
        defaults.update(torch.load(known.resume, map_location="cpu", weights_only=False)["config"])
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    aliases = {"env_id": ["--env-name", "--env_name"],
               "total_episodes": ["--n-episodes", "--n_episodes"],
               "wandb_project": ["--wandb_project"]}
    for field in fields(TrainingConfig):
        arguments = ["--" + field.name.replace("_", "-")] + aliases.get(field.name, [])
        default = defaults.get(field.name, field.default)
        if isinstance(field.default, bool):
            parser.add_argument(*arguments, dest=field.name, action=argparse.BooleanOptionalAction,
                                default=default)
        else:
            parser.add_argument(*arguments, dest=field.name, type=type(field.default), default=default)
    parser.add_argument("--debug", action="store_true", help="Disable W&B; does not shorten training")
    parser.add_argument("--mode", choices=["ppo", "ippo"], default="ppo")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = TrainingConfig(**{field.name: getattr(args, field.name) for field in fields(TrainingConfig)})
    if args.debug:
        config.wandb_mode = "disabled"
    if config.wandb_mode not in ("offline", "online", "disabled"):
        raise ValueError("--wandb-mode must be offline, online or disabled")
    print(json.dumps({"algorithm": "independent PPO", "episodes": config.total_episodes,
                      "environment": config.env_id, "run_dir": config.run_dir,
                      "wandb_mode": config.wandb_mode}), flush=True)
    return MultiAgent(config, resume=str(args.resume) if args.resume else None).train()


if __name__ == "__main__":
    main()
