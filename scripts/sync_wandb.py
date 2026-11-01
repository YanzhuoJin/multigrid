"""Authenticate interactively if needed, then upload saved W&B runs.

Run in your terminal: python scripts/sync_wandb.py
Credentials are handled by W&B itself, never stored in this project.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import urllib.request


def subprocess_environment():
    """Expose configured system proxies to W&B's background process."""
    environment = os.environ.copy()
    proxies = urllib.request.getproxies()
    for scheme in ('http', 'https'):
        name = f'{scheme}_proxy'
        if name not in environment and name.upper() not in environment:
            if proxies.get(scheme):
                environment[name.upper()] = proxies[scheme]
    if 'no_proxy' not in environment and 'NO_PROXY' not in environment:
        bypass = [proxies.get('no', ''), 'localhost', '127.0.0.1', '::1']
        environment['NO_PROXY'] = ','.join(value for value in bypass if value)
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('results/ippo_seed1'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve() if args.run_dir.is_absolute() else root / args.run_dir
    runs = sorted((run_dir / 'wandb').glob('offline-run-*'))
    if not runs:
        raise FileNotFoundError('No W&B offline runs found at ' + str(run_dir / 'wandb'))
    environment = subprocess_environment()
    subprocess.run([sys.executable, '-m', 'wandb', 'login'], check=True, env=environment)
    completed = []
    for run in runs:
        subprocess.run([sys.executable, '-m', 'wandb', 'sync', str(run)],
                       check=True, env=environment)
        completed.append(run.name)
    (run_dir / 'wandb_sync_status.json').write_text(json.dumps({
        'cloud_sync_command_succeeded': True, 'synced_runs': completed
    }, indent=2) + '\n')
    print('All W&B sync commands completed successfully. Check their output for dashboard URLs.')


if __name__ == '__main__':
    main()
