"""Package only source and final measured deliverables, without virtualenvs/caches."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('../socialrl_submission.zip'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    allowed_roots = {'envs', 'config', 'networks', 'tests', 'scripts', 'legacy'}
    final_runs = {'ippo_seed1', 'single_agent_seed1'}
    files = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        parts = relative.parts
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in {'.git', '.venv', '.pytest_cache', '__pycache__', '.vscode', '.mplconfig', '.cache'} for part in parts):
            continue
        if path.suffix in {'.pyc', '.pyo', '.zip'} or path.name == 'SUBMISSION_MANIFEST.json':
            continue
        include = len(parts) == 1 and (
            path.suffix in {'.py', '.md', '.txt', '.sh', '.json'}
            or path.name in {'LICENSE', '.gitignore'}
        )
        include |= parts[0] in allowed_roots
        if parts[0] == 'results' and len(parts) > 1:
            include |= parts[1] in final_runs or parts[1] == 'ippo_seed1_console.log'
        # Keep W&B offline data and metadata, omit debug logs and duplicate latest-run.
        if 'wandb' in parts and ('logs' in parts or 'latest-run' in parts or path.name.startswith('debug')):
            include = False
        if include:
            files.append(path)
    for required in ('LICENSE','README.md','REPORT.md','results/ippo_seed1/summary.json',
                     'results/ippo_seed1/checkpoints/final.pt','results/ippo_seed1/evaluation/summary.json',
                     'results/ippo_seed1/figures/learning_curve.png'):
        if root / required not in files:
            raise FileNotFoundError(required)
    manifest = {'algorithm': 'independent PPO', 'files': {str(p.relative_to(root)): {
        'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()
    } for p in files}}
    manifest_path = root / 'SUBMISSION_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files + [manifest_path]:
            archive.write(path, 'socialrl_submission/' + str(path.relative_to(root)))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    print(json.dumps({'output': str(output), 'files': len(files)+1, 'bytes': output.stat().st_size}, indent=2))

if __name__ == '__main__':
    main()
