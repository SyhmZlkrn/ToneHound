"""Report project storage and optionally prune reproducible experiment audio.

Default is read-only. --prune-experiments deletes only .npy/.wav files from the
seven fixed experiment directories below. It never removes directories, JSON
evidence, imported references, downloads, captures, tokens, models or tools.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIRS = (
    '.cache/renders_djent', '.cache/renders_funk', '.cache/renders_thall',
    '.cache/renders_v2', '.cache/renders_fixed',
    '.cache/reference_diagnostics', '.cache/reference_diagnostics_controls',
)


def safe_experiment_files(root: Path) -> list[Path]:
    checked_root = root.resolve()
    files = []
    for name in EXPERIMENT_DIRS:
        folder = root / name
        if not folder.exists():
            continue
        if folder.is_symlink() or not folder.resolve().is_relative_to(checked_root):
            raise ValueError(f'Experiment directory is outside the workspace: {name}')
        for path in folder.rglob('*'):
            if path.suffix.lower() not in {'.npy', '.wav'} or not path.is_file():
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
                raise ValueError(f'Experiment file resolves outside its checked directory: {name}')
            files.append(path)
    return sorted(files)


def audit(root: Path) -> dict:
    categories = {}
    for item in root.iterdir():
        if item.is_symlink():
            continue
        size = item.stat().st_size if item.is_file() else sum(
            p.stat().st_size for p in item.rglob('*') if p.is_file() and not p.is_symlink())
        categories[item.name] = size
    candidates = safe_experiment_files(root)
    groups = []
    for folder in EXPERIMENT_DIRS:
        paths = [p for p in candidates if p.is_relative_to(root / folder)]
        groups.append({'path': folder, 'files': len(paths), 'bytes': sum(p.stat().st_size for p in paths)})
    return {'project_bytes': sum(categories.values()), 'categories_bytes': dict(sorted(categories.items(), key=lambda i: -i[1])),
            'reproducible_experiments': groups,
            'prunable_bytes': sum(g['bytes'] for g in groups),
            'preserves': ['all JSON evidence', 'native/media', 'downloads', 'tone3000 profiles and credentials', 'embedding indexes', 'model weights', 'toolchain', 'assets']}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prune-experiments', action='store_true')
    parser.add_argument('--manifest', type=Path, help='Write the audit and exact candidate paths to a JSON file under .cache or dist.')
    args = parser.parse_args()
    report = audit(ROOT)
    paths = safe_experiment_files(ROOT)
    exact = [{'path': str(p.resolve()), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths]
    if args.manifest:
        manifest = args.manifest.resolve()
        if not any(manifest.is_relative_to((ROOT / name).resolve()) for name in ('.cache', 'dist')):
            parser.error('Manifest must be inside project .cache or dist.')
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({**report, 'candidates': exact}, indent=2), encoding='utf-8')
    if args.prune_experiments:
        deleted = 0
        for path, original in zip(paths, exact):
            # Revalidate both the absolute boundary and unchanged file before each delete.
            folder = next((ROOT / d).resolve() for d in EXPERIMENT_DIRS if path.is_relative_to(ROOT / d))
            if path.is_symlink() or not path.resolve().is_relative_to(folder):
                raise RuntimeError('Experiment path changed during cleanup; stopped.')
            stat = path.stat()
            if stat.st_size != original['bytes'] or stat.st_mtime_ns != original['mtime_ns']:
                raise RuntimeError('Experiment file changed during cleanup; stopped.')
            path.unlink()
            deleted += stat.st_size
        report['deleted_bytes'] = deleted
        report['deleted_files'] = len(paths)
    else:
        report['mode'] = 'dry-run; nothing deleted'
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
