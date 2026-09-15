"""Build a small, screened source ZIP without touching Git or local data.

The allowlist is release/source_manifest.json. A blocked scan reports paths and
rule names only, never candidate credential values. Does not prove arbitrary
secrets cannot exist; review the manifest before publishing.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DENIED_PARTS = {'.cache', '.git', '__pycache__', '.venv', 'vendor', 'dist'}
DENIED_SUFFIXES = {'.nam', '.wav', '.mp3', '.flac', '.webm', '.mp4', '.npy', '.npz', '.pt', '.pth', '.th', '.exe', '.dll', '.zip', '.safetensors'}
PATTERNS = {
    'TONE3000 credential': re.compile(rb't3k_(?:pub|sec)_[A-Za-z0-9_-]{24,}'),
    'GitHub credential': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})'),
    'JWT credential': re.compile(rb'eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}'),
    'private key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}


def local_credentials(root: Path) -> list[bytes]:
    """Read only known credential files; never include values in output."""
    values = []
    key = root / '.cache/tone3000/publishable_key.txt'
    if key.is_file():
        values.append(key.read_bytes().strip())
    token_path = root / '.cache/tone3000/tokens.json'
    if token_path.is_file():
        data = json.loads(token_path.read_text(encoding='utf-8-sig'))
        values.extend(str(data.get(k, '')).encode() for k in ('access_token', 'refresh_token'))
    return [v for v in values if len(v) >= 12]


def selected_files(root: Path, manifest: dict) -> list[Path]:
    paths = set()
    for pattern in manifest['include']:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(fnmatch.fnmatch(relative.as_posix(), p) for p in manifest.get('exclude', [])):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f'External file or symlink: {relative.as_posix()}')
            if DENIED_PARTS.intersection(relative.parts) or path.suffix.lower() in DENIED_SUFFIXES:
                raise ValueError(f'Disallowed source path: {relative.as_posix()}')
            if path.stat().st_size > manifest['max_file_bytes']:
                raise ValueError(f'Source file exceeds size limit: {relative.as_posix()}')
            paths.add(path)
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def inspect_files(root: Path, paths: list[Path]) -> tuple[list[dict], list[str]]:
    secrets = local_credentials(root)
    inventory, problems = [], []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        try:
            raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            problems.append(f'{relative}: not UTF-8 text')
        for label, pattern in PATTERNS.items():
            if pattern.search(raw):
                problems.append(f'{relative}: {label}')
        if any(secret in raw for secret in secrets):
            problems.append(f'{relative}: contains a local credential')
        inventory.append({'path': relative, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    return inventory, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/ToneHound-source.zip')
    parser.add_argument('--check', action='store_true', help='Scan only; do not write an archive.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'release/source_manifest.json').read_text(encoding='utf-8'))
    files = selected_files(ROOT, manifest)
    inventory, problems = inspect_files(ROOT, files)
    if problems:
        print('Export blocked by source screening:')
        print('\n'.join(problems))
        return 1
    summary = {'files': len(files), 'source_bytes': sum(p['bytes'] for p in inventory), 'credential_screen': 'passed', 'contains_runtime_assets': False}
    if not args.check:
        output = args.output.resolve()
        if not output.is_relative_to((ROOT / 'dist').resolve()):
            parser.error('Output must be inside the project dist directory.')
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                archive.write(path, 'ToneHound/' + path.relative_to(ROOT).as_posix())
            archive.writestr('ToneHound/SOURCE_INVENTORY.json', json.dumps({'summary': summary, 'files': inventory}, indent=2))
        summary.update(archive=str(output), archive_bytes=output.stat().st_size)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
