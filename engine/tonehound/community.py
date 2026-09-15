"""Verified local, cabinet-included additions. Never crawl a remote catalogue."""
import json
from pathlib import Path
from .tone_index import ToneEntry, file_key


def entries(root):
    from .tone_learning import digest
    folder = Path(root)/'.cache/community'
    try:
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return []
    found = []
    for row in manifest.get('entries', []):
        name = row.get('filename', '')
        if (Path(name).name != name or '/' in name or '\\' in name or
                row.get('rig') != 'full-rig' or '-Cab-' not in name):
            continue
        path = folder/'profiles'/name
        if not path.is_file() or digest(path) != row.get('sha256'):
            continue
        found.append(ToneEntry(file_key(path), path.stem, path, 'community', meta=row))
    return found


def metadata(path, root):
    for entry in entries(root):
        if entry.path.resolve() == Path(path).resolve():
            return {'creator': entry.meta['creator'], 'gear': 'Full rig / amp + cabinet',
                    'description': 'Cabinet-included capture, as labelled by the creator. '
                    + entry.meta['rig_evidence'] + '. Source collection declares GPLv3.',
                    'url': entry.meta['url'], 'metadata_version': 2,
                    'makes': [], 'models': [entry.name], 'tags': ['Full rig', 'Cabinet included']}
    return {}
