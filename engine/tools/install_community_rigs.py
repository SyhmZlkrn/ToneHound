"""Explicitly install the nine pinned, cabinet-included community captures.

Their collection declares GPLv3. Files and attribution stay in a local optional
pack, outside application/source releases. No TONE3000 API or bulk crawl.
"""
from pathlib import Path
import hashlib
import json
import urllib.request

from tonehound.nam_render import load


def install(root):
    root = Path(root)
    manifest = json.loads((root/'release/community_full_rigs.json').read_text())
    prefix = 'https://raw.githubusercontent.com/pelennor2170/NAM_models/' + manifest['commit'] + '/'
    destination = root/'.cache/community'
    (destination/'profiles').mkdir(parents=True, exist_ok=True)
    for row in manifest['entries']:
        name = row['filename']
        if (Path(name).name != name or '/' in name or '\\' in name or '-Cab-' not in name or
            row['rig'] != 'full-rig' or not row['url'].startswith(prefix)):
            raise ValueError('Manifest must contain pinned cabinet-included captures only')
        target = destination/'profiles'/name
        data = target.read_bytes() if target.is_file() else urllib.request.urlopen(row['url'],timeout=60).read(8*1024*1024+1)
        if len(data)>8*1024*1024 or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('Capture checksum mismatch: '+name)
        load(json.loads(data))  # Validate architecture/weight layout before installation.
        if not target.is_file():
            temp=target.with_suffix('.tmp');temp.write_bytes(data);temp.replace(target)
        print('Verified full rig: '+name)
    for name in ('COPYING','README.md'):
        (destination/name).write_bytes(urllib.request.urlopen(prefix+name,timeout=60).read())
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':
    install(Path(__file__).resolve().parents[2])
