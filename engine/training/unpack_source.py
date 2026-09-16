"""Unpack a private pilot bundle to Drive, with resumable extraction."""
from pathlib import Path
import argparse
import shutil
import sys
import zipfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common import digest, inside


def unpack(archive, destination):
    archive, destination = Path(archive), Path(destination).resolve()
    fingerprint = digest(archive)
    marker = destination/'ToneHound-ML'/'.source_bundle.sha256'
    if marker.is_file():
        if marker.read_text().strip() != fingerprint:
            raise ValueError('A different source bundle is already installed; use a new destination')
        print('Source bundle already unpacked')
        return
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        if sum(e.file_size for e in entries) > 8*1024**3:
            raise ValueError('Source bundle exceeds the 8 GiB input limit')
        if len({e.filename for e in entries}) != len(entries):
            raise ValueError('Duplicate ZIP entry')
        for entry in entries:
            inside(destination, entry.filename)
            if not (entry.filename.startswith('ToneHound-ML/source/') or
                    entry.filename == 'ToneHound-ML/pilot_report.json'):
                raise ValueError('Unexpected source-bundle member')
        for entry in entries:
            path = inside(destination, entry.filename)
            if entry.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name+'.unpacking')
            with z.open(entry) as source, temp.open('wb') as target:
                shutil.copyfileobj(source,target)
            temp.replace(path)
    if not (marker.parent/'source/manifest.json').is_file():
        raise ValueError('Source bundle has no manifest')
    marker.write_text(fingerprint+'\n',encoding='utf-8')
    print('Source bundle unpacked into ' + str(marker.parent))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    unpack(a.archive, a.destination)


if __name__ == '__main__':
    main()
