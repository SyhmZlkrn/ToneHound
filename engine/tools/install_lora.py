"""Install a trained LoRA export for the native matcher; optionally build its index."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common import digest, read_config, write_json


def install(source, root, name='LoRA 500-rig pilot'):
    from safetensors.torch import load
    from training.models import ToneEncoder
    source, root = Path(source), Path(root)
    with tempfile.TemporaryDirectory(prefix='tonehound-lora-') as temp:
        temp = Path(temp)
        if source.is_file():
            with zipfile.ZipFile(source) as archive:
                for filename, limit in [('config.yaml', 1024 * 1024), ('best.safetensors', 64 * 1024 * 1024)]:
                    found = [i for i in archive.infolist() if not i.is_dir() and Path(i.filename).name == filename]
                    if len(found) != 1 or found[0].file_size > limit:
                        raise ValueError('Export needs exactly one bounded ' + filename)
                    (temp / filename).write_bytes(archive.read(found[0]))
        else:
            if (source / 'config.yaml').stat().st_size > 1024 * 1024:
                raise ValueError('Training config is too large')
            shutil.copyfile(source / 'config.yaml', temp / 'config.yaml')
            weights = source / 'checkpoint/best.safetensors'
            if not weights.is_file():
                weights = source / 'best.safetensors'
            if weights.stat().st_size > 64 * 1024 * 1024:
                raise ValueError('LoRA checkpoint is too large')
            shutil.copyfile(weights, temp / 'best.safetensors')
        cfg = read_config(temp / 'config.yaml')
        if cfg['model']['mode'] != 'lora':
            raise ValueError('Select a LoRA run, including its learned head.')
        spec = dict(cfg['model'], gradient_checkpointing=False)
        model = ToneEncoder(spec)
        # Byte loading avoids Windows mmap locks retained by an error traceback
        # while TemporaryDirectory is trying to clean up a rejected export.
        model.load_trainable(load((temp / 'best.safetensors').read_bytes()))
        del model
        sha = digest(temp / 'best.safetensors')
        folder = root / '.cache/matching_models/lora'
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / (sha + '.safetensors')
        # A complete content-addressed file precedes the atomic manifest switch.
        with tempfile.NamedTemporaryFile(dir=folder, suffix='.tmp', delete=False) as stream:
            staging = Path(stream.name)
        try:
            shutil.copyfile(temp / 'best.safetensors', staging)
            if not destination.is_file() or digest(destination) != sha:
                staging.replace(destination)
        finally:
            staging.unlink(missing_ok=True)
        manifest = dict(schema=1, name=name, model=spec, checkpoint_sha256=sha)
        write_json(folder / 'model.json', manifest)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Training run directory or exported ZIP')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--name', default='LoRA 500-rig pilot')
    parser.add_argument('--prepare-index', action='store_true')
    args = parser.parse_args()
    print(json.dumps(install(args.source, args.root, args.name), indent=2), flush=True)
    if args.prepare_index:
        from tonehound.native_bridge import run
        result = run(dict(action='catalogue_index', matching_model='lora'), args.root,
                     lambda stage, fraction, message: print(message, flush=True))
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
