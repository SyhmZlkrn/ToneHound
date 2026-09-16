"""Build the pinned official NAM Core offline renderer (A1 and A2)."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, help='Existing clean pinned NAM Core checkout')
    p.add_argument('--work', type=Path, default=ROOT/'.cache/training_renderer')
    p.add_argument('--cmake', default='cmake')
    p.add_argument('--generator', default=None)
    p.add_argument('--jobs', type=int, default=2)
    a = p.parse_args()
    pin = next(d for d in json.loads((ROOT/'release/native_dependencies.json').read_text())['dependencies']
               if d['path'].endswith('/NeuralAmpModelerCore'))
    work = a.work.resolve(); work.mkdir(parents=True, exist_ok=True)
    source = a.source.resolve() if a.source else work/'NeuralAmpModelerCore'
    if not source.exists():
        run('git','clone','--no-checkout',pin['repository'],source)
        run('git','-C',source,'checkout','--detach',pin['revision'])
    commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()
    if commit != pin['revision']:
        raise ValueError('Renderer source revision differs from release pin')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'], text=True).strip():
        raise ValueError('Renderer checkout has local modifications')
    run('git','-C',source,'submodule','update','--init','--recursive')
    build = work/'build-tonehound'
    command = [a.cmake,'-S',str(ROOT/'engine/training/native'),'-B',str(build),
               '-DCMAKE_BUILD_TYPE=Release','-DNAM_CORE_ROOT='+source.as_posix()]
    if a.generator:
        command += ['-G',a.generator]
    run(*command)
    run(a.cmake,'--build',build,'--config','Release','--target','render','--parallel',a.jobs)
    candidates = [build/'render',build/'render.exe',build/'Release/render.exe']
    binary = next((c for c in candidates if c.is_file()), None)
    if binary is None:
        raise RuntimeError('Build completed but the render binary was not found')
    target = work/('nam-render'+binary.suffix)
    shutil.copy2(binary, target)
    import hashlib
    receipt = dict(schema=1, repository=pin['repository'], revision=pin['revision'],
                   binary_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    target.with_suffix('.json').write_text(json.dumps(receipt, indent=2),encoding='utf-8')
    print('NAM renderer: ' + str(target))


if __name__ == '__main__':
    main()
