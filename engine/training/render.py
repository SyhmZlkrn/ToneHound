"""Offline NAM backends with explicit renderer identity in dataset provenance."""
from pathlib import Path
import subprocess
import shutil
import tempfile

import numpy as np
import soundfile as sf
from training.common import digest, read_json

CORE_REVISION = '2563c0fd4cb1f9ce457d89a761738ea15097e1f3'


def verify_renderer(binary):
    binary = Path(binary).resolve()
    receipt = read_json(binary.with_suffix('.json'))
    if receipt.get('revision') != CORE_REVISION or receipt.get('binary_sha256') != digest(binary):
        raise ValueError('NAM Core renderer receipt/hash mismatch; rebuild it')
    return binary


class CoreRenderer:
    def __init__(self, path, binary):
        self.path = Path(path).resolve()
        self.binary = Path(binary).resolve()
        metadata = read_json(path)
        self.sample_rate = int(metadata.get('sample_rate') or 48000)
        if self.sample_rate <= 0:
            self.sample_rate = 48000
        # Official renderer prewarms the model; also retain a second of actual
        # preceding performance for excerpts after the beginning of the DI.
        self.receptive_field = self.sample_rate
        self.slimmable = metadata['architecture'] == 'SlimmableContainer'

    def render(self, audio, *, device='cpu'):
        with tempfile.TemporaryDirectory(prefix='tonehound_render_') as temp:
            temp = Path(temp)
            dry, wet = temp/'input.wav', temp/'output.wav'
            sf.write(dry, np.asarray(audio,dtype=np.float32), self.sample_rate, subtype='FLOAT')
            command = [str(self.binary)]
            if self.slimmable:
                command += ['--slim','1.0']
            model_path = self.path
            if not str(model_path).isascii():
                model_path = temp/'model.nam'
                shutil.copy2(self.path, model_path)
            command += [str(model_path),str(dry),str(wet)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, errors='replace', timeout=300)
            except subprocess.TimeoutExpired as error:
                raise ValueError('NAM Core render exceeded the 300-second limit') from error
            if result.returncode != 0:
                raise ValueError('NAM Core rendering failed: ' + result.stderr[-2000:])
            output, rate = sf.read(wet, dtype='float64', always_2d=True)
            if rate != self.sample_rate or output.shape != (len(audio),1):
                raise ValueError('Unexpected native renderer output rate/shape')
            return output[:,0]
