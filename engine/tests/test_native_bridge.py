"""Native file-job boundary: real audio import, crop bounds and useful failures."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tonehound import native_bridge as bridge
from tonehound.config import SAMPLE_RATE
from tonehound.ingest import Source


@pytest.fixture()
def song(tmp_path):
    path = tmp_path / 'reference with spaces.wav'
    t = np.arange(45 * SAMPLE_RATE) / SAMPLE_RATE
    sf.write(path, .1 * np.sin(2 * np.pi * 110 * t), SAMPLE_RATE)
    return path


def test_import_preserves_full_song_and_defaults_to_40_seconds(tmp_path, song):
    events = []
    result = bridge.run({'action': 'import', 'source': str(song)}, tmp_path,
                        lambda *event: events.append(event))
    assert result['duration'] == 45
    assert (result['start'], result['end']) == (0, 40)
    assert result['origin'] == 'file'
    assert sf.info(result['path']).duration == 45
    assert events[0][0] == 'importing'
    assert bridge.run({'action': 'import', 'source': str(song)}, tmp_path)['path'] == result['path']


def test_link_import_uses_same_decoder_and_keeps_provenance(tmp_path, monkeypatch):
    url = 'https://example.com/guitar-song'
    calls = []
    def load(spec, **kwargs):
        calls.append((spec, kwargs['seconds']))
        return Source(np.zeros(35 * SAMPLE_RATE, dtype=np.float32), 'Linked song', 'url', url=url)
    monkeypatch.setattr(bridge.ingest, 'load', load)
    result = bridge.run({'action': 'import', 'source': url}, tmp_path)
    assert calls == [(url, None)]
    assert result['url'] == url and result['origin'] == 'url'
    assert result['end'] == 35


def test_short_reference_is_rejected(tmp_path):
    path = tmp_path / 'short.wav'
    sf.write(path, np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    with pytest.raises(ValueError, match='at least 30 seconds'):
        bridge.run({'action': 'import', 'source': str(path)}, tmp_path)


@pytest.mark.parametrize('start,end', [(0, 29.9), (20, 51), (-1, 40), (0, float('nan')), (0, float('inf')), (40, 0)])
def test_invalid_crop_rejected(song, start, end):
    with pytest.raises(ValueError):
        bridge.validate_crop(song, start, end)


def test_exact_minimum_and_end_of_song_are_accepted(song):
    assert bridge.validate_crop(song, 15, 45) == 30
    assert bridge.validate_crop(song, 0, 40) == 40


def test_library_keeps_tone_identity_and_actual_photo(tmp_path):
    profiles = tmp_path / '.cache/tone3000/profiles'
    profiles.mkdir(parents=True)
    (profiles / 't3k-1061-987 Marshall.nam').write_text('{}')
    (profiles / '_ignore.nam').write_text('{}')
    bridge.write_json(tmp_path / '.cache/native/artwork/1061.json',
                      {'title': 'Marshall 1978 JMP 2203', 'image_path': 'actual-photo.jpg', 'creator': 'Creator'})
    rows = bridge.library(tmp_path)['profiles']
    assert len(rows) == 1
    assert rows[0]['tone_id'] == 1061
    assert rows[0]['model_id'] == 987
    assert rows[0]['image_path'] == 'actual-photo.jpg'


def test_job_failure_writes_machine_readable_error_and_log(tmp_path):
    request, response, progress = [tmp_path / name for name in ('request.json', 'response.json', 'progress.json')]
    bridge.write_json(request, {'action': 'unknown'})
    script = Path(bridge.__file__).resolve().parents[1] / 'tools/native_job.py'
    child = subprocess.run([sys.executable, str(script), '--request', str(request),
                            '--response', str(response), '--progress', str(progress)],
                           timeout=30, capture_output=True)
    assert child.returncode == 1
    result = json.loads(response.read_text())
    assert result['ok'] is False and 'Unknown native job' in result['message']
    assert 'Traceback' in response.with_suffix('.log').read_text()


def test_atomic_response_roundtrips_unicode(tmp_path):
    path = tmp_path / 'nested/result.json'
    bridge.write_json(path, {'name': '音色 – guitar'})
    assert bridge.read_json(path) == {'name': '音色 – guitar'}
    assert not list(path.parent.glob('*.tmp'))


def test_native_import_preserves_stereo_and_global_balance(tmp_path):
    t=np.arange(31*SAMPLE_RATE)/SAMPLE_RATE
    source=np.column_stack((.5*np.sin(2*np.pi*110*t),.15*np.sin(2*np.pi*397*t))).astype(np.float32)
    path=tmp_path/'stereo.wav';sf.write(path,source,SAMPLE_RATE,subtype='FLOAT')
    result=bridge.run({'action':'import','source':str(path)},tmp_path)
    audio,sr=sf.read(result['path'])
    assert sr == SAMPLE_RATE and result['channels'] == 2
    assert result['source_path'] == str(path.resolve())
    assert audio.shape == source.shape
    assert abs(np.corrcoef(audio.T)[0,1]) < .001
    assert np.sqrt(np.mean(audio[:,1]**2))/np.sqrt(np.mean(audio[:,0]**2)) == pytest.approx(.3,rel=1e-5)


def test_match_returns_stereo_eq_target_and_role_effects_for_entire_local_library(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tonehound import pipeline
    folder = tmp_path / '.cache/tone3000/profiles'
    folder.mkdir(parents=True)
    for i in range(55):
        (folder / f't3k-1-{i + 1} Profile.nam').write_text('{}')
    samples = np.column_stack((np.ones(31 * SAMPLE_RATE) * .1, np.ones(31 * SAMPLE_RATE) * -.1))
    path = tmp_path / 'reference.wav'
    sf.write(path, samples, SAMPLE_RATE)
    configurations = []
    class FakePipeline:
        def __init__(self, cfg):
            configurations.append(cfg)
        def match(self, source, progress=None):
            return SimpleNamespace(
                source=SimpleNamespace(samples=samples), stem=SimpleNamespace(samples=samples),
                to_json=lambda: {'matches': [], 'index_size': 55})
    monkeypatch.setattr(pipeline, 'Pipeline', FakePipeline)
    monkeypatch.delenv('TONEHOUND_DI_PATH', raising=False)
    result = bridge.run({'action': 'match', 'path': str(path), 'start': 0, 'end': 30,
                         'role': 'solo'}, tmp_path)
    assert configurations[0].profile_dir == folder
    assert len(list(configurations[0].profile_dir.glob('*.nam'))) == 55
    assert result['index_size'] == 55
    assert sf.info(result['stem_path']).channels == 2
    assert result['reference_features']['role'] == 'solo'
    assert result['effect_suggestion']['auto_apply']
    assert result['effect_suggestion']['delayEnabled']


def test_bridge_library_status_is_safe_without_authentication(tmp_path, monkeypatch):
    monkeypatch.delenv('TONE3000_PUBLISHABLE_KEY', raising=False)
    monkeypatch.delenv('TONEHOUND_DI_PATH', raising=False)
    result = bridge.run({'action': 'library_status'}, tmp_path)
    assert not result['connected'] and not result['configured']
    assert result['index_source']['kind'] == 'synthetic'
    assert 'token' not in json.dumps(result)


def test_stem_cache_reuses_identical_audio_and_preserves_stereo(tmp_path, monkeypatch):
    left = np.linspace(-.5, .5, 2000, dtype=np.float32)
    audio = np.column_stack((left, left * -.25))
    written = []
    original_write = sf.write
    def write(*args, **kwargs):
        written.append(args[0])
        return original_write(*args, **kwargs)
    monkeypatch.setattr(bridge.sf, 'write', write)
    first = bridge.cache_stem(tmp_path, audio)
    second = bridge.cache_stem(tmp_path, audio.astype(np.float64))
    assert first == second and len(written) == 1
    result, rate = sf.read(second, dtype='float32', always_2d=True)
    assert rate == SAMPLE_RATE
    np.testing.assert_array_equal(result, audio)
    assert len(list(first.parent.glob('*.wav'))) == 1
    assert not list(first.parent.glob('*.tmp.wav'))


def test_stem_identity_includes_channel_order_shape_and_sample_rate(tmp_path):
    audio = np.column_stack((np.ones(100, dtype=np.float32), np.zeros(100, dtype=np.float32)))
    paths = {bridge.cache_stem(tmp_path, audio), bridge.cache_stem(tmp_path, audio[:, ::-1]),
             bridge.cache_stem(tmp_path, audio, 44100), bridge.cache_stem(tmp_path, audio.ravel())}
    assert len(paths) == 4


def test_stem_cache_repairs_corruption_without_removing_legacy_references(tmp_path):
    audio = np.column_stack((np.ones(100, dtype=np.float32) * .25, np.zeros(100, dtype=np.float32)))
    path = bridge.cache_stem(tmp_path, audio)
    legacy = path.parent / 'stem-1234.wav'
    legacy.write_bytes(b'old session reference')
    # Same valid WAV header but different samples must also fail validation.
    sf.write(path, audio * 0, SAMPLE_RATE, subtype='FLOAT')
    assert bridge.cache_stem(tmp_path, audio) == path
    restored, _ = sf.read(path, dtype='float32', always_2d=True)
    np.testing.assert_array_equal(restored, audio)
    assert legacy.read_bytes() == b'old session reference'
