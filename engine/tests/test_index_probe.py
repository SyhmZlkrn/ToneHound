import hashlib

import numpy as np
import pytest
import soundfile as sf

from tonehound.index_probe import probe_info, resolve_probe


def test_clean_install_generates_reusable_deterministic_probe(tmp_path, monkeypatch):
    monkeypatch.delenv('TONEHOUND_DI_PATH', raising=False)
    assert probe_info(tmp_path)['kind'] == 'synthetic'
    path = resolve_probe(tmp_path)
    assert sf.info(path).duration == 15
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    assert resolve_probe(tmp_path) == path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
    second = resolve_probe(tmp_path / 'other')
    # libsndfile's FLOAT WAV PEAK chunk contains the write timestamp. The
    # deterministic contract is the decoded probe, not that container field.
    first_audio, first_rate = sf.read(path)
    second_audio, second_rate = sf.read(second)
    assert first_rate == second_rate
    np.testing.assert_array_equal(first_audio, second_audio)


def test_existing_user_di_and_custom_path_are_preserved(tmp_path, monkeypatch):
    monkeypatch.delenv('TONEHOUND_DI_PATH', raising=False)
    existing = tmp_path / 'assets/user_di/Djent DI.wav'
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b'user recording')
    assert resolve_probe(tmp_path) == existing.resolve()
    custom = tmp_path / 'my DI.wav'
    custom.write_bytes(b'another recording')
    monkeypatch.setenv('TONEHOUND_DI_PATH', str(custom))
    assert resolve_probe(tmp_path) == custom.resolve()
    assert existing.read_bytes() == b'user recording'


def test_bad_custom_path_is_not_silently_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv('TONEHOUND_DI_PATH', str(tmp_path / 'missing.wav'))
    with pytest.raises(FileNotFoundError, match='TONEHOUND_DI_PATH'):
        resolve_probe(tmp_path)
