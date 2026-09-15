import numpy as np
import pytest

from tonehound.reference_features import analyse_reference, estimate_tempo


def pulse(bpm=120, sr=12000, duration=30):
    x = np.zeros(sr * duration)
    n = np.arange(500)
    hit = np.sin(n * 1.1) * np.exp(-n / 60)
    for start in np.arange(0, len(x) - 500, sr * 60 / bpm).astype(int):
        x[start:start + len(hit)] += hit
    return x


def test_detects_measured_pulse_and_reports_half_double_ambiguity():
    result = estimate_tempo(pulse(), 12000)
    assert result['bpm'] == pytest.approx(120, abs=1)
    assert result['confidence'] > .7
    assert len(result['alternatives']) >= 1


def test_stereo_phase_cancellation_does_not_erase_pulse():
    x = pulse()
    result = estimate_tempo(np.column_stack((x, -x)), 12000)
    assert result['bpm'] == pytest.approx(120, abs=1)


@pytest.mark.parametrize('samples', [np.zeros(12000 * 10), np.ones(12000 * 10),
    np.sin(np.arange(12000 * 10) * .5), np.zeros(50)])
def test_absent_or_steady_signal_does_not_invent_tempo(samples):
    assert estimate_tempo(samples, 12000)['bpm'] is None


def test_role_presets_auto_apply_with_explicit_fallback_and_keep_screamer():
    features, solo = analyse_reference(np.zeros(12000 * 10), 12000, 'solo')
    assert features['bpm'] is None and solo['tempo_fallback']
    assert solo['auto_apply'] and solo['delayEnabled'] and solo['reverbEnabled']
    assert solo['delayBpm'] == 120 and solo['delayDivision'] == 1
    assert 'screamerEnabled' not in solo
    _, rhythm = analyse_reference(pulse(), 12000, 'rhythm')
    assert not rhythm['delayEnabled'] and not rhythm['reverbEnabled']
    assert rhythm['delayBpm'] == pytest.approx(120, abs=1)


def test_invalid_role_is_rejected():
    with pytest.raises(ValueError, match='rhythm or solo'):
        analyse_reference(pulse(), 12000, 'automatic')
