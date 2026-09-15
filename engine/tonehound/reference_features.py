"""Measured passage tempo and transparent role-based effect starting points.

This is independent of MERT amp retrieval. Tempo is estimated from changes in
the mix's spectral energy, retaining L/R energy even for out-of-phase audio.
It does not identify a song's original pedals, exact delay or production chain.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def estimate_tempo(samples: np.ndarray, sr: int) -> dict:
    result = {'bpm': None, 'confidence': 0.0, 'alternatives': [],
              'method': 'spectral_flux_autocorrelation',
              'note': 'No reliable pulse detected. Set BPM manually; solo delay starts at 120 BPM.'}
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or len(x) < 6 * sr or not np.all(np.isfinite(x)):
        return result
    if np.max(np.abs(x), initial=0.0) < 1e-5:
        return result
    # Bound analysis work to three minutes, independent of an imported song's
    # length. Rate conversion and spectra use each channel independently.
    x = x[:180 * sr]
    target_sr = 12000
    from math import gcd
    divisor = gcd(sr, target_sr)
    x = signal.resample_poly(x, target_sr // divisor, sr // divisor, axis=0)
    power = None
    for channel in x.T:
        _, _, spectrum = signal.stft(channel, target_sr, nperseg=512, noverlap=384,
                                      boundary=None, padded=False)
        current = np.abs(spectrum) ** 2
        power = current if power is None else power + current
    assert power is not None
    # Log compression stops one loud kick from overwhelming all quieter beats.
    log_energy = np.log1p(power[3:] / max(float(np.median(power[3:])), 1e-10))
    onset = np.maximum(np.diff(log_energy, axis=1), 0.0).mean(axis=0)
    if len(onset) < 10 or float(onset.std()) < 0.015:
        return result
    envelope = onset - np.mean(onset)
    correlation = signal.fftconvolve(envelope, envelope[::-1], mode='full')[len(envelope) - 1:]
    correlation /= np.maximum(np.arange(len(envelope), 0, -1), 1)
    if correlation[0] <= 1e-10:
        return result
    correlation /= correlation[0]
    rate = target_sr / 128.0
    lo, hi = int(rate * 60 / 240), min(len(correlation) - 1, int(rate * 60 / 40))
    peaks, _ = signal.find_peaks(correlation[lo:hi + 1], prominence=.02)
    if not len(peaks):
        return result
    lags = peaks + lo
    # Mild common-tempo prior resolves equal-strength subdivisions; alternatives
    # are still reported because audio alone cannot establish musical beat units.
    bpms = 60 * rate / lags
    preference = np.exp(-.15 * np.log2(bpms / 110.0) ** 2)
    best = int(lags[np.argmax(correlation[lags] * preference)])
    confidence = float(np.clip(correlation[best], 0.0, 1.0))
    if confidence < .15:
        return result
    # Quadratic interpolation removes the coarsest frame-grid quantisation.
    a, b, c = correlation[best - 1:best + 2]
    adjustment = .5 * (a - c) / (a - 2 * b + c) if abs(a - 2 * b + c) > 1e-9 else 0.0
    bpm = round(float(60 * rate / (best + np.clip(adjustment, -.5, .5))), 1)
    alternatives = [round(v, 1) for v in (bpm / 2, bpm * 2) if 40 <= v <= 240]
    result.update(bpm=bpm, confidence=round(confidence, 3), alternatives=alternatives,
                  note='Estimated pulse; half/double tempo may fit. Adjust BPM by ear. '
                       'Effects are role-based starting settings, not recovered recording settings.')
    return result


def analyse_reference(samples: np.ndarray, sr: int, role: str = 'rhythm') -> tuple[dict, dict]:
    if role not in ('solo', 'rhythm'):
        raise ValueError('Choose either rhythm or solo for the selected passage.')
    features = dict(estimate_tempo(samples, sr), role=role)
    bpm = features['bpm'] or 120.0
    solo = role == 'solo'
    # The user explicitly requested automatic application. Screamer is absent
    # here so their chosen overdrive state remains untouched: pulse analysis
    # supplies no evidence that a boost pedal belongs in the original chain.
    suggestion = {'auto_apply': True, 'role': role, 'delayEnabled': solo,
                  'delayMix': .18, 'delayFeedback': .25, 'delayDivision': 1,
                  'delayBpm': bpm, 'reverbEnabled': solo, 'reverbMix': .12,
                  'reverbSize': .5, 'tempo_fallback': features['bpm'] is None,
                  'note': ('Solo starting preset: dotted-eighth delay and light reverb. '
                           'BPM is estimated and can be overridden.' if solo else
                           'Rhythm starting preset: delay and reverb bypassed. Screamer is unchanged.')}
    return features, suggestion
