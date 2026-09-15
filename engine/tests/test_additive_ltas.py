"""Validate the identity that makes the joint (profile, IR) search affordable.

    LTAS(amp -> IR)  ~=  LTAS(amp) + ir_response(IR)

If this holds, a 300x200 pair search costs 60k vector additions instead of 60k
renders. If it does not, the joint search has to be redesigned -- which is why
this is a Phase 0 gate rather than an assumption buried in the matcher.

The identity is exact only where |H(f)|^2 is flat across a band, so error is
expected to concentrate wherever the cab response moves steeply: the low-end
roll-off, the presence peak, and the top-end shelf. What matters is the error
*inside the analysis band*, since that is the only region the matcher scores.

Run directly for a per-band report::

    python -m tests.test_additive_ltas
"""

import numpy as np

from tonehound.config import GUITAR, SAMPLE_RATE
from tonehound.features import band_centers, ir_response_db, ltas_db, third_octave_edges
from tonehound.irs import apply_ir, synthetic_cab_ir
from tonehound.probe import build_probe

# Error budget inside the analysis band, set from measurement rather than hope.
# Spectrum-weighted band averaging measures 0.28-0.98 dB worst-case across
# drives and cabs (flat averaging measured 2.4-3.7 dB); 1.5 dB leaves headroom
# without letting a regression through. Far tighter than the +/-12 dB the
# match-EQ can correct, so the approximation is not what limits match quality.
MAX_IN_BAND_ERR_DB = 1.5


def _amp(x: np.ndarray, drive: float) -> np.ndarray:
    """Stand-in for a NAM profile: a soft-clipping nonlinearity.

    Phase 0 has no .nam files yet, and the identity under test concerns the
    *cab*, which is linear. Any nonlinearity upstream serves equally well.
    """
    return np.tanh(drive * x)


def reference_render(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Fixed spectral weighting for IR band averages.

    Stands in for what Phase 1 will use: the mean spectrum of every amp render
    in the index. A single mid-drive render is a good proxy, because within a
    third-octave band the spectral shape is dominated by which harmonics of the
    (fixed) probe land there, not by which amp produced them.
    """
    return _amp(build_probe(sr), drive=3.0)


def _errors(drive: float, ir_seed: int, cfg=GUITAR, sr: int = SAMPLE_RATE,
            weighted: bool = True):
    probe = build_probe(sr)
    amped = _amp(probe, drive)
    h = synthetic_cab_ir(sr, seed=ir_seed)
    cabbed = apply_ir(amped, h)

    reference = reference_render(sr) if weighted else None
    measured = ltas_db(cabbed, sr, cfg)
    predicted = ltas_db(amped, sr, cfg) + ir_response_db(h, sr, cfg, reference=reference)

    # The identity fixes shape, not absolute level: FFT scaling conventions and
    # the sum-vs-mean band reduction leave a constant offset. The matcher
    # normalises level away, so compare shapes.
    err = (measured - measured.mean()) - (predicted - predicted.mean())

    centres = band_centers(third_octave_edges(cfg.ltas_lo_hz, cfg.ltas_hi_hz, cfg.n_bands))
    in_band = (centres >= cfg.analysis_lo_hz) & (centres <= cfg.analysis_hi_hz)
    return centres, err, in_band


def test_additive_ltas_holds_in_analysis_band():
    for drive in (1.0, 3.0, 12.0):
        for seed in (0, 1, 2):
            _, err, in_band = _errors(drive, seed)
            worst = np.abs(err[in_band]).max()
            assert worst < MAX_IN_BAND_ERR_DB, (
                f"drive={drive} ir_seed={seed}: worst in-band error {worst:.2f} dB "
                f"exceeds {MAX_IN_BAND_ERR_DB} dB"
            )


def test_additive_ltas_independent_of_drive():
    """The cab is linear, so the error must not grow with amp saturation.

    If it did, the decomposition would be entangled with the nonlinearity and
    taking scalar features from the amp render alone would be unsound.

    Deliberately tested on the *unweighted* path. Reference weighting improves
    accuracy but ties it to one drive level, which introduces a small genuine
    drive dependence -- a property of that weighting choice, not evidence of
    entanglement. Flat averaging isolates the structural claim.
    """
    for seed in (0, 1, 2):
        worst = []
        for drive in (1.0, 3.0, 12.0):
            _, err, in_band = _errors(drive, seed, weighted=False)
            worst.append(np.abs(err[in_band]).max())
        assert max(worst) - min(worst) < 0.5, f"ir_seed={seed}: error varies with drive: {worst}"


def main() -> int:
    print(f"{'band Hz':>10}  {'error dB':>9}   in-band")
    print("-" * 34)
    centres, err, in_band = _errors(drive=3.0, ir_seed=0)
    for c, e, b in zip(centres, err, in_band):
        print(f"{c:10.0f}  {e:+9.2f}   {'yes' if b else ''}")

    print()
    print(f"{'case':<24}{'flat worst':>12}{'weighted worst':>16}{'weighted rms':>14}")
    print("-" * 66)
    for drive in (1.0, 3.0, 12.0):
        for seed in (0, 1, 2):
            _, flat, m = _errors(drive, seed, weighted=False)
            _, wt, _ = _errors(drive, seed, weighted=True)
            print(
                f"drive={drive:<6} ir_seed={seed:<8}"
                f"{np.abs(flat[m]).max():>10.2f} dB"
                f"{np.abs(wt[m]).max():>14.2f} dB"
                f"{np.sqrt((wt[m] ** 2).mean()):>12.2f} dB"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
