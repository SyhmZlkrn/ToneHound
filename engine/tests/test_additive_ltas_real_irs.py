"""The additive-LTAS check again, against real cab IRs rather than synthetics.

Synthetic IRs are smooth by construction. Real cab captures carry mic-position
comb filtering and cabinet resonances that move far more steeply across a
third-octave band, which is exactly where the approximation is weakest. This is
the honest version of the test.

The IR directory is not part of the repo, so this skips when it is absent. Point
it somewhere else with::

    set TONEHOUND_IR_DIR=C:\\path\\to\\irs
"""

import os
import pathlib

import numpy as np
import pytest

from tonehound.config import GUITAR, SAMPLE_RATE
from tonehound.features import band_centers, ir_response_db, ltas_db, third_octave_edges
from tonehound.irs import apply_ir, load_ir
from tonehound.probe import build_probe

IR_DIR = pathlib.Path(os.environ.get("TONEHOUND_IR_DIR", r"C:\Users\NZXT\Downloads\Zilla Cabs"))

# Real IRs are less forgiving than synthetics, so this budget is looser than the
# synthetic test's 1.5 dB. It is still far inside the +/-12 dB the match-EQ can
# correct, and the matcher re-ranks its top candidates by real rendering anyway.
MAX_IN_BAND_ERR_DB = 2.5


def _ir_paths() -> list[pathlib.Path]:
    return sorted(IR_DIR.glob("*.wav")) if IR_DIR.is_dir() else []


requires_irs = pytest.mark.skipif(not _ir_paths(), reason=f"no IRs found in {IR_DIR}")


def _amp(x: np.ndarray, drive: float) -> np.ndarray:
    return np.tanh(drive * x)


def _measure(h: np.ndarray, amped: np.ndarray, reference, cfg=GUITAR, sr: int = SAMPLE_RATE):
    cabbed = apply_ir(amped, h)
    measured = ltas_db(cabbed, sr, cfg)
    predicted = ltas_db(amped, sr, cfg) + ir_response_db(h, sr, cfg, reference=reference)

    err = (measured - measured.mean()) - (predicted - predicted.mean())
    centres = band_centers(third_octave_edges(cfg.ltas_lo_hz, cfg.ltas_hi_hz, cfg.n_bands))
    in_band = (centres >= cfg.analysis_lo_hz) & (centres <= cfg.analysis_hi_hz)
    return np.abs(err[in_band]).max(), float(np.sqrt((err[in_band] ** 2).mean()))


@requires_irs
def test_additive_ltas_holds_for_real_irs():
    probe = build_probe()
    reference = _amp(probe, 3.0)

    failures = []
    for drive in (1.0, 3.0, 12.0):
        amped = _amp(probe, drive)
        for path in _ir_paths():
            worst, _ = _measure(load_ir(str(path)), amped, reference)
            if worst >= MAX_IN_BAND_ERR_DB:
                failures.append(f"{path.name} @ drive={drive}: {worst:.2f} dB")

    assert not failures, "additive approximation exceeded budget:\n  " + "\n  ".join(failures)


def main() -> int:
    paths = _ir_paths()
    if not paths:
        print(f"no IRs found in {IR_DIR}")
        return 1

    probe = build_probe()
    reference = _amp(probe, 3.0)
    amped = _amp(probe, 3.0)

    print(f"{len(paths)} real IRs from {IR_DIR}\n")
    print(f"{'IR':<38}{'flat worst':>12}{'weighted worst':>16}{'wtd rms':>10}")
    print("-" * 76)

    flat_worst, wtd_worst = [], []
    for path in paths:
        h = load_ir(str(path))
        f_w, _ = _measure(h, amped, None)
        w_w, w_r = _measure(h, amped, reference)
        flat_worst.append(f_w)
        wtd_worst.append(w_w)
        print(f"{path.stem[:37]:<38}{f_w:>9.2f} dB{w_w:>13.2f} dB{w_r:>8.2f} dB")

    print("-" * 76)
    print(f"{'median':<38}{np.median(flat_worst):>9.2f} dB{np.median(wtd_worst):>13.2f} dB")
    print(f"{'worst case':<38}{max(flat_worst):>9.2f} dB{max(wtd_worst):>13.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
