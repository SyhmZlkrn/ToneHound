"""Compare separation fidelity across probes, with a rank test.

sep_quality.py gives one continuous number per profile, so "does htdemucs do
better on real playing?" becomes a two-sample comparison over 34 paired-by-
profile measurements instead of a difference of two top-1 rates that each move
in steps of 0.029.

    PYTHONPATH=engine python engine/tools/sep_quality_compare.py \
        .cache/sq_synth.npz .cache/sq_djent.npz .cache/sq_funk.npz
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy import stats


def main(paths: list[str]) -> int:
    runs = [(pathlib.Path(p).stem.replace("sq_", ""), np.load(p, allow_pickle=True))
            for p in paths]

    print(f"{'probe':<10}{'n':>4}{'SI-SDR med':>12}{'IQR':>18}"
          f"{'LSD med':>10}{'gain vs mix':>13}{'stem lvl':>10}")
    print("-" * 77)
    for name, d in runs:
        s, l, g, v = d["sdr"], d["lsd"], d["gain"], d["level"]
        print(f"{name:<10}{len(s):>4}{np.median(s):>12.2f}"
              f"{f'[{np.percentile(s, 25):+.2f},{np.percentile(s, 75):+.2f}]':>18}"
              f"{np.median(l):>10.2f}{np.median(g):>13.2f}{np.median(v):>10.2f}")

    print(f"\n{'probe':<10}{'>10k mix':>12}{'>10k stem':>12}{'ratio dB':>10}"
          f"{'in-band frac':>14}")
    print("-" * 58)
    for name, d in runs:
        hm, he = np.median(d["himix"]), np.median(d["hiest"])
        print(f"{name:<10}{hm:>12.6f}{he:>12.6f}"
              f"{10 * np.log10((he + 1e-30) / (hm + 1e-30)):>10.1f}"
              f"{np.median(d['inband']):>14.4f}")

    print("\nMann-Whitney U on SI-SDR (one-sided: real DI > synthetic)")
    print("-" * 58)
    base = dict(runs).get("synth")
    if base is not None:
        for name, d in runs:
            if name == "synth":
                continue
            u = stats.mannwhitneyu(d["sdr"], base["sdr"], alternative="greater")
            delta = float(np.median(d["sdr"]) - np.median(base["sdr"]))
            # Rank-biserial: fraction of cross pairs where the real DI wins.
            n1, n2 = len(d["sdr"]), len(base["sdr"])
            print(f"  {name:<8} median +{delta:5.2f} dB   U={u.statistic:8.1f}  "
                  f"p={u.pvalue:.2e}   P(win)={u.statistic / (n1 * n2):.3f}")

    if len(runs) == 3:
        a, b = dict(runs).get("djent"), dict(runs).get("funk")
        if a is not None and b is not None:
            u = stats.mannwhitneyu(b["sdr"], a["sdr"], alternative="two-sided")
            print(f"\n  funk vs djent (two-sided): median "
                  f"{np.median(b['sdr']) - np.median(a['sdr']):+.2f} dB  p={u.pvalue:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
