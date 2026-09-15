"""Validate dropped DI takes before they are used as the probe.

The expensive failure is discovering after a 15-minute separation run that the
takes were amp recordings rather than clean DIs, or that half of each take is
silence. This checks cheaply and says so up front.

    PYTHONPATH=engine python engine/tools/prepare_di.py
"""

import argparse
import pathlib

import numpy as np

from tonehound.config import SAMPLE_RATE
from tonehound.probe import PROBE_LEVELS_DBFS, build_phrase
from tonehound.real_di import inspect, load_di, probe_from_di, warnings_for

ROOT = pathlib.Path(__file__).resolve().parents[2]
DI_DIR = ROOT / "assets" / "user_di"
AUDIO_EXT = {".wav", ".flac", ".aiff", ".aif", ".mp3", ".ogg"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=pathlib.Path, default=DI_DIR)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    args = ap.parse_args()

    takes = sorted(p for p in args.dir.glob("*") if p.suffix.lower() in AUDIO_EXT)
    if not takes:
        print(f"No audio found in {args.dir}")
        print("Read the README there for what to record, then re-run this.")
        return 1

    print(f"{len(takes)} take(s) in {args.dir}\n")

    # The synthetic probe, for comparison -- a real DI should look broadly like
    # this on the same measures, and clearly unlike a cabinet-filtered signal.
    ref = inspect(build_phrase(SAMPLE_RATE), SAMPLE_RATE)
    print(f"{'take':<34}{'secs':>7}{'peak':>8}{'crest':>8}{'centroid':>10}{'>5kHz':>9}")
    print("-" * 76)
    print(f"{'(synthetic probe, reference)':<34}{ref['seconds']:>7.1f}"
          f"{ref['peak_dbfs']:>8.1f}{ref['crest_db']:>8.1f}"
          f"{ref['centroid_hz']:>10.0f}{ref['high_5k_ratio']:>9.4f}")
    print()

    usable, total_s = [], 0.0
    for path in takes:
        try:
            di = load_di(path, SAMPLE_RATE, max_seconds=args.max_seconds)
        except Exception as exc:
            print(f"{path.name[:33]:<34}  FAILED TO LOAD: {type(exc).__name__}: {exc}")
            continue

        st = inspect(di, SAMPLE_RATE)
        print(f"{path.name[:33]:<34}{st['seconds']:>7.1f}{st['peak_dbfs']:>8.1f}"
              f"{st['crest_db']:>8.1f}{st['centroid_hz']:>10.0f}{st['high_5k_ratio']:>9.4f}")
        for w in warnings_for(st):
            print(f"{'':<34}  ! {w}")
        usable.append((path, di))
        total_s += st["seconds"]

    if not usable:
        print("\nNothing usable.")
        return 1

    print(f"\n{len(usable)} usable take(s), {total_s:.1f}s total")

    probe_len = sum(len(di) for _, di in usable) + len(usable) * int(0.25 * SAMPLE_RATE)
    print(f"Each take becomes a {len(PROBE_LEVELS_DBFS)}-level probe "
          f"({PROBE_LEVELS_DBFS} dBFS), mirroring the synthetic one.")
    print(f"Longest single-take probe: "
          f"{len(probe_from_di(max((d for _, d in usable), key=len))) / SAMPLE_RATE:.1f}s")

    print("\nNext: re-render the corpus through these instead of the synthetic probe,")
    print("then re-run the separation tests. Rendering 34 profiles takes a few minutes")
    print("per take, so start with the single best one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
