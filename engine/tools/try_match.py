"""Listen to what the matcher does with a real song snippet.

Point it at a song, and it will:
  1. pull the guitar out of the mix with UVR,
  2. fingerprint that stem and rank every profile in the index,
  3. render YOUR DI through the top matches,
  4. write an HTML page with audio players and open it.

The point is to hear the separated stem and the candidate tones side by side.
Numbers have been misleading us all session; ears are harder to fool.

Set expectations before listening: measured honestly (query = a different
performance from the index), top-1 is about 0.176 on the dev corpus, and every
"same amp" pair in that corpus shares a capturer, so the ranking may partly be
sorting by who made the capture. Treat a good match as encouraging and a bad one
as expected, not as evidence either way.

    PATH="$PWD/.cache/bin:$PATH" PYTHONPATH=engine python engine/tools/try_match.py \
        --song "path/to/snippet.mp3"
"""

import argparse
import html
import pathlib
import sys
import tempfile
import warnings
import webbrowser

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tonehound import nam_render
from tonehound.config import GUITAR, SAMPLE_RATE
from tonehound.features import fingerprint
from tonehound.real_di import load_di
from stem_test import self_retrieval, vectors, zscore_ref  # noqa: F401  (vectors, zscore_ref)
from stem_test_uvr import GUITAR_MODEL, separate

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "assets" / "dev_profiles"


def load_audio(path: pathlib.Path, sr: int, seconds: float | None) -> np.ndarray:
    x, file_sr = sf.read(str(path), always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        x = signal.resample_poly(x, sr, file_sr)
    if seconds:
        x = x[: int(seconds * sr)]
    peak = np.abs(x).max()
    return x / peak * 0.98 if peak > 0 else x


def write_mp3(path: pathlib.Path, x: np.ndarray, sr: int) -> None:
    peak = np.abs(x).max() + 1e-12
    sf.write(path, x / peak * 0.95, sr, format="MP3")


def player(title: str, filename: str, note: str = "") -> str:
    return f"""
    <div class="row">
      <div class="meta"><div class="title">{html.escape(title)}</div>
      <div class="note">{html.escape(note)}</div></div>
      <audio controls preload="none" src="{html.escape(filename)}"></audio>
    </div>"""


PAGE = """<!doctype html><meta charset="utf-8"><title>ToneHound - match preview</title>
<style>
 body{{font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;background:#14161a;
      color:#e6e8ec;margin:0;padding:32px;max-width:900px}}
 h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:30px 0 10px;
      color:#9aa3b2;text-transform:uppercase;letter-spacing:.06em}}
 .sub{{color:#8b94a3;margin:0 0 24px}}
 .row{{display:flex;align-items:center;gap:16px;padding:10px 12px;border-radius:8px;
      background:#1c1f26;margin-bottom:8px}}
 .row:nth-child(odd){{background:#191c22}}
 .meta{{flex:1;min-width:0}} .title{{font-weight:600;white-space:nowrap;
      overflow:hidden;text-overflow:ellipsis}}
 .note{{color:#8b94a3;font-size:13px}}
 audio{{height:34px;flex:0 0 300px}}
 .warn{{background:#2a2118;border-left:3px solid #c9903a;padding:12px 16px;
      border-radius:6px;color:#d9c9a8;margin:20px 0;font-size:14px}}
</style>
<h1>ToneHound match preview</h1>
<p class="sub">{song}</p>
<div class="warn"><b>How much to trust this.</b> Measured honestly, top-1 accuracy
is about 0.176 on the dev corpus, and every same-amp pair there shares a capturer,
so the ranking may partly be sorting by who made the capture rather than which amp
it is. A good match is encouraging; a bad one is expected. Listen to the isolated
guitar first &mdash; if that sounds wrong, nothing downstream can be right.</div>
<h2>What the engine hears</h2>
{inputs}
<h2>Top {n} matches &mdash; your DI through each</h2>
{matches}
<p class="sub" style="margin-top:28px">Index: {count} profiles. Distances are cosine
on the whitened fingerprint; lower is closer.</p>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", type=pathlib.Path, required=True, help="song snippet")
    ap.add_argument("--di", type=pathlib.Path,
                    default=ROOT / "assets" / "user_di" / "Djent DI.wav",
                    help="your DI, played through the matched amps")
    ap.add_argument("--cache", type=pathlib.Path, default=ROOT / ".cache" / "renders_djent")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--di-seconds", type=float, default=8.0)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "docs" / "try_match")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    sr, cfg = SAMPLE_RATE, GUITAR
    if not args.song.exists():
        print(f"No such file: {args.song}")
        return 1
    renders = sorted(args.cache.glob("*.npy"))
    if not renders:
        print(f"No index in {args.cache}. Run gate_b.py --cache {args.cache} first.")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"song  : {args.song.name}")
    print(f"DI    : {args.di.name}")
    print(f"index : {len(renders)} profiles\n")

    song = load_audio(args.song, sr, args.seconds)

    # 1. isolate the guitar -------------------------------------------------
    print("separating guitar (first run downloads the model) ...")
    from audio_separator.separator import Separator
    work = pathlib.Path(tempfile.mkdtemp(prefix="tonehound_try_"))
    sep = Separator(log_level=40, output_dir=str(work),
                    model_file_dir=str(ROOT / ".cache" / "uvr_models"),
                    use_soundfile=True)
    sep.load_model(model_filename=GUITAR_MODEL)
    stem = separate(song, sr, sep, work, "song")
    if stem is None:
        print("separator returned no guitar stem; matching the raw mix instead")
        stem = song

    # 2. rank ---------------------------------------------------------------
    print("fingerprinting and ranking ...")
    idx_fps = [fingerprint(np.load(p), sr, cfg) for p in renders]
    ish = np.array([f.shape_vector for f in idx_fps])
    isc = np.array([f.scalar_vector for f in idx_fps])
    ref = (zscore_ref(ish), zscore_ref(isc))
    index = vectors(ish, isc, ref)

    qf = fingerprint(stem, sr, cfg)
    q = vectors(qf.shape_vector[None, :], qf.scalar_vector[None, :], ref)
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    inn = index / (np.linalg.norm(index, axis=1, keepdims=True) + 1e-12)
    dist = (1.0 - qn @ inn.T)[0]
    order = np.argsort(dist)[: args.top]

    # 3. render the user's DI through each match ----------------------------
    di = load_di(args.di, sr, max_seconds=args.di_seconds)
    write_mp3(args.out / "00_song.mp3", song, sr)
    write_mp3(args.out / "01_isolated_guitar.mp3", stem, sr)
    write_mp3(args.out / "02_your_di.mp3", di, sr)

    rows = []
    for rank, i in enumerate(order, 1):
        name = renders[i].stem
        nam_path = PROFILE_DIR / f"{name}.nam"
        print(f"  [{rank}/{len(order)}] rendering {name[:50]}")
        fn = f"match_{rank}.mp3"
        try:
            model = nam_render.load(nam_path)
            write_mp3(args.out / fn, model.render(di), sr)
            rows.append(player(f"{rank}. {name}", fn, f"distance {dist[i]:.4f}"))
        except Exception as exc:
            rows.append(player(f"{rank}. {name}", "",
                               f"distance {dist[i]:.4f} - render failed: {exc}"))

    inputs = (player("Song snippet", "00_song.mp3", "what you gave it")
              + player("Isolated guitar", "01_isolated_guitar.mp3",
                       "what the matcher actually analysed - listen to this first")
              + player("Your DI, dry", "02_your_di.mp3", "no amp, for reference"))

    page = args.out / "index.html"
    page.write_text(PAGE.format(song=html.escape(args.song.name), inputs=inputs,
                                matches="".join(rows), n=len(order),
                                count=len(renders)), encoding="utf-8")
    print(f"\nwrote {page}")
    if not args.no_open:
        webbrowser.open(page.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
