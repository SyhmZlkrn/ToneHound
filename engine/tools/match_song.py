"""Match a song -- from a file or a link -- against the profile library.

The pipeline end to end, with an HTML page at the end so the result can be
*heard* rather than read off a number:

    1. fetch the song (yt-dlp for a link) and take a window out of it
    2. isolate the guitar with htdemucs_6s
    3. embed that stem with MERT
    4. rank the profile index by centred cosine
    5. render YOUR DI through the top matches and write players for all of it

Numbers have been misleading us all along; ears are harder to fool. The page
puts the isolated guitar first on purpose -- if separation failed, nothing below
it can be right, and that is visible in two seconds of listening.

    PATH="$PWD/.cache/bin:$PATH" PYTHONPATH=engine python engine/tools/match_song.py \
        --song "path/to/song.mp3"
    PYTHONPATH=engine python engine/tools/match_song.py --song "https://..." --top 8
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import sys
import time
import warnings
import webbrowser

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tonehound.config import SAMPLE_RATE
from tonehound.embed import EmbedConfig
from tonehound.pipeline import CAVEAT, Pipeline, PipelineConfig
from tonehound.tone_index import scan

warnings.filterwarnings("ignore")
warnings.filterwarnings("default", category=RuntimeWarning,
                        module=r"tonehound\.(tone_index|pipeline)")

ROOT = pathlib.Path(__file__).resolve().parents[2]


def write_mp3(path: pathlib.Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    peak = float(np.abs(x).max()) + 1e-12
    sf.write(path, np.asarray(x, dtype=np.float32) / peak * 0.95, sr, format="MP3")


def player(title: str, filename: str, note: str = "", link: str = "") -> str:
    anchor = (f' <a href="{html.escape(link)}" target="_blank">TONE3000</a>'
              if link else "")
    return f"""
    <div class="row">
      <div class="meta"><div class="title">{html.escape(title)}{anchor}</div>
      <div class="note">{html.escape(note)}</div></div>
      {'<audio controls preload="none" src="' + html.escape(filename) + '"></audio>'
       if filename else '<div class="note">no audio</div>'}
    </div>"""


PAGE = """<!doctype html><meta charset="utf-8"><title>ToneHound - {title}</title>
<style>
 body{{font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;background:#14161a;
      color:#e6e8ec;margin:0;padding:32px;max-width:940px}}
 h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:30px 0 10px;
      color:#9aa3b2;text-transform:uppercase;letter-spacing:.06em}}
 .sub{{color:#8b94a3;margin:0 0 24px}}
 .row{{display:flex;align-items:center;gap:16px;padding:10px 12px;border-radius:8px;
      background:#1c1f26;margin-bottom:8px}}
 .row:nth-child(odd){{background:#191c22}}
 .meta{{flex:1;min-width:0}} .title{{font-weight:600;white-space:nowrap;
      overflow:hidden;text-overflow:ellipsis}}
 .title a{{color:#6ea8fe;font-weight:400;font-size:13px;margin-left:8px}}
 .note{{color:#8b94a3;font-size:13px}}
 audio{{height:34px;flex:0 0 300px}}
 .warn{{background:#2a2118;border-left:3px solid #c9903a;padding:12px 16px;
      border-radius:6px;color:#d9c9a8;margin:20px 0;font-size:14px}}
 .bad{{background:#2a1818;border-left-color:#c95a3a;color:#d9b0a8}}
 table{{border-collapse:collapse;font-size:13px;color:#9aa3b2;margin-top:12px}}
 td{{padding:2px 14px 2px 0}}
</style>
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
{stemwarn}
<div class="warn"><b>How much to trust this.</b> {caveat}</div>
<h2>What the engine heard</h2>
{inputs}
<h2>Top {n} matches &mdash; your DI through each</h2>
{matches}
<h2>Run</h2>
<table>{timings}</table>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True,
                    help="a song file, or a URL to fetch with yt-dlp")
    ap.add_argument("--di", type=pathlib.Path,
                    default=ROOT / "assets" / "user_di" / "Djent DI.wav",
                    help="your DI, rendered through the matched profiles")
    ap.add_argument("--profiles", type=pathlib.Path,
                    default=ROOT / "assets" / "dev_profiles",
                    help="directory of .nam files to rank")
    ap.add_argument("--cache", type=pathlib.Path, default=ROOT / ".cache")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="how much of the song to analyse")
    ap.add_argument("--start", type=float, default=0.0,
                    help="offset into the song, in seconds")
    ap.add_argument("--index-seconds", type=float, default=15.0,
                    help="DI length each profile is rendered through")
    ap.add_argument("--layers", type=str, default="8,9,10,11",
                    help="MERT hidden-state layers to pool. 8-11 measured best; "
                         "the last layers of this genre-tuned checkpoint encode "
                         "what is played, not what it was played through.")
    ap.add_argument("--device", default="auto",
                    help="auto | cuda | cpu | mps -- where MERT runs")
    ap.add_argument("--render-device", default="cpu",
                    help="cpu (default) | auto | cuda[:index] -- opt-in float64 "
                         "GPU rendering for the index and auditions, with CPU fallback")
    ap.add_argument("--fp16", action="store_true",
                    help="half precision for MERT and htdemucs (CUDA only)")
    ap.add_argument("--shifts", type=int, default=None,
                    help="demucs shifts; 1 is about twice as fast as the "
                         "default 2, for slightly rougher separation")
    ap.add_argument("--di-seconds", type=float, default=8.0)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "docs" / "match_song")
    ap.add_argument("--json", type=pathlib.Path, default=None,
                    help="also write the result as JSON")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--no-audition", action="store_true",
                    help="skip rendering, just rank")
    args = ap.parse_args()

    entries = scan(args.profiles)
    if not entries:
        print(f"No .nam profiles in {args.profiles}")
        return 1

    cfg = PipelineConfig(
        profile_dir=args.profiles, cache_dir=args.cache, di_path=args.di,
        seconds=args.seconds, start_s=args.start,
        index_seconds=args.index_seconds, top=args.top,
        render_device=args.render_device,
        separator_autocast=args.fp16, separator_shifts=args.shifts,
        embed=EmbedConfig(layers=tuple(int(v) for v in args.layers.split(",")),
                          device=args.device,
                          dtype="float16" if args.fp16 else "float32"))

    last = [""]

    def progress(stage: str, fraction: float, message: str) -> None:
        line = f"  {stage:<11} {message}"
        if line != last[0]:
            print(line, flush=True)
            last[0] = line

    print(f"song     : {args.song}")
    print(f"DI       : {args.di.name}")
    print(f"profiles : {len(entries)} in {args.profiles}\n")

    pipe = Pipeline(cfg)
    devices = pipe.devices
    print(f"device   : separate on {devices['separate']}, "
          f"MERT on {devices['embed']}, NAM render on {devices['render']} (float64)\n")
    started = time.monotonic()
    result = pipe.match(args.song, progress=progress)
    wall = time.monotonic() - started

    print(f"\nstem used: {result.stem.stem}  ({result.stem.note})")
    print(f"index    : {result.index_size} profiles, {result.embed_dim}-d vectors\n")
    for m in result.matches:
        tone = f"  tone {m.entry.tone_id}" if m.entry.tone_id else ""
        print(f"  {m.rank}. {m.similarity:+.4f}  {m.entry.name[:64]}{tone}")

    # -- the page --------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    write_mp3(args.out / "00_song.mp3", result.source.samples)
    write_mp3(args.out / "01_stem.mp3", result.stem.samples)
    di_audio = pipe.di[: int(args.di_seconds * SAMPLE_RATE)]
    write_mp3(args.out / "02_your_di.mp3", di_audio)

    rows = []
    for m in result.matches:
        note = (f"similarity {m.similarity:+.4f} - closer than "
                f"{m.percentile:.0%} of the index")
        filename = ""
        if not args.no_audition:
            print(f"  rendering {m.rank}/{len(result.matches)}: {m.entry.name[:50]}")
            try:
                rendered = pipe.audition(m, di_audio, seconds=args.di_seconds)
                filename = f"match_{m.rank}.mp3"
                write_mp3(args.out / filename, rendered)
            except Exception as exc:
                note += f" - render failed: {exc}"
        rows.append(player(f"{m.rank}. {m.entry.name}", filename, note,
                           m.entry.tone3000_url or ""))

    inputs = (
        player("Song", "00_song.mp3",
               f"{result.source.duration_s:.0f}s from {result.source.origin}")
        + player(f"Isolated: {result.stem.stem}", "01_stem.mp3",
                 "what the matcher analysed - listen to this first")
        + player("Your DI, dry", "02_your_di.mp3", "no amp, for reference"))

    stemwarn = ""
    if not result.separation_ok:
        stemwarn = (f'<div class="warn bad"><b>Separation fell back.</b> '
                    f'{html.escape(result.stem.note)}</div>')

    timings = "".join(f"<tr><td>{k}</td><td>{v:.1f}s</td></tr>"
                      for k, v in {**result.timings, "total": wall}.items())
    subtitle = result.source.name + (f" &mdash; {result.source.url}"
                                     if result.source.url else "")

    page = args.out / "index.html"
    page.write_text(PAGE.format(
        title=html.escape(result.source.name[:70]), subtitle=subtitle,
        caveat=html.escape(CAVEAT), inputs=inputs, matches="".join(rows),
        n=len(result.matches), stemwarn=stemwarn, timings=timings),
        encoding="utf-8")
    print(f"\nwrote {page}  ({wall:.1f}s total)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_json(), indent=2), "utf-8")
        print(f"wrote {args.json}")

    if not args.no_open:
        webbrowser.open(page.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
