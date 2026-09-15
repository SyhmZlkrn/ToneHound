"""Connect to TONE3000, search it, and pull `.nam` files into a profile library.

The matcher ranks whatever `.nam` files it is pointed at. This is how that
directory gets filled from TONE3000 instead of by hand.

    PYTHONPATH=engine python engine/tools/tone3000_cli.py connect
    PYTHONPATH=engine python engine/tools/tone3000_cli.py search "5150" --limit 20
    PYTHONPATH=engine python engine/tools/tone3000_cli.py fetch --query "plexi" --limit 30
    PYTHONPATH=engine python engine/tools/tone3000_cli.py status

`connect` opens your browser at TONE3000's sign-in page; you approve the access
there and nothing here ever sees your password. It needs a publishable key from
TONE3000 -> Settings -> API Keys, supplied either as `TONE3000_PUBLISHABLE_KEY`
in the environment or written to `.cache/tone3000/publishable_key.txt`.

`fetch` downloads files, so it asks before pulling more than a handful unless
`--yes` is given. Every file lands as `t3k-<tone>-<model> <name>.nam`, which is
what lets a ranking point back at a Tone ID.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tonehound.tone3000 import KEY_ENV, Tone3000Client, Tone3000Error

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = ROOT / ".cache" / "tone3000" / "profiles"


def client(args: argparse.Namespace) -> Tone3000Client:
    return Tone3000Client(cache_dir=args.cache, redirect_uri=args.redirect_uri)


def cmd_status(args: argparse.Namespace) -> int:
    api = client(args)
    print(f"publishable key : {'set' if api.configured else 'MISSING (' + KEY_ENV + ')'}")
    print(f"redirect uri    : {api.redirect_uri}")
    print(f"connected       : {api.connected}")
    if api.connected:
        try:
            user = api.get_user()
            print(f"account         : {user.get('username', '?')}")
        except Tone3000Error as exc:
            print(f"account         : could not fetch ({exc.message})")
    local = sorted(DEFAULT_PROFILES.glob("*.nam"))
    print(f"downloaded      : {len(local)} profiles in {DEFAULT_PROFILES}")
    return 0


def cmd_connect(args: argparse.Namespace) -> int:
    api = client(args)
    if not api.configured:
        print(f"No publishable key. Set {KEY_ENV}, or write the t3k_pub_ key to\n"
              f"  {api.cache_dir / 'publishable_key.txt'}\n"
              "Get one from tone3000.com -> Settings -> API Keys.")
        return 1
    api.connect(timeout=args.timeout)
    print("Connected.")
    return cmd_status(args)


def cmd_disconnect(args: argparse.Namespace) -> int:
    client(args).disconnect()
    print("Disconnected; stored tokens removed.")
    return 0


def _rows(api: Tone3000Client, args: argparse.Namespace) -> list:
    return list(api.iter_tones(limit=args.limit, query=args.query,
                              sort=args.sort, gears=tuple(args.gears.split(",")),
                              fmt="nam"))


def cmd_search(args: argparse.Namespace) -> int:
    api = client(args)
    tones = _rows(api, args)
    if not tones:
        print("No tones matched.")
        return 0
    for tone in tones:
        makes = ", ".join(tone.makes[:3])
        print(f"  {tone.id:>7}  {tone.title[:48]:<48} {makes[:28]:<28} "
              f"by {tone.creator[:16]:<16} {tone.downloads:>6} dl")
    print(f"\n{len(tones)} tones.")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    api = client(args)
    dest = pathlib.Path(args.dest)
    tones = _rows(api, args)
    if not tones:
        print("No tones matched; nothing to download.")
        return 0

    plan = []
    for tone in tones:
        for model in api.list_models(tone.id, architecture=args.architecture):
            if model.model_url:
                plan.append((tone, model))
                if args.per_tone and len(
                        [p for p in plan if p[0].id == tone.id]) >= args.per_tone:
                    break

    print(f"{len(tones)} tones -> {len(plan)} model files into {dest}")
    if not plan:
        return 0
    if not args.yes:
        reply = input(f"Download {len(plan)} files? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Nothing downloaded.")
            return 1

    ok = 0
    for i, (tone, model) in enumerate(plan, 1):
        try:
            path = api.download_model(model, dest)
            ok += 1
            print(f"  [{i}/{len(plan)}] {path.name}")
        except Tone3000Error as exc:
            print(f"  [{i}/{len(plan)}] {tone.title[:40]}: {exc.message}")
    print(f"\n{ok}/{len(plan)} downloaded into {dest}")
    print(f"Index them with:\n  python engine/tools/match_song.py --profiles \"{dest}\" "
          "--song <song>")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=pathlib.Path, default=ROOT / ".cache")
    ap.add_argument("--redirect-uri", default=None)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("status").set_defaults(fn=cmd_status)
    connect = sub.add_parser("connect")
    connect.add_argument("--timeout", type=float, default=300.0)
    connect.set_defaults(fn=cmd_connect)
    sub.add_parser("disconnect").set_defaults(fn=cmd_disconnect)

    for name, fn in (("search", cmd_search), ("fetch", cmd_fetch)):
        p = sub.add_parser(name)
        p.add_argument("query", nargs="?", default="")
        p.add_argument("--query", dest="query", default=argparse.SUPPRESS)
        p.add_argument("--limit", type=int, default=25)
        p.add_argument("--sort", default="trending")
        p.add_argument("--gears", default="amp,amp-cab")
        if name == "fetch":
            p.add_argument("--dest", type=pathlib.Path, default=DEFAULT_PROFILES)
            p.add_argument("--architecture", type=int, default=None)
            p.add_argument("--per-tone", type=int, default=1,
                           help="cap files per tone; 0 for all")
            p.add_argument("--yes", action="store_true",
                           help="skip the confirmation before downloading")
        p.set_defaults(fn=fn)

    args = ap.parse_args()
    try:
        return args.fn(args)
    except Tone3000Error as exc:
        print(f"TONE3000 error [{exc.code}]: {exc.message}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
