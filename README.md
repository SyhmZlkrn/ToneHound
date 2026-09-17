# ToneHound

A native Windows guitar app and VST3 that finds playable Neural Amp Modeler
captures resembling the guitar sound in a selected song passage.

Import a WAV/MP3 or supported song link, select a passage of at least 30 seconds,
and audition the closest captures with your own guitar. ToneHound uses Demucs
for guitar separation, MERT for audio descriptions, and similarity search over
rendered NAM captures. Names and artist titles do not determine the ranking.

The desktop editor and live guitar processing use JUCE and the NAM C++ core.
Python performs analysis in separate worker processes. This repository includes
an older browser experiment under `engine/live`; the shipped interface is native.

The current studio controls include a searchable TONE3000 library, ten-second
played-guitar matching with eight editable EQ bands, graphical Screamer/delay/
reverb pedals, automatic solo/rhythm presets and pitch transpose. A persistent
descriptor catalogue and managed NAM cache support larger libraries without
installing every model. See [the studio guide](docs/STUDIO_FEATURES.md) and
[catalogue/MERT design](docs/CATALOGUE_AND_MERT.md).

## Train a larger matching model

Use the [Colab/Kaggle training walkthrough](docs/CLOUD_TRAINING.md) to prepare
full-rig captures with multiple independent DI performances, train a contrastive
MERT head or fine-tune MERT, and evaluate against a frozen baseline. The dataset
and checkpoints stay in private storage. The native app supports an installed
LoRA adapter and head through **Reference > Matching model > LoRA (pilot)**.
The existing Standard matcher remains available. See
[LoRA installation and usage](docs/LORA_MATCHING.md) for setup and pilot limits.

[Open the Colab notebook](https://colab.research.google.com/github/SyhmZlkrn/ToneHound/blob/main/notebooks/ToneHound_Colab.ipynb)

## Run the current development build

Open `dist/ToneHound/ToneHound.exe`. Choose your interface in **I / O**, select
the guitar input, load a `.nam` capture and enable monitoring. The VST3 bundle
uses the DAW's audio routing. See [native usage](native/README.md) and
[how ToneHound works](docs/HOW_TONEHOUND_WORKS.md).

The native bundle is small, but song matching additionally needs Python,
PyTorch, model weights and a local capture library. The current bundle expects
a configured source tree; it is not yet a self-contained installer for another
computer. Live guitar processing itself runs in C++.

## Set up a source checkout

Use Windows x64 and Python 3.10 for the validated development configuration.
Install Git, FFmpeg and Visual Studio 2022 Build Tools with the C++ desktop
workload and Windows SDK. CMake 3.24 or newer is required. Add FFmpeg to PATH.

Inspect prerequisites without changing the computer:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
```

Create an isolated `.venv`, install the recorded Python dependencies, and fetch
the pinned native source dependencies:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap.ps1 -Mode Install -Development
```

The default installs CPU PyTorch. An NVIDIA machine with suitable drivers can
use `-Torch cu128`. Bootstrap never replaces the global Python installation or
the existing `.cache/native/runtime.json`. When migrating an existing checkout,
set that file's `python` field to `.venv/Scripts/python.exe` using an absolute
path if you want the app to use the new environment. A new checkout receives
that setting automatically.

Native commits are recorded in `release/native_dependencies.json`; the script
fetches those revisions and their submodules into ignored `native/vendor`.
Python direct dependencies are pinned in `requirements-engine.txt`; transitive
dependencies still use pip resolution. The clean-environment install remains
to be verified on another computer; this is a reproducible setup recipe, not a
claim that an installer has been validated.

Build with the installed Microsoft compiler:

```powershell
cmake -S native -B native/build-vs -G "Visual Studio 17 2022" -A x64
cmake --build native/build-vs --config Release
.venv/Scripts/python.exe -m pytest
```

The existing `native/build.ps1` is a shortcut for this development computer's
portable toolchain. You do not need to copy that multi-gigabyte toolchain into
the repository. `native/package.ps1` packages its `build-msvc` outputs.

## Connect TONE3000 and prepare your library

Each user connects their own TONE3000 account. Set the environment variable
`TONE3000_PUBLISHABLE_KEY`, or save the key in the ignored local file
`.cache/tone3000/publishable_key.txt`. `.env.example` documents the names but is
not automatically loaded. Then use the account/library controls in the app or
the CLI:

```powershell
.venv/Scripts/python.exe engine/tools/tone3000_cli.py --help
.venv/Scripts/python.exe engine/tools/tone3000_cli.py connect
```

OAuth access and refresh tokens remain in `.cache/tone3000/tokens.json`. Neither
the tokens nor a developer's local key belongs in Git. Publishing source does
not transfer your account connection to other users.

NAM captures, TONE3000 images, user recordings and commercial songs are not
included in the source export. Import your own `.nam` files or download
captures through your account. The index uses your existing dry guitar recording,
a path configured with `TONEHOUND_DI_PATH`, or a deterministic synthetic probe
on a clean install; see [recording a DI](assets/user_di/README.md). The
model weights download on first analysis and then stay cached locally.

## Project structure

| Location | Responsibility |
| --- | --- |
| `native/Source` | Native editor, audio processor, device handling and worker bridge |
| `engine/tonehound` | Decode, separate, embed, index, catalog and analysis jobs |
| `engine/tools` | CLI, diagnostics and reproducible evaluation |
| `engine/tests` | Engine and worker-boundary tests |
| `scripts` | Setup, safe source export and storage audit |
| `release` | Source allowlist and pinned native dependencies |
| `.cache` | Local data only: accounts, models, indexes, downloaded and decoded audio |
| `docs` | User explanation, implementation notes and validation evidence |

Changes to audio features should include signal-based tests, not just GUI
checks. Keep slow analysis and file I/O off the audio callback. Incremental
profile embedding caches allow a larger library without reprocessing every
capture when one is added. See [release and storage notes](docs/RELEASE_AND_STORAGE.md)
for disk measurements, safe cleanup and distribution constraints.

## Prepare source for GitHub

The selected destination is [SyhmZlkrn/ToneHound](https://github.com/SyhmZlkrn/ToneHound).
These commands inspect and create a local source archive; they do not stage,
commit, push or publish:

```powershell
python scripts/export_source.py --check
python scripts/export_source.py
python -m unittest discover -s scripts/tests -v
```

Review `dist/ToneHound-source.zip` and its `SOURCE_INVENTORY.json`. The exporter
uses an allowlist, rejects binary/runtime paths, and screens for known local
credentials and common token forms without printing their values. This is an
additional check; review any new source/configuration before publication.

No project-wide license has been selected. JUCE licensing and the current
MERT checkpoint's noncommercial terms need an explicit release decision; see
the linked upstream notices in [release and storage notes](docs/RELEASE_AND_STORAGE.md).
