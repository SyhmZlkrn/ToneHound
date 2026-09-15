# Release, credentials and storage

ToneHound's development folder is much larger than its native app. The audited
folder contained **9,813,620,043 bytes (9.14 GiB)** on 9 September 2026, before
the current changes. Almost all of that was generated data and build tools.
The existing standalone-plus-VST3 folder was about **11.4 MiB**. This comparison
does not mean the full analysis runtime is 11.4 MiB: Python, PyTorch and MERT
weights are additional dependencies.

The applied cleanup removed **226 files / 4,050,645,472 bytes (3.77 GiB)** of
reproducible experiment output. After the new builds and validation, the
development tree measured **6,380,321,514 bytes (5.94 GiB)** before final
packaging. Remaining compiler tools, intermediate builds and saved reference
audio explain most of that size. These are excluded from the source export;
this update does not claim to shrink the complete AI runtime to a few MB.

## What occupied the drive

These are initial measurements, before any optional cleanup or new builds.
MiB and GiB use powers of 1024.

| Category | Initial size | Purpose |
| --- | ---: | --- |
| Old rendered experiment arrays | 3.36 GiB | Audio from earlier research runs, reproducible from the existing captures and probes |
| Portable build tools and installers | 2.62 GiB | Microsoft compiler/SDK, CMake and an unused compiler attempt |
| Native decoded reference media | 1.63 GiB | Stereo WAVs used by saved reference sessions |
| Stereo regression diagnostics | 0.41 GiB | Original/stem WAVs and embeddings; compact reports are separate JSON |
| Current native build products | 0.62 GiB | Object files and intermediate libraries |
| Native dependency checkouts | 0.15 GiB | Pinned JUCE/NAM source and their Git metadata |
| Original downloaded songs | 93 MiB | Source files for re-importing references |
| FFmpeg executable | 84 MiB | Audio decoding |
| Local development recordings/assets | 72 MiB | Personal DI takes and test songs |
| Separation weights | 52 MiB | Local htdemucs_6s weight file |
| TONE3000 captures | 11 MiB | The initial downloaded library |

The cached MERT checkpoint adds roughly **1.18 GiB outside this project**, in
the user's Hugging Face cache. The installed Python/PyTorch environment is also
outside the audited tree. Moving those bytes elsewhere does not make them go
away, so the report distinguishes total runtime needs from repository size.

## A small source repository

The root `.gitignore` excludes runtime caches, account files, personal audio,
NAM captures, weights, build products and downloaded vendor source. The
allowlist in `release/source_manifest.json` defines a tighter clean export:
native/Python source, tests, setup scripts, dependency pins and relevant text
documentation. It excludes historical scratch programs and copied API examples.

Run `python scripts/export_source.py` to create `dist/ToneHound-source.zip`.
`SOURCE_INVENTORY.json` inside records every exported path, size and SHA-256.
The process does not initialize Git, stage files, create a commit or contact
GitHub. A future release should upload built binaries as release artifacts,
while source history contains source and manifests. Vendor dependencies are
retrieved by exact revision using `scripts/bootstrap.ps1`; they are not copied
into every source release.

The user's chosen repository is
[SyhmZlkrn/ToneHound](https://github.com/SyhmZlkrn/ToneHound). Publication is
explicitly deferred by the current instruction. No automated publish step is
included in these scripts.

## Will the API key stay the same?

Uploading source does not rotate your TONE3000 key or change your local login.
The app currently reads a **publishable application key** from
`TONE3000_PUBLISHABLE_KEY` or `.cache/tone3000/publishable_key.txt`. It stores the
user-specific OAuth **access token**, **refresh token** and expiry in
`.cache/tone3000/tokens.json`. These have different roles: the key identifies
the application; OAuth tokens authorize that user's account.

The source export includes neither. Each installation signs in to its own
account. Current bootstrap documentation asks users to supply a key rather than
shipping this developer's local configuration. If an application-wide
publishable key is eventually distributed, configure that deliberately through
the provider's application settings; do not accidentally bundle the user's
tokens with it. If a credential has already been exposed publicly, deleting it
from a later commit does not invalidate the exposed value; rotate/revoke it
through the provider. This audit did not find or assume a prior exposure.

Credential screening reads only known local credential files to compare their
values against selected export text. Reports contain paths and rule names,
never credential values. `.env.example` is an empty configuration example;
ToneHound does not automatically read `.env` files.

## Reclaim reproducible experiments safely

Run this read-only audit:

```powershell
python scripts/storage_audit.py --manifest .cache/storage-prune-dry-run.json
```

It identifies only `.npy` and `.wav` files in seven fixed experiment folders:
`renders_djent`, `renders_funk`, `renders_thall`, `renders_v2`, `renders_fixed`,
`reference_diagnostics` and `reference_diagnostics_controls`, all under `.cache`.
The initial candidate set was **226 files / 4,050,645,472 bytes (3.77 GiB)**.
Full absolute paths, sizes and modification times are listed in the manifest.

When those reproducible experiments are no longer needed, the explicit command is:

```powershell
python scripts/storage_audit.py --prune-experiments --manifest .cache/storage-prune-applied.json
```

The tool rechecks resolved path boundaries and unchanged file metadata before
each removal. It never recursively deletes a folder. It preserves **all JSON
evidence, saved native reference media, original downloads, captures, user DI,
credentials, current embedding caches, model weights and build tools**. Running
an old experiment again may regenerate its arrays; compact results remain
available. Do not run cleanup while a diagnostic experiment is writing into
those folders. The normal app does not use those historical render arrays.

Saved reference audio is deliberately retained: removing it would break saved
song paths. A future reference-library removal control should show which
sessions depend on a file. The current tool therefore does not promise a
global cache size cap and does not remove tools or unused compiler attempts.

## Keep growth manageable

The new persistent SQLite catalogue separates descriptions from playable NAM
files. Its current 42 descriptors total 688,128 raw bytes; a portable pack is
644,314 bytes. Five hundred current-format descriptors need 7.8125 MiB before
metadata. The native Library tab can prepare/import/export these packs without
downloading all their NAMs. Existing model weights and reference audio remain
separate from this storage saving.

New managed NAM downloads use a **32 MiB working-cache target**. Eviction keeps
descriptors and removes only older unused managed captures. Active plugin
instances publish expiring leases; recent downloads and unindexed models are
protected. Those protections may temporarily exceed the target. The pre-existing
42 captures and local user imports remain unmanaged and preserved. This is not
a global disk quota. Selected evicted NAMs require the user's TONE3000 login and
network access to fetch again; identity/content hashes are checked before use.

SQLite checkpoints completed descriptors, so interrupted builds can resume.
Imported packs retain their source probe until an explicit rebuild. See
[CATALOGUE_AND_MERT.md](CATALOGUE_AND_MERT.md) for the curated 500-entry build
command, the actual-versus-synthetic test distinction and proposed training.
Temporary separator outputs are removed after they are read. Keep full renders
only when running an explicit diagnostic.

Maintain these boundaries as the app grows:

1. The native audio processor handles time-critical guitar DSP, without network
   calls, Python inference or disk writes in its callback.
2. Python analysis jobs exchange explicit JSON requests/results with the native
   bridge. Catalog growth, separation and MERT remain independently testable.
3. Stable capture IDs and content hashes connect downloads, embeddings and
   metadata. UI text is not the identity of a capture.
4. User data, reconstructible caches, source and release artifacts remain
   separate. No training set or personal account data belongs in source history.

`requirements-engine.txt` records the tested direct package versions;
`requirements-dev.txt` adds parity-test dependencies. The bootstrap defaults to
a dedicated virtual environment and CPU PyTorch, with an explicit CUDA option.
It preserves an existing global environment. Full transitive wheel locking,
clean-machine installer validation and automated artifact signing are future
release work, not completed by merely making a source ZIP.

## Actual dependency distribution constraints

The pinned **JUCE 8 modules are dual-licensed under AGPLv3 or JUCE's commercial
license**. A proprietary release and an AGPL release take different licensing
paths; this project has not selected one automatically.
[Pinned JUCE license](https://github.com/juce-framework/JUCE/blob/91ad83ae34a81e0833b1a2b0866f54846370ae53/LICENSE.md)

The current **MERT-v1-330M-finetuned-gtzan checkpoint is marked CC-BY-NC-4.0**.
Do not treat the present model choice as cleared for a commercial product.
[Current checkpoint model card](https://huggingface.co/dzur658/MERT-v1-330M-finetuned-gtzan)

The **NAM core and AudioDSPTools use MIT notices**. Their dependency notices,
including Eigen, still travel with a distributed build. JUCE also includes
the VST3 SDK (MIT in this pinned tree) and ASIO headers with separate Steinberg
terms. The native package already carries third-party notice files; retaining
those files alone is not a substitute for selecting the applicable license.
[NAM core license](https://github.com/sdatkinson/NeuralAmpModelerCore/blob/2563c0fd4cb1f9ce457d89a761738ea15097e1f3/LICENSE)

TONE3000 capture files/photos and user/imported songs are deliberately excluded
from the source export and binary bundle. Keep their source-page attribution
and obtain redistribution rights separately if a later installer bundles them.

Pitch uses the pinned Signalsmith Stretch and Signalsmith Linear libraries.
Both MIT notices are included in the native package. The live guitar processor
contains both pitch modes; no pitch model download is needed.
[Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch)
