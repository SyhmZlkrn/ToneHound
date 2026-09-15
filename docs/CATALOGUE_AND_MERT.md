# A larger ToneHound library without hundreds of installed NAM files

ToneHound now separates the **search library** from the **playable model cache**.
The search library holds compact numerical descriptions of captures. A NAM file
is needed to create that description or to play through the capture, but it does
not have to stay installed for every subsequent song comparison.

This update prepares the app for a curated library of about 500 captures. The
actual local library now has **51 real captures**: 42 TONE3000 captures plus nine
verified local additions labelled as cabinet-included full rigs. A separate test
imports and searches **500 synthetic 4,096-number descriptors without any NAM
files**. That establishes storage and retrieval behaviour, not the musical
accuracy or quality of a 500-capture collection.

## What happens when someone matches a song?

1. The app imports a file or supported URL and preserves stereo playback.
2. The user selects at least 30 seconds and chooses Rhythm or Solo.
3. Demucs attempts to isolate the guitar. MERT describes the resulting sound.
4. ToneHound compares that description with the stored capture descriptions.
   This step does not download every NAM in the library.
5. The chosen result's NAM is loaded from the working cache. If missing, the
   worker requests it from TONE3000 using its saved Model ID and the user's login.
6. The worker verifies the model identity, valid NAM structure and full SHA-256
   content hash before handing it to the native amp loader.
7. The user plays through the amp, optionally captures ten seconds for EQ matching,
   and adjusts the eight EQ bands and graphical pedals. Solo applies an editable
   delay/reverb preset using the estimated BPM; Rhythm switches those two off.

There is no artist-name rule that forces a particular amp. A first-place match
means the closest descriptor in the available collection, not proof of the
equipment used to record the song.

## How much storage does 500 need?

The current embedding combines four MERT layers into 4,096 float32 values.
That is 16,384 bytes per capture: **500 raw descriptors occupy 7.8125 MiB**.
Metadata and database overhead add to that; the portable pack is compressed.
The measured real 42-capture `.tonelib` pack is **644,314 bytes**.

| Stored item | Lifetime and purpose |
| --- | --- |
| `.cache/tone3000/catalogue.sqlite` | Persistent IDs, hashes, model/probe signatures and descriptors; retained after NAM eviction |
| `.cache/tone3000/profiles` | Playable NAM files; new managed downloads use a 32 MiB working-cache target |
| `.cache/tone3000/leases` | Small heartbeats protecting the captures selected in running plugin/app instances |
| `.tonelib` export | Portable descriptors and attribution IDs; no NAM files, recordings, credentials or signed download URLs |
| MERT and Demucs weights | Separate, persistent AI dependencies; not reduced by NAM eviction |
| Reference audio and local NAM imports | User/session data; not automatically removed by this cache |

The 32 MiB target applies to **managed NAM downloads**, not the whole app folder.
Active files, files downloaded in the last 30 seconds, and captures still waiting
to be indexed are protected. These can temporarily exceed the target. Older
unused managed files are evicted during indexing or selected-capture resolution.
Concurrent eviction is serialized through SQLite. A crashed instance's lease
expires after two minutes.

The 42 pre-existing captures remain preserved as unmanaged files. User-imported
NAM files are preserved too. New Library downloads and imported descriptor
entries participate in the managed cache. Offline playback works for models
still present; an evicted model needs TONE3000 access again. If the provider has
removed or changed a model, the app reports the failure instead of substituting
another file under its old descriptor. Re-add a changed capture and prepare its
descriptor again.

This does not make the entire AI runtime a few megabytes. The current MERT
checkpoint alone occupies about 1.18 GiB in the existing global model cache;
Python/PyTorch and separation weights are additional dependencies. The source
repository, native binaries and AI runtime are separate deliverables.

## Use the catalogue controls

In **Library**, search TONE3000 and add selected tones as before. Press
**Prepare descriptors** to index new captures. A subsequent match can also do
this preparation. **Export index** saves a `.tonelib` file; **Import index**
adds its descriptors without downloading its NAM collection. Imported entries
stay visible in the amp selector even when their NAM files are absent.
The `.tonelib` format currently carries TONE3000 identities. The nine community
full rigs remain a separate local optional pack, and their multi-DI vectors are
also present in the retrieval artifact; they are not added to a TONE3000 export.

Existing descriptors are retained when the recipient uses a different personal
DI recording. The descriptor records which probe created it; importing a pack
does not silently download and rebuild its whole catalogue. A changed embedding
configuration requires compatible descriptors or a rebuild. For consistent
release comparisons, build the entire pack with the same fixed probe and encoder
configuration. Mixed-probe packs must be rebuilt before export.

Developer commands, from the project root:

```powershell
python engine/tools/catalogue_cli.py status
python engine/tools/catalogue_cli.py build
python engine/tools/catalogue_cli.py export --pack dist/ToneHound-library.tonelib
python engine/tools/catalogue_cli.py import --pack path/to/library.tonelib
python engine/tools/catalogue_cli.py trim
```

TONE3000's [current API terms](https://www.tone3000.com/api/terms) restrict bulk
catalogue downloads and require written permission for preloading or bundling.
The technical capacity to index 500 files does not grant distribution rights.
Use creator-approved local packs, or obtain the required provider agreement
before preparing a preloaded TONE3000 catalogue. This update did not bulk-fetch
500 TONE3000 models.

For an authorised catalogue build, the developer tool accepts a reviewed JSON
array of Tone IDs and checkpoints descriptors as it goes:

```powershell
python engine/tools/catalogue_cli.py build --tone-ids reviewed-tone-ids.json --limit 500 --pack dist/ToneHound-library.tonelib
```

It resumes from completed descriptors. `build --rebuild` explicitly regenerates
them with the current probe and encoder, which may require downloading evicted
NAMs. See the provider's terms before using a batch build.

## More captures and better training solve different problems

Adding 500 captures gives the matcher more sounds to choose from. It does not
train MERT. Several captures can be the same amplifier at different settings;
500 captures are not necessarily 500 distinct amplifiers. A library dominated
by similar clean settings can still fail on distorted recordings.

MERT is pretrained to represent general musical and acoustic information. It is
not a database of the amps used on famous songs. ToneHound currently uses its
frozen features and compares rendered-capture vectors; it does not update MERT's
weights when someone adds a NAM or imports a song. See the primary
[MERT paper](https://arxiv.org/abs/2306.00107).

Multi-DI comparison training is now implemented. Read
[the training guide](MULTI_DI_TRAINING.md) for the measured controlled result,
DI and song splits, htdemucs remix evaluation, exact training method and commands.
MERT and htdemucs remain frozen; the fitted retrieval projection is separate.
Further improvements should follow these principles:

1. **Curate for coverage.** Include clean, edge-of-breakup, crunch and high-gain
   captures, with multiple amp families and clearly identified amp-only/full-rig
   chains. Verify levels, usable output, sample rates and metadata. Keep separate
   settings from becoming accidental duplicates.
2. **Vary the playing.** Render several independently recorded dry performances
   through each capture: sustained notes, chords, palm-muted riffs and lead lines,
   across guitars and tunings. A single DI cannot tell the model which features
   belong to the player and which belong to the amp.
3. **Teach similarity above frozen MERT first.** The current supervised projection
   reduces feature directions that vary between performances of one capture.
   Contrastive fine-tuning is another experiment to evaluate. Hard examples
   include similar amp families with different gain settings, rather than only
   obvious clean-versus-distorted pairs.
4. **Make practice examples resemble records.** Add controlled cabinet/microphone
   changes, mix EQ, compression, backing instruments, lossy encoding and room
   effects. Run examples through the same separation pipeline used on real songs.
   Track those transformations so the task does not reward accidental artefacts.
5. **Keep an honest examination set.** Hold out entire performances and source
   recordings, and evaluate unseen guitars, capture sessions and amp units where
   relevant. Do not split neighbouring chunks of the same take across training
   and testing. Report both exact-capture and amp-family results.
6. **Measure before replacing anything.** Compare frozen MERT, the trained
   projection and simpler spectral baselines. Track Recall@5/10, median rank and
   clean/distorted mistakes, then run blind listening comparisons. Only consider
   tuning MERT's later layers if the smaller model has a credible benefit and
   the held-out results justify the extra training and release complexity.

A projection model can remain separate from the encoder and be versioned with
its descriptor pack. Every changed feature transform needs an explicit signature
so old and new vectors cannot be mixed. Training artifacts now record the encoder
signature, NAM hashes, DI split provenance, projection and validation/test result.
The catalogue's base descriptors remain usable when the optional artifact is
absent or incompatible.

## Maintenance boundaries

`catalogue.py` owns persistent identity, descriptors, imports and cache lifetime.
`tone_index.py` owns rendering/embedding and similarity search. `pipeline.py`
coordinates analysis, while `native_bridge.py` exposes discrete worker actions.
The JUCE processor only resolves missing models on its loader worker; network,
SQLite and Python do not enter the live audio callback. Pitch, NAM, EQ and pedals
run in C++ with prepared buffers.

See [the workflow guide](HOW_TONEHOUND_WORKS.md),
[studio controls](STUDIO_FEATURES.md) and
[release/storage notes](RELEASE_AND_STORAGE.md) for the complete user flow and
distribution boundaries.
