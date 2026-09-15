# Studio update validation — 16 September 2026

## Multi-DI and Pitchproof-inspired controls

The 51-capture library uses the active frozen-MERT retrieval projection described
in [MULTI_DI_TRAINING.md](MULTI_DI_TRAINING.md). All seven local songs were used as
backing sources in controlled htdemucs remixes, with disjoint DI/song splits.
The separation-trained candidate did not beat the active model and was retained
without activation. Exact metrics and provenance are in
[retrieval_evaluation.json](retrieval_evaluation.json).

Native pitch tests passed detuning, 0/50/100% wet blend, exact zero transpose,
oversized blocks, low triads at 44.1/48/96 kHz, and attack preservation. The burst
test's pre-attack energy fell from 0.0471961 to zero while preserving an audible
shifted note. Fine tuning produced a shifted voice near the expected frequency
(within a 1 Hz search window), without a dry voice at 100% wet. The former
single-frequency amplitude check understated an off-bin sine; the revised test
measures the spectral peak and still checks dry/shifted voice isolation.

All nine added cabinet-included NAMs passed real native playback at three sample
rates. Existing EQ fitting/capture/state/pedal tests still pass. Pitch control
attachments, percentage formatting and immediate preset save/restore pass.
Pluginval strictness 5 passed after the final pitch build. UI inspection covered
the existing 1440 and 1280 layouts and the new pitch callout, with one value-format
correction and confirmation.

The final Python engine/release suite passed **428 tests**, with **40 skips** for
the missing optional historical corpus, in 168.02 seconds. A focused integration
run also passed 55 tests. Existing NumPy/Torch deprecation warnings remain.

The native worker imported the supplied remaster in stereo and ranked 51 trained
references at 0:22–1:00. It now selects a Peavey 5150 + Mesa 4x12 full rig first;
this does not establish the original recording's amplifier identity. The previous
JCM800-first result below is historical. Seven original `song_train` clips also
completed separation and ranking; they have no verified capture labels.

Raw logs are under `.cache/native`: `pitch-learning-dsp-final.txt`,
`pitch-ui-final.txt`, `community-all-native.txt`, `pluginval-pitch-final.txt`,
`learning-suite-final.txt`, `learning-integration-final.txt`, and
`learned-metallica-workflow.txt`. No user original audio or credentials were
included in the release/source archives, and no commit/push was made.

## Latest catalogue, pitch and eight-band update

The standalone and VST3 were rebuilt with the managed descriptor catalogue,
graphical stompboxes, eight editable EQ bands, amp-change EQ reset and header
pitch control. No commit or push was made.

| Check | Observed result |
| --- | --- |
| Full Python engine/release suite | 422 passed, 40 optional-corpus skips; 183.33 seconds |
| Final focused catalogue/index/pipeline/release checks | 61 passed after final path validation, atomic pack export and concurrent probe-file fixes |
| Synthetic 500-capture scale check | 500 × 4,096 float32 descriptors imported and searched without NAM downloads; correct self-query result |
| Cache safety | Unmanaged, active and unindexed files retained; concurrent eviction idempotent; invalid packs rejected before database writes |
| Real catalogue | 42 real captures indexed; 688,128 raw descriptor bytes; exported pack 644,314 bytes |
| Real missing-NAM recovery | Isolated empty cache imported the real pack; TONE3000 model 56 downloaded (283,524 bytes), identity/hash verified, then evicted with its descriptor retained |
| Native selector | Evicted catalogue entry remains selectable alongside the current local file, without downloading the whole library |
| Live pitch | Zero shift preserves exact dry samples; ±12 semitones on a 440 Hz test signal produces approximately 220/880 Hz; 1,920 samples added latency at 48 kHz |
| Studio pitch | All three notes of a low synthetic triad retained at 44.1/48/96 kHz; 8,467/9,216/18,432 latency samples (about 192 ms) |
| Eight-band EQ | Known spectral mismatch reduced from 2.1677 to 1.11257 dB RMS; automatic total response bounded to about ±6 dB |
| Real DI/EQ integration | Ten-second post-NAM capture, automatic application, manual 250 Hz adjustment, immediate host save and editor reopen all passed |
| Amp changes | All eight matched bands and Bass/Mid/Treble return to zero; correction bypassed |
| Native NAM and pitch reporting | Finite stereo audio at 44.1/48/96 kHz and varied block sizes; host receives latency changes for both pitch modes and zero bypass |
| Native reference workflow | Supplied remaster at 0:22–1:00: stereo import/playback, crop bounds, matching, capture load and automatic solo effects passed |
| VST3 | Final pluginval strictness 5 validation passed |
| Visual check | Batched native inspection at 1440×960 and minimum 1280 width, followed by one confirmation pass |

Pitch quality has an explicit trade-off. The short Live window smears closely
spaced low notes in the dense-chord diagnostic. Studio uses a longer window and
passes that test, with substantially more delay. Its approximately 192 ms
latency is intended for recorded guitar, not responsive monitoring. These
numerical tests do not establish parity with Neural DSP or substitute for
musician listening tests across pickups, playing styles and tunings.

No MERT weights were trained or fine-tuned. No curated 500-real-capture library
was created. The 500-entry test verifies storage and retrieval mechanics;
accuracy remains limited by corpus coverage and the existing representation.
The remaster still ranks JCM800 first and the Mark IIC+ full rig second; the new
storage design does not change that result into historical amp identification.

Raw current logs are in `.cache/native/`: `test-suite-catalogue.txt`,
`catalogue-focused-final.txt`, `catalogue-live-verify.txt`,
`catalogue-selector.txt`, `studio-dsp-final.txt`, `studio-eq-final.txt`,
`native-final.txt`, `catalogue-workflow.txt` and `pluginval-catalogue-final.txt`.
Screenshots use `catalogue-final*` and `catalogue-min-final*`. Test fixtures,
images, logs, credentials, recordings and NAM files are excluded from source.

## Previous studio baseline — 10 September 2026

The delivered native build includes the new amp-details frame, larger controls,
EQ matching, pedals, solo/rhythm selection and a paged TONE3000 library.
Changes remain local. No Git commit, push or repository publication was made.

## Checks completed

| Check | Result |
| --- | --- |
| Python engine and release-tool test command | 411 passed, 40 skipped; 89.37 seconds |
| Portable protocol and release tools, focused run | 22 passed; synthetic fixtures replace private recordings for protocol checks |
| Native DSP validator | Passed; known EQ spectral error fell from 1.81852 to 0.794292 dB RMS |
| Ten-second processor EQ integration | Passed using the user's 44.1 kHz DI through a real NAM |
| Saved host state | Matched EQ, manually changed delay mix/BPM, and enabled pedals survive reopening the editor |
| Native NAM audio | Finite, audible stereo output at 44.1/48/96 kHz and variable block sizes; monitoring off mutes guitar |
| New Mark IIC+ profiles | Both passed the native audio/state validator at all three sample rates |
| VST3 host validation | pluginval strictness 5 passed |
| Real native song workflow | Import, stereo L/R playback, crop bounds, play/stop, matching and automatic solo effects passed |
| Native UI | Four tabs inspected at 1440×960; minimum 1280-pixel width checked and clipped card footers fixed |
| Source screening | Allowlisted UTF-8 source passes credential, binary and size screening |
| Bootstrap | Read-only prerequisite/pinned-dependency check passed; a clean-machine installation was not attempted |

The 40 Python skips reflect the absent optional `assets/dev_profiles` corpus.
The older 592-test result in historical records used that corpus and is not the
current suite count. No missing-corpus tests were represented as passing.
Existing NumPy and TorchScript deprecation warnings remain.

Raw local reports: `test_suite_studio.txt`, `native_dsp_studio.txt`,
`native_eq_studio.txt`, `native_ui_studio.txt`, and `pluginval-studio.txt` in
`docs/`. Screenshots are under `.cache/native/studio-final*` and
`.cache/native/studio-min-final*`; local logs/images are excluded from Git export.

## Live catalogue and the Metallica correction

Initial API calls returned HTTP 403. Subsequent authenticated search, model
listing and downloads succeeded without changing the stored credentials.
Two creator-labelled Mark IIC+ amp/cab captures were downloaded and validated:

| TONE3000 tone | Model ID | Capture |
| --- | --- | --- |
| 51344 | 287284 | Mesa Boogie Mark IIC+ Full Rig |
| 47644 | 261032 | doubleboosted Mark IIC+ |

The library grew from 40 to 42. A repeated fetch added zero files, skipped two,
and reported zero errors. A real native match reported `index_size: 42`.
The new index preparation took 17.31 seconds on this computer; these figures
are observations, not a promise for another machine or a much larger library.

For the supplied remastered Master of Puppets recording at 0:22–1:00, the
unchanged pipeline returned JCM800 first (0.7929) and the Mark IIC+ full rig
second (0.7861). The user correctly identified Mark IIC+ as the relevant
recording amp. A similar-sounding JCM800 capture is not a correct equipment
identification. The interface now says **Closest library captures** and explains
that distinction after matching. No song-name override was introduced.

The passage tempo estimate was 105.5 BPM. Half/double-time ambiguity remains;
the user can change it. Automatic solo effects are editable starting settings,
not a reconstruction of the album's exact delay/reverb setup.

## Distribution limits

The native app/VST3 package still needs the configured Python/model environment
for song matching. Live guitar and effects run locally in C++. The source ZIP
excludes keys, tokens, user audio, captures, weights, vendor checkouts and build
tools. See `RELEASE_AND_STORAGE.md` for footprint, setup and dependency licences.

Matching accuracy, clean-machine installation, transitive Python wheel locking
and a signed installer remain further release work. This update does not
establish reliable identification of the actual amplifier used in an arbitrary
mixed/mastered recording.
