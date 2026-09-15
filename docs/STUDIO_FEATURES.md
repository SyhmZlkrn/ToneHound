# ToneHound: library, EQ matching and pedals

This update keeps the native standalone and VST3 workflow. The Amplifier,
EQ match, Pedals and Library tabs share the same reference passage and guitar
signal. No browser is used for the interface; TONE3000 account sign-in opens
the provider's authentication page when you press Connect account.

## Find and add more captures

Open **Library**, connect your own TONE3000 account, and search by amp, maker
or capture name. Browse the results with Previous and Next. Select a tone and
press **Add selected tone** to download up to three of its captures. Existing
models are skipped, and invalid downloads are rejected before entering the
library. Progress, cancellation and download errors appear in the panel.

The matcher has no 40-profile ceiling. Forty was the initial downloaded set.
Every compatible indexed catalogue entry participates in the next match,
including captures whose NAM has been evicted from the working cache. New
captures are rendered and indexed incrementally; an unchanged descriptor is
reused. Searching the website does not
download its entire catalogue or create an index of every result.

The local library now contains 42 captures after a verified TONE3000 download
of two Mark IIC+ amp/cab captures. Repeating that download skipped both existing
files. You can continue adding tones using the same Library controls.

Results are the closest sounds in the downloaded library, not proof of the
recording's original equipment. In the supplied Master of Puppets remaster
passage, a Mark IIC+ capture now ranks second while JCM800 remains first. The
original amp family being known does not force the audio matcher to select it.

For a new checkout, the index can use a deterministic generated guitar probe.
An existing `assets/user_di/Djent DI.wav` remains preferred when present.
Set `TONEHOUND_DI_PATH` to choose a different dry guitar recording. Changing
that recording does not invalidate imported/persisted catalogue descriptors.
An explicit catalogue rebuild uses the new probe throughout.

Use **Prepare descriptors**, **Import index** and **Export index** to maintain a
portable `.tonelib` collection. Comparison uses its small descriptors; only
selected or newly indexed NAMs need downloading. New managed captures have a
32 MiB working-cache target, with protection for active/unindexed/recent files.
See [CATALOGUE_AND_MERT.md](CATALOGUE_AND_MERT.md) for the 500-capture design,
actual measurements, cache behaviour and proposed MERT training programme.

The amp information frame shows available creator-supplied make/model data,
description, creator and source link beside the uncropped photo. Missing or
unavailable details are labelled; the app does not invent the recording rig.

## Match EQ using your guitar

1. Import a song, choose a passage of at least 30 seconds, and find matching tones.
2. Select a capture you like and enable guitar monitoring. Keep amp bypass off.
3. Open **EQ match** and press **Play & capture 10 seconds**.
4. Play varied notes or chords in the style and tuning of the reference passage.
5. After ten seconds, the fitted correction is applied. Use **EQ on / EQ bypassed**
   to compare, or **Reset curve** to remove it.

The target is the separated reference stem from the match. Your measured signal
is taken after the selected amp, Screamer when enabled, and the manual tone
controls. Reference playback, delay, reverb and output volume are excluded from
the application's measurement path. A hardware loopback or microphone can
still capture sound physically, so use the guitar input on your interface.

ToneHound compares average frequency balance after removing the overall level
difference. It fits an eight-band correction, limits its automatic total response
to approximately ±6 dB, and fades changes in. Edit the faders at **63, 125, 250,
500 Hz, 1, 2, 4 and 8 kHz**, or type exact gains, up to ±12 dB per band. Manual
adjustments can exceed the automatic fit's total limit. Double-click a fader to
return it to zero. Silence and insufficient frequency
coverage are rejected. FFT analysis and fitting run on a worker thread; the
audio callback records into a prepared buffer and applies prepared filters.

Changing the capture resets all eight bands **and Bass/Mid/Treble to zero**, and
bypasses the correction. Changing the reference or crop
invalidates the old EQ target; run matching again for the new passage. Matched
EQ and pedal parameters are saved with the plugin state. This is spectral
matching, not a new NAM capture or a reconstruction of an amp's physical knobs.
Playing, pickups and separation quality still affect the result.

## Pitch transpose

The header's **Pitch** control shifts the live guitar from **−12 to +12 semitones**
before Screamer and NAM. The reference song is unaffected. Zero semitones **and
zero fine-tuning cents** passes the dry samples unchanged and adds no pitch latency.

The quality selector displays the trade-off: **Live: 40 ms** responds faster but
can smear closely spaced low notes; **Studio: 192 ms** uses a longer analysis
window for clearer chords and is intended for recorded guitar. Studio's delay
is unsuitable for responsive live monitoring. These are additional processing
delays, before the interface/DAW buffer and any NAM resampling. The plugin reports
its active latency to the host. **Pitch settings** opens quality, wet/dry blend,
fine tuning (−50 to +50 cents), and **Preserve pick attacks**. All are saved and
automatable. A 100% wet blend transposes the guitar; lower values mix in a
latency-aligned dry voice for harmony. This is a fixed interval, not key-aware
harmonisation. Attack preservation shapes the shifted envelope without adding
unshifted audio; disable it if sustained chords pump.

The controls take inspiration from [Pitchproof](https://aegeanmusic.com/pitchproof-specs).
ToneHound continues to use its own Signalsmith-based processing, not Pitchproof
or Neural DSP code. Pitchproof itself documents a preference for monophonic
signals. ToneHound retains its separate Studio mode for clearer low chords.

## Rhythm, solo and tempo

Choose the **Passage type** before matching. **Solo** automatically applies
a conservative delay and reverb preset after matching. **Rhythm** switches
those two effects off for a dry starting sound. Screamer remains a manual choice.

Tempo is estimated from the selected reference audio. It can be detected at
half or double the intended beat, especially with fast riffs. The UI labels
it as an estimate; adjust **Tempo (BPM)** in Pedals if necessary. If a reliable
tempo is unavailable, the solo preset uses 120 BPM as an editable fallback.
The software does not claim to recover the original song's exact effect settings.

The Pedals tab has three coloured stompboxes with rotary knobs, editable values,
metal footswitches and an on/bypassed indicator. Click a footswitch to enable the
effect; keyboard focus and activation are also supported. It contains:

| Effect | Position | Controls |
| --- | --- | --- |
| Screamer | Before NAM | Enable, Drive, Tone, Level |
| Delay | After matched EQ | Enable, Mix, Feedback, Tempo, note division |
| Reverb | After delay | Enable, Mix, Size |

Screamer is a native mid-focused soft-clipping overdrive. Delay supports quarter,
dotted eighth, eighth and triplet eighth notes, with stereo echoes. Reverb adds
a stereo decay. Each effect can be bypassed and adjusted independently. The
reference song continues to play in stereo through its separate playback path.

## Release and storage

The source export excludes local credentials, recordings, model weights, NAM
captures, build tools and caches. The standalone/VST3 package is separate from
the Python/model runtime needed for matching. See **RELEASE_AND_STORAGE.md**
for the measured footprint, source export, dependency setup and license notes.

The September 2026 cleanup removed 226 reproducible experiment files totalling
4,050,645,472 bytes (3.77 GiB). Original audio, saved reference media, profiles,
tokens, weights and JSON evidence were retained. Identical separated stems now
reuse a content-addressed cache entry. Saved reference media is retained to
avoid breaking DAW sessions; there is no automatic global storage cap yet.

GitHub source: [SyhmZlkrn/ToneHound](https://github.com/SyhmZlkrn/ToneHound).
Only the screened source is published; local accounts and training assets remain
outside Git. Publication uses the repository owner's Git identity.
