# Native EQ matching and pedals: validation

The native DSP component was compiled with the local MSVC release toolchain and
the existing JUCE static library. Its isolated validator completed successfully
on 10 September 2026. The component tests do not require Python, network access,
an audio interface, or a NAM capture. Full application and VST3 validation are
separate checks.

The reproducible test entry point is `runMatchDSPTests()` in
`native/Source/MatchDSPTests.cpp`; the DSP implementation is in
`native/Source/MatchDSP.{h,cpp}`.

## Measured result

A deterministic broadband guitar-range test signal was filtered by known
positive and negative EQ bands. The matcher compared that reference with the
unfiltered signal. Across the 15 analysis frequencies, the error between the
known target EQ and the applied correction fell from **1.81852 dB RMS to
0.794292 dB RMS**. This verifies an actual reduction in a known spectral
difference; it is not a listening evaluation or a claim of exact song recovery.

The test also verifies the complete fitted response remains within ±6 dB,
including frequencies between the filter centres.

## Passed checks

- An inverted right channel produces the same reference power spectrum as the
  left channel instead of cancelling it.
- Target analysis continues when the selected amp resets its old correction.
- The recording reaches exactly ten seconds of samples and reports 50% progress
  at five seconds.
- A completed recording produces a correction on the worker thread.
- Saved correction gains restore at a different sample rate.
- Changing the reference invalidates its previous correction.
- Silence and a single narrow-band note are rejected as unsuitable measurements.
- Bypassed pedals leave the dry signal unchanged.
- The overdrive changes the signal and produces finite samples.
- At 120 BPM, the quarter-note delay produces the expected half-second echo,
  with distinct left and right outputs.
- Reverb produces a finite stereo decay after an impulse.

## Processing contract

The measurement is taken after the selected NAM and the manual bass, mid and
treble controls, before matched EQ, delay, reverb, output gain and reference
playback. A preamp screamer is included in the measured amp tone when enabled.
The reference player is added only after the guitar chain, so its samples cannot
enter the recording through the application's internal routing.

FFT analysis, file reading, curve fitting, coefficient construction and memory
reclamation run outside the audio callback. The callback writes to a prepared
ten-second buffer and uses preallocated filters and delay buffers. The matched
EQ is a regularized, smoothed 15-band IIR correction with gain limits and short
crossfades. It adds no fixed block latency.

The correction compares average spectral shape after removing the overall level
difference. Play varied notes or chords in the style of the selected passage.
Different notes, tuning, pickups, playing dynamics and source-separation errors
can still influence the result. The screamer is a native overdrive effect, and
solo/rhythm settings are starting presets rather than recovered effect settings
from the original recording.
