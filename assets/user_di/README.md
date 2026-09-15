# Drop your guitar DI recordings here

These replace the synthesised Karplus-Strong probe when testing separation.
Everything measured about UVR so far used synthetic audio, and `htdemucs_6s`
was trained on real music -- so real playing is the one thing that can settle
whether separation is genuinely as damaging as it currently measures.

## What to record

**A clean DI** -- guitar straight into your interface. No amp, no amp sim, no
pedals, no reverb. Just the raw pickup signal. This is the single most common
thing to get wrong: a recording *of an amp* is not a DI, and `prepare_di.py`
will warn you if a take looks cabinet-filtered.

**Record your own playing.** Do not use commercial recordings or someone else's
material.

## How much

- 2 to 4 takes, **30-60 seconds each**
- 44.1 kHz or 48 kHz WAV, mono or stereo (both are handled)
- Play fairly densely -- long silences waste the take

## What to play

Mirror what the synthetic probe covers, because that is what the fingerprint
measures:

1. **Palm-muted low-E chugs** -- transients and low-end behaviour
2. **Open power chords** -- intermodulation, how the amp sags under a load
3. **Single-note lead lines, mid-neck** -- midrange voicing
4. **Clean arpeggios** -- headroom and where compression starts
5. **A chromatic run** -- so every frequency region gets excited

Varied dynamics matter: play some of it soft and some hard. Amps are nonlinear,
so how hard you hit is part of the tone being measured.

If you have several guitars, one take each (humbucker and single-coil, say) is
more useful than several takes on one guitar.

## Then

```bash
PYTHONPATH=engine python engine/tools/prepare_di.py
```

It validates each take, flags anything that looks like an amp recording rather
than a DI, and reports what it found.
