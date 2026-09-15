"""Instrument-specific analysis configuration.

The engine is parameterised by instrument so bass can be added later without a
retrofit. Only guitar ships in v1.
"""

from dataclasses import dataclass

SAMPLE_RATE = 48_000
"""NAM's anchor rate. Everything in the engine works at this rate."""


@dataclass(frozen=True)
class InstrumentConfig:
    name: str

    # Band-limiting applied to a separated stem before analysis, to reject
    # bleed from neighbouring instruments.
    analysis_lo_hz: float
    analysis_hi_hz: float

    # LTAS grid. Deliberately wider than the analysis band so that roll-off
    # shape either side is still captured.
    ltas_lo_hz: float
    ltas_hi_hz: float
    n_bands: int

    # High band used as a saturation cue ("fizz").
    fizz_lo_hz: float
    fizz_hi_hz: float

    probe_asset: str


GUITAR = InstrumentConfig(
    name="guitar",
    analysis_lo_hz=80.0,
    analysis_hi_hz=10_000.0,
    ltas_lo_hz=40.0,
    ltas_hi_hz=16_000.0,
    n_bands=48,
    fizz_lo_hz=4_000.0,
    fizz_hi_hz=8_000.0,
    probe_asset="probe_guitar_di.wav",
)

# Deferred to a later version. Kept here so the shape of the extension is
# explicit: bass needs a 30 Hz analysis floor, because an 80 Hz high-pass would
# discard the fundamental of every note below E2.
#
# BASS = InstrumentConfig(
#     name="bass",
#     analysis_lo_hz=30.0, analysis_hi_hz=6_000.0,
#     ltas_lo_hz=25.0, ltas_hi_hz=10_000.0, n_bands=48,
#     fizz_lo_hz=2_000.0, fizz_hi_hz=5_000.0,
#     probe_asset="probe_bass_di.wav",
# )
