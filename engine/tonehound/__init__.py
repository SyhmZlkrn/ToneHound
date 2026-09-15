"""ToneHound analysis engine.

Non-real-time. Everything here runs in a sidecar process, never on an audio
thread.
"""

from .config import GUITAR, SAMPLE_RATE, InstrumentConfig

__all__ = ["GUITAR", "SAMPLE_RATE", "InstrumentConfig"]
