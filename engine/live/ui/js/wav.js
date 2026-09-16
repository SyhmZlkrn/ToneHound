/*
  Tiny WAV writer, used only by mock mode.

  The real engine hands the browser `/audio/{id}.wav` URLs, so nothing in the
  live path needs this. Mock mode has to invent audio the <audio> element can
  actually play -- a data-less demo where every player is silent would hide the
  one thing the results panel exists to do (let you listen). Encoding PCM16 by
  hand is ~40 lines and keeps the page dependency-free.
*/
(function (global) {
  "use strict";

  function encodeWav(channelData, sampleRate) {
    const n = channelData.length;
    const buf = new ArrayBuffer(44 + n * 2);
    const view = new DataView(buf);
    const str = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };

    str(0, "RIFF"); view.setUint32(4, 36 + n * 2, true); str(8, "WAVE");
    str(12, "fmt "); view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);            // PCM
    view.setUint16(22, 1, true);            // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    str(36, "data"); view.setUint32(40, n * 2, true);

    for (let i = 0; i < n; i++) {
      const s = Math.max(-1, Math.min(1, channelData[i]));
      view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  function wavUrl(channelData, sampleRate) {
    return URL.createObjectURL(encodeWav(channelData, sampleRate));
  }

  /* Peaks in the same shape as GET /peaks/{id} so mock and live agree. */
  function peaksOf(samples, bins) {
    const min = new Array(bins), max = new Array(bins), rms = new Array(bins);
    const step = samples.length / bins;
    for (let b = 0; b < bins; b++) {
      const a = Math.floor(b * step), z = Math.min(samples.length, Math.floor((b + 1) * step));
      let lo = 0, hi = 0, acc = 0, count = 0;
      for (let i = a; i < z; i++) {
        const v = samples[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        acc += v * v; count++;
      }
      min[b] = lo; max[b] = hi; rms[b] = count ? Math.sqrt(acc / count) : 0;
    }
    return { bins, min, max, rms };
  }

  global.EZWav = { encodeWav, wavUrl, peaksOf };
})(window);
