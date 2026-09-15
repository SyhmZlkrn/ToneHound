#pragma once
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>
#include <array>
#include <atomic>
#include <vector>

namespace tonehound {
constexpr size_t matchBands = 8;
using MatchGains = std::array<double, matchBands>;
double matchFrequency(size_t band);

struct MatchFilter {
    double b0=1,b1=0,b2=0,a1=0,a2=0,z1=0,z2=0;
    void set(double frequency, double gainDB, double rate);
    float process(float sample) noexcept;
    double response(double frequency, double rate) const;
};

struct Spectrum {
    std::array<double,matchBands> power{};
    double rms=0;
    int audibleFrames=0;
};
// Offline only. Stereo power is accumulated independently, never phase-cancelled.
Spectrum analyseSpectrum(const float* const* data, int channels, int samples, double rate);
MatchGains fitMatchCurve(const Spectrum& target, const Spectrum& played, double rate);

class MatchEQ {
public:
    MatchEQ() = default;
    ~MatchEQ();
    void prepare(double rate);
    void loadTarget(const juce::File&, bool preserveCurve = false);
    void clearTarget();
    bool startCapture();
    void cancelCapture();
    void reset();
    juce::String status() const;
    float progress() const noexcept;
    bool targetReady() const noexcept { return targetValid.load(); }
    bool ready() const noexcept { return curveValid.load(); }
    bool isCapturing() const noexcept { return phase.load()==capturing; }
    MatchGains gains() const;
    unsigned fitRevision() const noexcept { return fittedRevision.load(); }
    void restore(const MatchGains&, bool valid);
    std::vector<std::pair<float,float>> curve() const;
    // Audio thread only, allocation- and lock-free. Call beginBlock once per block.
    void beginBlock(bool enabled) noexcept;
    void record(float postAmp) noexcept;
    float process(float sample) noexcept;
    void collectRetired();
private:
    enum Phase { idle, capturing, analysing };
    struct Packet { MatchGains gains{}; std::array<MatchFilter,matchBands> filters{}; Packet* next=nullptr; };
    void publish(const MatchGains&, bool valid);
    void setMessage(juce::String);
    juce::ThreadPool worker{1};
    mutable juce::CriticalSection lock;
    Spectrum target;
    MatchGains savedGains{};
    juce::String detail="Match an amp to prepare the reference EQ";
    std::vector<float> capture;
    std::atomic<double> sampleRate{48000};
    std::atomic<size_t> captured{0};
    std::atomic<int> phase{idle};
    std::atomic<unsigned> generation{0};
    std::atomic<unsigned> fittedRevision{0};
    std::atomic<bool> targetValid{false},curveValid{false},busy{false};
    std::atomic<Packet*> pending{nullptr},retired{nullptr};
    std::array<MatchFilter,matchBands> filters{},oldFilters{};
    juce::SmoothedValue<float> blend,transition;
};

struct PedalSettings {
    bool screamer=false, delay=false, reverb=false;
    float drive=.25f,tone=.5f,level=.5f;
    float delayMix=.18f,feedback=.25f,bpm=120;
    int division=1;
    float reverbMix=.12f,roomSize=.5f;
};
class PedalChain {
public:
    void prepare(double rate);
    void update(const PedalSettings&) noexcept;
    float preAmp(float sample) noexcept;
    void postAmp(float mono,float& left,float& right) noexcept;
private:
    double sampleRate=48000,highPassMemory=0,previousInput=0,toneMemory=0;
    double highPassPole=.99,tonePole=.5;
    juce::SmoothedValue<float> screamBlend,drive,tone,level,delayMix,feedback,delaySamples,reverbMix;
    std::vector<float> delayLeft,delayRight;
    size_t position=0;
    float delayLowLeft=0,delayLowRight=0;
    juce::Reverb room;
    PedalSettings previous;
};
} // namespace tonehound
