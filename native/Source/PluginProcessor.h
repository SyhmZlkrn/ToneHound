#pragma once
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>
#include "Bridge.h"
#include "MatchDSP.h"
#include "PitchShift.h"
#include <NAM/get_dsp.h>
#include <array>
#include <atomic>

class ModelGraph;
struct PreviewAudio;

class ToneHoundProcessor final : public juce::AudioProcessor, private juce::Timer
{
public:
    ToneHoundProcessor();
    ~ToneHoundProcessor() override;
    static juce::AudioProcessorValueTreeState::ParameterLayout parameters();
    void prepareToPlay(double, int) override;
    void releaseResources() override {}
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    bool isBusesLayoutSupported(const BusesLayout&) const override;
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "ToneHound"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override;
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return "Default"; }
    void changeProgramName(int, const juce::String&) override {}
    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;

    void loadProfile(const juce::File&, juce::String displayName = {}, bool preserveMatchedEQ = false);
    bool loadProfileForTest(const juce::File&, juce::String& error);
    void loadPreview(const juce::File&);
    void collectRetired(); // message thread; never frees a model on the audio thread
    juce::String profileName() const;
    juce::String profilePath() const;
    juce::String modelMessage() const;
    void setModelMessage(juce::String);
    void setParameter(const juce::String& id, float actual);
    void setSelection(double start, double end);
    juce::var referenceInfo() const;
    juce::var matchInfo() const;
    bool hasMatches() const;
    juce::String matchExplanation() const;
    void setReference(juce::var);
    void setMatches(juce::var, juce::String note = {});
    void loadEqTarget(const juce::File& file, bool preserveCurve = false) { if(!preserveCurve)resetEqMatch();eqMatch.loadTarget(file,preserveCurve); }
    void clearEqTarget() { eqMatch.clearTarget(); resetEqMatch(); }
    bool startEqCapture();
    void cancelEqCapture() { eqMatch.cancelCapture(); }
    void resetEqMatch();
    juce::String eqMatchStatus() const { return eqMatch.status(); }
    float eqCaptureProgress() const { return eqMatch.progress(); }
    bool eqTargetReady() const { return eqMatch.targetReady(); }
    bool eqMatchReady() const { return eqMatch.ready(); }
    bool eqCapturing() const { return eqMatch.isCapturing(); }
    std::vector<std::pair<float,float>> eqCurve() const { return eqMatch.curve(); }
    void applyPerformanceSuggestion(bool solo, float bpm);

    juce::AudioProcessorValueTreeState state;
    juce::File root;
    AnalysisBridge analysis, artwork;
    juce::AudioDeviceManager* deviceManager = nullptr; // standalone only; DAW owns I/O in VST3
    bool standalone = false;
    std::atomic<float> inputPeak{0}, outputPeak{0};
    std::atomic<bool> modelReady{false}, loading{false}, audioFault{false};
    std::atomic<bool> previewPlaying{false}, previewRestart{false}, previewLoop{true};
    std::atomic<int> matchingModel{0}; // 0 Standard, 1 installed LoRA; analysis only
    std::atomic<double> previewPosition{0}, selectionStart{0}, selectionEnd{40}, rate{48000};

private:
    void timerCallback() override;
    void syncEqParameters();
    juce::CriticalSection eqStateLock;
    void writeCacheLease();
    juce::File cacheLease;
    int leaseTicks=0;
    unsigned lastFitRevision=0;
    tonehound::MatchGains lastBandParameters{};
    std::array<std::atomic<float>*,8> bandValues{};
    std::atomic<float>* pitchValue=nullptr;
    std::atomic<float>* pitchStudioValue=nullptr;
    std::atomic<float>* pitchBlendValue=nullptr;
    std::atomic<float>* pitchCentsValue=nullptr;
    std::atomic<float>* pitchAttackValue=nullptr;
    std::atomic<int> ampLatency{0},currentPitchLatency{0};
    GuitarPitchShift pitchShift;
    struct Biquad {
        double b0=1,b1=0,b2=0,a1=0,a2=0,z1=0,z2=0;
        float process(float x) { auto y=b0*x+z1; z1=b1*x-a1*y+z2; z2=b2*x-a2*y; return (float)y; }
        void set(int type, double frequency, double gainDB, double sr);
        void reset() { z1=z2=0; }
    } eq[3];
    juce::ThreadPool loader{1};
    mutable juce::CriticalSection infoLock;
    juce::var reference, matches;
    juce::String lastMatchNote;
    juce::String currentName, currentPath, message = "Choose a capture to begin";
    std::atomic<unsigned> loadVersion{0}, previewVersion{0};
    ModelGraph* active = nullptr;
    std::atomic<ModelGraph*> pendingModel{nullptr}, retiredModels{nullptr};
    PreviewAudio* preview = nullptr;
    std::atomic<PreviewAudio*> pendingPreview{nullptr}, retiredPreviews{nullptr};
    std::array<double, 512> mono{}, wet{};
    double previewCursor = 0, gateEnvelope = 0, gateGain = 0;
    juce::SmoothedValue<float> inputGain, outputGain;
    juce::SmoothedValue<float> monitorGain;
    std::array<std::atomic<float>*, 9> values{};
    std::array<std::atomic<float>*,14> effectValues{};
    tonehound::MatchEQ eqMatch;
    tonehound::PedalChain pedals;
    float previousEQ[3]{1000,1000,1000};
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ToneHoundProcessor)
};
