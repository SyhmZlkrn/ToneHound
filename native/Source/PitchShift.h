#pragma once
#include <juce_dsp/juce_dsp.h>
#include <signalsmith-stretch/signalsmith-stretch.h>

// Polyphonic spectral transpose before the amp. At zero semitones, samples
// pass through unchanged and no extra pitch latency is reported.
class GuitarPitchShift {
public:
    void prepare(double sampleRate);
    void process(double* data,int count,int semitones,bool studio=false,float blend=1.f,float cents=0.f,bool attack=true);
    int latencyFor(int semitones,bool studio=false,float cents=0.f) const noexcept {return semitones==0 && cents==0.f?0:latencies[studio?1:0];}
private:
    std::array<signalsmith::stretch::SignalsmithStretch<float>,2> engines;
    std::array<float,512> input{},output{};
    int previous=0;
    std::array<int,2> latencies{};
    bool previousStudio=false;
    float previousCents=0;
    std::vector<double> dryDelay;
    size_t dryPosition=0;
    double dryEnvelope=0,wetEnvelope=0,attackCoefficient=0,releaseCoefficient=0;
    juce::SmoothedValue<float> fade,wetMix,attackMix;
};
