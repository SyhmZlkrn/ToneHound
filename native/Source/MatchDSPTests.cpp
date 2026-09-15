#include "MatchDSP.h"
#include "PitchShift.h"
#include <iostream>
#include <random>

// Called by the native validator. No device, profile, network or Python is needed.
void runMatchDSPTests()
{
    const auto require=[](bool condition,const char* message){if(!condition)throw std::runtime_error(message);};
    {
        GuitarPitchShift pitch;pitch.prepare(48000);
        std::array<double,256> block{};
        for(int semitones:{0,-12,12}){
            pitch.prepare(48000);int crossings=0;double previous=0;int measured=0;
            for(int offset=0;offset<96000;offset+=256){
                const int n=juce::jmin(256,96000-offset);
                for(int i=0;i<n;++i)block[(size_t)i]=.1*std::sin(2*juce::MathConstants<double>::pi*440*(offset+i)/48000.);
                const auto dry=block;pitch.process(block.data(),n,semitones);
                for(int i=0;i<n;++i){auto value=block[(size_t)i];require(std::isfinite(value),"Pitch generated non-finite output");
                    if(semitones==0)require(value==dry[(size_t)i],"Zero transpose changed dry samples");
                    if(offset+i>24000){if(previous<=0 && value>0)++crossings;++measured;}previous=value;
                }
            }
            const auto measuredHz=crossings*48000./measured,expected=440*std::pow(2.,semitones/12.);
            require(std::abs(measuredHz-expected)<2.,"Pitch frequency does not match selected semitones");
            std::cout<<"Pitch "<<semitones<<" st: "<<measuredHz<<" Hz, latency "<<pitch.latencyFor(semitones)<<" samples"<<std::endl;
        }
        for(double rate:{44100.,48000.,96000.}){
            pitch.prepare(rate);const int count=(int)(rate*2);std::vector<double> shifted((size_t)count);
            const double notes[]{110.,138.591315,164.813778};
            auto started=juce::Time::getMillisecondCounterHiRes();
            for(int offset=0;offset<count;offset+=256){
                auto n=juce::jmin(256,count-offset);
                for(int i=0;i<n;++i){block[(size_t)i]=0;for(auto hz:notes)block[(size_t)i]+=.07*std::sin(2*juce::MathConstants<double>::pi*hz*(offset+i)/rate);}
                pitch.process(block.data(),n,-5,true);
                for(int i=0;i<n;++i){require(std::isfinite(block[(size_t)i]),"Polyphonic pitch produced invalid audio");shifted[(size_t)(offset+i)]=block[(size_t)i];}
            }
            auto elapsed=juce::Time::getMillisecondCounterHiRes()-started;
            for(auto hz:notes){
                double peak=0;const auto expected=hz*std::pow(2.,-5./12.);
                for(double delta=-2;delta<=2;delta+=.25){
                    double real=0,imag=0;
                    for(int i=(int)rate;i<count;++i){const auto phase=2*juce::MathConstants<double>::pi*(expected+delta)*i/rate;
                        real+=shifted[(size_t)i]*std::cos(phase);imag+=shifted[(size_t)i]*std::sin(phase);}
                    peak=juce::jmax(peak,2*std::hypot(real,imag)/rate);
                }
                require(peak>.012,"Pitch lost a note in the transposed chord");
            }
            std::cout<<"Studio polyphonic pitch "<<rate<<" Hz: three chord notes retained; "<<elapsed/20<<"% realtime, "<<pitch.latencyFor(-5,true)<<" latency samples"<<std::endl;
        }
        // A burst exposes pre-echo that a steady sine/chord cannot detect.
        std::array<double,2> beforeAttack{};
        for(int fix=0;fix<2;++fix){
            pitch.prepare(48000);std::vector<double> burst(96000,0.);
            for(int i=24000;i<28800;++i)burst[(size_t)i]=.15*std::sin(2*juce::MathConstants<double>::pi*220*(i-24000)/48000.);
            pitch.process(burst.data(),(int)burst.size(),7,false,1.f,0.f,fix==1);
            const int onset=24000+pitch.latencyFor(7);
            for(int i=onset-1920;i<onset;++i)beforeAttack[(size_t)fix]+=burst[(size_t)i]*burst[(size_t)i];
            double audible=0;for(int i=onset;i<onset+4800;++i)audible+=burst[(size_t)i]*burst[(size_t)i];
            require(audible>1.,"Attack preservation silenced a picked note");
        }
        require(beforeAttack[1]<beforeAttack[0]*.1,"Attack preservation did not reduce pre-echo");
        std::cout<<"Pitch pre-attack energy "<<beforeAttack[0]<<" -> "<<beforeAttack[1]<<std::endl;
        for(float mix:{0.f,.5f,1.f}){
            pitch.prepare(48000);std::vector<double> signal(96000);
            for(int i=0;i<96000;++i)signal[(size_t)i]=.1*std::sin(2*juce::MathConstants<double>::pi*440*i/48000.);
            pitch.process(signal.data(),(int)signal.size(),0,false,mix,25.f,true);
            auto amplitude=[&](double hz){double re=0,im=0;for(int i=48000;i<96000;++i){auto phase=2*juce::MathConstants<double>::pi*hz*i/48000.;re+=signal[(size_t)i]*std::cos(phase);im+=signal[(size_t)i]*std::sin(phase);}return 2*std::hypot(re,im)/48000.;};
            double dry=amplitude(440),wet=0;const double expected=440*std::pow(2.,.25/12.);
            // Resolve the short-window shifter's peak within 1 Hz; an off-bin
            // 1-second projection otherwise understates a correctly shifted voice.
            for(double delta=-1.;delta<=1.;delta+=.05)wet=juce::jmax(wet,amplitude(expected+delta));
            std::cout<<"Pitch blend "<<mix<<": dry="<<dry<<", shifted="<<wet<<std::endl;
            require(pitch.latencyFor(0,false,25)>0,"Fine tuning latency is missing");
            if(mix==0)require(dry>.09 && wet<.015,"Dry blend is not delay-aligned dry audio");
            if(mix==1)require(wet>.075 && dry<.015,"Fine tuning pitch or fully wet isolation failed");
            if(mix==.5)require(dry>.035 && wet>.035,"Harmony blend lost one voice");
        }
        std::cout<<"PASS: fine tuning, dry/wet harmony, oversized blocks and attack isolation"<<std::endl;
    }
    constexpr double sr=48000.;constexpr int count=480000;
    std::vector<float> played(count),reference(count),opposite(count);
    std::mt19937 random(7183);std::uniform_real_distribution<float> noise(-.12f,.12f);
    tonehound::MatchFilter low,high;low.set(226.,4.,sr);high.set(3620.,-4.,sr);
    for(int i=0;i<count;++i){played[(size_t)i]=noise(random);reference[(size_t)i]=high.process(low.process(played[(size_t)i]));opposite[(size_t)i]=-reference[(size_t)i];}
    const float* source[]{played.data()};const float* target[]{reference.data()};const float* stereo[]{reference.data(),opposite.data()};
    auto sourceSpectrum=tonehound::analyseSpectrum(source,1,count,sr),targetSpectrum=tonehound::analyseSpectrum(target,1,count,sr);
    auto stereoSpectrum=tonehound::analyseSpectrum(stereo,2,count,sr);
    require(std::abs(stereoSpectrum.rms-targetSpectrum.rms)<1.e-7,"EQ target stereo power was cancelled");
    auto gain=tonehound::fitMatchCurve(targetSpectrum,sourceSpectrum,sr);
    std::array<tonehound::MatchFilter,tonehound::matchBands> filters;
    for(size_t i=0;i<filters.size();++i)filters[i].set(tonehound::matchFrequency(i),gain[i],sr);
    double before=0,after=0;
    for(size_t i=0;i<filters.size();++i) {
        auto hz=tonehound::matchFrequency(i),expected=low.response(hz,sr)+high.response(hz,sr),actual=0.;
        for(auto& f:filters)actual+=f.response(hz,sr);
        before+=expected*expected;after+=(expected-actual)*(expected-actual);
    }
    require(after<before*.55,"Fitted EQ did not reduce a known spectral mismatch");
    std::cout<<"EQ spectral error: "<<std::sqrt(before/filters.size())<<" -> "<<std::sqrt(after/filters.size())<<" dB RMS"<<std::endl;
    for(int i=0;i<512;++i){double db=0;auto hz=30.*std::pow(20000./30.,i/511.);for(auto& f:filters)db+=f.response(hz,sr);require(std::isfinite(db)&&std::abs(db)<=6.01,"Matched EQ exceeded its total gain bound");}
    bool rejected=false;std::vector<float> silence(count,0);const float* quiet[]{silence.data()};
    try{tonehound::analyseSpectrum(quiet,1,count,sr);}catch(const std::exception&){rejected=true;}
    require(rejected,"Silent EQ capture was accepted");
    rejected=false;for(int i=0;i<count;++i)silence[(size_t)i]=.1f*std::sin((float)(2*juce::MathConstants<double>::pi*440*i/sr));
    try{auto thin=tonehound::analyseSpectrum(quiet,1,count,sr);tonehound::fitMatchCurve(targetSpectrum,thin,sr);}catch(const std::exception&){rejected=true;}
    require(rejected,"Single-note frequency coverage was accepted for a full EQ curve");
    auto referenceFile=juce::File::getSpecialLocation(juce::File::tempDirectory).getNonexistentChildFile("tonehound-eq-validation",".wav");
    struct TemporaryReference {juce::File file;~TemporaryReference(){file.deleteFile();}} cleanup{referenceFile};
    {
        auto stream=referenceFile.createOutputStream();require(stream!=nullptr,"Could not create the EQ test reference");
        juce::WavAudioFormat format;
        std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(),sr,2,24,{},0));
        require(writer!=nullptr && writer->writeFromFloatArrays(stereo,2,count),"Could not write the EQ test reference");
    }
    const auto waitUntil=[&](const std::function<bool()>& predicate){auto deadline=juce::Time::getMillisecondCounterHiRes()+5000;do{if(predicate())return true;juce::Thread::sleep(5);}while(juce::Time::getMillisecondCounterHiRes()<deadline);return false;};
    tonehound::MatchEQ match;match.prepare(sr);match.loadTarget(referenceFile);match.reset();
    require(waitUntil([&]{return match.targetReady();}),"Resetting the amp cancelled target preparation");
    require(waitUntil([&]{return match.startCapture();}),"Could not arm the 10-second EQ capture");
    for(int i=0;i<count/2;++i){if(i%256==0)match.beginBlock(true);match.record(played[(size_t)i]);match.process(played[(size_t)i]);}
    require(match.isCapturing() && std::abs(match.progress()-.5f)<1.e-6,"EQ capture did not track 5 of 10 seconds");
    for(int i=count/2;i<count;++i){if(i%256==0)match.beginBlock(true);match.record(played[(size_t)i]);match.process(played[(size_t)i]);}
    require(!match.isCapturing() && waitUntil([&]{return match.ready();}),"10-second EQ capture did not produce an applied curve");
    auto saved=match.gains();tonehound::MatchEQ restored;restored.prepare(96000);restored.restore(saved,true);
    require(restored.ready() && restored.gains()==saved,"Matched EQ did not restore at another sample rate");
    restored.loadTarget(referenceFile,true);
    require(restored.ready() && restored.gains()==saved,"Reopening the editor immediately discarded its saved EQ");
    require(waitUntil([&]{return restored.targetReady();}) && restored.ready() && restored.gains()==saved,"Reference preparation discarded the restored EQ curve");
    match.clearTarget();require(!match.targetReady() && !match.ready(),"Changing the reference retained a stale EQ curve");
    match.loadTarget(referenceFile);require(waitUntil([&]{return match.targetReady();}),"Target reload failed");
    require(waitUntil([&]{return match.startCapture();}),"Could not arm the silence rejection test");
    for(int i=0;i<count;++i)match.record(0);
    require(waitUntil([&]{return match.status().containsIgnoreCase("too little");}) && !match.ready(),"Silent player capture produced a usable correction");
    tonehound::PedalChain pedals;pedals.prepare(sr);tonehound::PedalSettings settings;pedals.update(settings);
    for(int i=0;i<10000;++i){auto x=played[(size_t)i];auto pre=pedals.preAmp(x);float l,r;pedals.postAmp(pre,l,r);require(pre==x && l==x && r==x,"Bypassed pedals changed the dry signal");}
    settings.screamer=true;settings.drive=.7f;pedals.update(settings);double changed=0;
    for(int i=0;i<10000;++i){auto x=.08f*std::sin((float)(2*juce::MathConstants<double>::pi*220*i/sr));auto y=pedals.preAmp(x);require(std::isfinite(y),"Screamer produced invalid samples");if(i>2000)changed+=std::abs(y-x);}
    require(changed>20,"Screamer did not change the guitar signal");
    pedals.prepare(sr);settings={};settings.delay=true;settings.delayMix=.5f;settings.feedback=.4f;settings.division=0;settings.bpm=120;pedals.update(settings);
    float l=0,r=0;for(int i=0;i<10000;++i)pedals.postAmp(0,l,r);
    double leftEcho=0,rightEcho=0,difference=0,early=0;
    for(int i=0;i<50000;++i){pedals.postAmp(i==0?.25f:0,l,r);require(std::isfinite(l)&&std::isfinite(r),"Delay produced invalid samples");if(i>100 && i<23900)early+=std::abs(l);if(i>=23990&&i<24020)leftEcho+=std::abs(l);if(i>=24270&&i<24320)rightEcho+=std::abs(r);difference+=std::abs(l-r);}
    require(early<1.e-5 && leftEcho>.05 && rightEcho>.05 && difference>.1,"Stereo delay did not follow the selected BPM/division");
    pedals.prepare(sr);settings={};settings.reverb=true;settings.reverbMix=.5f;pedals.update(settings);
    for(int i=0;i<5000;++i)pedals.postAmp(0,l,r);
    double tail=0,stereoTail=0;
    for(int i=0;i<96000;++i){pedals.postAmp(i==0?.25f:0,l,r);require(std::isfinite(l)&&std::isfinite(r),"Reverb produced invalid samples");if(i>1000){tail+=l*l+r*r;stereoTail+=std::abs(l-r);}}
    require(tail>1.e-5 && stereoTail>.01,"Reverb did not produce a stereo tail");
    std::cout<<"PASS: EQ improvement/bounds, stereo power, 10-second capture/restore/invalidation, silence/coverage rejection, dry bypass, screamer, tempo delay and stereo reverb"<<std::endl;
}
