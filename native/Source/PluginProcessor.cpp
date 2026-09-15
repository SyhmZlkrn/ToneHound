#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <functional>
#include <iostream>
#include <mutex>
// AudioDSPTools' resampler retains this single constant from its iPlug origin.
namespace iplug { inline constexpr double PI=juce::MathConstants<double>::pi; }
#define DEFAULT_BLOCK_SIZE 512
#include <ResamplingContainer/ResamplingContainer.h>
#undef DEFAULT_BLOCK_SIZE

class ModelGraph
{
public:
    ModelGraph(const juce::File& file, double sr) : hostRate(sr)
    {
        // Upstream Lanczos tables are initialised lazily; serialize construction
        // across plugin instances, away from every audio callback.
        static std::mutex construction;const std::lock_guard<std::mutex> guard(construction);
        nam::DspLoadOptions options; options.prewarm = false;
        dsp = nam::get_dsp(std::filesystem::path(file.getFullPathName().toWideCharPointer()), options);
        if (dsp->NumInputChannels()!=1 || dsp->NumOutputChannels()!=1)
            throw std::runtime_error("This capture needs a mono input and output.");
        const auto modelRate = dsp->GetExpectedSampleRate() > 0 ? dsp->GetExpectedSampleRate() : 48000.0;
        dsp->Reset(modelRate, 4096);
        if (std::abs(modelRate-sr)>0.5) {
            resampler = std::make_unique<dsp::ResamplingContainer<double,1,12>>(modelRate);
            resampler->Reset(sr, 512); latency = resampler->GetLatency();
        }
        processFunction = [this](double** in, double** out, int n) { dsp->process(in,out,n); };
    }
    void process(double* in, double* out, int n)
    {
        double* inputs[]{in}; double* outputs[]{out};
        if (resampler) resampler->ProcessBlock(inputs, outputs, n, processFunction);
        else dsp->process(inputs, outputs, n);
    }
    ModelGraph* next = nullptr;
    double hostRate;
    int latency=0;
private:
    std::unique_ptr<nam::DSP> dsp;
    std::unique_ptr<dsp::ResamplingContainer<double,1,12>> resampler;
    std::function<void(double**, double**, int)> processFunction;
};

struct PreviewAudio {
    juce::AudioBuffer<float> samples;
    double rate=48000;
    PreviewAudio* next=nullptr;
};
template<typename T> static void retire(std::atomic<T*>& head, T* item)
{
    if (!item) return;
    item->next=head.load();
    while (!head.compare_exchange_weak(item->next,item)) {}
}
template<typename T> static void freeRetired(std::atomic<T*>& head)
{
    auto* p=head.exchange(nullptr);
    while (p) { auto* next=p->next; delete p; p=next; }
}

juce::AudioProcessorValueTreeState::ParameterLayout ToneHoundProcessor::parameters()
{
    juce::AudioProcessorValueTreeState::ParameterLayout p;
    auto add=[&](const char* id,const char* label,float lo,float hi,float value) {
        p.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{id,1},label,
              juce::NormalisableRange<float>(lo,hi,0.1f),value));
    };
    add("input","Input",-24,24,0); add("output","Output",-36,12,-6);
    add("gate","Noise gate",-100,-20,-80);
    add("bass","Bass",-12,12,0); add("mid","Mid",-12,12,0); add("treble","Treble",-12,12,0);
    p.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"monitor",1},"Guitar monitoring",true));
    p.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"bypass",1},"Bypass amp",false));
    p.add(std::make_unique<juce::AudioParameterChoice>(juce::ParameterID{"channel",1},"Guitar input",juce::StringArray{"Input 1","Input 2"},0));
    auto toggle=[&](const char* id,const char* label){p.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{id,1},label,false));};
    auto unit=[&](const char* id,const char* label,float value){p.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{id,1},label,juce::NormalisableRange<float>(0,1,.001f),value));};
    toggle("eqMatchEnabled","Matched EQ");
    toggle("screamerEnabled","Screamer");unit("screamerDrive","Screamer drive",.25f);unit("screamerTone","Screamer tone",.5f);unit("screamerLevel","Screamer level",.5f);
    toggle("delayEnabled","Delay");unit("delayMix","Delay mix",.18f);
    p.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"delayFeedback",1},"Delay feedback",juce::NormalisableRange<float>(0,.8f,.001f),.25f));
    p.add(std::make_unique<juce::AudioParameterChoice>(juce::ParameterID{"delayDivision",1},"Delay division",juce::StringArray{"Quarter","Dotted eighth","Eighth","Triplet eighth"},1));
    add("delayBpm","Delay BPM",40,240,120);
    toggle("reverbEnabled","Reverb");unit("reverbMix","Reverb mix",.12f);unit("reverbSize","Reverb size",.5f);
    p.add(std::make_unique<juce::AudioParameterChoice>(juce::ParameterID{"performanceRole",1},"Performance role",juce::StringArray{"Rhythm","Solo"},0));
    p.add(std::make_unique<juce::AudioParameterInt>(juce::ParameterID{"pitchSemitones",1},"Transpose",-12,12,0));
    for(int i=0;i<8;++i){auto id="eqBand"+juce::String(i+1);add(id.toRawUTF8(),("EQ "+juce::String(tonehound::matchFrequency((size_t)i),0)+" Hz").toRawUTF8(),-12,12,0);}
    p.add(std::make_unique<juce::AudioParameterChoice>(juce::ParameterID{"pitchQuality",1},"Pitch quality",juce::StringArray{"Live","Studio"},0));
    unit("pitchBlend","Pitch wet blend",1.f);
    add("pitchCents","Pitch fine tuning",-50,50,0);
    p.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"pitchAttack",1},"Preserve pitch attacks",true));
    return p;
}
ToneHoundProcessor::ToneHoundProcessor()
    : AudioProcessor(BusesProperties().withInput("Guitar",juce::AudioChannelSet::stereo(),true)
                                       .withOutput("Output",juce::AudioChannelSet::stereo(),true)),
      state(*this,nullptr,"ToneHound",parameters()), root(findProjectRoot()), analysis(root), artwork(root)
{
    const char* ids[]{"input","output","gate","bass","mid","treble","monitor","bypass","channel"};
    for(size_t i=0;i<values.size();++i) values[i]=state.getRawParameterValue(ids[i]);
    const char* effectIds[]{"eqMatchEnabled","screamerEnabled","screamerDrive","screamerTone","screamerLevel","delayEnabled","delayMix","delayFeedback","delayDivision","delayBpm","reverbEnabled","reverbMix","reverbSize","performanceRole"};
    for(size_t i=0;i<effectValues.size();++i)effectValues[i]=state.getRawParameterValue(effectIds[i]);
    for(size_t i=0;i<bandValues.size();++i)bandValues[i]=state.getRawParameterValue("eqBand"+juce::String((int)i+1));
    pitchValue=state.getRawParameterValue("pitchSemitones");
    pitchStudioValue=state.getRawParameterValue("pitchQuality");
    pitchBlendValue=state.getRawParameterValue("pitchBlend");
    pitchCentsValue=state.getRawParameterValue("pitchCents");
    pitchAttackValue=state.getRawParameterValue("pitchAttack");
    cacheLease=root.getChildFile(".cache/tone3000/leases").getChildFile(juce::Uuid().toString()+".json");
    startTimerHz(30);
    inputGain.setCurrentAndTargetValue(1); outputGain.setCurrentAndTargetValue(0.501187f);
    monitorGain.setCurrentAndTargetValue(1);
}
ToneHoundProcessor::~ToneHoundProcessor()
{
    stopTimer();
    ++loadVersion; ++previewVersion; loader.removeAllJobs(true,-1);
    cacheLease.deleteFile();
    delete pendingModel.exchange(nullptr); delete active;
    delete pendingPreview.exchange(nullptr); delete preview;
    collectRetired();
}
bool ToneHoundProcessor::isBusesLayoutSupported(const BusesLayout& l) const
{
    return (l.getMainInputChannelSet()==juce::AudioChannelSet::mono() || l.getMainInputChannelSet()==juce::AudioChannelSet::stereo())
        && (l.getMainOutputChannelSet()==juce::AudioChannelSet::mono() || l.getMainOutputChannelSet()==juce::AudioChannelSet::stereo());
}
void ToneHoundProcessor::prepareToPlay(double sr,int)
{
    rate=sr;
    inputGain.reset(sr,0.015); outputGain.reset(sr,0.015); monitorGain.reset(sr,0.015);
    gateEnvelope=gateGain=0;
    for(auto& filter:eq) filter.reset();
    for(auto& value:previousEQ) value=1000;
    pedals.prepare(sr);eqMatch.prepare(sr);pitchShift.prepare(sr);
    currentPitchLatency=pitchShift.latencyFor(values[7]->load()>.5f?0:(int)pitchValue->load(),pitchStudioValue->load()>.5f,values[7]->load()>.5f?0:pitchCentsValue->load());
    setLatencySamples(ampLatency.load()+currentPitchLatency.load());
    auto path=profilePath();
    if(path.isNotEmpty()) loadProfile(juce::File(path),profileName(),true);
}
void ToneHoundProcessor::setParameter(const juce::String& id,float value)
{
    if(auto* p=state.getParameter(id)) { p->beginChangeGesture(); p->setValueNotifyingHost(p->convertTo0to1(value)); p->endChangeGesture(); }
}
void ToneHoundProcessor::writeCacheLease()
{
    auto path=profilePath();if(path.isEmpty())return;
    cacheLease.getParentDirectory().createDirectory();
    cacheLease.replaceWithText(juce::JSON::toString(object({{"path",path}})));
}
void ToneHoundProcessor::resetEqMatch()
{
    const juce::ScopedLock guard(eqStateLock);
    eqMatch.reset();lastFitRevision=eqMatch.fitRevision();lastBandParameters.fill(0);
    for(int i=0;i<8;++i)setParameter("eqBand"+juce::String(i+1),0);
    setParameter("eqMatchEnabled",0);
}
void ToneHoundProcessor::timerCallback()
{
    collectRetired();
    if(++leaseTicks>=150){leaseTicks=0;writeCacheLease();}
    const auto latency=ampLatency.load()+currentPitchLatency.load();
    if(getLatencySamples()!=latency)setLatencySamples(latency);
    syncEqParameters();
}
void ToneHoundProcessor::syncEqParameters()
{
    const juce::ScopedLock guard(eqStateLock);
    if(eqMatch.fitRevision()!=lastFitRevision){
        lastFitRevision=eqMatch.fitRevision();lastBandParameters=eqMatch.gains();
        for(int i=0;i<8;++i)setParameter("eqBand"+juce::String(i+1),(float)lastBandParameters[(size_t)i]);
    }
    tonehound::MatchGains bands{};for(size_t i=0;i<8;++i)bands[i]=bandValues[i]->load();
    // Host automation and slider edits are applied away from the audio thread.
    if(bands!=lastBandParameters && !eqMatch.isCapturing()){
        lastBandParameters=bands;eqMatch.restore(bands,true);
    }
}
double ToneHoundProcessor::getTailLengthSeconds() const
{
    // Includes the longest permitted delay (1.5 s) at feedback .8 to below -60 dB.
    return effectValues[5]->load()>.5f || effectValues[10]->load()>.5f ? 50.0 : 0.1;
}
bool ToneHoundProcessor::startEqCapture()
{
    if(!modelReady || loading || audioFault || values[7]->load()>.5f || values[6]->load()<.5f)return false;
    if(!eqMatch.startCapture())return false;
    // A completed capture becomes audible automatically; the DSP bypasses it while recording.
    setParameter("eqMatchEnabled",1);return true;
}
void ToneHoundProcessor::applyPerformanceSuggestion(bool solo,float bpm)
{
    setParameter("performanceRole",solo?1.f:0.f);
    setParameter("delayBpm",std::isfinite(bpm) && bpm>0?juce::jlimit(40.f,240.f,bpm):120.f);
    setParameter("delayEnabled",solo?1.f:0.f);setParameter("reverbEnabled",solo?1.f:0.f);
    if(solo){setParameter("delayMix",.18f);setParameter("delayFeedback",.25f);setParameter("delayDivision",1);setParameter("reverbMix",.12f);setParameter("reverbSize",.5f);}
}
juce::String ToneHoundProcessor::profileName() const { const juce::ScopedLock g(infoLock); return currentName; }
juce::String ToneHoundProcessor::profilePath() const { const juce::ScopedLock g(infoLock); return currentPath; }
juce::String ToneHoundProcessor::modelMessage() const { const juce::ScopedLock g(infoLock); return message; }
juce::var ToneHoundProcessor::referenceInfo() const { const juce::ScopedLock g(infoLock);return reference.clone(); }
juce::var ToneHoundProcessor::matchInfo() const { const juce::ScopedLock g(infoLock);return matches.clone(); }
bool ToneHoundProcessor::hasMatches() const { const juce::ScopedLock g(infoLock);return matches.isArray(); }
juce::String ToneHoundProcessor::matchExplanation() const { const juce::ScopedLock g(infoLock);return lastMatchNote; }
void ToneHoundProcessor::setReference(juce::var r) { const juce::ScopedLock g(infoLock);reference=r.clone(); }
void ToneHoundProcessor::setMatches(juce::var r,juce::String note) { const juce::ScopedLock g(infoLock);matches=r.clone();lastMatchNote=std::move(note); }
void ToneHoundProcessor::setModelMessage(juce::String m) { const juce::ScopedLock g(infoLock); message=std::move(m); }
void ToneHoundProcessor::loadProfile(const juce::File& file,juce::String name,bool preserveMatchedEQ)
{
    // A captured correction belongs to the selected amp; do not carry it to another capture.
    if(!preserveMatchedEQ){resetEqMatch();for(auto id:{"bass","mid","treble"})setParameter(id,0);}
    auto version=++loadVersion;
    loading=true; setModelMessage("Loading capture...");
    // Publish the requested selection before starting the worker. An editor
    // opened during host-state restoration must retain this amp and its EQ.
    { const juce::ScopedLock g(infoLock);
      currentPath=file.getFullPathName(); currentName=name.isEmpty()?file.getFileNameWithoutExtension():name; }
    writeCacheLease();
    const auto sr=rate.load();
    loader.addJob([this,file,name,version,sr] {
        try {
            if(!file.existsAsFile() && file.getParentDirectory()==root.getChildFile(".cache/tone3000/profiles")){
                setModelMessage("Downloading selected capture...");
                AnalysisBridge resolver(root);resolver.submit(object({{"action","profile_resolve"},{"filename",file.getFileName()}}));
                auto deadline=juce::Time::getMillisecondCounterHiRes()+90000;
                while(resolver.snapshot().busy && version==loadVersion.load() && juce::Time::getMillisecondCounterHiRes()<deadline)juce::Thread::sleep(20);
                if(version!=loadVersion.load())return;
                auto result=resolver.snapshot();
                if(result.busy || !(bool)result.response["ok"])throw std::runtime_error(("Capture download failed: "+result.message).toStdString());
            }
            if(!file.existsAsFile() || !file.hasFileExtension("nam")) throw std::runtime_error("Choose a valid .nam capture.");
            auto graph=std::make_unique<ModelGraph>(file,sr);
            if(version!=loadVersion.load()) return;
            ampLatency=graph->latency;setLatencySamples(graph->latency+currentPitchLatency.load());
            auto* replaced=pendingModel.exchange(graph.release()); delete replaced;
            { const juce::ScopedLock g(infoLock);
              message="Capture ready"; }
            audioFault=false; modelReady=true;
        } catch(const std::exception& e) { if(version==loadVersion.load()) {modelReady=false;setModelMessage("Could not load capture: "+juce::String(e.what()));} }
        if(version==loadVersion.load()) loading=false;
        collectRetired();
    });
}
bool ToneHoundProcessor::loadProfileForTest(const juce::File& file,juce::String& error)
{
    try {
        auto graph=std::make_unique<ModelGraph>(file,rate.load());
        ampLatency=graph->latency;setLatencySamples(graph->latency+currentPitchLatency.load());
        delete pendingModel.exchange(graph.release());
        modelReady=true; audioFault=false;
        { const juce::ScopedLock g(infoLock); currentPath=file.getFullPathName(); currentName=file.getFileNameWithoutExtension(); }
        return true;
    } catch(const std::exception& e) { error=e.what(); return false; }
}
void ToneHoundProcessor::loadPreview(const juce::File& file)
{
    previewPlaying=false;
    auto version=++previewVersion;
    loader.addJob([this,file,version] {
        juce::AudioFormatManager formats; formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
        if(!reader || reader->lengthInSamples>std::numeric_limits<int>::max()) { setModelMessage("Could not open reference audio."); return; }
        auto audio=std::make_unique<PreviewAudio>(); audio->rate=reader->sampleRate;
        audio->samples.setSize(juce::jmin(2,(int)reader->numChannels),(int)reader->lengthInSamples);
        reader->read(&audio->samples,0,audio->samples.getNumSamples(),0,true,true);
        if(version!=previewVersion.load()) return;
        delete pendingPreview.exchange(audio.release()); previewRestart=true;
        collectRetired();
    });
}
void ToneHoundProcessor::setSelection(double start,double end)
{
    selectionStart=juce::jmax(0.0,start); selectionEnd=juce::jmax(start+30.0,end); previewRestart=true;
}
void ToneHoundProcessor::collectRetired() { freeRetired(retiredModels); freeRetired(retiredPreviews); eqMatch.collectRetired(); }

void ToneHoundProcessor::Biquad::set(int type,double frequency,double gainDB,double sr)
{
    const auto A=std::pow(10.0,gainDB/40.0), w=2*juce::MathConstants<double>::pi*juce::jmin(frequency,0.45*sr)/sr;
    const auto c=std::cos(w), s=std::sin(w), alpha=s/std::sqrt(2.0), beta=2*std::sqrt(A)*alpha;
    double a0;
    if(type==0) {
        b0=A*((A+1)-(A-1)*c+beta); b1=2*A*((A-1)-(A+1)*c); b2=A*((A+1)-(A-1)*c-beta);
        a0=(A+1)+(A-1)*c+beta; a1=-2*((A-1)+(A+1)*c); a2=(A+1)+(A-1)*c-beta;
    } else if(type==2) {
        b0=A*((A+1)+(A-1)*c+beta); b1=-2*A*((A-1)+(A+1)*c); b2=A*((A+1)+(A-1)*c-beta);
        a0=(A+1)-(A-1)*c+beta; a1=2*((A-1)-(A+1)*c); a2=(A+1)-(A-1)*c-beta;
    } else {
        const auto al=s/(2*0.75); b0=1+al*A; b1=-2*c; b2=1-al*A; a0=1+al/A; a1=-2*c; a2=1-al/A;
    }
    b0/=a0; b1/=a0; b2/=a0; a1/=a0; a2/=a0;
}
void ToneHoundProcessor::processBlock(juce::AudioBuffer<float>& buffer,juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals; midi.clear();
    if(auto* p=pendingModel.exchange(nullptr)) { retire(retiredModels,active); active=p; }
    if(auto* p=pendingPreview.exchange(nullptr)) { retire(retiredPreviews,preview); preview=p; previewCursor=0; }
    const auto sr=rate.load();
    const auto channels=buffer.getNumChannels(), n=buffer.getNumSamples();
    const int inputChannel=juce::jlimit(0,juce::jmax(0,getTotalNumInputChannels()-1),(int)values[8]->load());
    inputGain.setTargetValue(juce::Decibels::decibelsToGain(values[0]->load()));
    outputGain.setTargetValue(juce::Decibels::decibelsToGain(values[1]->load()));
    monitorGain.setTargetValue(values[6]->load()>0.5f?1.0f:0.0f);
    const bool bypass=values[7]->load()>0.5f;
    const int semitones=bypass?0:(int)pitchValue->load();
    const bool studioPitch=pitchStudioValue->load()>.5f;
    const float pitchCents=bypass?0.f:pitchCentsValue->load();
    currentPitchLatency=pitchShift.latencyFor(semitones,studioPitch,pitchCents);
    tonehound::PedalSettings settings;
    settings.screamer=!bypass && effectValues[1]->load()>.5f;settings.drive=effectValues[2]->load();settings.tone=effectValues[3]->load();settings.level=effectValues[4]->load();
    settings.delay=!bypass && effectValues[5]->load()>.5f;settings.delayMix=effectValues[6]->load();settings.feedback=effectValues[7]->load();settings.division=(int)effectValues[8]->load();settings.bpm=effectValues[9]->load();
    settings.reverb=!bypass && effectValues[10]->load()>.5f;settings.reverbMix=effectValues[11]->load();settings.roomSize=effectValues[12]->load();
    pedals.update(settings);eqMatch.beginBlock(!bypass && effectValues[0]->load()>.5f);
    const auto gateDB=values[2]->load(), threshold=juce::Decibels::decibelsToGain(gateDB);
    const double release=std::exp(-1.0/(0.08*sr)), attack=std::exp(-1.0/(0.001*sr));
    const double frequencies[]{180,850,4200};
    for(int i=0;i<3;++i) { auto db=values[(size_t)(3+i)]->load(); if(db!=previousEQ[i]) { eq[i].set(i,frequencies[i],db,sr); previousEQ[i]=db; } }
    float inPeak=0,outPeak=0;
    auto play=previewPlaying.load() && preview;
    auto begin=selectionStart.load(), end=selectionEnd.load();
    if(previewRestart.exchange(false)) previewCursor=begin*(preview?preview->rate:48000);
    for(int offset=0;offset<n;offset+=512) {
        auto count=juce::jmin(512,n-offset);
        for(int i=0;i<count;++i) {
            auto x=channels>0?buffer.getSample(inputChannel,offset+i):0.0f;
            if(!std::isfinite(x)) x=0;
            inPeak=juce::jmax(inPeak,std::abs(x));
            const auto level=std::abs(x);
            gateEnvelope=level+(gateEnvelope-level)*(level>gateEnvelope?attack:release);
            const auto wanted=gateDB<=-99 || gateEnvelope>threshold?1.0:0.0;
            gateGain=wanted+(gateGain-wanted)*(wanted>gateGain?attack:release);
            mono[(size_t)i]=(float)(x*inputGain.getNextValue()*gateGain);
        }
        pitchShift.process(mono.data(),count,semitones,studioPitch,pitchBlendValue->load(),pitchCents,pitchAttackValue->load()>.5f);
        for(int i=0;i<count;++i)mono[(size_t)i]=pedals.preAmp((float)mono[(size_t)i]);
        if(active && modelReady.load() && !bypass && !audioFault.load() && std::abs(active->hostRate-sr)<0.5) {
            try { active->process(mono.data(),wet.data(),count); }
            catch(...) { audioFault=true; std::fill_n(wet.data(),count,0.0); }
        } else if(bypass) std::copy_n(mono.data(),count,wet.data());
        else std::fill_n(wet.data(),count,0.0);
        for(int i=0;i<count;++i) {
            float guitar=(float)wet[(size_t)i];
            if(!std::isfinite(guitar)){guitar=0;audioFault=true;}
            if(!bypass) for(auto& filter:eq) guitar=filter.process(guitar);
            eqMatch.record(guitar); // Reference playback is added only after this measurement point.
            guitar=eqMatch.process(guitar);
            float guitarLeft,guitarRight;pedals.postAmp(guitar,guitarLeft,guitarRight);
            const auto gain=outputGain.getNextValue()*monitorGain.getNextValue();
            guitarLeft*=gain;guitarRight*=gain;
            const double previewEnd=preview?juce::jmin(end*preview->rate,(double)preview->samples.getNumSamples()):0;
            if(play && previewCursor>=previewEnd) {
                if(previewLoop.load()) previewCursor=begin*preview->rate;
                else { play=false; previewPlaying=false; }
            }
            for(int ch=0;ch<channels;++ch) {
                float backing=0;
                if(play && previewCursor>=0 && previewCursor<previewEnd && preview->samples.getNumSamples()>1) {
                    int a=juce::jlimit(0,preview->samples.getNumSamples()-1,(int)previewCursor);
                    int b=juce::jmin(a+1,preview->samples.getNumSamples()-1);
                    auto* data=preview->samples.getReadPointer(juce::jmin(ch,preview->samples.getNumChannels()-1));
                    backing=(data[a]+(data[b]-data[a])*(float)(previewCursor-a))*0.35f;
                }
                float y=(channels==1?.5f*(guitarLeft+guitarRight):(ch==0?guitarLeft:guitarRight))+backing;
                if(!std::isfinite(y)) { y=0; audioFault=true; }
                y=juce::jlimit(-1.0f,1.0f,y);
                buffer.setSample(ch,offset+i,y); outPeak=juce::jmax(outPeak,std::abs(y));
            }
            if(play) previewCursor+=preview->rate/sr;
        }
    }
    if(preview) previewPosition=previewCursor/preview->rate;
    inputPeak.store(juce::jmax(inPeak,inputPeak.load()*0.9f)); outputPeak.store(juce::jmax(outPeak,outputPeak.load()*0.9f));
}
void ToneHoundProcessor::getStateInformation(juce::MemoryBlock& data)
{
    const juce::ScopedLock guard(eqStateLock);
    // Include edits made immediately before a host save, even without a timer tick.
    syncEqParameters();
    auto tree=state.copyState(); tree.setProperty("profile",profilePath(),nullptr);
    tree.setProperty("profileName",profileName(),nullptr);
    tree.setProperty("reference",juce::JSON::toString(referenceInfo()),nullptr);
    tree.setProperty("matches",juce::JSON::toString(matchInfo()),nullptr);
    tree.setProperty("matchNote",matchExplanation(),nullptr);
    tree.setProperty("selectionStart",selectionStart.load(),nullptr); tree.setProperty("selectionEnd",selectionEnd.load(),nullptr);
    juce::Array<juce::var> curve;for(auto gain:eqMatch.gains())curve.add(gain);
    tree.setProperty("eqMatchCurve",juce::JSON::toString(curve),nullptr);tree.setProperty("eqMatchValid",eqMatch.ready(),nullptr);
    if(auto xml=tree.createXml()) copyXmlToBinary(*xml,data);
}
void ToneHoundProcessor::setStateInformation(const void* data,int size)
{
    const juce::ScopedLock guard(eqStateLock);
    if(auto xml=getXmlFromBinary(data,size)) {
        auto tree=juce::ValueTree::fromXml(*xml);
        if(!tree.hasType(state.state.getType())) return;
        state.replaceState(tree);
        const bool restoreMatchedEQ=effectValues[0]->load()>.5f;
        setReference(juce::JSON::parse(tree["reference"].toString())); setMatches(juce::JSON::parse(tree["matches"].toString()),tree["matchNote"].toString());
        auto path=tree["profile"].toString(); if(path.isNotEmpty()) loadProfile(juce::File(path),tree["profileName"].toString(),true);
        selectionStart=tree.getProperty("selectionStart",0.0); selectionEnd=tree.getProperty("selectionEnd",40.0);
        auto ref=referenceInfo()["path"].toString(); if(ref.isNotEmpty()) loadPreview(juce::File(ref));
        auto stored=juce::JSON::parse(tree["eqMatchCurve"].toString());tonehound::MatchGains gain{};
        if(stored.isArray() && (stored.size()==(int)tonehound::matchBands || stored.size()==15)) {
            for(size_t i=0;i<gain.size();++i){
                if(stored.size()==8)gain[i]=(double)stored[(int)i];
                else {
                    // Approximate migration of the earlier half-octave curve.
                    auto position=juce::jlimit(0.,14.,2*std::log2(tonehound::matchFrequency(i)/80.));
                    auto lo=(int)position,hi=juce::jmin(14,lo+1);
                    gain[i]=(double)stored[lo]+(position-lo)*((double)stored[hi]-(double)stored[lo]);
                }
            }
            for(int i=0;i<8;++i){
                auto value=gain[(size_t)i];
                setParameter("eqBand"+juce::String(i+1),std::isfinite(value)?(float)juce::jlimit(-12.,12.,value):0.f);
                gain[(size_t)i]=bandValues[(size_t)i]->load();
            }
            eqMatch.restore(gain,(bool)tree.getProperty("eqMatchValid",false));
            lastBandParameters=gain;lastFitRevision=eqMatch.fitRevision();
            setParameter("eqMatchEnabled",restoreMatchedEQ?1.f:0.f);
        }
    }
}
juce::AudioProcessorEditor* ToneHoundProcessor::createEditor() { return new ToneHoundEditor(*this); }
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new ToneHoundProcessor(); }
