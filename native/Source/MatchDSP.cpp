#include "MatchDSP.h"
#include <complex>
#include <functional>

namespace tonehound {
double matchFrequency(size_t i) { constexpr double frequencies[]{63,125,250,500,1000,2000,4000,8000}; return frequencies[juce::jmin(i,matchBands-1)]; }
void MatchFilter::set(double frequency,double gainDB,double rate)
{
    const auto a=std::pow(10.,gainDB/40.),w=2.*juce::MathConstants<double>::pi*juce::jmin(frequency,rate*.45)/rate;
    const auto alpha=std::sin(w)/(2.*0.9),den=1.+alpha/a;
    b0=(1.+alpha*a)/den;b1=-2.*std::cos(w)/den;b2=(1.-alpha*a)/den;
    a1=b1;a2=(1.-alpha/a)/den;
}
float MatchFilter::process(float x) noexcept
{
    const auto y=b0*x+z1;z1=b1*x-a1*y+z2;z2=b2*x-a2*y;
    return (float)y;
}
double MatchFilter::response(double frequency,double rate) const
{
    auto z=std::exp(std::complex<double>(0.,-2.*juce::MathConstants<double>::pi*frequency/rate));
    return 20.*std::log10(std::max(1.e-12,std::abs((b0+b1*z+b2*z*z)/(1.+a1*z+a2*z*z))));
}

namespace {
constexpr int fftSize=4096;
// Fixed memory regardless of reference duration; evenly sample at most 1,500 frames.
Spectrum measure(int channels,juce::int64 samples,double rate,
                 const std::function<bool(juce::int64,juce::AudioBuffer<float>&)>& read)
{
    if(channels<1 || samples<fftSize || rate<8000) throw std::runtime_error("The reference is too short for EQ analysis.");
    juce::dsp::FFT fft(12);
    juce::AudioBuffer<float> frame(channels,fftSize);
    std::array<float,fftSize*2> fftData{};
    std::array<double,fftSize/2+1> accumulated{};
    const auto totalFrames=1+(samples-fftSize)/(fftSize/2);
    const auto frames=(int)std::min<juce::int64>(1500,totalFrames);
    double square=0;size_t seen=0;int audible=0;
    for(int f=0;f<frames;++f) {
        const auto pos=frames>1?(juce::int64)((double)f*(double)(samples-fftSize)/(frames-1)):0;
        if(!read(pos,frame)) throw std::runtime_error("Could not read reference audio.");
        double frameSquare=0;
        for(int ch=0;ch<channels;++ch) {
            fftData.fill(0);
            for(int i=0;i<fftSize;++i) {
                double x=frame.getSample(ch,i);if(!std::isfinite(x))x=0;
                frameSquare+=x*x;
                fftData[(size_t)i]=(float)(x*(.5-.5*std::cos(2.*juce::MathConstants<double>::pi*i/(fftSize-1))));
            }
            fft.performRealOnlyForwardTransform(fftData.data(),true);
            for(size_t i=0;i<accumulated.size();++i) accumulated[i]+=(double)fftData[2*i]*fftData[2*i]+(double)fftData[2*i+1]*fftData[2*i+1];
        }
        square+=frameSquare;seen+=(size_t)channels*fftSize;
        if(frameSquare/((double)channels*fftSize)>1.e-7)++audible;
    }
    Spectrum result;result.rms=std::sqrt(square/std::max<size_t>(1,seen));result.audibleFrames=audible;
    for(size_t band=0;band<matchBands;++band) {
        double sum=0,weight=0;const auto center=matchFrequency(band);
        for(size_t bin=1;bin<accumulated.size();++bin) {
            const auto hz=(double)bin*rate/fftSize;
            const auto distance=std::abs(std::log2(hz/center));
            const auto w=std::max(0.,1.-distance/.5);
            sum+=w*accumulated[bin];weight+=w;
        }
        result.power[band]=sum/std::max(1.,weight*frames*channels*fftSize*fftSize);
    }
    if(result.rms<2.e-4 || audible<juce::jmin(8,frames)) throw std::runtime_error("Too little guitar signal. Check the input and play continuously for 10 seconds.");
    return result;
}
}
Spectrum analyseSpectrum(const float* const* data,int channels,int samples,double rate)
{
    return measure(channels,samples,rate,[&](juce::int64 start,juce::AudioBuffer<float>& block) {
        for(int ch=0;ch<channels;++ch)block.copyFrom(ch,0,data[ch]+(size_t)start,fftSize);
        return true;
    });
}
MatchGains fitMatchCurve(const Spectrum& target,const Spectrum& played,double rate)
{
    const auto targetMax=*std::max_element(target.power.begin(),target.power.end());
    const auto sourceMax=*std::max_element(played.power.begin(),played.power.end());
    if(targetMax<1.e-14 || sourceMax<1.e-14) throw std::runtime_error("Not enough signal to compute a reliable EQ curve.");
    MatchGains desired{},weight{},smooth{};double level=0,weightSum=0;
    for(size_t i=0;i<matchBands;++i) {
        const auto t=target.power[i]/targetMax,s=played.power[i]/sourceMax;
        // Suppress unexcited bands; an absent harmonic is not evidence for a large boost.
        weight[i]=juce::jlimit(0.,1.,std::min(t,s)/.002);
        if(matchFrequency(i)>rate*.42)weight[i]=0;
        desired[i]=10.*std::log10(std::max(target.power[i],targetMax*1.e-6)/std::max(played.power[i],sourceMax*1.e-6));
        level+=desired[i]*weight[i];weightSum+=weight[i];
    }
    if(weightSum<4)throw std::runtime_error("Play varied notes or chords across the strings; the sample has too little frequency coverage.");
    level/=weightSum;
    for(size_t i=0;i<matchBands;++i)desired[i]=juce::jlimit(-6.,6.,.8*(desired[i]-level));
    for(size_t i=0;i<matchBands;++i) {
        double sum=2.*desired[i]*weight[i],w=2.*weight[i];
        if(i>0){sum+=desired[i-1]*weight[i-1];w+=weight[i-1];}
        if(i+1<matchBands){sum+=desired[i+1]*weight[i+1];w+=weight[i+1];}
        smooth[i]=w>0?sum/w:0;
    }
    std::array<std::array<double,matchBands>,matchBands> basis{},matrix{};
    MatchGains rhs{},gains{};
    for(size_t j=0;j<matchBands;++j) {
        MatchFilter filter;filter.set(matchFrequency(j),1.,rate);
        for(size_t i=0;i<matchBands;++i)basis[i][j]=filter.response(matchFrequency(i),rate);
    }
    for(size_t i=0;i<matchBands;++i) {
        for(size_t k=0;k<matchBands;++k)rhs[i]+=basis[k][i]*weight[k]*smooth[k];
        for(size_t j=0;j<matchBands;++j) {
            for(size_t k=0;k<matchBands;++k)matrix[i][j]+=basis[k][i]*weight[k]*basis[k][j];
            if(i==j)matrix[i][j]+=.25+(i>0?.2:0)+(i+1<matchBands?.2:0);
            if(i+1==j || j+1==i)matrix[i][j]-=.2;
        }
    }
    for(int iteration=0;iteration<60;++iteration)for(size_t i=0;i<matchBands;++i) {
        auto value=rhs[i];for(size_t j=0;j<matchBands;++j)if(i!=j)value-=matrix[i][j]*gains[j];
        gains[i]=juce::jlimit(-6.,6.,value/matrix[i][i]);
    }
    // Bound the actual summed response, not just each individual band's gain.
    for(int pass=0;pass<3;++pass) {
        std::array<MatchFilter,matchBands> filters;
        for(size_t i=0;i<matchBands;++i)filters[i].set(matchFrequency(i),gains[i],rate);
        double peak=0;
        for(int point=0;point<256;++point) {
            auto hz=30.*std::pow(std::min(20000.,rate*.48)/30.,point/255.);double response=0;
            for(auto& f:filters)response+=f.response(hz,rate);
            peak=std::max(peak,std::abs(response));
        }
        if(peak<=6.)break;
        for(auto& gain:gains)gain*=5.95/peak;
    }
    return gains;
}

MatchEQ::~MatchEQ()
{
    ++generation;phase=idle;worker.removeAllJobs(true,-1);
    delete pending.exchange(nullptr);collectRetired();
}
void MatchEQ::prepare(double sr)
{
    ++generation;phase=idle;worker.removeAllJobs(true,-1);busy=false;
    sampleRate=sr;capture.resize((size_t)std::ceil(sr*10.));captured=0;
    blend.reset(sr,.025);blend.setCurrentAndTargetValue(0);
    transition.reset(sr,.025);transition.setCurrentAndTargetValue(1);
    for(auto& f:filters)f={};for(auto& f:oldFilters)f={};
    publish(gains(),curveValid.load());
}
void MatchEQ::setMessage(juce::String s){const juce::ScopedLock guard(lock);detail=std::move(s);}
juce::String MatchEQ::status() const
{
    if(phase==capturing)return "Play guitar: "+juce::String(10.*(1.-progress()),1)+" seconds remaining";
    if(phase==analysing)return "Calculating your EQ curve...";
    const juce::ScopedLock guard(lock);return detail;
}
float MatchEQ::progress() const noexcept{return capture.empty()?0:(float)captured.load()/(float)capture.size();}
MatchGains MatchEQ::gains() const{const juce::ScopedLock guard(lock);return savedGains;}
void MatchEQ::collectRetired()
{
    auto* item=retired.exchange(nullptr);while(item){auto* next=item->next;delete item;item=next;}
}
void MatchEQ::publish(const MatchGains& gain,bool valid)
{
    auto packet=std::make_unique<Packet>();packet->gains=gain;
    for(size_t i=0;i<matchBands;++i)packet->filters[i].set(matchFrequency(i),gain[i],sampleRate.load());
    {const juce::ScopedLock guard(lock);savedGains=gain;}
    delete pending.exchange(packet.release());curveValid=valid;collectRetired();
}
void MatchEQ::loadTarget(const juce::File& file,bool preserveCurve)
{
    const juce::ScopedLock control(lock);
    auto version=++generation;phase=idle;targetValid=false;busy=true;
    setMessage("Preparing the reference EQ...");
    if(!preserveCurve)publish({},false);
    worker.addJob([this,file,version] {
        try {
            juce::AudioFormatManager formats;formats.registerBasicFormats();
            std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
            if(!reader)throw std::runtime_error("The separated guitar reference could not be opened. Match the song again.");
            auto spectrum=measure(juce::jmin(2,(int)reader->numChannels),reader->lengthInSamples,reader->sampleRate,
                [&](juce::int64 offset,juce::AudioBuffer<float>& block){
                    if(version!=generation.load())throw std::runtime_error("Cancelled");
                    return reader->read(&block,0,fftSize,offset,true,true);
                });
            const juce::ScopedLock guard(lock);
            if(version==generation.load()) {
                target=spectrum;
                targetValid=true;setMessage("Reference ready. Play varied notes or chords for 10 seconds.");
            }
        }catch(const std::exception& e){const juce::ScopedLock guard(lock);if(version==generation.load())setMessage(e.what());}
        {const juce::ScopedLock guard(lock);if(version==generation.load())busy=false;}
        collectRetired();
    });
}
void MatchEQ::clearTarget()
{
    const juce::ScopedLock control(lock);
    ++generation;phase=idle;targetValid=false;busy=false;
    publish({},false);setMessage("Match an amp to prepare the reference EQ");
}
bool MatchEQ::startCapture()
{
    const juce::ScopedLock control(lock);
    if(!targetValid || capture.empty() || busy.exchange(true))return false;
    const auto version=++generation;captured=0;
    Spectrum comparison;{const juce::ScopedLock guard(lock);comparison=target;}
    phase=capturing;
    worker.addJob([this,version,comparison] {
        while(version==generation.load() && phase.load()==capturing)juce::Thread::sleep(10);
        if(version!=generation.load())return;
        try {
            const float* data[]{capture.data()};
            auto played=analyseSpectrum(data,1,(int)captured.load(),sampleRate.load());
            auto curve=fitMatchCurve(comparison,played,sampleRate.load());
            const juce::ScopedLock guard(lock);
            if(version==generation.load()){publish(curve,true);++fittedRevision;setMessage("EQ matched. Adjust the eight bands or bypass to compare.");}
        }catch(const std::exception& e){const juce::ScopedLock guard(lock);if(version==generation.load())setMessage(e.what());}
        {const juce::ScopedLock guard(lock);if(version==generation.load()){phase=idle;busy=false;}}
    });
    return true;
}
void MatchEQ::cancelCapture()
{
    const juce::ScopedLock control(lock);
    ++generation;phase=idle;
    // Queue the reset after the worker relinquishes the capture memory.
    const auto version=generation.load();
    worker.addJob([this,version]{const juce::ScopedLock guard(lock);if(version==generation.load())busy=false;});
    setMessage(targetValid?"Capture cancelled. Ready to try again.":"Match an amp to prepare the reference EQ");
}
void MatchEQ::reset()
{
    const juce::ScopedLock control(lock);
    // Changing the amp may coincide with preparation of a new song target.
    // Cancel the player measurement, without cancelling that target's file job.
    if(phase.load()!=idle)cancelCapture();
    publish({},false);
    if(targetValid)setMessage("Reference ready. Play varied notes or chords for 10 seconds.");
}
void MatchEQ::restore(const MatchGains& gain,bool valid)
{
    const juce::ScopedLock control(lock);
    MatchGains sanitized{};for(size_t i=0;i<matchBands;++i)sanitized[i]=std::isfinite(gain[i])?juce::jlimit(-12.,12.,gain[i]):0;
    publish(sanitized,valid);
    if(valid)setMessage("EQ curve ready. Adjust a band or capture a new match.");
}
std::vector<std::pair<float,float>> MatchEQ::curve() const
{
    auto gain=gains();std::array<MatchFilter,matchBands> f;
    for(size_t i=0;i<matchBands;++i)f[i].set(matchFrequency(i),gain[i],sampleRate.load());
    std::vector<std::pair<float,float>> result;result.reserve(96);
    for(int i=0;i<96;++i){auto hz=40.*std::pow(16000./40.,i/95.);double db=0;for(auto& filter:f)db+=filter.response(hz,sampleRate.load());result.emplace_back((float)hz,(float)db);}
    return result;
}
void MatchEQ::beginBlock(bool enabled) noexcept
{
    if(auto* packet=pending.exchange(nullptr)) {
        oldFilters=filters;filters=packet->filters;transition.setCurrentAndTargetValue(0);transition.setTargetValue(1);
        packet->next=retired.load();while(!retired.compare_exchange_weak(packet->next,packet)){}
    }
    blend.setTargetValue(enabled && curveValid.load() && phase.load()!=capturing?1.f:0.f);
}
void MatchEQ::record(float x) noexcept
{
    if(phase.load()!=capturing)return;
    auto index=captured.load();
    if(index<capture.size())capture[index++]=std::isfinite(x)?x:0;
    captured.store(index);
    if(index>=capture.size()){int expected=capturing;phase.compare_exchange_strong(expected,analysing);}
}
float MatchEQ::process(float x) noexcept
{
    auto y=x;for(auto& f:filters)y=f.process(y);
    auto t=transition.getNextValue();
    if(t<1){auto old=x;for(auto& f:oldFilters)old=f.process(old);y=old+t*(y-old);}
    return x+blend.getNextValue()*(y-x);
}

void PedalChain::prepare(double sr)
{
    sampleRate=sr;delayLeft.assign((size_t)std::ceil(sr*3.2)+4,0);delayRight=delayLeft;position=0;
    delayLowLeft=delayLowRight=0;highPassMemory=previousInput=toneMemory=0;
    highPassPole=std::exp(-2.*juce::MathConstants<double>::pi*180./sr);
    for(auto* s:{&screamBlend,&drive,&tone,&level,&delayMix,&feedback,&reverbMix})s->reset(sr,.025);
    delaySamples.reset(sr,.1);delaySamples.setCurrentAndTargetValue((float)(sr*.375));
    screamBlend.setCurrentAndTargetValue(0);delayMix.setCurrentAndTargetValue(0);reverbMix.setCurrentAndTargetValue(0);
    drive.setCurrentAndTargetValue(.25f);tone.setCurrentAndTargetValue(.5f);level.setCurrentAndTargetValue(.5f);feedback.setCurrentAndTargetValue(.25f);
    room.setSampleRate(sr);room.reset();previous.roomSize=-1;
}
void PedalChain::update(const PedalSettings& s) noexcept
{
    screamBlend.setTargetValue(s.screamer?1.f:0.f);drive.setTargetValue(s.drive);tone.setTargetValue(s.tone);level.setTargetValue(s.level);
    delayMix.setTargetValue(s.delay?s.delayMix:0);feedback.setTargetValue(juce::jlimit(0.f,.8f,s.feedback));
    const double beats[]{1.,.75,.5,1./3.};
    delaySamples.setTargetValue((float)(sampleRate*60./juce::jlimit(40.f,240.f,s.bpm)*beats[juce::jlimit(0,3,s.division)]));
    reverbMix.setTargetValue(s.reverb?s.reverbMix:0);
    if(s.roomSize!=previous.roomSize) {
        juce::Reverb::Parameters p;p.roomSize=.15f+.7f*s.roomSize;p.damping=.6f;p.wetLevel=1.f/3.f;p.dryLevel=0;p.width=1;
        room.setParameters(p);
    }
    previous=s;
}
float PedalChain::preAmp(float x) noexcept
{
    const auto hp=highPassPole*(highPassMemory+x-previousInput);previousInput=x;highPassMemory=hp;
    const auto gain=2.+28.*drive.getNextValue();
    const auto clipped=std::tanh(hp*gain)/std::tanh(gain);
    tonePole=std::exp(-2.*juce::MathConstants<double>::pi*(1000.+7000.*tone.getNextValue())/sampleRate);
    toneMemory=clipped+(toneMemory-clipped)*tonePole;
    const auto processed=(float)(toneMemory*(.15+1.7*level.getNextValue()));
    return x+screamBlend.getNextValue()*(processed-x);
}
void PedalChain::postAmp(float x,float& left,float& right) noexcept
{
    if(delayLeft.empty()){left=right=x;return;}
    const auto delay=delaySamples.getNextValue();
    const auto read=[&](const std::vector<float>& line,double distance) {
        double index=(double)position-distance;while(index<0)index+=(double)line.size();
        auto a=(size_t)index,b=(a+1)%line.size();return line[a]+(float)(index-a)*(line[b]-line[a]);
    };
    auto dl=read(delayLeft,delay),dr=read(delayRight,delay*1.012);
    const auto pole=(float)std::exp(-2.*juce::MathConstants<double>::pi*4500./sampleRate);
    delayLowLeft=dl+(delayLowLeft-dl)*pole;delayLowRight=dr+(delayLowRight-dr)*pole;
    const auto feedbackGain=feedback.getNextValue(),mix=delayMix.getNextValue();
    // Cross feedback gives a stereo echo. Never feed the reference player into this chain.
    delayLeft[position]=(mix>0?x:0)+feedbackGain*delayLowRight;
    delayRight[position]=(mix>0?x:0)+feedbackGain*delayLowLeft;
    position=(position+1)%delayLeft.size();
    left=x+mix*dl;right=x+mix*dr;
    auto wetL=left,wetR=right;room.processStereo(&wetL,&wetR,1);
    const auto rm=reverbMix.getNextValue();left+=rm*wetL;right+=rm*wetR;
}
} // namespace tonehound
