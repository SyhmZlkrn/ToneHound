#include "PitchShift.h"
void GuitarPitchShift::prepare(double sr)
{
    // Preallocate both modes; no reconfiguration/allocation in the callback.
    engines[0].configure(1,(int)std::round(sr*.032),(int)std::round(sr*.008),true);
    engines[1].configure(1,(int)std::round(sr*.192),(int)std::round(sr*.048),false);
    for(size_t i=0;i<engines.size();++i){engines[i].reset();latencies[i]=engines[i].inputLatency()+engines[i].outputLatency();}
    previous=0;previousStudio=false;previousCents=0;
    dryDelay.assign((size_t)*std::max_element(latencies.begin(),latencies.end())+1,0.);
    dryPosition=0;dryEnvelope=wetEnvelope=0;
    attackCoefficient=std::exp(-1./(.002*sr));releaseCoefficient=std::exp(-1./(.012*sr));
    fade.reset(sr,.01);fade.setCurrentAndTargetValue(1);
    wetMix.reset(sr,.02);wetMix.setCurrentAndTargetValue(1);
    attackMix.reset(sr,.02);attackMix.setCurrentAndTargetValue(1);
}
void GuitarPitchShift::process(double* data,int count,int semitones,bool studio,float blend,float cents,bool attack)
{
    semitones=juce::jlimit(-12,12,semitones);cents=juce::jlimit(-50.f,50.f,cents);
    wetMix.setTargetValue(juce::jlimit(0.f,1.f,blend));attackMix.setTargetValue(attack?1.f:0.f);
    auto& stretch=engines[studio?1:0];
    if(semitones!=previous || studio!=previousStudio || cents!=previousCents){
        const bool reset=(previous==0 && previousCents==0) || (semitones==0 && cents==0) || studio!=previousStudio;
        if(reset){
            stretch.reset();std::fill(dryDelay.begin(),dryDelay.end(),0.);dryPosition=0;dryEnvelope=wetEnvelope=0;
        }
        stretch.setTransposeSemitones(semitones+cents/100.f);previous=semitones;previousCents=cents;
        previousStudio=studio;
        // Do not repeatedly mute held notes during automation of fine tuning.
        if(reset){fade.setCurrentAndTargetValue(0);fade.setTargetValue(1);}
    }
    if(semitones==0 && cents==0)return;
    // Work with arbitrary host blocks while keeping the callback allocation free.
    for(int offset=0;offset<count;offset+=(int)input.size()){
        const int n=juce::jmin((int)input.size(),count-offset);
        for(int i=0;i<n;++i)input[(size_t)i]=(float)data[offset+i];
        const float* in[]{input.data()};float* out[]{output.data()};stretch.process(in,n,out,n);
        for(int i=0;i<n;++i){
            dryDelay[dryPosition]=data[offset+i];
            const auto read=(dryPosition+dryDelay.size()-(size_t)latencies[studio?1:0])%dryDelay.size();
            const double dry=dryDelay[read],wet=output[(size_t)i];
            dryPosition=(dryPosition+1)%dryDelay.size();
            auto follow=[&](double sample,double& envelope){const double power=sample*sample;
                const double coefficient=power>envelope?attackCoefficient:releaseCoefficient;
                envelope=power+coefficient*(envelope-power);};
            follow(dry,dryEnvelope);follow(wet,wetEnvelope);
            // Match the delayed DI's attack envelope without leaking its pitch.
            // Bounded correction avoids amplifying near-silence or lost bins.
            const double correction=juce::jlimit(0.,2.,std::sqrt(dryEnvelope/(wetEnvelope+1.e-12)));
            const float a=attackMix.getNextValue(),mix=wetMix.getNextValue();
            data[offset+i]=(dry*(1-mix)+wet*mix*(1+a*(correction-1)))*fade.getNextValue();
        }
    }
}
