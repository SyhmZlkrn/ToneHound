#include "PluginEditor.h"
#include <iostream>
#include <chrono>
void runMatchDSPTests();

static void require(bool ok,const juce::String& message)
{if(!ok) throw std::runtime_error(message.toStdString());}
static void pump(int ms) {juce::MessageManager::getInstance()->runDispatchLoopUntil(ms);}
static juce::String option(const juce::StringArray& args,const juce::String& name)
{auto i=args.indexOf(name);return i>=0?args[i+1]:juce::String{};}

int main(int argc,char** argv)
{
    juce::ScopedJuceInitialiser_GUI gui;
    juce::StringArray args;for(int i=1;i<argc;++i)args.add(argv[i]);
    try {
        if(args.contains("--dsp-check")){runMatchDSPTests();return 0;}
        auto root=findProjectRoot();
        if(args.contains("--pitch-snapshot")){
            ToneHoundProcessor p;
            auto panel=makePitchPanel(p);
            auto* blend=dynamic_cast<juce::Slider*>(panel->findChildWithID("pitch.blend"));
            auto* cents=dynamic_cast<juce::Slider*>(panel->findChildWithID("pitch.cents"));
            auto* quality=dynamic_cast<juce::ComboBox*>(panel->findChildWithID("pitch.quality"));
            require(blend && cents && quality,"Pitch controls missing");
            blend->setValue(.5,juce::sendNotificationSync);cents->setValue(-12,juce::sendNotificationSync);quality->setSelectedId(2,juce::sendNotificationSync);
            require(blend->getTextFromValue(.5)=="50% wet" && std::abs(blend->getValueFromText("50% wet")-.5)<.001,"Pitch blend percentage is not readable/editable");
            require(std::abs(p.state.getRawParameterValue("pitchBlend")->load()-.5f)<.001f && p.state.getRawParameterValue("pitchCents")->load()==-12 && p.state.getRawParameterValue("pitchQuality")->load()==1,"Pitch controls are not connected to audio parameters");
            juce::MemoryBlock saved;p.getStateInformation(saved);ToneHoundProcessor restored;restored.setStateInformation(saved.getData(),(int)saved.getSize());
            require(restored.state.getRawParameterValue("pitchCents")->load()==-12 && std::abs(restored.state.getRawParameterValue("pitchBlend")->load()-.5f)<.001f,"Pitch preset did not restore");
            auto image=panel->createComponentSnapshot(panel->getLocalBounds());juce::File output(option(args,"--pitch-snapshot"));output.getParentDirectory().createDirectory();auto stream=output.createOutputStream();require(stream!=nullptr,"Cannot write pitch snapshot");stream->setPosition(0);stream->truncate();juce::PNGImageFormat png;require(png.writeImageToStream(image,*stream),"Cannot export pitch settings");
            std::cout<<"PASS: pitch settings attachments and preset restore"<<std::endl;return 0;
        }
        auto profiles=root.getChildFile(".cache/tone3000/profiles").findChildFiles(juce::File::findFiles,false,"*.nam");
        require(!profiles.isEmpty(),"No NAM profiles found");
        auto file=option(args,"--profile").isEmpty()?profiles[0]:juce::File(option(args,"--profile"));
        if(args.contains("--catalogue-ui-check")){
            ToneHoundProcessor p;p.prepareToPlay(48000,256);juce::String error;require(p.loadProfileForTest(file,error),error);
            p.root=root.getChildFile(".cache/native/catalogue-ui-check").getChildFile(juce::Uuid().toString());
            auto missing=p.root.getChildFile(".cache/tone3000/profiles/t3k-1-1 Cached descriptor.nam");
            auto manifest=p.root.getChildFile(".cache/native/library.json");manifest.getParentDirectory().createDirectory();
            juce::Array<juce::var> rows;rows.add(object({{"path",missing.getFullPathName()},{"name","Cached descriptor"},{"tone_id",1},{"model_id",1}}));
            require(manifest.replaceWithText(juce::JSON::toString(object({{"profiles",rows}}))),"Could not create isolated catalogue fixture");
            auto editor=std::unique_ptr<juce::AudioProcessorEditor>(p.createEditor());
            auto* selector=dynamic_cast<juce::ComboBox*>(editor->findChildWithID("amp.selector"));
            require(selector && selector->getNumItems()==2 && selector->getItemText(0)=="Cached descriptor","Evicted catalogue capture disappeared from amp selector");
            require(!missing.existsAsFile(),"Browsing descriptors unexpectedly downloaded a NAM");
            std::cout<<"PASS: native selector retains evicted descriptors and the current local capture without downloading the library"<<std::endl;return 0;
        }
        if(args.contains("--bridge-check")) {
            AnalysisBridge bridge(root);
            require(bridge.submit(object({{"action","match"},{"path",root.getChildFile("assets/song_train/Periphery - Garden In The Bones (Audio).mp3").getFullPathName()},{"start",0},{"end",40}})),"Job not accepted");
            pump(350);bridge.cancel();
            auto deadline=juce::Time::getMillisecondCounterHiRes()+10000;
            while(bridge.snapshot().busy && juce::Time::getMillisecondCounterHiRes()<deadline)pump(20);
            require(!bridge.snapshot().busy && bridge.snapshot().response["code"].toString()=="cancelled","Worker cancellation failed");
            require(bridge.submit(object({{"action","library"}})),"Worker did not accept the next job");
            deadline=juce::Time::getMillisecondCounterHiRes()+20000;
            while(bridge.snapshot().busy && juce::Time::getMillisecondCounterHiRes()<deadline)pump(20);
            require((bool)bridge.snapshot().response["ok"],"Job after cancellation failed");
            std::cout<<"PASS: owned worker cancellation and subsequent job"<<std::endl;return 0;
        }
        auto ioSnapshot=option(args,"--io-snapshot");
        if(ioSnapshot.isNotEmpty()) {
            auto driver=option(args,"--driver");require(driver.isNotEmpty(),"Specify the ASIO driver with --driver");
            juce::AudioDeviceManager devices;juce::XmlElement setup("DEVICESETUP");
            setup.setAttribute("deviceType","ASIO");setup.setAttribute("audioInputDeviceName",driver);setup.setAttribute("audioOutputDeviceName",driver);
            setup.setAttribute("audioDeviceRate",48000);setup.setAttribute("audioDeviceBufferSize",256);
            auto error=devices.initialise(2,2,&setup,false);require(error.isEmpty(),error);
            ToneHoundProcessor p;p.standalone=true;p.deviceManager=&devices;p.setParameter("monitor",0);
            struct Player : juce::AudioProcessorPlayer {
                std::atomic<int> blocks{0};
                void audioDeviceIOCallbackWithContext(const float* const* inputs,int ni,float* const* outputs,int no,int n,const juce::AudioIODeviceCallbackContext& context) override
                {++blocks;juce::AudioProcessorPlayer::audioDeviceIOCallbackWithContext(inputs,ni,outputs,no,n,context);}
            } player;
            player.setProcessor(&p);devices.addAudioCallback(&player);p.loadProfile(file);pump(1200);
            auto count=player.blocks.load();devices.removeAudioCallback(&player);player.setProcessor(nullptr);
            require(count>0 && p.modelReady && !p.audioFault,"Interface callback or NAM load failed");
            auto panel=makeAudioIOPanel(p);auto image=panel->createComponentSnapshot(panel->getLocalBounds());
            juce::File output(ioSnapshot);output.getParentDirectory().createDirectory();auto stream=output.createOutputStream();require(stream!=nullptr,"Cannot write I/O snapshot");
            stream->setPosition(0);stream->truncate();
            juce::PNGImageFormat png;require(png.writeImageToStream(image,*stream),"Cannot export I/O panel");
            auto* device=devices.getCurrentAudioDevice();std::cout<<"PASS: "<<device->getName()<<", "<<device->getCurrentSampleRate()<<" Hz, "<<device->getCurrentBufferSizeSamples()<<" samples, "<<count<<" native audio callbacks (monitor off)"<<std::endl;
            return 0;
        }
        auto snapshot=option(args,"--snapshot");
        if(snapshot.isNotEmpty()) {
            ToneHoundProcessor p;p.standalone=true;p.setParameter("monitor",0);p.setRateAndBufferSizeDetails(48000,256);p.prepareToPlay(48000,256);
            if(option(args,"--profile").isNotEmpty())p.loadProfile(file);
            auto editor=std::unique_ptr<ToneHoundEditor>(static_cast<ToneHoundEditor*>(p.createEditor()));
            if(option(args,"--width").isNotEmpty()){auto width=option(args,"--width").getIntValue();editor->setSize(width,(int)std::round(width/1.5));}
            if(args.contains("--solo"))if(auto* role=dynamic_cast<juce::ComboBox*>(editor->findChildWithID("reference.role")))role->setSelectedId(2,juce::sendNotificationSync);
            auto reference=option(args,"--reference");
            if(reference.isNotEmpty()) editor->loadReferenceForPreview(juce::File(reference));
            auto deadline=juce::Time::getMillisecondCounterHiRes()+120000;
            do {pump(100);} while((p.loading || p.analysis.snapshot().busy || p.artwork.snapshot().busy) && juce::Time::getMillisecondCounterHiRes()<deadline);
            pump(700);
            require(!p.loading,"Capture load timed out");
            require(p.modelReady,"Capture failed: "+p.modelMessage());
            if(reference.isNotEmpty())require(p.referenceInfo().isObject(),"Reference import failed: "+p.analysis.snapshot().message);
            if(reference.isNotEmpty()) {
                auto* start=dynamic_cast<juce::TextEditor*>(editor->findChildWithID("crop.start"));
                auto* end=dynamic_cast<juce::TextEditor*>(editor->findChildWithID("crop.end"));
                require(start && end,"Native crop controls missing");
                start->setText("99:00",false);start->onReturnKey();
                require(p.selectionEnd-p.selectionStart>=30,"Start field broke crop minimum");
                start->setText("0",false);start->onReturnKey();end->setText("-9",false);end->onReturnKey();
                require(p.selectionStart==0 && p.selectionEnd==30,"End field did not enforce 30 seconds");
                end->setText("40",false);end->onReturnKey();
                if(option(args,"--end").isNotEmpty()) {end->setText(option(args,"--end"),false);end->onReturnKey();}
                if(option(args,"--start").isNotEmpty()) {start->setText(option(args,"--start"),false);start->onReturnKey();}
                auto* play=dynamic_cast<juce::Button*>(editor->findChildWithID("reference.play"));
                require(play!=nullptr,"Reference transport missing");play->onClick();
                require(p.previewPlaying,"Native play control did not start transport");
                juce::AudioBuffer<float> buffer(2,512);juce::MidiBuffer midi;float peak=0;double stereoDifference=0;
                for(int i=0;i<300;++i){
                    buffer.clear();p.processBlock(buffer,midi);peak=juce::jmax(peak,buffer.getMagnitude(0,512));
                    for(int sample=0;sample<512;++sample) stereoDifference+=std::abs(buffer.getSample(0,sample)-buffer.getSample(1,sample));
                }
                require(peak>1e-6,"Reference preview produced no sound");play->onClick();require(!p.previewPlaying,"Stop control did not stop transport");
                if(args.contains("--expect-stereo")) {
                    require((int)p.referenceInfo()["channels"]==2,"Import lost stereo channels");
                    require(stereoDifference>1,"Native reference playback collapsed left and right channels");
                    std::cout<<"PASS: stereo import and distinct left/right reference playback"<<std::endl;
                }
                pump(100);
                std::cout<<"PASS: native import job, crop bounds, reference playback and stop"<<std::endl;
            }
            if(args.contains("--match")) {
                bool clicked=false;
                for(auto* child:editor->getChildren()) if(auto* button=dynamic_cast<juce::TextButton*>(child);button && button->getButtonText()=="Find matching tones") {button->onClick();clicked=true;break;}
                require(clicked,"Find matching tones control missing");
                deadline=juce::Time::getMillisecondCounterHiRes()+240000;
                do {pump(100);} while((p.analysis.snapshot().busy || p.loading) && juce::Time::getMillisecondCounterHiRes()<deadline);
                pump(1000);
                require(p.hasMatches() && p.modelReady,"Native matching failed: "+p.analysis.snapshot().message);
                auto result=p.matchInfo();require(result.isArray() && result.size()>0,"No native result cards");
                require(p.profilePath()==result[0]["path"].toString(),"Top result did not load its NAM capture");
                std::cout<<"PASS: native match button, Python pipeline, result cards and automatic capture load"<<std::endl;
                std::cout<<"Top capture: "<<result[0]["name"].toString()<<std::endl;
                if(args.contains("--solo")){
                    require(p.state.getRawParameterValue("delayEnabled")->load()>.5f,"Solo match did not enable delay");
                    require(p.state.getRawParameterValue("reverbEnabled")->load()>.5f,"Solo match did not enable reverb");
                    require(p.state.getRawParameterValue("delayBpm")->load()>=40,"Solo tempo invalid");
                    std::cout<<"PASS: automatic solo effects and estimated BPM"<<std::endl;
                }
                deadline=juce::Time::getMillisecondCounterHiRes()+30000;
                while(p.artwork.snapshot().busy && juce::Time::getMillisecondCounterHiRes()<deadline) pump(100);
                pump(500);
            }
            auto image=editor->createComponentSnapshot(editor->getLocalBounds());
            juce::File output(snapshot);output.getParentDirectory().createDirectory();
            auto stream=output.createOutputStream();require(stream!=nullptr,"Cannot write snapshot");
            stream->setPosition(0);stream->truncate();
            juce::PNGImageFormat png;require(png.writeImageToStream(image,*stream),"PNG export failed");
            std::cout << "Native editor snapshot: " << snapshot << std::endl;
            stream.reset();
            if(args.contains("--all-pages"))for(int page=0;page<4;++page){
                auto* tab=dynamic_cast<juce::TextButton*>(editor->findChildWithID("page."+juce::String(page)));
                require(tab!=nullptr,"Studio tab missing");tab->onClick();pump(100);
                auto capture=editor->createComponentSnapshot(editor->getLocalBounds());
                auto target=output.getSiblingFile(output.getFileNameWithoutExtension()+"-page"+juce::String(page)+".png");
                auto out=target.createOutputStream();require(out!=nullptr,"Cannot write studio screenshot");out->setPosition(0);out->truncate();
                require(png.writeImageToStream(capture,*out),"Studio PNG export failed");
            }
            return 0;
        }
        if(args.contains("--devices")) {
            juce::AudioDeviceManager devices;devices.initialise(0,0,nullptr,false);
            for(auto* type:devices.getAvailableDeviceTypes()) {
                type->scanForDevices();std::cout<<type->getTypeName()<<std::endl;
                std::cout<<"  Inputs: "<<type->getDeviceNames(true).joinIntoString(" | ")<<std::endl;
                std::cout<<"  Outputs: "<<type->getDeviceNames(false).joinIntoString(" | ")<<std::endl;
            }
            return 0;
        }
        auto render=option(args,"--render");
        if(render.isNotEmpty()) {
            juce::AudioFormatManager formats;formats.registerBasicFormats();
            std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(juce::File(render)));
            require(reader!=nullptr,"Cannot decode DI");
            const int length=(int)juce::jmin(reader->lengthInSamples,(juce::int64)(reader->sampleRate*15));
            juce::AudioBuffer<float> source(1,length);reader->read(&source,0,length,0,true,false);
            ToneHoundProcessor p;p.setRateAndBufferSizeDetails(reader->sampleRate,256);p.prepareToPlay(reader->sampleRate,256);
            p.setParameter("gate",-100);p.setParameter("output",0);
            juce::String error;require(p.loadProfileForTest(file,error),error);
            juce::MidiBuffer midi;juce::AudioBuffer<float> buffer(2,256);
            for(int i=0;i<40;++i){buffer.clear();p.processBlock(buffer,midi);}
            auto output=juce::File(option(args,"--output"));output.getParentDirectory().createDirectory();
            auto stream=output.createOutputStream();require(stream!=nullptr,"Cannot write DI render");stream->setPosition(0);stream->truncate();
            juce::WavAudioFormat wav;std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),reader->sampleRate,1,24,{},0));
            require(writer!=nullptr,"Cannot create WAV writer");
            for(int offset=0;offset<length;offset+=256) {
                int n=juce::jmin(256,length-offset);buffer.setSize(2,n,false,false,true);buffer.clear();buffer.copyFrom(0,0,source,0,offset,n);
                p.processBlock(buffer,midi);require(!p.audioFault,"NAM render fault");require(writer->writeFromAudioSampleBuffer(buffer,0,n),"WAV write failed");
            }
            std::cout<<"Rendered actual DI through native NAM: "<<output.getFullPathName()<<std::endl;return 0;
        }
        juce::Array<juce::var> reports;
        if(args.contains("--eq-integration")) {
            const int blockSize=256;
            juce::AudioFormatManager formats;formats.registerBasicFormats();
            std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(root.getChildFile("assets/user_di/Djent DI.wav")));
            require(reader!=nullptr && reader->sampleRate>0,"Cannot decode integration DI");
            const double sr=reader->sampleRate;const int length=(int)std::round(sr*10);
            require(reader->lengthInSamples>=length,"Ten-second DI required");
            juce::AudioBuffer<float> source(1,length),rendered(1,length);reader->read(&source,0,length,0,true,false);
            ToneHoundProcessor p;p.setRateAndBufferSizeDetails(sr,blockSize);p.prepareToPlay(sr,blockSize);
            p.setParameter("gate",-100);p.setParameter("output",0);p.setParameter("monitor",1);
            juce::String error;require(p.loadProfileForTest(file,error),error);
            juce::MidiBuffer midi;juce::AudioBuffer<float> buffer(2,blockSize);
            for(int offset=0;offset<length;offset+=blockSize){
                int n=juce::jmin(blockSize,length-offset);buffer.setSize(2,n,false,false,true);buffer.clear();buffer.copyFrom(0,0,source,0,offset,n);p.processBlock(buffer,midi);rendered.copyFrom(0,offset,buffer,0,0,n);
            }
            auto target=root.getChildFile(".cache/native/eq-integration-target.wav");
            auto stream=target.createOutputStream();require(stream!=nullptr,"Cannot write EQ integration target");stream->setPosition(0);stream->truncate();
            juce::WavAudioFormat wav;std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),sr,1,24,{},0));
            require(writer!=nullptr && writer->writeFromAudioSampleBuffer(rendered,0,length),"EQ target write failed");writer.reset();
            p.loadEqTarget(target);auto deadline=juce::Time::getMillisecondCounterHiRes()+10000;
            while(!p.eqTargetReady() && juce::Time::getMillisecondCounterHiRes()<deadline)pump(10);
            require(p.eqTargetReady(),"EQ target preparation: "+p.eqMatchStatus());
            require(p.startEqCapture(),"Processor EQ capture failed to arm");
            for(int offset=0;offset<length;offset+=blockSize){
                int n=juce::jmin(blockSize,length-offset);buffer.setSize(2,n,false,false,true);buffer.clear();buffer.copyFrom(0,0,source,0,offset,n);p.processBlock(buffer,midi);
            }
            deadline=juce::Time::getMillisecondCounterHiRes()+10000;
            while(!p.eqMatchReady() && juce::Time::getMillisecondCounterHiRes()<deadline){pump(10);p.collectRetired();}
            require(p.eqMatchReady() && p.eqCaptureProgress()>=1,"Processor EQ did not complete: "+p.eqMatchStatus());
            p.applyPerformanceSuggestion(true,120);require(p.state.getRawParameterValue("delayEnabled")->load()>.5f,"Solo delay missing");
            p.setParameter("delayMix",.37f);p.setParameter("delayBpm",137);
            juce::MemoryBlock state;p.getStateInformation(state);ToneHoundProcessor restored;restored.setStateInformation(state.getData(),(int)state.getSize());
            require(restored.eqMatchReady(),"Matched EQ was not saved in processor state");
            auto original=restored.eqCurve();auto editor=std::unique_ptr<juce::AudioProcessorEditor>(restored.createEditor());pump(100);
            require(restored.eqMatchReady() && restored.eqCurve()==original,"Opening editor erased saved EQ");
            require(restored.state.getRawParameterValue("delayEnabled")->load()>.5f,"Pedal state did not restore");
            require(std::abs(restored.state.getRawParameterValue("delayMix")->load()-.37f)<.001f && restored.state.getRawParameterValue("delayBpm")->load()==137,"Opening editor reset manual pedal settings");
            restored.setParameter("eqBand3",5);restored.setParameter("eqMatchEnabled",1);
            juce::MemoryBlock immediateState;restored.getStateInformation(immediateState);
            ToneHoundProcessor immediate;immediate.setStateInformation(immediateState.getData(),(int)immediateState.getSize());
            require(immediate.state.getRawParameterValue("eqBand3")->load()==5 && immediate.eqMatchReady(),"Immediate host save lost manual EQ edit");
            pump(80);
            require(restored.eqMatchReady(),"Manual eight-band EQ did not become ready");
            auto manual=restored.eqCurve();float boost=0;for(auto point:manual)if(point.first>220 && point.first<280)boost=juce::jmax(boost,point.second);
            require(boost>2,"Manual 250 Hz band did not change the actual EQ response");
            restored.setParameter("bass",5);restored.setParameter("mid",-3);restored.setParameter("treble",4);
            restored.loadProfile(profiles[profiles.size()>1?1:0]);
            require(!restored.eqMatchReady(),"Changing amp retained matched EQ");
            for(auto id:{"bass","mid","treble","eqBand1","eqBand2","eqBand3","eqBand4","eqBand5","eqBand6","eqBand7","eqBand8"})
                require(restored.state.getRawParameterValue(id)->load()==0,"Changing amp retained EQ settings");
            std::cout<<"PASS: real NAM / user DI ten-second EQ capture, auto apply, solo preset, immediate eight-band EQ save, editor/pedal state restoration and amp-change EQ reset"<<std::endl;return 0;
        }
        for(double sr:{44100.,48000.,96000.}) {
            ToneHoundProcessor p;p.setRateAndBufferSizeDetails(sr,256);p.prepareToPlay(sr,256);
            p.setParameter("gate",-100);p.setParameter("output",0);
            juce::String error;require(p.loadProfileForTest(file,error),error);
            juce::MidiBuffer midi;double energy=0,seconds=0;int samples=0;
            for(int block:{1,64,127,256,512,1024}) {
                juce::AudioBuffer<float> buffer(2,block);
                for(int iteration=0;iteration<40;++iteration) {
                    for(int i=0;i<block;++i) {auto x=.07f*std::sin((float)(2*juce::MathConstants<double>::pi*110*(samples+i)/sr));buffer.setSample(0,i,x);buffer.setSample(1,i,0);}
                    auto before=std::chrono::steady_clock::now();p.processBlock(buffer,midi);
                    seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();samples+=block;
                    for(int ch=0;ch<2;++ch) for(int i=0;i<block;++i) {
                        auto y=buffer.getSample(ch,i);require(std::isfinite(y)&&std::abs(y)<=1,"Non-finite / unbounded audio");energy+=y*y;
                        require(buffer.getSample(0,i)==buffer.getSample(1,i),"Mono guitar must reach both outputs");
                    }
                }
            }
            require(energy>1e-5 && !p.audioFault,"NAM produced no audio or reported an error");
            for(int quality:{0,1}){
                p.setParameter("pitchQuality",(float)quality);p.setParameter("pitchSemitones",-5);
                juce::AudioBuffer<float> shifted(2,256);shifted.clear();p.processBlock(shifted,midi);pump(60);
                const int expectedPitch=(int)std::round(sr*(quality==0?.032:.192))+(quality==0?(int)std::round(sr*.008):0);
                require(p.getLatencySamples()>=expectedPitch,"Host was not informed of pitch latency");
            }
            p.setParameter("pitchSemitones",0);juce::AudioBuffer<float> dryPitch(2,256);dryPitch.clear();p.processBlock(dryPitch,midi);pump(60);
            require(p.getLatencySamples()<sr*.01,"Zero transpose retained pitch latency");
            p.setParameter("monitor",0);
            juce::AudioBuffer<float> silence(2,1024);
            for(int i=0;i<10;++i){silence.clear();p.processBlock(silence,midi);}
            require(silence.getMagnitude(0,silence.getNumSamples())==0,"Monitor off leaked guitar output");
            p.setSelection(12,52);p.setParameter("bass",4.5f);juce::MemoryBlock state;p.getStateInformation(state);
            ToneHoundProcessor restored;restored.setStateInformation(state.getData(),(int)state.getSize());
            require(std::abs(restored.state.getRawParameterValue("bass")->load()-4.5f)<.01,"Host parameter state did not restore");
            require(restored.selectionStart==12 && restored.selectionEnd==52,"Crop state did not restore");
            p.loadProfile(root.getChildFile(".cache/native/"+juce::Uuid().toString()+".nam"));
            auto failedDeadline=juce::Time::getMillisecondCounterHiRes()+5000;
            while(p.loading && juce::Time::getMillisecondCounterHiRes()<failedDeadline)pump(20);
            require(!p.modelReady,"Failed capture silently retained the previous amp");
            reports.add(object({{"sample_rate",sr},{"samples",samples},{"processing_seconds",seconds},{"realtime_load",seconds/(samples/sr)},{"energy",energy},{"latency_samples",p.getLatencySamples()}}));
            std::cout<<sr<<" Hz: "<<seconds/(samples/sr)*100<<"% realtime, latency "<<p.getLatencySamples()<<" samples"<<std::endl;
            pump(30);
        }
        auto report=root.getChildFile("docs/native_validation.json");report.replaceWithText(juce::JSON::toString(object({{"profile",file.getFullPathName()},{"checks","finite audio, stereo output, variable blocks, 44.1/48/96 kHz, monitor off, host parameter/crop restore"},{"runs",reports}})));
        std::cout<<"PASS: native NAM audio and state checks"<<std::endl;
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<std::endl;return 1;}
}
