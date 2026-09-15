#include "PluginEditor.h"

class ToneHoundApplication final : public juce::JUCEApplication
{
public:
    const juce::String getApplicationName() override { return "ToneHound"; }
    const juce::String getApplicationVersion() override { return "0.1.0"; }
    bool moreThanOneInstanceAllowed() override { return false; }
    void initialise(const juce::String&) override
    {
        settings=juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory).getChildFile("ToneHound");
        settings.createDirectory();
        processor=std::make_unique<ToneHoundProcessor>();
        processor->standalone=true;processor->deviceManager=&devices;
        juce::MemoryBlock state;
        if(settings.getChildFile("session.bin").loadFileAsData(state))
            processor->setStateInformation(state.getData(),(int)state.getSize());
        processor->setParameter("monitor",0); // explicit monitoring on each standalone launch
        auto saved=juce::XmlDocument::parse(settings.getChildFile("audio.xml"));
        auto error=devices.initialise(2,2,saved.get(),true);
        player.setProcessor(processor.get());devices.addAudioCallback(&player);
        if(error.isNotEmpty()) processor->setModelMessage("Open I/O to choose an audio device: "+error);
        window=std::make_unique<Window>(*processor);
    }
    void shutdown() override
    {
        window.reset();
        devices.removeAudioCallback(&player);player.setProcessor(nullptr);devices.closeAudioDevice();
        if(processor) {juce::MemoryBlock state;processor->getStateInformation(state);settings.getChildFile("session.bin").replaceWithData(state.getData(),state.getSize());}
        if(auto xml=devices.createStateXml()) xml->writeTo(settings.getChildFile("audio.xml"));
        processor.reset();
    }
    void systemRequestedQuit() override { quit(); }
    void anotherInstanceStarted(const juce::String&) override { if(window) window->toFront(true); }
private:
    class Window final : public juce::DocumentWindow
    {
    public:
        explicit Window(ToneHoundProcessor& p):DocumentWindow("ToneHound",palette::background,allButtons)
        {
            setUsingNativeTitleBar(true);setResizable(true,false);
            auto* editor=p.createEditor();setContentOwned(editor,true);
            getConstrainer()->setFixedAspectRatio(1.5);
            setResizeLimits(1280,854,1920,1280);
            centreWithSize(getWidth(),getHeight());setVisible(true);
        }
        void closeButtonPressed() override {juce::JUCEApplication::getInstance()->systemRequestedQuit();}
    };
    juce::File settings;
    juce::AudioDeviceManager devices;
    juce::AudioProcessorPlayer player;
    std::unique_ptr<ToneHoundProcessor> processor;
    std::unique_ptr<Window> window;
};
START_JUCE_APPLICATION(ToneHoundApplication)
