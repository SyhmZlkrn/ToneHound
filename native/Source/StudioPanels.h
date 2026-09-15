#pragma once
#include "PluginProcessor.h"

// Secondary workspaces share the processor but own their controls and jobs.
// Keeping these out of the editor makes the main performance surface simpler.
class EqMatchPanel final : public juce::Component, private juce::Timer
{
public:
    explicit EqMatchPanel(ToneHoundProcessor&);
    ~EqMatchPanel() override;
    void paint(juce::Graphics&) override;
    void resized() override;
private:
    void timerCallback() override;
    ToneHoundProcessor& processor;
    juce::TextButton capture{"Play & capture 10 seconds"}, cancel{"Cancel capture"}, reset{"Reset curve"}, enable{"EQ enabled"};
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> attachment;
    std::array<juce::Slider,8> bands;
    std::array<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>,8> bandAttachments;
};

class StompSwitch final : public juce::Button {
public:
    StompSwitch():juce::Button("Pedal bypass"){}
    void paintButton(juce::Graphics&,bool,bool) override;
};

class PedalPanel final : public juce::Component
{
public:
    explicit PedalPanel(ToneHoundProcessor&);
    void paint(juce::Graphics&) override;
    void resized() override;
private:
    struct Control {
        juce::String name;
        juce::Slider slider;
        std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;
    };
    ToneHoundProcessor& processor;
    std::array<StompSwitch,3> toggles;
    std::array<std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment>,3> buttons;
    std::vector<std::unique_ptr<Control>> controls;
    juce::ComboBox division;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> divisionAttachment;
};

class LibraryPanel final : public juce::Component, private juce::ListBoxModel, private juce::Timer
{
public:
    explicit LibraryPanel(ToneHoundProcessor&);
    ~LibraryPanel() override;
    void paint(juce::Graphics&) override;
    void resized() override;
    std::function<void()> libraryChanged;
private:
    int getNumRows() override;
    void paintListBoxItem(int,juce::Graphics&,int,int,bool) override;
    void selectedRowsChanged(int) override;
    void timerCallback() override;
    void search(int page);
    ToneHoundProcessor& processor;
    AnalysisBridge bridge;
    juce::TextEditor query;
    juce::TextButton searchButton{"Search"}, connect{"Connect account"}, add{"Add selected tone"}, previous{"Previous"}, next{"Next"}, cancel{"Cancel"};
    juce::TextButton prepare{"Prepare descriptors"},importPack{"Import index"},exportPack{"Export index"};
    std::unique_ptr<juce::FileChooser> chooser;
    juce::ListBox rows{"TONE3000 results",this};
    juce::var tones;
    juce::String status="Search TONE3000 and add captures to your matching library.";
    int completion=0,page=1,pages=1;
};
