#pragma once
#include "PluginProcessor.h"

namespace palette {
inline const juce::Colour background{0xff111416}, panel{0xff1a1e20}, raised{0xff24292b},
    line{0xff485153}, text{0xffeeeee7}, muted{0xffadb7b8}, accent{0xffd8be88}, green{0xff9cbfa7};
}

class EqMatchPanel;
class PedalPanel;
class LibraryPanel;

std::unique_ptr<juce::Component> makeAudioIOPanel(ToneHoundProcessor&);
std::unique_ptr<juce::Component> makePitchPanel(ToneHoundProcessor&);

class EZLookAndFeel : public juce::LookAndFeel_V4
{
public:
    EZLookAndFeel();
    static EZLookAndFeel& instance();
    void drawRotarySlider(juce::Graphics&,int,int,int,int,float,float,float,juce::Slider&) override;
    void drawButtonBackground(juce::Graphics&,juce::Button&,const juce::Colour&,bool,bool) override;
    juce::Font getTextButtonFont(juce::TextButton&,int) override;
    juce::Font getComboBoxFont(juce::ComboBox&) override;
    juce::Font getLabelFont(juce::Label&) override;
    juce::Font getPopupMenuFont() override;
};

class CropWaveform : public juce::Component, private juce::ChangeListener
{
public:
    CropWaveform();
    void setFile(const juce::File&,double duration,double start,double end);
    void setSelection(double,double);
    void paint(juce::Graphics&) override;
    void mouseDown(const juce::MouseEvent&) override;
    void mouseDrag(const juce::MouseEvent&) override;
    std::function<void(double,double)> changed;
    double duration=0,start=0,end=40,playhead=-1;
private:
    void changeListenerCallback(juce::ChangeBroadcaster*) override { repaint(); }
    double timeAt(float x) const;
    int drag=0;
    double mouseTime=0,dragStart=0,dragEnd=0;
    juce::AudioFormatManager formats;
    juce::AudioThumbnailCache cache{8};
    juce::AudioThumbnail thumbnail{256,formats,cache};
};

class Dial : public juce::Component
{
public:
    Dial(ToneHoundProcessor&,juce::String id,juce::String label);
    void paint(juce::Graphics&) override;
    void resized() override;
private:
    juce::String name;
    juce::Slider slider;
    juce::AudioProcessorValueTreeState::SliderAttachment attachment;
};

class CaptureCard : public juce::Button
{
public:
    CaptureCard() : juce::Button("Capture") {}
    void paintButton(juce::Graphics&,bool,bool) override;
    juce::String title,detail;
    juce::Image image;
    bool selected=false;
    int rank=0;
};

class ToneHoundEditor final : public juce::AudioProcessorEditor, private juce::Timer,
                          public juce::FileDragAndDropTarget
{
public:
    explicit ToneHoundEditor(ToneHoundProcessor&);
    ~ToneHoundEditor() override;
    void paint(juce::Graphics&) override;
    void resized() override;
    bool isInterestedInFileDrag(const juce::StringArray&) override;
    void filesDropped(const juce::StringArray&,int,int) override;
    void fileDragEnter(const juce::StringArray&,int,int) override { dragging=true; repaint(); }
    void fileDragExit(const juce::StringArray&) override { dragging=false; repaint(); }
    void loadReferenceForPreview(const juce::File&); // deterministic native screenshot/QA entry point
private:
    void timerCallback() override;
    void loadLibrary();
    void selectCapture(juce::var);
    juce::var rowFor(const juce::File&);
    void updateCards();
    void importSong(juce::String);
    void chooseFile(bool nam);
    void setImportMode(bool link);
    void updateSelection(double,double);
    void readSelection(bool editingStart);
    void showIO();
    void receive(const AnalysisBridge::Snapshot&);
    void refreshPhoto();
    void setPage(int);
    void applyRoleEffects();
    static juce::String clockText(double);
    static double parseTime(const juce::String&);

    ToneHoundProcessor& processor;
    juce::ComboBox profiles, channel;
    juce::Slider pitchControl;
    juce::TextButton pitchSettings{"Pitch settings"};
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> pitchAttachment;
    juce::ComboBox passageRole;
    juce::ComboBox matchingModel;
    std::array<juce::TextButton,4> pageTabs;
    std::unique_ptr<EqMatchPanel> eqPanel;
    std::unique_ptr<PedalPanel> pedalPanel;
    std::unique_ptr<LibraryPanel> libraryPanel;
    juce::TextEditor captureDescription;
    int page=0;
    juce::TextButton browse{"Load .nam"}, io{"I / O"}, monitor{"MONITOR"}, bypass{"BYPASS"};
    juce::TextButton fileTab{"FILE"}, linkTab{"LINK"}, importButton{"Import MP3 / WAV"}, fetchButton{"Import"};
    juce::TextButton previewButton{"Play selection"}, loopButton{"Loop"}, findButton{"Find matching tones"}, cancelButton{"Cancel"};
    juce::TextButton matchHelp{"About these matches"};
    juce::TextEditor url, startField, endField;
    juce::HyperlinkButton sourceLink{"View capture on TONE3000",juce::URL{}};
    CropWaveform waveform;
    std::array<std::unique_ptr<Dial>,6> dials;
    std::array<CaptureCard,6> cards;
    std::unique_ptr<juce::FileChooser> chooser;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> monitorAttachment,bypassAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> channelAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> roleAttachment;
    std::vector<juce::var> library;
    juce::var selectedCapture;
    juce::Image ampPhoto;
    int lastJob=0,lastArtwork=0,requestedArtwork=0;
    juce::String status="Import a reference, select a passage, then find matching captures.";
    juce::String referenceName, referenceOrigin, stemNote;
    bool linkMode=false,dragging=false;
    double duration=0;
    int referenceChannels=0;
    float scale=1;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ToneHoundEditor)
};
