#include "PluginEditor.h"
#include "StudioPanels.h"

namespace {
juce::Font font(float size,bool bold=false) { return juce::Font(juce::FontOptions("Segoe UI",size,bold?juce::Font::bold:juce::Font::plain)); }
void label(juce::Graphics& g,juce::String text,juce::Rectangle<float> r,float size,juce::Colour colour=palette::text,bool bold=false,juce::Justification justify=juce::Justification::centredLeft)
{ g.setColour(colour); g.setFont(font(size,bold)); g.drawText(text,r.toNearestInt(),justify,true); }
void panel(juce::Graphics& g,juce::Rectangle<float> r)
{ g.setColour(palette::panel); g.fillRoundedRectangle(r,10); g.setColour(palette::line.withAlpha(.65f)); g.drawRoundedRectangle(r,.0f+10,1); }
juce::String niceName(juce::String name)
{
    if(name.startsWith("t3k-")) name=name.fromFirstOccurrenceOf(" ",false,false);
    return name.replaceCharacter('_',' ').trim();
}
void photo(juce::Graphics& g,const juce::Image& image,juce::Rectangle<float> area)
{
    if(!image.isValid()) return;
    juce::Graphics::ScopedSaveState save(g);
    juce::Path p; p.addRoundedRectangle(area,5); g.reduceClipRegion(p);
    // Keep the complete TONE3000 photograph at its original aspect ratio.
    g.drawImageWithin(image,(int)area.getX(),(int)area.getY(),(int)area.getWidth(),(int)area.getHeight(),juce::RectanglePlacement::centred);
}
}

EZLookAndFeel& EZLookAndFeel::instance() { static EZLookAndFeel theme; return theme; }
EZLookAndFeel::EZLookAndFeel()
{
    setColour(juce::ResizableWindow::backgroundColourId,palette::background);
    setColour(juce::TextButton::buttonColourId,palette::raised);
    setColour(juce::TextButton::buttonOnColourId,palette::accent);
    setColour(juce::TextButton::textColourOffId,palette::text);
    setColour(juce::TextButton::textColourOnId,palette::background);
    setColour(juce::ComboBox::backgroundColourId,palette::raised);
    setColour(juce::ComboBox::outlineColourId,palette::line);
    setColour(juce::ComboBox::textColourId,palette::text);
    setColour(juce::ComboBox::arrowColourId,palette::accent);
    setColour(juce::PopupMenu::backgroundColourId,palette::panel);
    setColour(juce::PopupMenu::textColourId,palette::text);
    setColour(juce::PopupMenu::highlightedBackgroundColourId,palette::raised.brighter(.15f));
    setColour(juce::TextEditor::backgroundColourId,palette::background);
    setColour(juce::TextEditor::textColourId,palette::text);
    setColour(juce::TextEditor::outlineColourId,palette::line);
    setColour(juce::TextEditor::focusedOutlineColourId,palette::accent);
    setColour(juce::Label::textColourId,palette::text);
    setColour(juce::Slider::textBoxTextColourId,palette::text);
    setColour(juce::Slider::textBoxOutlineColourId,juce::Colours::transparentBlack);
    setColour(juce::Slider::textBoxBackgroundColourId,juce::Colours::transparentBlack);
    setColour(juce::Slider::thumbColourId,palette::accent);
    setColour(juce::ToggleButton::textColourId,palette::text);
    setColour(juce::ToggleButton::tickColourId,palette::accent);
    setColour(juce::ScrollBar::thumbColourId,palette::accent);
    setColour(juce::ListBox::backgroundColourId,palette::background);
    setColour(juce::ListBox::textColourId,palette::text);
    setColour(juce::HyperlinkButton::textColourId,palette::muted);
}
juce::Font EZLookAndFeel::getTextButtonFont(juce::TextButton&,int) { return font(16,true); }
juce::Font EZLookAndFeel::getComboBoxFont(juce::ComboBox&) {return font(17);}
juce::Font EZLookAndFeel::getLabelFont(juce::Label&) {return font(16);}
juce::Font EZLookAndFeel::getPopupMenuFont() {return font(16);}
void EZLookAndFeel::drawButtonBackground(juce::Graphics& g,juce::Button& b,const juce::Colour& c,bool hover,bool down)
{
    auto r=b.getLocalBounds().toFloat().reduced(.5f);
    auto colour=b.getToggleState()?b.findColour(juce::TextButton::buttonOnColourId):c;
    if(hover) colour=colour.brighter(.07f); if(down) colour=colour.darker(.08f);
    g.setColour(colour.withMultipliedAlpha(b.isEnabled()?1.0f:.4f)); g.fillRoundedRectangle(r,5);
    g.setColour((b.getToggleState()||b.hasKeyboardFocus(true)?palette::accent:palette::line).withMultipliedAlpha(b.isEnabled()?1.f:.3f)); g.drawRoundedRectangle(r,5,b.hasKeyboardFocus(true)?2.f:1.f);
}
void EZLookAndFeel::drawRotarySlider(juce::Graphics& g,int x,int y,int w,int h,float pos,float a,float b,juce::Slider&)
{
    auto size=(float)juce::jmin(w,h)-12; juce::Rectangle<float> r((float)x+((float)w-size)/2,(float)y+((float)h-size)/2,size,size);
    auto centre=r.getCentre(); auto radius=size/2;
    g.setColour(juce::Colours::black.withAlpha(.45f)); g.fillEllipse(r.translated(0,4).expanded(1));
    juce::Path track; track.addCentredArc(centre.x,centre.y,radius,radius,0,a,b,true);
    g.setColour(palette::line); g.strokePath(track,juce::PathStrokeType(2.5f));
    juce::Path active; active.addCentredArc(centre.x,centre.y,radius,radius,0,a,a+pos*(b-a),true);
    g.setColour(palette::accent); g.strokePath(active,juce::PathStrokeType(2.5f));
    auto knob=r.reduced(6);
    g.setGradientFill(juce::ColourGradient(juce::Colour(0xff555b5d),knob.getTopLeft(),juce::Colour(0xff222729),knob.getBottomRight(),false));
    g.fillEllipse(knob); g.setColour(juce::Colour(0xff707779).withAlpha(.5f)); g.drawEllipse(knob,1);
    auto inner=knob.reduced(4);
    g.setGradientFill(juce::ColourGradient(juce::Colour(0xff34393c),inner.getTopLeft(),juce::Colour(0xff1c2022),inner.getBottomRight(),false)); g.fillEllipse(inner);
    auto angle=a+pos*(b-a); auto tip=centre.getPointOnCircumference(radius-13,angle);
    auto tail=centre.getPointOnCircumference(radius-24,angle);
    g.setColour(palette::accent); g.drawLine({tail,tip},3);
}

CropWaveform::CropWaveform() { formats.registerBasicFormats(); thumbnail.addChangeListener(this); setMouseCursor(juce::MouseCursor::LeftRightResizeCursor); }
void CropWaveform::setFile(const juce::File& file,double d,double s,double e)
{ duration=d;start=s;end=e;thumbnail.setSource(new juce::FileInputSource(file));repaint(); }
void CropWaveform::setSelection(double s,double e) { start=s;end=e;repaint(); }
double CropWaveform::timeAt(float x) const { return duration*juce::jlimit(0.0,1.0,(double)(x-10)/juce::jmax(1,getWidth()-20)); }
void CropWaveform::paint(juce::Graphics& g)
{
    auto r=getLocalBounds().toFloat(); g.setColour(palette::background);g.fillRoundedRectangle(r,6);
    auto plot=getLocalBounds().reduced(10,25);
    for(int i=1;i<5;++i) { g.setColour(palette::line.withAlpha(.45f));auto x=plot.getX()+plot.getWidth()*i/5;g.drawVerticalLine(x,(float)plot.getY(),(float)plot.getBottom()); }
    g.setColour(palette::muted.withAlpha(.25f));g.drawHorizontalLine(plot.getCentreY(),(float)plot.getX(),(float)plot.getRight());
    if(duration<=0) { label(g,"Your reference waveform",r,13,palette::muted,false,juce::Justification::centred);return; }
    g.setColour(palette::muted.withAlpha(.55f));thumbnail.drawChannels(g,plot,0,duration,.88f);
    auto sx=(float)plot.getX()+(float)(start/duration)*plot.getWidth();
    auto ex=(float)plot.getX()+(float)(end/duration)*plot.getWidth();
    juce::Rectangle<float> selection(sx,5,juce::jmax(1.f,ex-sx),(float)getHeight()-10);
    g.setColour(palette::accent.withAlpha(.12f));g.fillRect(selection);
    {juce::Graphics::ScopedSaveState save(g);g.reduceClipRegion(selection.toNearestInt());g.setColour(palette::accent);thumbnail.drawChannels(g,plot,0,duration,.88f);}
    g.setColour(palette::accent);g.drawRect(selection,1.2f);
    for(auto x:{sx,ex}) {g.fillRoundedRectangle(x-3,selection.getCentreY()-13,6,26,3);}
    if(playhead>=0 && playhead<duration) {g.setColour(palette::text);g.drawVerticalLine((int)(plot.getX()+playhead/duration*plot.getWidth()),8,(float)getHeight()-8);}
    label(g,"0:00",{10,(float)getHeight()-20,60,16},10,palette::muted);
    auto endText=juce::String((int)duration/60)+":"+juce::String((int)duration%60).paddedLeft('0',2);
    label(g,endText,{(float)getWidth()-65,(float)getHeight()-20,55,16},10,palette::muted,false,juce::Justification::centredRight);
}
void CropWaveform::mouseDown(const juce::MouseEvent& e)
{
    if(duration<30) return;
    mouseTime=timeAt(e.position.x);dragStart=start;dragEnd=end;
    double tolerance=duration*10.0/juce::jmax(1,getWidth());
    drag=std::abs(mouseTime-start)<tolerance?1:std::abs(mouseTime-end)<tolerance?2:(mouseTime>start&&mouseTime<end)?3:(mouseTime<start?1:2);
}
void CropWaveform::mouseDrag(const juce::MouseEvent& e)
{
    if(!drag || duration<30) return;
    auto delta=timeAt(e.position.x)-mouseTime;
    if(drag==1) start=juce::jlimit(0.0,end-30,dragStart+delta);
    if(drag==2) end=juce::jlimit(start+30,duration,dragEnd+delta);
    if(drag==3) {start=juce::jlimit(0.0,duration-(dragEnd-dragStart),dragStart+delta);end=start+(dragEnd-dragStart);}
    if(changed) changed(start,end);repaint();
}
Dial::Dial(ToneHoundProcessor& p,juce::String id,juce::String labelText)
    : name(std::move(labelText)),attachment(p.state,id,slider)
{
    addAndMakeVisible(slider);slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow,false,90,20);
    slider.setRotaryParameters(juce::MathConstants<float>::pi*1.22f,juce::MathConstants<float>::pi*2.78f,true);
    slider.setTextValueSuffix(" dB");slider.setDoubleClickReturnValue(true,id=="gate"?-80:id=="output"?-6:0);
    if(id=="gate") slider.textFromValueFunction=[](double v){return v<=-99?juce::String("Off"):juce::String(v,1)+" dB";};
    slider.setTooltip(id=="input"?"Drive the capture harder or softer. Double-click to reset.":id=="gate"?"Mute low-level input noise. Fully left disables the gate.":"Tone shaping after the capture. Double-click to reset.");
}
void Dial::paint(juce::Graphics& g) {label(g,name,{0,0,(float)getWidth(),22},14,palette::muted,true,juce::Justification::centred);}
void Dial::resized() {slider.setBounds(getLocalBounds().withTrimmedTop(20));}
void CaptureCard::paintButton(juce::Graphics& g,bool hover,bool)
{
    auto r=getLocalBounds().toFloat().reduced(.5f);
    g.setColour(selected?palette::accent.withAlpha(.10f):hover?palette::raised:palette::panel);g.fillRoundedRectangle(r,6);
    g.setColour(selected?palette::accent:palette::line);g.drawRoundedRectangle(r,6,1);
    if(image.isValid()) photo(g,image,{10,13,56,50});
    else {g.setColour(palette::line);g.fillRoundedRectangle(10,13,56,50,3);label(g,"NAM",{10,13,56,50},13,palette::muted,true,juce::Justification::centred);}
    const auto detailsY=(float)getHeight()-23;
    g.setFont(font(15,true));g.setColour(palette::text);g.drawFittedText(niceName(title),78,12,getWidth()-90,(int)detailsY-16,juce::Justification::topLeft,getHeight()<86?2:3,1.0f);
    label(g,(rank>0?juce::String(rank)+"  /  ":"")+detail,{78,detailsY,r.getWidth()-90,19},13,selected?palette::accent:palette::muted);
}

ToneHoundEditor::ToneHoundEditor(ToneHoundProcessor& p) : AudioProcessorEditor(p),processor(p)
{
    setLookAndFeel(&EZLookAndFeel::instance());setOpaque(true);
    setResizable(true,true);setResizeLimits(1280,854,1920,1280);getConstrainer()->setFixedAspectRatio(1.5);setSize(1440,960);
    for(auto* c:std::initializer_list<juce::Component*>{&profiles,&channel,&browse,&io,&monitor,&bypass,&fileTab,&linkTab,&importButton,&fetchButton,&previewButton,&loopButton,&findButton,&cancelButton,&url,&startField,&endField,&sourceLink,&waveform}) addAndMakeVisible(c);
    const char* ids[]{"input","gate","bass","mid","treble","output"}; const char* names[]{"INPUT","GATE","BASS","MID","TREBLE","OUTPUT"};
    for(size_t i=0;i<6;++i) {dials[i]=std::make_unique<Dial>(p,ids[i],names[i]);addAndMakeVisible(*dials[i]);addAndMakeVisible(cards[i]);}
    const char* pages[]{"Amplifier","EQ match","Pedals","Library"};
    for(int i=0;i<4;++i){pageTabs[i].setButtonText(pages[i]);pageTabs[i].onClick=[this,i]{setPage(i);};pageTabs[i].setComponentID("page."+juce::String(i));addAndMakeVisible(pageTabs[i]);}
    eqPanel=std::make_unique<EqMatchPanel>(p);pedalPanel=std::make_unique<PedalPanel>(p);libraryPanel=std::make_unique<LibraryPanel>(p);
    libraryPanel->libraryChanged=[this]{loadLibrary();updateCards();};
    for(auto* c:std::initializer_list<juce::Component*>{eqPanel.get(),pedalPanel.get(),libraryPanel.get(),&captureDescription,&passageRole})addAndMakeVisible(c);
    captureDescription.setReadOnly(true);captureDescription.setMultiLine(true,true);captureDescription.setScrollbarsShown(true);
    captureDescription.setFont(font(16));captureDescription.setName("Capture description");captureDescription.setColour(juce::TextEditor::outlineColourId,juce::Colours::transparentBlack);
    passageRole.addItem("Rhythm - dry / tight",1);passageRole.addItem("Solo - delay / reverb",2);passageRole.setSelectedId(1,juce::dontSendNotification);
    roleAttachment=std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(p.state,"performanceRole",passageRole);
    passageRole.setName("Passage type");passageRole.setComponentID("reference.role");passageRole.onChange=[this]{applyRoleEffects();};
    addAndMakeVisible(matchingModel);
    matchingModel.addItem("Standard",1);matchingModel.addItem("LoRA (pilot)",2);
    matchingModel.setItemEnabled(2,processor.root.getChildFile(".cache/matching_models/lora/model.json").existsAsFile());
    matchingModel.setName("Matching model");matchingModel.setComponentID("reference.matchingModel");
    matchingModel.setTooltip("LoRA uses your trained pilot adapter and head for song matching. Standard uses the existing MERT matcher. LoRA is experimental.");
    matchingModel.setSelectedId(processor.matchingModel.load()+1,juce::dontSendNotification);
    matchingModel.onChange=[this]{
        processor.matchingModel=matchingModel.getSelectedId()==2?1:0;
        processor.setMatches(juce::var{});updateCards();
        status=matchingModel.getText()+" selected. Find matching tones to refresh the results.";repaint();
    };
    profiles.setComponentID("amp.selector");
    channel.addItem("Input 1",1);channel.addItem("Input 2",2);
    channelAttachment=std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(p.state,"channel",channel);
    pitchControl.setSliderStyle(juce::Slider::IncDecButtons);pitchControl.setTextBoxStyle(juce::Slider::TextBoxLeft,false,56,30);
    pitchControl.setTextValueSuffix(" st");pitchControl.setName("Pitch transpose in semitones");pitchControl.setDoubleClickReturnValue(true,0);
    pitchControl.setTooltip("Transpose before the amp, -12 to +12 semitones. Open Pitch settings for blend, fine tuning and attack preservation. Zero semitones and zero cents bypass pitch processing.");
    pitchControl.setComponentID("pitch.transpose");addAndMakeVisible(pitchControl);
    pitchAttachment=std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(p.state,"pitchSemitones",pitchControl);
    pitchSettings.setName("Open pitch settings");pitchSettings.setComponentID("pitch.settings");
    addAndMakeVisible(pitchSettings);
    pitchSettings.onClick=[this]{juce::CallOutBox::launchAsynchronously(makePitchPanel(processor),pitchSettings.getBounds(),this);};
    monitor.setClickingTogglesState(true);bypass.setClickingTogglesState(true);
    monitorAttachment=std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(p.state,"monitor",monitor);
    bypassAttachment=std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(p.state,"bypass",bypass);
    fileTab.onClick=[this]{setImportMode(false);};linkTab.onClick=[this]{setImportMode(true);};
    importButton.onClick=[this]{chooseFile(false);};browse.onClick=[this]{chooseFile(true);};
    fetchButton.onClick=[this]{importSong(url.getText().trim());};url.onReturnKey=fetchButton.onClick;
    url.setTextToShowWhenEmpty("Paste a song URL",palette::muted);url.setFont(font(16));
    for(auto* f:{&startField,&endField}) {f->setFont(font(16));f->setJustification(juce::Justification::centred);f->setSelectAllWhenFocused(true);}
    startField.onReturnKey=[this]{readSelection(true);};startField.onFocusLost=startField.onReturnKey;
    endField.onReturnKey=[this]{readSelection(false);};endField.onFocusLost=endField.onReturnKey;
    startField.setComponentID("crop.start");endField.setComponentID("crop.end");
    previewButton.setComponentID("reference.play");io.setComponentID("audio.io");
    waveform.changed=[this](double s,double e){updateSelection(s,e);};
    previewButton.onClick=[this]{processor.previewPlaying=!processor.previewPlaying.load();processor.previewRestart=true;};
    loopButton.setClickingTogglesState(true);loopButton.setToggleState(true,juce::dontSendNotification);
    loopButton.onClick=[this]{processor.previewLoop=loopButton.getToggleState();};
    findButton.setColour(juce::TextButton::buttonColourId,palette::accent);findButton.setColour(juce::TextButton::textColourOffId,palette::background);
    findButton.onClick=[this] {
        processor.previewPlaying=false;
        processor.analysis.submit(object({{"action","match"},{"path",processor.referenceInfo()["path"]},
                                          {"start",waveform.start},{"end",waveform.end},{"role",passageRole.getSelectedId()==2?"solo":"rhythm"},
                                          {"matching_model",processor.matchingModel.load()==1?"lora":"standard"}}));
    };
    cancelButton.onClick=[this]{processor.analysis.cancel();};io.onClick=[this]{showIO();};
    profiles.onChange=[this]{auto i=profiles.getSelectedId()-1;if(i>=0&&i<(int)library.size())selectCapture(library[(size_t)i]);};
    sourceLink.setJustificationType(juce::Justification::centredLeft);
    loadLibrary();setImportMode(false);
    addAndMakeVisible(matchHelp);matchHelp.setColour(juce::TextButton::buttonColourId,palette::background);
    matchHelp.onClick=[this]{juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::InfoIcon,"About these matches",processor.matchExplanation(),"OK",this);};
    if(auto reference=processor.referenceInfo();reference.isObject()) {
        duration=(double)reference["duration"];referenceName=reference["name"].toString();referenceOrigin=reference["origin"].toString();
        referenceChannels=(int)reference["channels"];
        stemNote=reference["stem_note"].toString();
        passageRole.setSelectedId(reference["role"].toString()=="solo"?2:1,juce::dontSendNotification);
        auto target=reference["stem_path"].toString();if(target.isNotEmpty())processor.loadEqTarget(juce::File(target),true);
        waveform.setFile(juce::File(reference["path"].toString()),duration,processor.selectionStart,processor.selectionEnd);
    }
    updateSelection(waveform.start,waveform.end);updateCards();setPage(0);startTimerHz(24);resized();
}
ToneHoundEditor::~ToneHoundEditor() {stopTimer();setLookAndFeel(nullptr);}
juce::var ToneHoundEditor::rowFor(const juce::File& file)
{
    auto stem=file.getFileNameWithoutExtension();int id=0;
    if(!stem.startsWith("t3k-")){
        auto saved=readJson(processor.root.getChildFile(".cache/native/library.json"));
        if(auto* rows=saved["profiles"].getArray())for(const auto& row:*rows)
            if(row["path"].toString()==file.getFullPathName())return row;
    }
    if(stem.startsWith("t3k-")) id=stem.substring(4).upToFirstOccurrenceOf("-",false,false).getIntValue();
    auto meta=readJson(processor.root.getChildFile(".cache/native/artwork").getChildFile(juce::String(id)+".json"));
    auto catalog=readJson(processor.root.getChildFile(".cache/tone3000/metadata").getChildFile(juce::String(id)+".json"));
    if(!meta.isObject())meta=object({});
    if(catalog.isObject())for(auto key:{"title","creator","gear","url","description","makes","models","metadata_version"})
        if(meta[key].isVoid())meta.getDynamicObject()->setProperty(key,catalog[key]);
    auto title=meta["title"].toString();
    return object({{"path",file.getFullPathName()},{"name",title.isEmpty()?niceName(stem):title},
                   {"model_name",stem},{"tone_id",id},{"creator",meta["creator"]},{"gear",meta["gear"]},
                   {"image_path",meta["image_path"]},{"url",meta["url"]},{"description",meta["description"]},{"makes",meta["makes"]},{"models",meta["models"]},{"metadata_version",meta["metadata_version"]}});
}
void ToneHoundEditor::loadLibrary()
{
    profiles.clear(juce::dontSendNotification);library.clear();
    for(auto directory:{".cache/tone3000/profiles","assets/dev_profiles"}) {
        auto files=processor.root.getChildFile(directory).findChildFiles(juce::File::findFiles,false,"*.nam");files.sort();
        for(auto& f:files) if(!f.getFileName().startsWith("_")) {
            auto row=rowFor(f);library.push_back(row);profiles.addItem(row["name"].toString(),(int)library.size());
        }
    }
    // The descriptor catalogue also contains captures whose NAM was evicted.
    auto savedLibrary=readJson(processor.root.getChildFile(".cache/native/library.json"));
    if(auto* rows=savedLibrary["profiles"].getArray())for(const auto& row:*rows){
        auto file=juce::File(row["path"].toString());
        const bool allowed=file.getParentDirectory()==processor.root.getChildFile(".cache/tone3000/profiles") ||
            (file.getParentDirectory()==processor.root.getChildFile(".cache/community/profiles") && file.existsAsFile());
        if(!allowed || !file.hasFileExtension("nam"))continue;
        bool known=false;for(const auto& existing:library)if(existing["path"]==row["path"]){known=true;break;}
        if(!known){library.push_back(row);profiles.addItem(row["name"].toString(),(int)library.size());}
    }
    int selected=0;auto path=processor.profilePath();
    for(size_t i=0;i<library.size();++i) if(library[i]["path"].toString()==path) selected=(int)i;
    if(path.isNotEmpty() &&
       (library.empty() || library[(size_t)selected]["path"].toString()!=path)) selectCapture(rowFor(juce::File(path)));
    else if(!library.empty()) {profiles.setSelectedId(selected+1,juce::dontSendNotification);selectCapture(library[(size_t)selected]);}
}
void ToneHoundEditor::selectCapture(juce::var row)
{
    selectedCapture=row;auto path=row["path"].toString();
    if(path!=processor.profilePath() || (!processor.modelReady && !processor.loading)) processor.loadProfile(juce::File(path),row["name"].toString());
    bool found=false;
    for(size_t i=0;i<library.size();++i) if(library[i]["path"].toString()==path) {profiles.setSelectedId((int)i+1,juce::dontSendNotification);found=true;}
    if(!found) {library.push_back(row);profiles.addItem(row["name"].toString(),(int)library.size());profiles.setSelectedId((int)library.size(),juce::dontSendNotification);}
    refreshPhoto();
    auto id=(int)row["tone_id"];
    if(id>0 && (!ampPhoto.isValid() || (int)selectedCapture["metadata_version"]!=2) && processor.artwork.submit(object({{"action","artwork"},{"tone_id",id}}))) requestedArtwork=id;
    updateCards();repaint();
}
void ToneHoundEditor::refreshPhoto()
{
    auto id=(int)selectedCapture["tone_id"];
    auto meta=readJson(processor.root.getChildFile(".cache/native/artwork").getChildFile(juce::String(id)+".json"));
    if(meta.isObject()) {
        for(auto key:{"image_path","creator","gear","url","description","makes","models","metadata_version"})
            if(!meta[key].isVoid())selectedCapture.getDynamicObject()->setProperty(key,meta[key]);
    }
    auto imagePath=selectedCapture["image_path"].toString();ampPhoto=imagePath.isEmpty()?juce::Image{}:juce::ImageFileFormat::loadFrom(juce::File(imagePath));
    auto source=selectedCapture["url"].toString();if(source.isEmpty() && id>0) source="https://www.tone3000.com/tones/"+juce::String(id);
    sourceLink.setURL(juce::URL(source));sourceLink.setButtonText(id>0?"View capture on TONE3000":"View capture source");
    auto description=selectedCapture["description"].toString();
    captureDescription.setText(description.isNotEmpty()?description:(int)selectedCapture["metadata_version"]==2?"No description supplied for this capture.":"Description unavailable. Connect TONE3000 in Library to refresh capture details.",false);
    sourceLink.setVisible(source.isNotEmpty() && page==0);
}
void ToneHoundEditor::updateCards()
{
    auto matches=processor.matchInfo();auto* ranked=matches.getArray();
    for(size_t i=0;i<cards.size();++i) {
        juce::var row;
        if(ranked && (int)i<ranked->size()) row=ranked->getReference((int)i);
        else if(!ranked && i<library.size()) row=library[i];
        auto& card=cards[i];card.setVisible(row.isObject());if(!row.isObject())continue;
        card.title=row["name"].toString();card.rank=ranked?(int)i+1:0;
        card.detail=ranked?"similarity "+juce::String((double)row["similarity"],2):"PLAY CAPTURE";
        auto imagePath=row["image_path"].toString();if(imagePath.isEmpty()) imagePath=processor.root.getChildFile(".cache/native/artwork").getChildFile(row["tone_id"].toString()+".jpg").getFullPathName();
        card.image=juce::ImageFileFormat::loadFrom(juce::File(imagePath));card.selected=row["path"].toString()==selectedCapture["path"].toString();
        card.onClick=[this,row]{selectCapture(row);};card.setTooltip(row["name"].toString()+"\nClick to play this capture with your guitar.");card.repaint();
    }
}
void ToneHoundEditor::setImportMode(bool link)
{
    linkMode=link;fileTab.setToggleState(!link,juce::dontSendNotification);linkTab.setToggleState(link,juce::dontSendNotification);
    url.setVisible(link);fetchButton.setVisible(link);importButton.setVisible(!link);repaint();
}
void ToneHoundEditor::chooseFile(bool nam)
{
    chooser=std::make_unique<juce::FileChooser>(nam?"Load a NAM capture":"Import a reference song",processor.root,nam?"*.nam":"*.mp3;*.wav",true);
    juce::Component::SafePointer<ToneHoundEditor> safe(this);
    chooser->launchAsync(juce::FileBrowserComponent::openMode|juce::FileBrowserComponent::canSelectFiles,[safe,nam](const juce::FileChooser& fc) {
        if(!safe)return;auto f=fc.getResult();if(!f.existsAsFile())return;
        if(nam) safe->selectCapture(safe->rowFor(f));else safe->importSong(f.getFullPathName());
    });
}
void ToneHoundEditor::importSong(juce::String source)
{
    if(source.isEmpty()) {status="Choose an MP3/WAV file or enter a song URL.";return;}
    if(processor.analysis.snapshot().busy) return;
    processor.previewPlaying=false;
    processor.clearEqTarget();
    processor.analysis.submit(object({{"action","import"},{"source",source}}));
}
juce::String ToneHoundEditor::clockText(double value)
{
    auto minutes=(int)value/60;return juce::String(minutes).paddedLeft('0',2)+":"+juce::String(value-minutes*60,1).paddedLeft('0',4);
}
double ToneHoundEditor::parseTime(const juce::String& text)
{return text.contains(":")?text.upToFirstOccurrenceOf(":",false,false).getDoubleValue()*60+text.fromFirstOccurrenceOf(":",false,false).getDoubleValue():text.getDoubleValue();}
void ToneHoundEditor::updateSelection(double s,double e)
{
    if(duration>=30) {s=juce::jlimit(0.0,duration-30,s);e=juce::jlimit(s+30,duration,e);}
    if(s!=processor.selectionStart || e!=processor.selectionEnd){
        processor.clearEqTarget();auto reference=processor.referenceInfo();
        if(auto* info=reference.getDynamicObject()){info->removeProperty("stem_path");info->removeProperty("reference_features");processor.setReference(reference);}
    }
    auto cached=readJson(processor.root.getChildFile(".cache/native/library.json"))["profiles"];
    if(auto* entries=cached.getArray())for(auto row:*entries){
        const auto path=row["path"].toString();bool found=false;
        for(const auto& local:library)if(local["path"].toString()==path){found=true;break;}
        if(!found && path.isNotEmpty() && (int)row["model_id"]>0){library.push_back(row);profiles.addItem(row["name"].toString(),(int)library.size());}
    }
    waveform.setSelection(s,e);processor.setSelection(s,e);
    if(!startField.hasKeyboardFocus(true))startField.setText(clockText(s),false);
    if(!endField.hasKeyboardFocus(true))endField.setText(clockText(e),false);
    repaint();
}
void ToneHoundEditor::readSelection(bool editingStart)
{
    if(duration<30)return;
    auto s=waveform.start,e=waveform.end;
    auto entered=parseTime(editingStart?startField.getText():endField.getText());
    if(!std::isfinite(entered)) entered=editingStart?s:e;
    if(editingStart) s=juce::jlimit(0.0,e-30,entered);else e=juce::jlimit(s+30,duration,entered);
    updateSelection(s,e);startField.setText(clockText(waveform.start),false);endField.setText(clockText(waveform.end),false);
}
void ToneHoundEditor::receive(const AnalysisBridge::Snapshot& job)
{
    if(!(bool)job.response["ok"]) {status=job.response["message"].toString();return;}
    auto result=job.response["result"];
    if(job.action=="import") {
        processor.setReference(result);processor.setMatches(juce::var{});duration=(double)result["duration"];
        referenceChannels=(int)result["channels"];
        referenceName=result["name"].toString();referenceOrigin=result["origin"].toString();
        waveform.setFile(juce::File(result["path"].toString()),duration,(double)result["start"],(double)result["end"]);
        processor.loadPreview(juce::File(result["path"].toString()));updateSelection(waveform.start,waveform.end);
        status="Drag the handles or enter times. Select at least 30 seconds.";stemNote.clear();
    } else if(job.action=="match") {
        processor.setMatches(result["matches"],result["caveat"].toString());
        stemNote="Compared the "+result["stem"]["used"].toString()+" stem";
        if(!(bool)result["stem"]["ok"])stemNote=result["stem"]["note"].toString();
        auto reference=processor.referenceInfo();
        if(auto* info=reference.getDynamicObject()) {
            info->setProperty("stem_note",stemNote);info->setProperty("stem_path",result["stem_path"]);
            info->setProperty("reference_features",result["reference_features"]);info->setProperty("role",passageRole.getSelectedId()==2?"solo":"rhythm");processor.setReference(reference);
        }
        status=(result["retrieval"]["mode"].toString()=="lora"?"LoRA (pilot): ":"Standard: ")+juce::String("closest sounds in your library. Audition the captures to compare.");
        if(auto* rows=result["matches"].getArray();rows&&!rows->isEmpty())selectCapture(rows->getReference(0));
        processor.loadEqTarget(juce::File(result["stem_path"].toString()));applyRoleEffects();
    }
    updateCards();repaint();
}
void ToneHoundEditor::applyRoleEffects()
{
    auto reference=processor.referenceInfo();
    if(auto* info=reference.getDynamicObject()){info->setProperty("role",passageRole.getSelectedId()==2?"solo":"rhythm");processor.setReference(reference);}
    if(processor.hasMatches())processor.applyPerformanceSuggestion(passageRole.getSelectedId()==2,(float)(double)reference["reference_features"]["bpm"]);
}
void ToneHoundEditor::setPage(int selected)
{
    page=selected;
    for(int i=0;i<4;++i)pageTabs[i].setToggleState(i==page,juce::dontSendNotification);
    for(auto* c:std::initializer_list<juce::Component*>{&profiles,&browse,&sourceLink,&captureDescription})c->setVisible(page==0);
    sourceLink.setVisible(page==0 && selectedCapture["url"].toString().isNotEmpty());
    eqPanel->setVisible(page==1);pedalPanel->setVisible(page==2);libraryPanel->setVisible(page==3);repaint();
}
void ToneHoundEditor::timerCallback()
{
    processor.collectRetired();auto job=processor.analysis.snapshot();
    if(job.completion!=lastJob) {lastJob=job.completion;receive(job);}
    auto art=processor.artwork.snapshot();
    if(art.completion!=lastArtwork) {
        lastArtwork=art.completion;refreshPhoto();updateCards();
        // Selection may change while the previous capture's photo is fetching.
        const auto wanted=(int)selectedCapture["tone_id"];
        if(wanted>0 && (!ampPhoto.isValid() || (int)selectedCapture["metadata_version"]!=2) && wanted!=requestedArtwork &&
           processor.artwork.submit(object({{"action","artwork"},{"tone_id",wanted}}))) requestedArtwork=wanted;
    }
    importButton.setEnabled(!job.busy);fetchButton.setEnabled(!job.busy);
    matchingModel.setEnabled(!job.busy);
    matchingModel.setSelectedId(processor.matchingModel.load()+1,juce::dontSendNotification);
    findButton.setEnabled(!job.busy && duration>=30);findButton.setButtonText(job.busy?"Working...":"Find matching tones");
    cancelButton.setVisible(job.busy);previewButton.setEnabled(duration>=30 && !job.busy);
    matchHelp.setVisible(processor.hasMatches());
    previewButton.setButtonText(processor.previewPlaying?"Stop playback":"Play selection");
    waveform.playhead=processor.previewPlaying?processor.previewPosition.load():-1;waveform.repaint();
    monitor.setButtonText(monitor.getToggleState()?"MONITOR ON":"MONITOR OFF");
    repaint();
}
bool ToneHoundEditor::isInterestedInFileDrag(const juce::StringArray& files)
{return !files.isEmpty() && juce::File(files[0]).hasFileExtension("mp3;wav;nam");}
void ToneHoundEditor::filesDropped(const juce::StringArray& files,int,int)
{dragging=false;if(files.isEmpty())return;auto f=juce::File(files[0]);if(f.hasFileExtension("nam"))selectCapture(rowFor(f));else importSong(files[0]);repaint();}
void ToneHoundEditor::loadReferenceForPreview(const juce::File& file) {importSong(file.getFullPathName());}

void ToneHoundEditor::paint(juce::Graphics& g)
{
    g.fillAll(palette::background);g.addTransform(juce::AffineTransform::scale(scale));
    label(g,"TONEHOUND",{24,15,240,40},30,palette::text,true);
    label(g,"Find your tone. Make it yours.",{26,53,255,21},14,palette::muted);
    auto meter=[&](float x,const char* title,float level) {
        label(g,title,{x,21,42,19},14,palette::muted,true);
        auto db=juce::Decibels::gainToDecibels(level,-60.f);auto v=juce::jlimit(0.f,1.f,(db+60)/60);
        for(int i=0;i<24;++i){g.setColour((float)i/24<v?(i>21?juce::Colour(0xffda8d75):palette::green):palette::line.withAlpha(.5f));g.fillRoundedRectangle(x+43+i*4.4f,23,2.7f,13,1);}
        label(g,db<=-59?"-inf dB":juce::String(db,1)+" dB",{x+43,44,110,21},13,palette::muted);
    };
    meter(280,"IN",processor.inputPeak);meter(430,"OUT",processor.outputPeak);
    panel(g,{24,136,944,374});panel(g,{24,528,944,148});panel(g,{992,88,424,830});
    if(page==0) {
        juce::Rectangle<float> stage{44,210,504,278};
        g.setColour(juce::Colour(0xff101314));g.fillRoundedRectangle(stage,6);
        if(ampPhoto.isValid())photo(g,ampPhoto,stage);
        else {
            label(g,processor.loading?"Loading capture...":"NAM",stage,38,palette::muted,true,juce::Justification::centred);
            label(g,"Capture photograph",{54,450,484,25},15,palette::muted,false,juce::Justification::centred);
        }
        g.setColour(palette::line);g.drawRoundedRectangle({568,210,380,278},6,1);
        auto metadata=[](const juce::var& value){
            if(auto* a=value.getArray()){juce::StringArray names;for(auto& v:*a)names.add(v.isObject()?v["name"].toString():v.toString());return names.joinIntoString(", ");}
            return value.toString();
        };
        auto identity=metadata(selectedCapture["makes"]),models=metadata(selectedCapture["models"]);
        if(models.isNotEmpty())identity+=(identity.isEmpty()?"":" / ")+models;
        label(g,"Make / model",{586,224,344,20},14,palette::accent,true);
        g.setColour(palette::text);g.setFont(font(17,true));
        g.drawFittedText(identity.isEmpty()?"Details unavailable":identity,{586,250,344,42},juce::Justification::topLeft,2,1.0f);
        auto creator=selectedCapture["creator"].toString();
        label(g,creator.isEmpty()?"Local NAM capture":"Capture by "+creator,{586,298,344,22},14,palette::muted);
    }
    label(g,"Reference",{1012,106,380,32},25,palette::text,true);
    label(g,linkMode?"Paste a supported song link":"MP3 / WAV - drop a file or browse",{1012,245,380,24},15,palette::muted);
    label(g,referenceName.isEmpty()?"No reference loaded":referenceName,{1012,283,380,25},17,palette::text,true);
    label(g,duration>0?clockText(duration)+"  /  "+(referenceChannels==2?"Stereo":referenceChannels==1?"Mono":"Re-import for stereo"):"Choose a song at least 30 seconds long",{1012,314,380,20},14,palette::muted);
    label(g,"Start",{1012,532,174,20},14,palette::muted,true);label(g,"End",{1214,532,178,20},14,palette::muted,true);
    label(g,duration>0?juce::String(waveform.end-waveform.start,1)+" seconds selected / minimum 30":"Default selection: 40 seconds",{1012,603,380,22},14,palette::accent);
    label(g,"Passage type",{1012,689,192,21},15,palette::text,true);
    label(g,"Matching model",{1220,689,172,21},15,palette::text,true);
    auto features=processor.referenceInfo()["reference_features"];
    const auto bpm=(double)features["bpm"];
    juce::String effects=passageRole.getSelectedId()==2?"Auto delay + reverb":"Dry rhythm preset";
    if(processor.hasMatches())effects+=bpm>0?" / est. "+juce::String(bpm,0)+" BPM":" / tempo adjustable";
    label(g,effects,{1012,761,380,23},14,palette::muted);
    const auto job=processor.analysis.snapshot();
    g.setFont(font(15));g.setColour(palette::muted);g.drawFittedText(job.busy?job.message:status,{1012,853,job.busy?280:380,51},juce::Justification::topLeft,3,1.0f);
    if(job.busy){g.setColour(palette::line);g.fillRoundedRectangle(1012,844,380,3,1);g.setColour(palette::accent);float v=(float)job.progress;g.fillRoundedRectangle(1012,844,380*juce::jlimit(0.f,1.f,v<0?.35f:v),3,1);}
    label(g,processor.hasMatches()?"Closest library captures":"Your captures",{24,692,430,31},23,palette::text,true);
    label(g,juce::String((int)library.size())+" library captures",{512,697,242,23},15,palette::muted,false,juce::Justification::centredRight);
    g.setColour(palette::line);g.drawHorizontalLine(928,24,1416);
    auto ready=processor.modelReady.load()&&!processor.audioFault.load();g.setColour(ready?palette::green:palette::muted);g.fillEllipse(26,943,6,6);
    label(g,processor.audioFault?"Audio stopped. Reload the capture.":processor.loading?"Loading capture...":processor.modelMessage(),{42,932,660,26},14,palette::muted);
    label(g,juce::String(processor.rate.load()/1000,1)+" kHz / "+juce::String(processor.getBlockSize())+" samples / "+(processor.standalone?"Standalone":"VST3"),{992,932,424,26},14,palette::muted,false,juce::Justification::centredRight);
    if(dragging){g.setColour(palette::accent.withAlpha(.1f));g.fillRoundedRectangle(992,88,424,830,10);g.setColour(palette::accent);g.drawRoundedRectangle(992,88,424,830,10,2);}
}
void ToneHoundEditor::resized()
{
    scale=getWidth()/1440.f;
    auto bounds=[&](juce::Component& c,float x,float y,float w,float h){c.setBounds(juce::Rectangle<float>(x*scale,y*scale,w*scale,h*scale).toNearestInt());};
    bounds(channel,714,23,145,38);bounds(monitor,879,23,172,38);bounds(bypass,1071,23,144,38);bounds(io,1235,23,181,38);
    bounds(pitchSettings,584,9,122,29);bounds(pitchControl,584,42,122,30);
    for(int i=0;i<4;++i)bounds(pageTabs[i],24+(float)i*239,88,227,36);
    bounds(profiles,44,156,716,38);bounds(browse,776,156,172,38);
    bounds(sourceLink,586,452,344,25);bounds(captureDescription,580,328,356,114);
    for(auto* component:std::initializer_list<juce::Component*>{eqPanel.get(),pedalPanel.get(),libraryPanel.get()})if(component){
        component->setTransform(juce::AffineTransform());component->setBounds(0,0,904,334);
        component->setTransform(juce::AffineTransform::scale(scale).translated(44*scale,156*scale));
    }
    for(size_t i=0;i<6;++i){if(dials[i])bounds(*dials[i],32+(float)i*156,542,150,121);bounds(cards[i],24+(float)(i%3)*320,730+(float)(i/3)*98,304,90);}
    bounds(fileTab,1012,153,184,36);bounds(linkTab,1208,153,184,36);
    bounds(importButton,1012,200,380,40);bounds(url,1012,200,280,40);bounds(fetchButton,1304,200,88,40);
    bounds(waveform,1012,344,380,170);bounds(startField,1012,555,178,38);bounds(endField,1214,555,178,38);
    bounds(previewButton,1012,636,275,40);bounds(loopButton,1301,636,91,40);
    bounds(passageRole,1012,718,192,38);bounds(matchingModel,1220,718,172,38);
    bounds(findButton,1012,794,380,43);bounds(cancelButton,1304,881,88,30);
    bounds(matchHelp,775,693,193,31);
}

namespace {
class PitchContent final : public juce::Component
{
public:
    explicit PitchContent(ToneHoundProcessor& p)
    {
        setLookAndFeel(&EZLookAndFeel::instance());setSize(440,326);
        quality.addItem("Live / 40 ms",1);quality.addItem("Studio / 192 ms",2);
        quality.setName("Pitch quality and added latency");quality.setComponentID("pitch.quality");
        qualityAttachment=std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(p.state,"pitchQuality",quality);
        for(auto* slider:{&blend,&cents}){
            slider->setSliderStyle(juce::Slider::LinearHorizontal);slider->setTextBoxStyle(juce::Slider::TextBoxRight,false,90,32);
            slider->setDoubleClickReturnValue(true,slider==&blend?1.:0.);addAndMakeVisible(*slider);
        }
        blend.setName("Pitch wet blend");blend.setComponentID("pitch.blend");
        cents.setName("Pitch fine tuning in cents");cents.setComponentID("pitch.cents");cents.setTextValueSuffix(" cents");
        blendAttachment=std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(p.state,"pitchBlend",blend);
        // APVTS installs its formatter; set the musician-facing percentage after it.
        blend.textFromValueFunction=[](double value){return juce::String(value*100,0)+"% wet";};
        blend.valueFromTextFunction=[](const juce::String& text){return text.getDoubleValue()/100.;};blend.updateText();
        centsAttachment=std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(p.state,"pitchCents",cents);
        attack.setButtonText("Preserve pick attacks");attack.setComponentID("pitch.attack");
        attack.setTooltip("Shape the shifted sound with the original pick envelope. Turn off if the envelope pumps on sustained chords.");
        attackAttachment=std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(p.state,"pitchAttack",attack);
        addAndMakeVisible(quality);addAndMakeVisible(attack);
        quality.setBounds(172,58,244,34);blend.setBounds(162,111,254,36);cents.setBounds(162,164,254,36);attack.setBounds(24,215,392,34);
    }
    ~PitchContent() override {setLookAndFeel(nullptr);}
    void paint(juce::Graphics& g) override
    {
        g.fillAll(palette::panel);label(g,"Pitch",{24,15,392,30},25,palette::text,true);
        label(g,"Quality / delay",{24,60,143,30},16);label(g,"Blend",{24,114,130,30},16);label(g,"Fine tuning",{24,167,135,30},16);
        g.setFont(font(15));g.setColour(palette::muted);
        g.drawFittedText("100% wet transposes; a blend adds harmony. Studio is for recorded guitar and clearer chords. Zero semitones + zero cents bypasses pitch.",{24,264,392,54},juce::Justification::topLeft,3);
    }
private:
    juce::ComboBox quality;juce::Slider blend,cents;juce::ToggleButton attack;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> qualityAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> blendAttachment,centsAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> attackAttachment;
};
class IOContent final : public juce::Component
{
public:
    explicit IOContent(ToneHoundProcessor& p):processor(p)
    {
        setLookAndFeel(&EZLookAndFeel::instance());setSize(650,p.deviceManager?560:260);
        if(p.deviceManager) {selector=std::make_unique<juce::AudioDeviceSelectorComponent>(*p.deviceManager,1,2,2,2,false,false,true,false);addAndMakeVisible(*selector);resized();}
    }
    ~IOContent() override {setLookAndFeel(nullptr);}
    void paint(juce::Graphics& g) override {
        g.fillAll(palette::background);label(g,"Audio I / O",{22,12,590,37},24,palette::text,true);
        if(selector) {
            label(g,"Choose your interface, input channels, sample rate and buffer size.",{22,53,607,23},15,palette::muted);
            label(g,"Use Input 1 / Input 2 in the amp header to choose your guitar channel.",{22,(float)getHeight()-42,605,23},14,palette::muted);
        } else {
            g.setColour(palette::text);g.setFont(font(16));g.drawFittedText("Your DAW manages audio devices, sample rate and buffer size. Open its audio preferences to change the interface.",{24,75,600,76},juce::Justification::topLeft,3);
            label(g,juce::String(processor.rate.load(),0)+" Hz  /  "+juce::String(processor.getBlockSize())+" samples",{24,169,595,24},16,palette::accent);
            label(g,"Select the guitar channel using Input 1 / Input 2 in ToneHound.",{24,211,595,22},15,palette::muted);
        }
    }
    void resized() override {if(selector)selector->setBounds(20,89,getWidth()-40,getHeight()-145);}
private: ToneHoundProcessor& processor;std::unique_ptr<juce::AudioDeviceSelectorComponent> selector;
};
}
void ToneHoundEditor::showIO()
{
    juce::DialogWindow::LaunchOptions opts;opts.content.setOwned(makeAudioIOPanel(processor).release());
    opts.dialogTitle="ToneHound - Audio I/O";opts.dialogBackgroundColour=palette::background;
    opts.escapeKeyTriggersCloseButton=true;opts.useNativeTitleBar=true;opts.resizable=false;opts.componentToCentreAround=this;opts.launchAsync();
}
std::unique_ptr<juce::Component> makeAudioIOPanel(ToneHoundProcessor& p) {return std::make_unique<IOContent>(p);}
std::unique_ptr<juce::Component> makePitchPanel(ToneHoundProcessor& p) {return std::make_unique<PitchContent>(p);}
