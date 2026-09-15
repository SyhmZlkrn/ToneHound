#include "PluginEditor.h"
#include "StudioPanels.h"

namespace {
juce::Font uiFont(float size,bool bold=false) {return juce::Font(juce::FontOptions("Segoe UI",size,bold?juce::Font::bold:juce::Font::plain));}
void text(juce::Graphics& g,const juce::String& value,juce::Rectangle<int> bounds,float size=15,juce::Colour colour=palette::text,int lines=1)
{g.setColour(colour);g.setFont(uiFont(size));g.drawFittedText(value,bounds,juce::Justification::centredLeft,lines,1.0f);}
juce::String joined(const juce::var& value)
{
    if(auto* a=value.getArray()){juce::StringArray result;for(const auto& v:*a)result.add(v.isObject()?v["name"].toString():v.toString());return result.joinIntoString(", ");}
    return value.toString();
}
}

EqMatchPanel::EqMatchPanel(ToneHoundProcessor& p):processor(p)
{
    for(auto* c:{&capture,&cancel,&reset,&enable})addAndMakeVisible(c);
    capture.setComponentID("eq.capture");enable.setComponentID("eq.enable");reset.setComponentID("eq.reset");
    enable.setClickingTogglesState(true);
    attachment=std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(p.state,"eqMatchEnabled",enable);
    capture.onClick=[this]{processor.startEqCapture();};
    cancel.onClick=[this]{processor.cancelEqCapture();};reset.onClick=[this]{processor.resetEqMatch();};
    capture.setTooltip("After matching a song, enable monitoring and play representative guitar for ten seconds. Backing audio is excluded.");
    for(size_t i=0;i<bands.size();++i){
        auto& slider=bands[i];slider.setSliderStyle(juce::Slider::LinearVertical);
        slider.setTextBoxStyle(juce::Slider::TextBoxBelow,false,86,25);slider.setTextValueSuffix(" dB");
        slider.setDoubleClickReturnValue(true,0);slider.setName(juce::String(tonehound::matchFrequency(i),0)+" Hz EQ gain");
        slider.setComponentID("eq.band"+juce::String((int)i+1));
        slider.onDragStart=[this]{processor.setParameter("eqMatchEnabled",1);};
        bandAttachments[i]=std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(p.state,"eqBand"+juce::String((int)i+1),slider);
        addAndMakeVisible(slider);
    }
    startTimerHz(10);
}
EqMatchPanel::~EqMatchPanel(){stopTimer();}
void EqMatchPanel::timerCallback()
{
    const auto progress=processor.eqCaptureProgress();
    const auto busy=processor.eqCapturing();
    capture.setEnabled(processor.eqTargetReady() && processor.modelReady && !processor.loading && !busy
                       && processor.state.getRawParameterValue("monitor")->load()>.5f
                       && processor.state.getRawParameterValue("bypass")->load()<.5f);
    cancel.setVisible(busy);enable.setEnabled(processor.eqMatchReady());reset.setEnabled(processor.eqMatchReady());
    for(auto& band:bands)band.setEnabled(!busy);
    enable.setButtonText(enable.getToggleState()?"EQ on":"EQ bypassed");repaint();
}
void EqMatchPanel::paint(juce::Graphics& g)
{
    text(g,"Eight-band EQ",{0,0,getWidth(),30},23);
    text(g,"Capture your guitar for 10 seconds, then shape each band. Double-click a fader to reset.",{0,32,getWidth(),24},15,palette::muted);
    const auto width=getWidth()/8;
    for(int i=0;i<8;++i){
        auto area=juce::Rectangle<int>(i*width+5,61,width-10,189);
        g.setColour(palette::background);g.fillRoundedRectangle(area.toFloat(),5);
        auto hz=tonehound::matchFrequency((size_t)i);
        text(g,hz<1000?juce::String(hz,0)+" Hz":juce::String(hz/1000,0)+" kHz",area.withHeight(25).reduced(12,0),15,palette::accent);
        g.setColour(palette::line);g.drawHorizontalLine(154,(float)area.getX()+14,(float)area.getRight()-14);
    }
    const auto progress=processor.eqCaptureProgress();
    if(progress>0 && progress<1){g.setColour(palette::accent);g.fillRect(0,265,(int)(getWidth()*progress),3);}
    text(g,processor.eqMatchStatus(),{0,267,getWidth(),24},15,palette::muted);
}
void EqMatchPanel::resized()
{
    capture.setBounds(0,getHeight()-40,252,40);enable.setBounds(266,getHeight()-40,160,40);
    reset.setBounds(440,getHeight()-40,145,40);cancel.setBounds(getWidth()-164,getHeight()-40,164,40);
    const auto width=getWidth()/8;for(int i=0;i<8;++i)bands[(size_t)i].setBounds(i*width+10,87,width-20,158);
}

void StompSwitch::paintButton(juce::Graphics& g,bool hover,bool down)
{
    const auto cx=getWidth()*.5f;
    g.setColour(getToggleState()?palette::green:juce::Colour(0xff202629));g.fillEllipse(cx-4,0,8,8);
    juce::Rectangle<float> ring(cx-18,17,36,36);
    g.setGradientFill(juce::ColourGradient(juce::Colour(0xffc2c8c9),ring.getTopLeft(),juce::Colour(0xff52595c),ring.getBottomRight(),false));g.fillEllipse(ring);
    g.setColour(down?juce::Colour(0xff555c5d):hover?juce::Colour(0xffd5dadb):juce::Colour(0xff9ca3a5));g.fillEllipse(ring.reduced(7));
    if(hasKeyboardFocus(true)){g.setColour(palette::accent);g.drawEllipse(ring.expanded(3),2);}
    ::text(g,getToggleState()?"ON":"BYPASSED",{0,58,getWidth(),20},13,palette::text);
}

PedalPanel::PedalPanel(ToneHoundProcessor& p):processor(p)
{
    const char* enabled[]{"screamerEnabled","delayEnabled","reverbEnabled"};
    const char* names[]{"Screamer","Delay","Reverb"};
    for(int i=0;i<3;++i){toggles[i].setButtonText(names[i]);toggles[i].setClickingTogglesState(true);addAndMakeVisible(toggles[i]);buttons[i]=std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(p.state,enabled[i],toggles[i]);toggles[i].setTooltip("Enable or bypass this pedal.");}
    for(auto pair:std::initializer_list<std::pair<const char*,const char*>>{
        {"screamerDrive","Drive"},{"screamerTone","Tone"},{"screamerLevel","Level"},
        {"delayMix","Mix"},{"delayFeedback","Feedback"},{"delayBpm","Tempo (BPM)"},
        {"reverbMix","Mix"},{"reverbSize","Size"}}){
        auto c=std::make_unique<Control>();c->name=pair.second;c->slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        c->slider.setTextBoxStyle(juce::Slider::TextBoxBelow,false,78,24);c->slider.setName(pair.second);
        c->slider.setRotaryParameters(juce::MathConstants<float>::pi*1.22f,juce::MathConstants<float>::pi*2.78f,true);
        c->attachment=std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(p.state,pair.first,c->slider);
        addAndMakeVisible(c->slider);controls.push_back(std::move(c));
    }
    division.addItemList({"Quarter note","Dotted eighth","Eighth note","Triplet eighth"},1);addAndMakeVisible(division);
    divisionAttachment=std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(p.state,"delayDivision",division);
    division.setName("Delay note division");
}
void PedalPanel::paint(juce::Graphics& g)
{
    const auto width=(getWidth()-40)/3;
    const juce::Colour colours[]{juce::Colour(0xff25453b),juce::Colour(0xff273f50),juce::Colour(0xff49384c)};
    const char* names[]{"SCREAMER","DELAY","REVERB"};
    for(int i=0;i<3;++i){
        const int x=i*(width+20);auto box=juce::Rectangle<float>((float)x,0,(float)width,(float)getHeight()-32);
        g.setColour(colours[i]);g.fillRoundedRectangle(box,10);
        g.setColour(colours[i].brighter(.25f));g.drawRoundedRectangle(box.reduced(.5f),10,1);
        for(auto corner:{box.getTopLeft().translated(10,10),box.getTopRight().translated(-10,10),box.getBottomLeft().translated(10,-10),box.getBottomRight().translated(-10,-10)}){
            g.setColour(juce::Colour(0xffa9afb0));g.fillEllipse(corner.x-2,corner.y-2,4,4);
        }
        text(g,names[i],{x+24,15,width-48,30},23);
        if(i!=1)text(g,i==0?"MID-FOCUSED OVERDRIVE":"STEREO SPACE",{x+24,185,width-48,22},14);
    }
    for(size_t i=0;i<controls.size();++i){int col=i<3?0:i<6?1:2,slot=i<3?(int)i:i<6?(int)i-3:(int)i-6;
        text(g,controls[i]->name,{col*(width+20)+12+slot*91,59,88,23},15);}
    text(g,"Guitar > Pitch > Screamer > Amp + EQ > Delay > Reverb. Click a footswitch to enable or bypass.",{0,getHeight()-24,getWidth(),24},14,palette::muted);
}
void PedalPanel::resized()
{
    const auto width=(getWidth()-40)/3;
    for(int i=0;i<3;++i)toggles[i].setBounds(i*(width+20)+width/2-48,218,96,78);
    for(size_t i=0;i<controls.size();++i){int col=i<3?0:i<6?1:2,slot=i<3?(int)i:i<6?(int)i-3:(int)i-6;controls[i]->slider.setBounds(col*(width+20)+10+slot*91,84,88,88);}
    division.setBounds(width+44,179,width-48,33);
}

LibraryPanel::LibraryPanel(ToneHoundProcessor& p):processor(p),bridge(p.root)
{
    for(auto* c:std::initializer_list<juce::Component*>{&query,&searchButton,&connect,&add,&previous,&next,&cancel,&rows})addAndMakeVisible(c);
    for(auto* c:{&prepare,&importPack,&exportPack})addAndMakeVisible(c);
    prepare.onClick=[this]{bridge.submit(object({{"action","catalogue_index"}}));};
    auto choosePack=[this](bool importing){
        chooser=std::make_unique<juce::FileChooser>(importing?"Import tone descriptors":"Export tone descriptors",processor.root,"*.tonelib");
        juce::Component::SafePointer<LibraryPanel> safe(this);
        chooser->launchAsync((importing?juce::FileBrowserComponent::openMode:juce::FileBrowserComponent::saveMode)|juce::FileBrowserComponent::canSelectFiles,[safe,importing](const juce::FileChooser& c){
            if(!safe || c.getResult()==juce::File{})return;
            safe->bridge.submit(object({{"action",importing?"catalogue_import":"catalogue_export"},{"path",c.getResult().withFileExtension("tonelib").getFullPathName()}}));
        });
    };
    importPack.onClick=[choosePack]{choosePack(true);};exportPack.onClick=[choosePack]{choosePack(false);};
    query.setTextToShowWhenEmpty("Search amp, maker or capture",palette::muted);query.setFont(uiFont(16));query.setName("Search TONE3000");
    rows.setRowHeight(46);rows.setMultipleSelectionEnabled(false);rows.setColour(juce::ListBox::outlineColourId,palette::line);rows.setOutlineThickness(1);
    searchButton.onClick=[this]{search(1);};query.onReturnKey=searchButton.onClick;
    previous.onClick=[this]{search(page-1);};next.onClick=[this]{search(page+1);};
    connect.onClick=[this]{bridge.submit(object({{"action","library_connect"}}));};
    cancel.onClick=[this]{bridge.cancel();};
    add.onClick=[this]{auto n=rows.getSelectedRow();if(n>=0 && n<tones.size())bridge.submit(object({{"action","library_fetch"},{"tone_ids",juce::var(juce::Array<juce::var>{tones[n]["id"]})},{"max_models",3},{"models_per_tone",3},{"managed_cache",true}}));};
    bridge.submit(object({{"action","library_status"}}));startTimerHz(8);
}
LibraryPanel::~LibraryPanel(){stopTimer();}
int LibraryPanel::getNumRows(){return tones.size();}
void LibraryPanel::selectedRowsChanged(int){repaint();}
void LibraryPanel::paintListBoxItem(int row,juce::Graphics& g,int w,int h,bool selected)
{
    if(row<0 || row>=tones.size())return;
    g.fillAll(selected?palette::raised:palette::background);
    auto tone=tones[row];text(g,tone["title"].toString(),{12,2,w-24,23},16,selected?palette::accent:palette::text);
    auto detail=joined(tone["makes"])+"  "+joined(tone["models"]);if(detail.trim().isEmpty())detail=tone["gear"].toString();
    text(g,detail+"  /  "+tone["creator"].toString(),{12,25,w-24,h-25},13,palette::muted);
}
void LibraryPanel::search(int requestedPage)
{
    if(bridge.snapshot().busy)return;page=juce::jmax(1,requestedPage);
    bridge.submit(object({{"action","library_search"},{"query",query.getText().trim()},{"page",page},{"page_size",20}}));
}
void LibraryPanel::timerCallback()
{
    const auto job=bridge.snapshot();
    if(job.completion!=completion){
        completion=job.completion;
        if(!(bool)job.response["ok"])status=job.response["message"].toString();
        else {
            const auto result=job.response["result"];
            if(job.action=="library_search"){
                tones=result["tones"];pages=juce::jmax(1,(int)result["total_pages"]);rows.updateContent();rows.deselectAllRows();
                status=juce::String(tones.size())+" results. Select a tone to add up to 3 captures.";
            } else if(job.action=="library_fetch") {
                const auto added=result["added"].size(),skipped=result["skipped"].size(),failed=result["errors"].size();
                status=juce::String(added)+" added / "+juce::String(skipped)+" already present / "+juce::String(failed)+" failed.";
                if(failed>0)status+=" "+result["errors"][0]["message"].toString();
                else if(added>0)status+=" New captures are indexed on your next match.";
                if(libraryChanged)libraryChanged();
            } else if(job.action.startsWith("catalogue_")){
                status=job.action=="catalogue_export"?"Descriptor pack exported. It contains no NAM files.":juce::String((int)result["indexed_count"])+" descriptors ready. NAM files are downloaded when needed.";
                if(libraryChanged)libraryChanged();
            } else {
                status=(bool)result["connected"]?"TONE3000 login saved. Search to find and add more captures.":"Connect your own TONE3000 account to search and download captures.";
            }
        }
    }
    query.setEnabled(!job.busy);searchButton.setEnabled(!job.busy);connect.setEnabled(!job.busy);
    prepare.setEnabled(!job.busy);importPack.setEnabled(!job.busy);exportPack.setEnabled(!job.busy);
    previous.setEnabled(!job.busy&&page>1);next.setEnabled(!job.busy&&page<pages);
    add.setEnabled(!job.busy&&rows.getSelectedRow()>=0);cancel.setVisible(job.busy);
    if(job.busy)status=job.message;repaint();
}
void LibraryPanel::paint(juce::Graphics& g)
{
    text(g,"Keep descriptors. Cache NAM files as needed (32 MiB working cache).",{0,45,getWidth()-195,26},15,palette::muted);
    text(g,"Page "+juce::String(page)+" / "+juce::String(pages),{230,getHeight()-82,140,34},14,palette::muted);
    text(g,status,{0,getHeight()-42,getWidth()-(cancel.isVisible()?100:0),40},14,palette::muted,2);
}
void LibraryPanel::resized()
{
    query.setBounds(0,0,getWidth()-320,38);searchButton.setBounds(getWidth()-306,0,116,38);connect.setBounds(getWidth()-176,0,176,38);
    rows.setBounds(0,82,getWidth(),getHeight()-171);
    prepare.setBounds(getWidth()-190,45,190,29);
    importPack.setBounds(360,getHeight()-82,120,34);exportPack.setBounds(490,getHeight()-82,120,34);
    previous.setBounds(0,getHeight()-82,105,34);next.setBounds(116,getHeight()-82,105,34);
    add.setBounds(getWidth()-215,getHeight()-82,215,34);cancel.setBounds(getWidth()-88,getHeight()-42,88,34);
}
