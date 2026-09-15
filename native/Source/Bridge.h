#pragma once
#include <juce_core/juce_core.h>

juce::File findProjectRoot();
juce::var readJson(const juce::File& file);
juce::var object(std::initializer_list<std::pair<juce::Identifier, juce::var>> values);

class AnalysisBridge final : private juce::Thread
{
public:
    struct Snapshot {
        bool busy = false;
        int completion = 0;
        juce::String action, message = "Ready", stage;
        double progress = 0.0;
        juce::var response;
    };
    explicit AnalysisBridge(juce::File root);
    ~AnalysisBridge() override;
    bool submit(juce::var request);
    void cancel();
    Snapshot snapshot() const;
private:
    void run() override;
    juce::File root;
    mutable juce::CriticalSection lock;
    Snapshot state;
    juce::var pending;
    std::atomic<bool> cancelled{false};
};
