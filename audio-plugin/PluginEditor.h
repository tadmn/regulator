#pragma once

#include "PluginProcessor.h"

class AudioPluginAudioProcessorEditor final : public juce::AudioProcessorEditor,
                                              public juce::Timer,
                                              public juce::FileDragAndDropTarget
{
public:
    explicit AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor&);
    ~AudioPluginAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

    void timerCallback() override;

    bool isInterestedInFileDrag(const juce::StringArray& /*files*/) override { return true; }
    void filesDropped(const juce::StringArray& files, int x, int y) override;
    void fileDragEnter (const juce::StringArray&, int, int) override {}

private:
    AudioPluginAudioProcessor& processorRef;

    juce::Label predictionLabel, modelFileLabel;
    juce::Slider delaySlider;

    juce::AudioProcessorValueTreeState::SliderAttachment delaySliderAttachment;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};
