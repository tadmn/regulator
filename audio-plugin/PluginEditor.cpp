#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    addAndMakeVisible(mCentroidLabel);
    mCentroidLabel.setJustificationType(juce::Justification::centred);
    mCentroidLabel.setFont({juce::FontOptions(25.f)});

    addAndMakeVisible(mProcessingTimeLabel);
    mProcessingTimeLabel.setJustificationType(juce::Justification::centred);
    mProcessingTimeLabel.setFont({juce::FontOptions(25.f)});

    setSize (400, 300);
    startTimer(30);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
}

void AudioPluginAudioProcessorEditor::resized()
{
    auto b = getLocalBounds();
    mCentroidLabel.setBounds(b.removeFromBottom(getHeight() / 2));
    mProcessingTimeLabel.setBounds(b);
}

void AudioPluginAudioProcessorEditor::timerCallback()
{
    mCentroidLabel.setText(juce::String(juce::roundToInt(processorRef.mSpectralCentroid.load(std::memory_order_relaxed))),
                   juce::dontSendNotification);

    mProcessingTimeLabel.setText(juce::String(processorRef.mProcessingTime_ms
                                                                   .load(std::memory_order_relaxed), 2),
                                 juce::dontSendNotification);
}
