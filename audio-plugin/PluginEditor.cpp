#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    addAndMakeVisible(mLabel);
    mLabel.setJustificationType(juce::Justification::centred);
    mLabel.setFont({juce::FontOptions(25.f)});

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
    mLabel.setBounds(getLocalBounds());
}

void AudioPluginAudioProcessorEditor::timerCallback()
{
    mLabel.setText(juce::String(juce::roundToInt(processorRef.mSpectralCentroid.load(std::memory_order_relaxed))),
                   juce::dontSendNotification);
}
