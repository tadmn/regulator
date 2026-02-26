#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    addAndMakeVisible(centroidLabel);
    centroidLabel.setJustificationType(juce::Justification::centred);
    centroidLabel.setFont({juce::FontOptions(25.f)});

    addAndMakeVisible(predictionLabel);
    predictionLabel.setJustificationType(juce::Justification::centred);
    predictionLabel.setFont({juce::FontOptions(25.f)});

    addAndMakeVisible(modelFileLabel);
    modelFileLabel.setJustificationType(juce::Justification::centred);

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
    modelFileLabel.setBounds(b.removeFromTop(22 ));
    predictionLabel.setBounds(b.removeFromTop(getHeight() / 2));
    centroidLabel.setBounds(b);
}

void AudioPluginAudioProcessorEditor::filesDropped(const juce::StringArray& files, int /*x*/, int /*y*/) {
    const std::string path = files[0].toStdString();;
    if (auto result = processorRef.loadModel(path); ! result) {
        juce::AlertWindow::showMessageBoxAsync({}, "", "Failed to load " + path + "\n\n" + result.msg);
    }
}

void AudioPluginAudioProcessorEditor::timerCallback()
{
    predictionLabel.setText(juce::String(processorRef.modelProcessor.prediction, 2), juce::dontSendNotification);
    centroidLabel.setText(juce::String(juce::roundToInt(processorRef.modelProcessor.avgCentroid.load(std::memory_order_relaxed))),
                   juce::dontSendNotification);

    {
        auto p = processorRef.getModelFile().getFullPathName();
        if (p.isEmpty()) {
            p = "No model loaded";
        }

        modelFileLabel.setText(p, juce::dontSendNotification);
    }
}
