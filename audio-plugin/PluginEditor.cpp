#include "PluginProcessor.h"
#include "PluginEditor.h"

AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p) {
    addAndMakeVisible(predictionLabel);
    predictionLabel.setJustificationType(juce::Justification::centred);
    predictionLabel.setFont({juce::FontOptions(27.f)});

    addAndMakeVisible(modelFileLabel);
    modelFileLabel.setJustificationType(juce::Justification::centred);

    setSize (400, 300);
    startTimer(30);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor() {}

void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g) {
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
}

void AudioPluginAudioProcessorEditor::resized() {
    auto b = getLocalBounds();
    modelFileLabel.setBounds(b.removeFromTop(22 ));
    predictionLabel.setBounds(b);
}

void AudioPluginAudioProcessorEditor::filesDropped(const juce::StringArray& files, int /*x*/, int /*y*/) {
    const std::string path = files[0].toStdString();;
    if (auto result = processorRef.loadModel(path); ! result) {
        juce::AlertWindow::showMessageBoxAsync({}, "", "Failed to load " + path + "\n\n" + result.msg);
    }
}

void AudioPluginAudioProcessorEditor::timerCallback() {
    predictionLabel.setText(juce::String(processorRef.modelProcessor.prediction, 2), juce::dontSendNotification);

    {
        auto p = processorRef.getModelFile().getFullPathName();
        if (p.isEmpty()) {
            p = "No model loaded";
        }

        modelFileLabel.setText(p, juce::dontSendNotification);
    }
}
