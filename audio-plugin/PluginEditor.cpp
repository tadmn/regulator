#include "PluginProcessor.h"
#include "PluginEditor.h"

AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p),
      delaySliderAttachment(p.params.apvts, Param::delay.id, delaySlider) {
    addAndMakeVisible(predictionLabel);
    predictionLabel.setJustificationType(juce::Justification::centred);
    predictionLabel.setFont({juce::FontOptions(27.f)});

    addAndMakeVisible(modelFileLabel);
    modelFileLabel.setJustificationType(juce::Justification::centred);

    addAndMakeVisible(delaySlider);

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
    predictionLabel.setBounds(b.removeFromTop(50));
    delaySlider.setBounds(b);
}

void AudioPluginAudioProcessorEditor::filesDropped(const juce::StringArray& files, int /*x*/, int /*y*/) {
    const std::string path = files[0].toStdString();
    processorRef.params.setModelPath(path);
}

void AudioPluginAudioProcessorEditor::timerCallback() {
    predictionLabel.setText(juce::String(processorRef.modelProcessor.prediction, 2), juce::dontSendNotification);

    {
        auto p = processorRef.params.getModelPath();
        if (p.empty()) {
            p = "<Drag n' drop a .tflite file here to load the model>";
        }

        modelFileLabel.setText(p, juce::dontSendNotification);
    }
}
