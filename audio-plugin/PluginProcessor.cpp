#include "PluginProcessor.h"
#include "PluginEditor.h"

struct ScopedSuspendProcessing {
    explicit ScopedSuspendProcessing(AudioPluginAudioProcessor& p) : processor(p) {
        processor.suspendProcessing(true);
    }

    ~ScopedSuspendProcessing() {
        processor.suspendProcessing(false);
    }

    AudioPluginAudioProcessor& processor;
};

//==============================================================================
AudioPluginAudioProcessor::AudioPluginAudioProcessor()
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::mono(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::mono(), true)
                     #endif
                       ) {
}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor() {}

//==============================================================================
tb::Result AudioPluginAudioProcessor::loadModel(const std::string& path) {
    ScopedSuspendProcessing ssp(*this);
    modelFile = "";

    if (auto result = modelProcessor.loadModel(path); ! result) {
        return result;
    }

    modelFile = path;
    return {};
}

void AudioPluginAudioProcessor::prepareToPlay (double sampleRate, int /*samplesPerBlock*/) {
    modelProcessor.prepare();
    gain.reset(sampleRate, 0.01);
    gain.setCurrentAndTargetValue(1.f);
}

void AudioPluginAudioProcessor::releaseResources() {}

bool AudioPluginAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const {
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}

void AudioPluginAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                             juce::MidiBuffer& /*midiMessages*/) {
    juce::ScopedNoDenormals noDenormals;

    auto audio = choc::buffer::createChannelArrayView(buffer.getArrayOfWritePointers(),
                                                      buffer.getNumChannels(),
                                                      buffer.getNumSamples());

    modelProcessor.process(audio, getSampleRate());

    {
        auto g = std::clamp(modelProcessor.prediction.load(std::memory_order_relaxed), 0.f, 1.f);
        if (g > 0.7f) {
            g = 1.f;
        } else if (g < 0.3f) {
            g = 0.f;
        }

        gain.setTargetValue(g);
        gain.applyGain(buffer, buffer.getNumSamples());
    }
}

//==============================================================================
void AudioPluginAudioProcessor::getStateInformation (juce::MemoryBlock& destData) {
    juce::MemoryOutputStream outputStream(destData, false);
    outputStream << modelFile.getFullPathName();
}

void AudioPluginAudioProcessor::setStateInformation (const void* data, int sizeInBytes) {
    juce::MemoryInputStream inputStream(data, sizeInBytes, false);
    loadModel(inputStream.readString().toStdString());
}

//==============================================================================
const juce::String AudioPluginAudioProcessor::getName() const {
    return JucePlugin_Name;
}

bool AudioPluginAudioProcessor::acceptsMidi() const {
#if JucePlugin_WantsMidiInput
    return true;
#else
    return false;
#endif
}

bool AudioPluginAudioProcessor::producesMidi() const {
#if JucePlugin_ProducesMidiOutput
    return true;
#else
    return false;
#endif
}

bool AudioPluginAudioProcessor::isMidiEffect() const {
#if JucePlugin_IsMidiEffect
    return true;
#else
    return false;
#endif
}

double AudioPluginAudioProcessor::getTailLengthSeconds() const {
    return 0.0;
}

int AudioPluginAudioProcessor::getNumPrograms() {
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
    // so this should be at least 1, even if you're not really implementing programs.
}

int AudioPluginAudioProcessor::getCurrentProgram() {
    return 0;
}

void AudioPluginAudioProcessor::setCurrentProgram (int /*index*/) {
    juce::ignoreUnused (index);
}

const juce::String AudioPluginAudioProcessor::getProgramName (int /*index*/) {
    return {};
}

void AudioPluginAudioProcessor::changeProgramName (int /*index*/, const juce::String& /*newName*/) {
}

//==============================================================================
bool AudioPluginAudioProcessor::hasEditor() const { return true; }

juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor() {
    return new AudioPluginAudioProcessorEditor(*this);
}

//==============================================================================
// This creates new instances of the plugin
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new AudioPluginAudioProcessor();
}
