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

AudioPluginAudioProcessor::AudioPluginAudioProcessor()
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::mono(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::mono(), true)
                     #endif
                       )
    , params(*this) {
    params.onChange(Param::modelPath) = [this]{ loadModel(); };
}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor() {}

void AudioPluginAudioProcessor::loadModel() {
    const auto path = params.getModelPath();
    if (path.empty())
        return;

    const ScopedSuspendProcessing ssp(*this);

    const auto result = modelProcessor.loadModel(params.getModelPath());
    if (! result) {
        showWarningMessage("Failed to load model: " + result.msg);
        params.setModelPath("");
    }
}

void AudioPluginAudioProcessor::showWarningMessage(const std::string msg) {
    juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, JucePlugin_Name, msg);
}

void AudioPluginAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock) {
    jassert(getMainBusNumInputChannels() == getMainBusNumOutputChannels());

    modelProcessor.prepare();
    gain.reset(sampleRate, 0.6);
    gain.setCurrentAndTargetValue(1.f);

    juce::dsp::ProcessSpec spec {
        .sampleRate = sampleRate,
        .maximumBlockSize = static_cast<uint32_t>(samplesPerBlock),
        .numChannels = static_cast<uint32_t>(getMainBusNumInputChannels())
    };

    delay.setMaximumDelayInSamples(sampleRate * params.getRange(Param::delay).end / 1'000.0);
    delay.prepare(spec);
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
        delay.setDelay(int(params.getDelay() * getSampleRate() / 1'000.0));

        juce::dsp::AudioBlock<float> block { buffer };
        juce::dsp::ProcessContextReplacing<float> ctx { block };
        delay.process(ctx);
    }

    {
        auto g = std::clamp(modelProcessor.prediction.load(std::memory_order_relaxed), 0.f, 1.f);
        gain.setTargetValue(g > 0.5f ? 1.f : 0.f);
        gain.applyGain(buffer, buffer.getNumSamples());
    }
}

void AudioPluginAudioProcessor::getStateInformation (juce::MemoryBlock& destData) {
    params.getStateInformation(destData);
}

void AudioPluginAudioProcessor::setStateInformation (const void* data, int sizeInBytes) {
    params.setStateInformation(data, sizeInBytes);
}

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

bool AudioPluginAudioProcessor::hasEditor() const { return true; }

juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor() {
    return new AudioPluginAudioProcessorEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new AudioPluginAudioProcessor();
}
