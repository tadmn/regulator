#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

//==============================================================================
// Parameter ID tags — each is a unique type, enabling onChange() overloading.
// Use the inline instances (Param::delay, etc.) everywhere.
namespace Param {
struct DelayTag {
    static constexpr auto id = "delay";
};

struct ThresholdTag {
    static constexpr auto id = "threshold";
};

struct ModelPathTag {
    static constexpr auto id = "model_path";
};

inline constexpr DelayTag delay{};
inline constexpr ThresholdTag threshold{};
inline constexpr ModelPathTag modelPath{};
}

//==============================================================================
// Parameter layout factory
inline juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ Param::DelayTag::id, 1 },
        "Delay",
        juce::NormalisableRange<float>(0.f, 6'000.f, 1.f),
        2'000.f,
        juce::AudioParameterFloatAttributes{}
        .withLabel("ms")
        .withStringFromValueFunction([](float v, int) {
            return juce::String(static_cast<int>(v)) + " ms";
        })
        .withValueFromStringFunction([](const juce::String& s) {
            return s.retainCharacters("0123456789.").getFloatValue();
        })
        ));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ Param::ThresholdTag::id, 1 },
        "Threshold",
        juce::NormalisableRange<float>(0.f, 1.f, 0.001f),
        0.5f,
        juce::AudioParameterFloatAttributes{}
        .withStringFromValueFunction([](float v, int) {
            return juce::String(v, 3);
        })
        .withValueFromStringFunction([](const juce::String& s) {
            return s.getFloatValue();
        })
        ));

    return { params.begin(), params.end() };
}

//==============================================================================
/**
 *  ParameterListener
 *
 *  Internal helper that bridges juce::AudioProcessorParameter::Listener into
 *  an assignable std::function callback. One instance per float parameter.
 */
class ParameterListener : private juce::AudioProcessorParameter::Listener {
public:
    using Callback = std::function<void()>;

    explicit ParameterListener(juce::RangedAudioParameter& param)
        : floatParam(&param) {
        param.addListener(this);
    }

    ~ParameterListener() override {
        floatParam->removeListener(this);
    }

    Callback onFloatChanged;

private:
    void parameterValueChanged(int, float) override {
        if (!onFloatChanged)
            return;

        juce::MessageManager::callAsync([this] {
            if (onFloatChanged)
                onFloatChanged();
        });
    }

    void parameterGestureChanged(int, bool) override {}

    juce::RangedAudioParameter* floatParam = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ParameterListener)
};

//==============================================================================
/**
 *  PluginParameters
 *
 *  Owns the APVTS and exposes every parameter via:
 *    - Typed getters          (audio-thread safe for float params)
 *    - onChange() references  (assign lambdas, always called on the message thread)
 *
 *  Usage
 *  -----
 *      parameters.onChange(delay)     = [](float ms)              { ... };
 *      parameters.onChange(threshold) = [](float t)               { ... };
 *      parameters.onChange(modelPath) = [](const juce::String& p) { ... };
 *
 *  Assign nullptr to remove a callback:
 *      parameters.onChange(delay) = nullptr;
 */
class PluginParameters : private juce::ValueTree::Listener {
public:
    //==========================================================================
    explicit PluginParameters(juce::AudioProcessor& processor)
        : apvts(processor, nullptr, "PARAMETERS", createParameterLayout()) {
        delayParam = dynamic_cast<juce::AudioParameterFloat*>(
            apvts.getParameter(Param::DelayTag::id));
        thresholdParam = dynamic_cast<juce::AudioParameterFloat*>(
            apvts.getParameter(Param::ThresholdTag::id));

        jassert(delayParam != nullptr);
        jassert(thresholdParam != nullptr);

        apvts.state.setProperty(Param::ModelPathTag::id, juce::String{}, nullptr);
        apvts.state.addListener(this);

        delayListener = std::make_unique<ParameterListener>(*delayParam);
        thresholdListener = std::make_unique<ParameterListener>(*thresholdParam);

        delayListener->onFloatChanged = [this] {
            if (_onDelay) _onDelay();
        };
        thresholdListener->onFloatChanged = [this] {
            if (_onThreshold) _onThreshold();
        };
    }

    ~PluginParameters() override {
        apvts.state.removeListener(this);
    }

    //==========================================================================
    // onChange() — returns a writable reference to the callback for that param.
    ParameterListener::Callback& onChange(Param::DelayTag)     { return _onDelay; }
    ParameterListener::Callback& onChange(Param::ThresholdTag) { return _onThreshold; }
    ParameterListener::Callback& onChange(Param::ModelPathTag) { return _onModelPath; }

    //==========================================================================
    // Getters
    const juce::NormalisableRange<float>& getRange(Param::DelayTag) const noexcept {
        return delayParam->getNormalisableRange();
    }
    const juce::NormalisableRange<float>& getRange(Param::ThresholdTag) const noexcept {
        return thresholdParam->getNormalisableRange();
    }

    float getDelay()     const noexcept { return delayParam->get(); }
    float getThreshold() const noexcept { return thresholdParam->get(); }

    std::string getModelPath() const {
        return apvts.state.getProperty(Param::ModelPathTag::id).toString().toStdString();
    }

    void setModelPath(const std::string& path) {
        apvts.state.setProperty(Param::ModelPathTag::id, juce::String(path), nullptr);
    }

    //==========================================================================
    // State persistence
    void getStateInformation(juce::MemoryBlock& destData) {
        auto xml = apvts.copyState().createXml();
        juce::AudioProcessor::copyXmlToBinary(*xml, destData);
    }

    void setStateInformation(const void* data, int sizeInBytes) {
        auto xml = juce::AudioProcessor::getXmlFromBinary(data, sizeInBytes);
        if (xml != nullptr)
            apvts.replaceState(juce::ValueTree::fromXml(*xml));
        // valueTreeRedirected() fires automatically, re-attaches this listener
        // and invokes _onModelPath via callAsync.
    }

    //==========================================================================
    juce::AudioProcessorValueTreeState apvts;

private:
    //==========================================================================
    // juce::ValueTree::Listener

    void valueTreePropertyChanged(juce::ValueTree&,
                                  const juce::Identifier& property) override {
        if (property != juce::Identifier(Param::ModelPathTag::id) || !_onModelPath)
            return;

        juce::MessageManager::callAsync([this] {
            if (_onModelPath) _onModelPath();
        });
    }

    // Called by replaceState() — the old tree is gone, re-attach to the new one.
    void valueTreeRedirected(juce::ValueTree& redirectedTree) override {
        redirectedTree.addListener(this);

        if (_onModelPath) {
            juce::MessageManager::callAsync([this] {
                if (_onModelPath) _onModelPath();
            });
        }
    }

    //==========================================================================
    juce::AudioParameterFloat* delayParam     = nullptr;
    juce::AudioParameterFloat* thresholdParam = nullptr;

    std::unique_ptr<ParameterListener> delayListener;
    std::unique_ptr<ParameterListener> thresholdListener;

    ParameterListener::Callback _onDelay;
    ParameterListener::Callback _onThreshold;
    ParameterListener::Callback _onModelPath;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginParameters)
};