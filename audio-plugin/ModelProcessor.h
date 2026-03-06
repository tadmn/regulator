#pragma once

#include <tb_LiteRt.h>

#include <choc_DisableAllWarnings.h>
#include <boost/circular_buffer.hpp>
#include <choc_ReenableAllWarnings.h>

#include "FeatureExtractor.h"

class ModelProcessor {
public:
    static constexpr double modelSampleRate      = 22050.0;
    static constexpr int    modelInputFeatureSets = 126;

    // Each fifo slot is a full FeatureArray; capacity = one clip's worth of sets
    using FeatureFifo = boost::circular_buffer<FeatureExtractor::FeatureArray>;

    ModelProcessor() :
        featuresFifo(modelInputFeatureSets),
        featureExtractor(modelSampleRate) {}

    ~ModelProcessor() {}

    tb::Result loadModel(const std::string& modelPath) {
        prepare();
        return { model.loadModel(modelPath) };
    }

    void prepare() {
        featureExtractor.settle();
        featuresFifo.clear();
        prediction = 1.f;
    }

    void process(choc::buffer::ChannelArrayView<float> audioIn, double inSampleRate) {
        if (! model.modelLoaded())
            return;

        featureExtractor.process(audioIn, inSampleRate,
            [this](const FeatureExtractor::FeatureArray& featureSet) {
                featuresFifo.push_front(featureSet);

                if (featuresFifo.full()) {
                    featuresFifo.linearize();

                    // Flatten the circular buffer into a contiguous float span:
                    //   (modelInputFeatureSets × kNumFeatures) floats
                    static constexpr int kN = modelInputFeatureSets * FeatureExtractor::kNumFeatures;
                    std::array<float, kN> flat = {};
                    auto* dst = flat.data();
                    for (const auto& fs : featuresFifo) {
                        std::copy(fs.begin(), fs.end(), dst);
                        dst += FeatureExtractor::kNumFeatures;
                    }

                    std::array<float, 2> out = {};
                    if (auto r = model.process(std::span<const float>(flat), out); ! r.empty())
                        throw std::runtime_error(r);

                    prediction = out[0];
                }
            });
    }

    std::atomic<float> prediction = 1.f;

private:
    FeatureFifo      featuresFifo;
    FeatureExtractor featureExtractor;
    tb::LiteRt       model;

public:
    ModelProcessor(ModelProcessor const &)            = delete;
    ModelProcessor& operator=(ModelProcessor const &) = delete;
};