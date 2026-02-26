#pragma once

#include <tb_LiteRt.h>

#include <choc_DisableAllWarnings.h>
#include <boost/circular_buffer.hpp>
#include <choc_ReenableAllWarnings.h>

#include "FeatureExtractor.h"

class ModelProcessor {
public:
    static constexpr double modelSampleRate = 22050.0;

    ModelProcessor() : featuresFifo(129) {}
    ~ModelProcessor() {}

    void loadModel(const std::string& modelPath) {
        model.loadModel(modelPath);
        prepare();
    }

    void prepare() {
        featureExtractor.prepare();
        featuresFifo.clear();
        avgCentroid = 0.f;
        prediction = 1.f;
    }

    void process(choc::buffer::ChannelArrayView<float> audioIn, double inSampleRate) {
        featureExtractor.process(audioIn, inSampleRate, modelSampleRate,
             [this](float f) {
                 featuresFifo.push_front(f);
                 if (featuresFifo.full()) {
                     featuresFifo.linearize();
                     std::span<float> in(
                         featuresFifo.array_one().first,
                         featuresFifo.size());
                     std::array<float, 2> out = {};
                     if (model.process(in, out)) {
                         avgCentroid = std::reduce(in.begin(), in.end()) / in.size();
                         prediction = out[0];
                     }
                 }
             });
    }

    std::atomic<float> avgCentroid = 0.f;
    std::atomic<float> prediction = 1.f;

private:
    FeatureExtractor featureExtractor;
    boost::circular_buffer<float> featuresFifo;
    tb::LiteRt model;

public:
    ModelProcessor(ModelProcessor const &) = delete;
    ModelProcessor & operator=(ModelProcessor const &) = delete;
};