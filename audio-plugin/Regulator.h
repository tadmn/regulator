
#pragma once

#include <tb_SampleRateConverter.h>
#include <tb_FifoBuffer.h>
#include <tb_Windowing.h>
#include <FastFourier.h>
#include <choc_SampleBuffers.h>

class Regulator {
public:
    static constexpr int kMaxInBufferFrames = 1024;
    static constexpr double kInSampleRate = 44100.0;
    static constexpr double kOutSampleRate = 22050.0;
    static constexpr int kFftSize = 2048;
    static constexpr int kHopSize = 512;
    static constexpr int kNumChannels = 1;

    static_assert(kOutSampleRate < kInSampleRate);

    struct Features {
        float spectralCentroid = 0.f;
    };

    Regulator() :
        mResampler(kNumChannels, tb::SampleRateConverter::Quality::BestQuality),
        mResampledBuffer(kNumChannels, kFftSize),
        mWindow(tb::window<float>(tb::WindowType::Hamming, kFftSize)),
        mFifoBuffer(kNumChannels, kFftSize),
        mFftInBuffer(kNumChannels, kFftSize),
        mFft(kFftSize) {
        settle();
    }

    ~Regulator() {}

    void settle() {
        mResampler.reset();
        mFifoBuffer.clear();
    }

    void process(choc::buffer::ChannelArrayView<float> audioIn, std::function<void(const Features&)> applyFeatures) {
        tb_assert(audioIn.getNumChannels() == kNumChannels);
        tb_assert(audioIn.getNumFrames() <= kMaxInBufferFrames);
        tb_assert(applyFeatures);

        while (audioIn.getNumFrames() > 0) {
            auto [remainingIn, resampled] =
                mResampler.process(audioIn, mResampledBuffer, kInSampleRate, kOutSampleRate);
            
            audioIn = remainingIn;

            while (resampled.getNumFrames() > 0) {
                resampled = mFifoBuffer.push(resampled);
                if (mFifoBuffer.isFull()) {
                    // Make a copy of the accumulated samples, so that when we window them we don't affect
                    // the overlapping samples in the next chunk
                    copy(mFftInBuffer, mFifoBuffer.getBuffer());

                    // Apply windowing
                    applyGainPerFrame(mFftInBuffer, [this](auto i) { return mWindow[i]; });

                    mFft.forward(mFftInBuffer.getIterator(0).sample, mFftOut.data());

                    // Calculate frequency resolution (Hz per bin)
                    const double freqResolution = kOutSampleRate / (2.0 * (mFftOut.size() - 1));

                    double weightedSum = 0.0;
                    double magnitudeSum = 0.0;

                    // Calculate weighted sum and total magnitude
                    for (size_t i = 0; i < mFftOut.size(); ++i) {
                        double frequency = i * freqResolution;
                        double magnitude = std::abs(mFftOut[i]);

                        weightedSum += frequency * magnitude;
                        magnitudeSum += magnitude;
                    }

                    double spectralCentroid = 0.0;
                    if (magnitudeSum != 0.0) // Avoid division by zero
                        spectralCentroid = weightedSum / magnitudeSum;

                    Features features {
                        .spectralCentroid = static_cast<float>(spectralCentroid)
                    };

                    applyFeatures(features);

                    // Shift samples in FFT buffer
                    mFifoBuffer.pop(kHopSize);
                }
            }
        }
    }

private:
    tb::SampleRateConverter mResampler;
    choc::buffer::ChannelArrayBuffer<float> mResampledBuffer;
    std::vector<float> mWindow;
    tb::FifoBuffer<float> mFifoBuffer;
    choc::buffer::ChannelArrayBuffer<float> mFftInBuffer;
    FastFourier mFft;
    std::array<std::complex<float>, kFftSize> mFftOut;
};