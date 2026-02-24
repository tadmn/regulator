#pragma once

#include <tb_SampleRateConverter.h>
#include <tb_FifoBuffer.h>
#include <tb_Windowing.h>
#include <tb_AudioFeatures.h>
#include <FastFourier.h>
#include <choc_SampleBuffers.h>

class FeatureExtractor {
public:
    static constexpr int kMaxInBufferFrames = 1024;
    static constexpr int kFftSize = 2048;
    static constexpr int kHopSize = 512;
    static constexpr int kNumChannels = 1;
    static constexpr auto kSrcQuality = tb::SampleRateConverter::Quality::MediumQuality;

    FeatureExtractor() :
        mResampler(kNumChannels, kSrcQuality),
        mResampledBuffer(kNumChannels, kFftSize),
        mWindow(tb::window<float>(tb::WindowType::Hamming, kFftSize)),
        mFifoBuffer(kNumChannels, kFftSize),
        mFftInBuffer(kNumChannels, kFftSize),
        mFft(kFftSize) {
        prepare();
    }

    ~FeatureExtractor() {}

    void prepare() {
        mResampler.reset();
        mFifoBuffer.clear();
    }

    void process(choc::buffer::ChannelArrayView<float> audioIn,
                 double inSampleRate,
                 double featuresSampleRate,
                 const std::function<void(float f)>& applyFeatureSet) {
        tb_assert(audioIn.getNumChannels() == kNumChannels);
        tb_assert(audioIn.getNumFrames() <= kMaxInBufferFrames);
        tb_assert(applyFeatureSet);
        tb_assert(featuresSampleRate <= inSampleRate);

        while (audioIn.getNumFrames() > 0) {
            auto [remainingIn, resampled] =
                mResampler.process(audioIn, mResampledBuffer, inSampleRate, featuresSampleRate);

            audioIn = remainingIn;

            while (resampled.getNumFrames() > 0) {
                resampled = mFifoBuffer.push(resampled);
                if (mFifoBuffer.isFull()) {
                    // Make a copy of the accumulated samples, so that when we window them we don't affect
                    // the overlapping samples in the next chunk
                    copy(mFftInBuffer, mFifoBuffer.getBuffer());

                    // Apply windowing
                    applyGainPerFrame(mFftInBuffer, [this](auto i) { return mWindow[i]; });

                    std::array<std::complex<float>, kFftSize / 2 + 1> fftOut = {};
                    mFft.forward(mFftInBuffer.getIterator(0).sample, fftOut.data());

                    std::array<float, kFftSize / 2 + 1> fftPowerMag = {};
                    for (size_t i = 0; i < fftOut.size(); ++i) {
                        const auto mag = std::abs(fftOut[i]);
                        fftPowerMag[i] = mag * mag;
                    }

                    const float centroid = tb::spectralCentroid(fftPowerMag, featuresSampleRate);

                    applyFeatureSet(centroid);

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
};