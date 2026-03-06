#pragma once

#include <tb_SampleRateConverter.h>
#include <tb_FifoBuffer.h>
#include <tb_Windowing.h>
#include <tb_AudioFeatures.h>
#include <FastFourier.h>
#include <choc_SampleBuffers.h>

#include <array>
#include <complex>
#include <cmath>

class FeatureExtractor {
public:
    static constexpr int  kFftSize     = 2048;
    static constexpr int  kHopSize     = 512;
    static constexpr int  kNumChannels = 1;
    static constexpr int  kNumMelBins  = 64;

    // Total features per STFT frame:
    //   64 log-mel bins  +  1 RMS energy  +  1 spectral flux  =  66
    static constexpr int  kNumFeatures = kNumMelBins + 2;

    static constexpr auto kSrcQuality  = tb::SampleRateConverter::Quality::MediumQuality;

    using FeatureArray = std::array<float, kNumFeatures>;

    explicit FeatureExtractor(double featuresSampleRate) :
        mFeaturesSampleRate(featuresSampleRate),
        mResampler(kNumChannels, kSrcQuality),
        mResampledBuffer(kNumChannels, kFftSize),
        mWindow(tb::window<float>(tb::WindowType::Hann, kFftSize)),
        mFifoBuffer(kNumChannels, kFftSize),
        mFftInBuffer(kNumChannels, kFftSize),
        mFft(kFftSize),
        mFilterbank(tb::melFilterbank(kNumMelBins, kFftSize / 2 + 1, featuresSampleRate))
    {
        mPrevMagnitude.fill(0.f);
        settle();
    }

    ~FeatureExtractor() {}

    void settle() {
        mResampler.reset();
        mFifoBuffer.clear();
        mPrevMagnitude.fill(0.f);   // reset flux state on clip boundary
    }

    void process(choc::buffer::ChannelArrayView<float> audioIn,
                 double inSampleRate,
                 const std::function<void(const FeatureArray&)>& applyFeatureSet)
    {
        tb_assert(audioIn.getNumChannels() == kNumChannels);
        tb_assert(applyFeatureSet);
        tb_assert(mFeaturesSampleRate <= inSampleRate);

        while (audioIn.getNumFrames() > 0) {
            auto [remainingIn, resampled] =
                mResampler.process(audioIn, mResampledBuffer, inSampleRate, mFeaturesSampleRate);

            audioIn = remainingIn;

            while (resampled.getNumFrames() > 0) {
                resampled = mFifoBuffer.push(resampled);
                if (mFifoBuffer.isFull()) {
                    applyFeatureSet(extractFeatureSet());
                    mFifoBuffer.pop(kHopSize);
                }
            }
        }
    }

private:
    FeatureArray extractFeatureSet()
    {
        // ── Copy + window ────────────────────────────────────────────────
        copy(mFftInBuffer, mFifoBuffer.getBuffer());
        applyGainPerFrame(mFftInBuffer, [this](auto i) { return mWindow[i]; });

        // ── Forward FFT ──────────────────────────────────────────────────
        std::array<std::complex<float>, kFftSize / 2 + 1> fftOut = {};
        mFft.forward(mFftInBuffer.getIterator(0).sample, fftOut.data());

        // ── Power spectrum & magnitude spectrum ──────────────────────────
        std::array<float, kFftSize / 2 + 1> fftPowerMag = {};
        std::array<float, kFftSize / 2 + 1> fftMag      = {};
        for (std::size_t i = 0; i < fftOut.size(); ++i) {
            const float mag = std::abs(fftOut[i]);
            fftMag[i]       = mag;
            fftPowerMag[i]  = mag * mag;
        }

        // ── Log-mel spectrogram (64 bins) ────────────────────────────────
        FeatureArray features = {};
        tb::applyMelFilterbank(fftPowerMag, mFilterbank,
                               std::span<float>(features.data(), kNumMelBins));

        // ── RMS energy (time-domain, pre-window) ─────────────────────────
        {
            auto* ptr = mFifoBuffer.getBuffer().getIterator(0).sample;
            features[kNumMelBins] = tb::rmsEnergy(
                std::span<const float>(ptr, static_cast<std::size_t>(kFftSize)));
        }

        // ── Spectral flux ────────────────────────────────────────────────
        features[kNumMelBins + 1] = tb::spectralFlux(mPrevMagnitude, fftMag);
        mPrevMagnitude = fftMag;

        return features;
    }

    double                                      mFeaturesSampleRate;
    tb::SampleRateConverter                     mResampler;
    choc::buffer::ChannelArrayBuffer<float>     mResampledBuffer;
    std::vector<float>                          mWindow;
    tb::FifoBuffer<float>                       mFifoBuffer;
    choc::buffer::ChannelArrayBuffer<float>     mFftInBuffer;
    FastFourier                                 mFft;
    std::vector<std::vector<float>>             mFilterbank;
    std::array<float, kFftSize / 2 + 1>        mPrevMagnitude;
};