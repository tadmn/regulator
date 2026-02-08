
#pragma once

#include <tb_FifoBuffer.h>
#include <tb_Windowing.h>
#include <FastFourier.h>
#include <choc_SampleBuffers.h>

class Regulator {
public:
    static constexpr double kSampleRate = 22050.0;
    static constexpr int kFftSize = 2048;
    static constexpr int kHopSize = 512;
    static constexpr int kNumChannels = 1;

    Regulator() :
        mWindow(tb::window<float>(tb::WindowType::Hamming, kFftSize)),
        mFifoBuffer(kNumChannels, kFftSize),
        mFftInBuffer({ .numChannels = kNumChannels, .numFrames = kFftSize }),
        mFft(kFftSize) {
        settle();
    }

    ~Regulator() {}

    void settle() {
        mSpectralCentroid = 0.0;
        mFifoBuffer.clear();
    }

    void process(choc::buffer::ChannelArrayView<float> audio) {
        tb_assert(audio.getNumChannels() == kNumChannels);
        while (audio.getNumFrames() > 0) {
            audio = mFifoBuffer.push(audio);
            if (mFifoBuffer.isFull()) {
                // Make a copy of the accumulated samples, so that when we window them we don't affect
                // the overlapping samples in the next chunk
                copy(mFftInBuffer, mFifoBuffer.getBuffer());

                // Apply windowing
                applyGainPerFrame(mFftInBuffer, [this](auto i) { return mWindow[i]; });

                mFft.forward(mFftInBuffer.getIterator(0).sample, mFftOut.data());

                // Calculate frequency resolution (Hz per bin)
                const double freqResolution = kSampleRate / (2.0 * (mFftOut.size() - 1));

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

                mSpectralCentroid = spectralCentroid;

                // Shift samples in FFT buffer
                mFifoBuffer.pop(kHopSize);
            }
        }
    }

    double spectralCentroid() const { return mSpectralCentroid; }

private:
    std::vector<float> mWindow;
    tb::FifoBuffer<float> mFifoBuffer;
    choc::buffer::MonoBuffer<float> mFftInBuffer;
    FastFourier mFft;
    std::array<std::complex<float>, kFftSize> mFftOut;
    double mSpectralCentroid = 0.0;
};