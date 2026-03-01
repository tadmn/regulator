#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sndfile.h>

#include <string>
#include <vector>
#include <thread>
#include <future>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <chrono>
#include <iomanip>

#include <FeatureExtractor.h>

namespace py = pybind11;

struct FeatureSet {
    float spectralCentroid = 0.f;

    static constexpr std::size_t numFeatures = 1;

    void flatten(float* dst) const {
        *dst = spectralCentroid;
    }
};

using Clip = std::vector<FeatureSet>;

std::vector<float> loadWav(const std::string& path) {
    SF_INFO info = {};
    SNDFILE* file = sf_open(path.c_str(), SFM_READ, &info);
    if (! file) {
        std::cerr << "Failed to open " << path << ": " << sf_strerror(nullptr) << "\n";
        return {};
    }

    std::vector<float> samples(info.frames);
    sf_count_t read = sf_readf_float(file, samples.data(), info.frames);
    sf_close(file);

    if (read != info.frames)
        std::cerr << "Read " << read << " frames, expected " << info.frames << "\n";

    return samples;
}

static std::vector<Clip> extractFeaturesForFile(
        const std::string& filePath,
        std::size_t        setsPerClip,
        std::size_t        clipHopFrames)
{
    auto wav = loadWav(filePath);
    if (wav.empty()) {
        std::cerr << filePath << " is empty" << "\n";
        return {};
    }

    float* f[1];
    f[0] = wav.data();
    const auto audio = choc::buffer::createChannelArrayView(f, 1, wav.size());

    FeatureExtractor extractor;
    std::vector<Clip> clips;
    const auto clipFrames = FeatureExtractor::kFftSize + (setsPerClip - 1) * FeatureExtractor::kHopSize;

    uint32_t startFrame = 0;
    while (true) {
        uint32_t endFrame = startFrame + clipFrames;
        if (endFrame > wav.size()) {
            return clips;
        }

        Clip clip;
        const auto clipAudio = audio.getFrameRange({startFrame, endFrame});
        extractor.process(clipAudio,
                          22050.0,
                          22050.0,
                          [&clip](float featureSet) {
                              clip.push_back({ .spectralCentroid = featureSet });
                          });

        tb_assert(clip.size() == setsPerClip);
        clips.push_back(std::move(clip));

        startFrame += clipHopFrames;
        extractor.settle();
    }
}


// ─────────────────────────────────────────────────────────────
//  Progress printer (thread-safe).
//
//     Prints a single updating line:
//       [====>    ] 42/100 files (42%) | 3 clips/s | ETA 0:00:28
// ─────────────────────────────────────────────────────────────

class ProgressReporter {
public:
    explicit ProgressReporter(std::size_t total)
        : total_(total)
        , startTime_(std::chrono::steady_clock::now())
    {
        std::cout << "Extracting features from " << total_ << " file"
                  << (total_ == 1 ? "" : "s") << "...\n";
    }

    // Call after each file finishes. Thread-safe.
    void fileCompleted(std::size_t clipsProduced) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++completed_;
        totalClips_ += clipsProduced;
        print();
    }

    // Print final summary line.
    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto elapsed = elapsedSeconds();
        std::cout << "\rDone. " << completed_ << " file"
                  << (completed_ == 1 ? "" : "s") << ", "
                  << totalClips_ << " clip"
                  << (totalClips_ == 1 ? "" : "s") << " in "
                  << formatDuration(elapsed) << ".          \n";
    }

private:
    void print() {
        // Bar
        constexpr int barWidth = 30;
        const double pct = total_ > 0 ? double(completed_) / double(total_) : 1.0;
        const int filled = static_cast<int>(pct * barWidth);
        std::string bar(filled, '=');
        if (filled < barWidth) { bar += '>'; bar += std::string(barWidth - filled - 1, ' '); }

        // Rate & ETA
        double elapsed = elapsedSeconds();
        double rate    = elapsed > 0.0 ? double(completed_) / elapsed : 0.0;
        double eta     = (rate > 0.0 && completed_ < total_)
                             ? double(total_ - completed_) / rate
                             : 0.0;

        std::cout << "\r[" << bar << "] "
                  << completed_ << "/" << total_
                  << " (" << std::fixed << std::setprecision(0) << (pct * 100.0) << "%)"
                  << " | " << std::setprecision(1) << rate << " files/s"
                  << " | ETA " << formatDuration(eta)
                  << std::flush;
    }

    double elapsedSeconds() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - startTime_).count();
    }

    static std::string formatDuration(double seconds) {
        auto s = static_cast<int>(seconds);
        return std::to_string(s / 60) + ":" + (s % 60 < 10 ? "0" : "") + std::to_string(s % 60);
    }

    std::size_t  total_;
    std::size_t  completed_  = 0;
    std::size_t  totalClips_ = 0;
    std::mutex   mutex_;
    std::chrono::steady_clock::time_point startTime_;
};


// ─────────────────────────────────────────────────────────────
// 5.  Main entry point exposed to Python.
//
//   extractFeatures(paths, setsPerClip, clipHopFrames, nThreads=0)
//
//   Returns np.ndarray of shape:
//     (totalClips, setsPerClip, FeatureSet::numFeatures)
//   dtype float32, C-contiguous (ready for Conv1D / Conv2D).
//
//   totalClips = sum of clips produced across all files.
// ─────────────────────────────────────────────────────────────

py::array_t<float> extractFeatures(
        const std::vector<std::string>& paths,
        std::size_t                     setsPerClip,
        std::size_t                     clipHopFrames,
        std::size_t                     nThreads = 0)
{
    if (paths.empty())
        return py::array_t<float>(
            std::vector<std::size_t>{0, setsPerClip, FeatureSet::numFeatures});

    if (setsPerClip == 0)
        throw std::invalid_argument("framesPerClip must be > 0");
    if (clipHopFrames == 0)
        throw std::invalid_argument("clipHopFrames must be > 0");

    if (nThreads == 0)
        nThreads = std::max(1u, std::thread::hardware_concurrency());

    const std::size_t nFiles = paths.size();
    nThreads = std::min(nThreads, nFiles);

    // perFileResults[i] = all clips extracted from file i.
    // Pre-sized so threads write to non-overlapping indices — no mutex needed.
    std::vector<std::vector<Clip>> perFileResults(nFiles);

    {
        py::gil_scoped_release release;

        ProgressReporter progress(nFiles);

        const std::size_t chunkSize = (nFiles + nThreads - 1) / nThreads;

        std::vector<std::future<void>> futures;
        futures.reserve(nThreads);

        for (std::size_t t = 0; t < nThreads; ++t) {
            const std::size_t start = t * chunkSize;
            const std::size_t end   = std::min(start + chunkSize, nFiles);

            futures.push_back(std::async(std::launch::async, [&, start, end] {
                for (std::size_t i = start; i < end; ++i) {
                    perFileResults[i] = extractFeaturesForFile(paths[i], setsPerClip, clipHopFrames);
                    progress.fileCompleted(perFileResults[i].size());
                }
            }));
        }

        for (auto& f : futures)
            f.get();   // re-throws any worker exception on this thread

        progress.finish();

    } // GIL reacquired here

    // ── Count total clips ────────────────────────────────────────────────
    std::size_t totalClips = 0;
    for (const auto& fileClips : perFileResults)
        totalClips += fileClips.size();

    // ── Allocate 3D result array: (totalClips, framesPerClip, numFeatures)
    py::array_t<float> result(std::vector<std::size_t>{
        totalClips,
        setsPerClip,
        FeatureSet::numFeatures
    });
    float* buf = result.mutable_data();

    // Row stride in floats: one clip = framesPerClip * numFeatures floats
    const std::size_t clipStride  = setsPerClip * FeatureSet::numFeatures;
    const std::size_t frameStride = FeatureSet::numFeatures;

    std::size_t clipIdx = 0;
    for (const auto& fileClips : perFileResults) {
        for (const auto& clip : fileClips) {
            float* clipPtr = buf + clipIdx * clipStride;
            for (std::size_t f = 0; f < setsPerClip; ++f)
                clip[f].flatten(clipPtr + f * frameStride);
            ++clipIdx;
        }
    }

    return result;
}


// ─────────────────────────────────────────────────────────────
// 6.  pybind11 module definition
// ─────────────────────────────────────────────────────────────

PYBIND11_MODULE(audio_features, m) {
    m.doc() = "Multi-threaded audio feature extraction";

    py::class_<FeatureSet>(m, "FeatureSet")
        .def(py::init<>())
        .def_readwrite("spectralCentroid", &FeatureSet::spectralCentroid)
        .def("__repr__", [](const FeatureSet& fs) {
            return "<FeatureSet spectralCentroid=" +
                   std::to_string(fs.spectralCentroid) + ">";
        });

    m.def("extractFeatures",
          &extractFeatures,
          py::arg("paths"),
          py::arg("framesPerClip"),
          py::arg("clipHopFrames"),
          py::arg("nThreads") = 0,
          R"doc(
Extract audio features for a list of files using chunked std::async.

Each file is split into overlapping clips of framesPerClip STFT frames,
with a hop of clipHopFrames between clip start positions. All clips across
all files are stacked into a single 3D NumPy array.

Parameters
----------
paths         : list[str]   Paths to audio files.
framesPerClip : int         STFT frames per clip (e.g. 129).
clipHopFrames : int         Frame-space hop between clip windows.
nThreads      : int         Worker threads (0 = hardware_concurrency).

Returns
-------
np.ndarray, shape (totalClips, framesPerClip, numFeatures), dtype float32
          )doc");
}