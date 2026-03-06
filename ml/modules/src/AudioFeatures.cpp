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

// FeatureSet is now just the fixed-size array produced by FeatureExtractor.
// numFeatures is read directly from FeatureExtractor::kNumFeatures so there
// is a single source of truth — changing the extractor automatically flows
// through to the numpy array shape.
using FeatureSet = std::array<float, FeatureExtractor::kNumFeatures>;
using Clip       = std::vector<FeatureSet>;

static constexpr std::size_t kNumFeatures = FeatureExtractor::kNumFeatures;

struct WavFile {
    std::vector<float> samples;
    double             sampleRate = 0.0;
};

WavFile loadWav(const std::string& path) {
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

    return { std::move(samples), static_cast<double>(info.samplerate) };
}

static std::vector<Clip> extractFeaturesForFile(
        const std::string& filePath,
        std::size_t        setsPerClip,
        std::size_t        clipHopFrames)
{
    auto wav = loadWav(filePath);
    if (wav.samples.empty())
        throw std::runtime_error("File " + filePath + " is empty");

    float* f[1];
    f[0] = wav.samples.data();
    const auto audio = choc::buffer::createChannelArrayView(f, 1, wav.samples.size());

    FeatureExtractor extractor(wav.sampleRate);
    std::vector<Clip> clips;
    const auto clipFrames = FeatureExtractor::kFftSize + (setsPerClip - 1) * FeatureExtractor::kHopSize;

    uint32_t startFrame = 0;
    while (true) {
        uint32_t endFrame = startFrame + clipFrames;
        if (endFrame > wav.samples.size())
            return clips;

        Clip clip;
        const auto clipAudio = audio.getFrameRange({startFrame, endFrame});
        extractor.process(clipAudio,
                          wav.sampleRate,
                          [&clip](const FeatureExtractor::FeatureArray& featureSet) {
                              clip.push_back(featureSet);
                          });

        if (clip.size() != setsPerClip)
            throw std::runtime_error("clip size mismatch");

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
        , startTime_(std::chrono::steady_clock::now()) {}

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
// Internal helper: run multithreaded extraction and return
// perFileResults. Shared by both public entry points.
// ─────────────────────────────────────────────────────────────

static std::vector<std::vector<Clip>> extractAllFiles(
        const std::vector<std::string>& paths,
        std::size_t                     setsPerClip,
        std::size_t                     clipHopFrames,
        std::size_t                     nThreads)
{
    const std::size_t nFiles = paths.size();
    nThreads = std::min(nThreads, nFiles);

    // Pre-sized so threads write to non-overlapping indices — no mutex needed.
    std::vector<std::vector<Clip>> perFileResults(nFiles);

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

    return perFileResults;
}


// ─────────────────────────────────────────────────────────────
// Helper: flatten perFileResults into a 3D numpy array.
// ─────────────────────────────────────────────────────────────

static py::array_t<float> flattenToArray(
        const std::vector<std::vector<Clip>>& perFileResults,
        std::size_t                           setsPerClip)
{
    std::size_t totalClips = 0;
    for (const auto& fileClips : perFileResults)
        totalClips += fileClips.size();

    std::vector<std::size_t> shape   = { totalClips, setsPerClip, kNumFeatures };
    std::vector<std::size_t> strides = {
        setsPerClip * kNumFeatures * sizeof(float),
        kNumFeatures * sizeof(float),
        sizeof(float)
    };
    py::array_t<float> result(shape, strides);
    float* buf = result.mutable_data();

    const std::size_t clipStride  = setsPerClip * kNumFeatures;
    const std::size_t frameStride = kNumFeatures;

    std::size_t clipIdx = 0;
    for (const auto& fileClips : perFileResults) {
        for (const auto& clip : fileClips) {
            float* clipPtr = buf + clipIdx * clipStride;
            for (std::size_t f = 0; f < setsPerClip; ++f)
                std::copy(clip[f].begin(), clip[f].end(), clipPtr + f * frameStride);
            ++clipIdx;
        }
    }

    return result;
}


// ─────────────────────────────────────────────────────────────
//  extractFeatures  (original API — unchanged)
//
//   Returns np.ndarray shape (totalClips, setsPerClip, numFeatures)
// ─────────────────────────────────────────────────────────────

py::array_t<float> extractFeatures(
        const std::vector<std::string>& paths,
        std::size_t                     setsPerClip,
        std::size_t                     clipHopFrames,
        std::size_t                     nThreads = 0)
{
    if (paths.empty())
        throw std::runtime_error("paths is empty");
    if (setsPerClip == 0)
        throw std::invalid_argument("framesPerClip must be > 0");
    if (clipHopFrames == 0)
        throw std::invalid_argument("clipHopFrames must be > 0");

    if (nThreads == 0)
        nThreads = std::max(1u, std::thread::hardware_concurrency());

    std::vector<std::vector<Clip>> perFileResults;
    {
        py::gil_scoped_release release;
        perFileResults = extractAllFiles(paths, setsPerClip, clipHopFrames, nThreads);
    }

    return flattenToArray(perFileResults, setsPerClip);
}


// ─────────────────────────────────────────────────────────────
//  extractFeaturesWithCounts  (new)
//
//   Same multithreaded extraction as extractFeatures, but also
//   returns a per-file clip count vector so the caller can
//   reconstruct file_indices without sacrificing batch throughput.
//
//   Returns
//   -------
//   tuple[np.ndarray, list[int]]
//     [0]  float32 array  shape (totalClips, setsPerClip, numFeatures)
//     [1]  list of ints   length == len(paths), clipsPerFile[i] is the
//          number of clips extracted from paths[i]  (may be 0 if the
//          file was too short to yield even one clip)
// ─────────────────────────────────────────────────────────────

std::pair<py::array_t<float>, std::vector<std::size_t>>
extractFeaturesWithCounts(
        const std::vector<std::string>& paths,
        std::size_t                     setsPerClip,
        std::size_t                     clipHopFrames,
        std::size_t                     nThreads = 0)
{
    if (paths.empty())
        throw std::runtime_error("paths is empty");
    if (setsPerClip == 0)
        throw std::invalid_argument("framesPerClip must be > 0");
    if (clipHopFrames == 0)
        throw std::invalid_argument("clipHopFrames must be > 0");

    if (nThreads == 0)
        nThreads = std::max(1u, std::thread::hardware_concurrency());

    std::vector<std::vector<Clip>> perFileResults;
    {
        py::gil_scoped_release release;
        perFileResults = extractAllFiles(paths, setsPerClip, clipHopFrames, nThreads);
    }

    // Build clip-count vector (one entry per input file, preserving order).
    std::vector<std::size_t> clipsPerFile(perFileResults.size());
    for (std::size_t i = 0; i < perFileResults.size(); ++i)
        clipsPerFile[i] = perFileResults[i].size();

    return { flattenToArray(perFileResults, setsPerClip), clipsPerFile };
}


// ─────────────────────────────────────────────────────────────
// pybind11 module definition
// ─────────────────────────────────────────────────────────────

PYBIND11_MODULE(audio_features, m) {
    m.doc() = "Multi-threaded audio feature extraction";

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

    m.def("extractFeaturesWithCounts",
          &extractFeaturesWithCounts,
          py::arg("paths"),
          py::arg("framesPerClip"),
          py::arg("clipHopFrames"),
          py::arg("nThreads") = 0,
          R"doc(
Same as extractFeatures but also returns a per-file clip count list.

Use this when you need to reconstruct which clips came from which file
(e.g. for GroupShuffleSplit file-level train/val splitting) without
sacrificing multi-threaded batch throughput.

Parameters
----------
paths         : list[str]   Paths to audio files.
framesPerClip : int         STFT frames per clip.
clipHopFrames : int         Frame-space hop between clip windows.
nThreads      : int         Worker threads (0 = hardware_concurrency).

Returns
-------
tuple[np.ndarray, list[int]]
  [0]  float32 array shape (totalClips, framesPerClip, numFeatures)
  [1]  list of ints, one per input file — clips extracted from that file
       (0 if the file was too short to yield a single clip)
          )doc");
}