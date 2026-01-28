#include <algorithm>
#include <cctype>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <thread>
#include <mutex>
#include <future>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace fs = std::filesystem;
namespace py = pybind11;

struct DocChunk {
    std::string text;
    std::string source;
};

struct IngestionStats {
    size_t files_seen = 0;
    size_t text_files = 0;
    size_t total_chunks = 0;
    size_t total_bytes = 0;
    double read_seconds = 0.0;
    double chunk_seconds = 0.0;
    double total_seconds = 0.0;
};

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string read_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return "";
    std::string content;
    file.seekg(0, std::ios::end);
    content.resize(static_cast<size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    if (!content.empty()) {
        file.read(&content[0], static_cast<std::streamsize>(content.size()));
    }
    return content;
}

std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    current.reserve(32);
    for (unsigned char ch : text) {
        if (std::isspace(ch)) {
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
                current.reserve(32);
            }
        } else {
            current.push_back(static_cast<char>(ch));
        }
    }
    if (!current.empty()) {
        tokens.push_back(std::move(current));
    }
    return tokens;
}

std::string join_tokens(const std::vector<std::string>& tokens, size_t start, size_t end) {
    if (start >= end || start >= tokens.size()) return {};
    end = std::min(end, tokens.size());
    std::string out;
    size_t estimated = 0;
    for (size_t i = start; i < end; ++i) estimated += tokens[i].size() + 1;
    out.reserve(estimated);
    for (size_t i = start; i < end; ++i) {
        if (i > start) out.push_back(' ');
        out.append(tokens[i]);
    }
    return out;
}

std::vector<std::string> chunk_text(const std::string& text, int chunk_size, int overlap) {
    if (chunk_size <= 0) return {};
    std::vector<std::string> tokens = tokenize(text);
    if (tokens.empty()) return {};

    std::vector<std::string> chunks;
    const size_t step = static_cast<size_t>(chunk_size - overlap);
    for (size_t start = 0; start < tokens.size(); start += step) {
        size_t end = start + static_cast<size_t>(chunk_size);
        std::string chunk = join_tokens(tokens, start, end);
        if (chunk.empty()) break;
        chunks.push_back(std::move(chunk));
        if (end >= tokens.size()) break;
    }
    return chunks;
}

bool is_text_extension(const fs::path& path) {
    static const std::unordered_set<std::string> extensions = {
        ".txt", ".md", ".markdown", ".rst", ".py", ".json", ".yaml", ".yml",
        ".toml", ".csv", ".ts", ".js", ".html", ".css", ".cpp", ".cc", ".c",
        ".h", ".hpp", ".java", ".go", ".rs", ".sh"
    };
    std::string ext = to_lower(path.extension().string());
    return extensions.find(ext) != extensions.end();
}

}  // namespace

class IngestionEngine {
public:
    // This method runs WITHOUT the Python GIL (Global Interpreter Lock).
    // It uses all CPU cores to read and chunk files in parallel.
    std::pair<std::vector<DocChunk>, IngestionStats> process_file_list_parallel(
        const std::vector<std::string>& paths,
        int chunk_size,
        int overlap
    ) const {
        IngestionStats global_stats;
        std::vector<DocChunk> global_chunks;

        // Determine thread count (e.g., 16 on your Ryzen)
        unsigned int n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) n_threads = 4;

        // Mutex for merging results
        std::mutex merge_mutex;

        // Split work into chunks for each thread
        size_t total_files = paths.size();
        std::vector<std::thread> workers;

        auto total_start = Clock::now();

        // Worker lambda
        auto worker_task = [&](size_t start_idx, size_t end_idx) {
            std::vector<DocChunk> local_chunks;
            IngestionStats local_stats;

            for (size_t i = start_idx; i < end_idx; ++i) {
                fs::path file_path(paths[i]);
                if (!fs::exists(file_path) || !fs::is_regular_file(file_path)) continue;

                local_stats.files_seen++;
                if (!is_text_extension(file_path)) continue;
                local_stats.text_files++;

                auto read_start = Clock::now();
                std::string content = read_file(file_path);
                local_stats.read_seconds += elapsed_seconds(read_start, Clock::now());
                local_stats.total_bytes += content.size();

                auto chunk_start = Clock::now();
                // Basic chunking (streaming omitted for simplicity in parallel batch)
                std::vector<std::string> parts = chunk_text(content, chunk_size, overlap);
                local_stats.chunk_seconds += elapsed_seconds(chunk_start, Clock::now());

                for (auto& part : parts) {
                    if (!part.empty()) {
                        local_stats.total_chunks++;
                        local_chunks.push_back(DocChunk{std::move(part), file_path.string()});
                    }
                }
            }

            // Merge into global
            std::lock_guard<std::mutex> lock(merge_mutex);
            global_stats.files_seen += local_stats.files_seen;
            global_stats.text_files += local_stats.text_files;
            global_stats.total_chunks += local_stats.total_chunks;
            global_stats.total_bytes += local_stats.total_bytes;
            global_stats.read_seconds += local_stats.read_seconds;
            global_stats.chunk_seconds += local_stats.chunk_seconds;

            global_chunks.insert(
                global_chunks.end(),
                std::make_move_iterator(local_chunks.begin()),
                std::make_move_iterator(local_chunks.end())
            );
        };

        // Launch threads
        size_t files_per_thread = (total_files + n_threads - 1) / n_threads;
        for (unsigned int t = 0; t < n_threads; ++t) {
            size_t start = t * files_per_thread;
            size_t end = std::min(start + files_per_thread, total_files);
            if (start < end) {
                workers.emplace_back(worker_task, start, end);
            }
        }

        for (auto& t : workers) {
            if (t.joinable()) t.join();
        }

        global_stats.total_seconds = elapsed_seconds(total_start, Clock::now());
        return {std::move(global_chunks), global_stats};
    }
};

PYBIND11_MODULE(rag_core, m) {
    py::class_<DocChunk>(m, "DocChunk")
        .def_readonly("text", &DocChunk::text)
        .def_readonly("source", &DocChunk::source);

    py::class_<IngestionStats>(m, "IngestionStats")
        .def_readonly("files_seen", &IngestionStats::files_seen)
        .def_readonly("text_files", &IngestionStats::text_files)
        .def_readonly("total_chunks", &IngestionStats::total_chunks)
        .def_readonly("total_bytes", &IngestionStats::total_bytes)
        .def_readonly("read_seconds", &IngestionStats::read_seconds)
        .def_readonly("chunk_seconds", &IngestionStats::chunk_seconds)
        .def_readonly("total_seconds", &IngestionStats::total_seconds);

    py::class_<IngestionEngine>(m, "IngestionEngine")
        .def(py::init<>())
        .def("process_file_list_parallel", &IngestionEngine::process_file_list_parallel,
             py::call_guard<py::gil_scoped_release>()); // RELEASES GIL FOR SPEED
}