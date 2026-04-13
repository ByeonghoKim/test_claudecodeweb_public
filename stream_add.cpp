/*
 * stream_add.cpp — Stream 64 MB float vectors and time just the add step.
 *
 * Layout
 * ------
 *  VectorPair   – RAII struct holding two 64 MB float arrays
 *  StreamSource – input_iterator that yields VectorPairs one at a time
 *  timed_add()  – adds two vectors; timer wraps ONLY the add loop
 *  main()       – drives the stream, prints per-pair stats and a summary
 *
 * Build:
 *   g++ -O2 -std=c++17 -o stream_add stream_add.cpp
 *
 * The -O2 flag lets the compiler vectorise the add loop (AVX/SSE) without
 * removing it, so the timing reflects realistic hardware throughput.
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr std::size_t MB              = 1024ULL * 1024;
static constexpr std::size_t VECTOR_BYTES    = 64 * MB;              // 64 MB
static constexpr std::size_t ELEMENTS        = VECTOR_BYTES / sizeof(float); // 16 777 216
static constexpr int         NUM_PAIRS       = 8;

// ---------------------------------------------------------------------------
// 1. Data types
// ---------------------------------------------------------------------------
struct VectorPair {
    int                 index;
    std::vector<float>  a;
    std::vector<float>  b;
};

// ---------------------------------------------------------------------------
// 2. StreamSource — generates VectorPairs lazily (one at a time)
// ---------------------------------------------------------------------------
class StreamSource {
public:
    explicit StreamSource(int num_pairs, std::uint64_t seed = 42)
        : num_pairs_(num_pairs), rng_(seed)
    {
        advance();   // prime first pair
    }

    bool has_next() const { return current_index_ < num_pairs_; }

    VectorPair next() {
        VectorPair p = std::move(current_);
        ++current_index_;
        if (has_next()) advance();
        return p;
    }

private:
    void advance() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        current_.index = current_index_;
        current_.a.resize(ELEMENTS);
        current_.b.resize(ELEMENTS);
        for (std::size_t i = 0; i < ELEMENTS; ++i) {
            current_.a[i] = dist(rng_);
            current_.b[i] = dist(rng_);
        }
    }

    int             num_pairs_;
    int             current_index_ = 0;
    std::mt19937_64 rng_;
    VectorPair      current_;
};

// ---------------------------------------------------------------------------
// 3. Timed add — isolates *just* the addition loop
// ---------------------------------------------------------------------------
struct AddResult {
    std::vector<float>  result;
    double              elapsed_sec;
};

AddResult timed_add(const std::vector<float>& a, const std::vector<float>& b)
{
    std::vector<float> result(a.size());

    // ── just the add ──────────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] + b[i];

    auto t1 = std::chrono::high_resolution_clock::now();
    // ──────────────────────────────────────────────────────────────────────

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    return {std::move(result), elapsed};
}

// ---------------------------------------------------------------------------
// 4. Runtime estimation helpers
// ---------------------------------------------------------------------------
double estimate_remaining(const std::vector<double>& times, int pairs_left)
{
    if (times.empty()) return std::numeric_limits<double>::quiet_NaN();
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    return mean * pairs_left;
}

double throughput_gbs(double elapsed_sec)
{
    double bytes_touched = 3.0 * static_cast<double>(VECTOR_BYTES); // read a, read b, write result
    return (bytes_touched / elapsed_sec) / (1024.0 * 1024.0 * 1024.0);
}

// ---------------------------------------------------------------------------
// 5. Main
// ---------------------------------------------------------------------------
int main()
{
    auto sep62 = std::string(62, '=');
    auto dash62 = std::string(62, '-');

    std::cout << sep62 << "\n";
    std::cout << "  Stream-add benchmark (C++)\n";
    std::cout << "  Vector size : " << (VECTOR_BYTES / MB) << " MB"
              << "  (" << ELEMENTS << " float32 elements)\n";
    std::cout << "  Pairs       : " << NUM_PAIRS << "\n";
    std::cout << sep62 << "\n";

    StreamSource source(NUM_PAIRS);
    std::vector<double> add_times;
    add_times.reserve(NUM_PAIRS);

    while (source.has_next()) {
        VectorPair pair = source.next();
        int pairs_done = pair.index + 1;
        int pairs_left = NUM_PAIRS - pairs_done;

        // ── just the add ────────────────────────────────────────────────
        AddResult ar = timed_add(pair.a, pair.b);
        // ────────────────────────────────────────────────────────────────

        add_times.push_back(ar.elapsed_sec);
        double est = estimate_remaining(add_times, pairs_left);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  pair " << std::setw(2) << pairs_done << "/" << NUM_PAIRS
                  << " | add = " << std::setw(7) << ar.elapsed_sec * 1000.0 << " ms"
                  << " | " << std::setw(5) << std::setprecision(2) << throughput_gbs(ar.elapsed_sec) << " GB/s"
                  << " | est. remaining add time = ";
        if (pairs_left > 0)
            std::cout << std::setprecision(1) << est * 1000.0 << " ms\n";
        else
            std::cout << "done\n";

        // Release result memory immediately (benchmarking, not accumulating)
        ar.result.clear();
        ar.result.shrink_to_fit();
    }

    double total   = std::accumulate(add_times.begin(), add_times.end(), 0.0);
    double mean    = total / add_times.size();
    double minimum = *std::min_element(add_times.begin(), add_times.end());
    double maximum = *std::max_element(add_times.begin(), add_times.end());

    std::cout << dash62 << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Total add time : " << std::setw(8) << total   * 1000.0 << " ms\n";
    std::cout << "  Mean  add time : " << std::setw(8) << mean    * 1000.0 << " ms"
              << "  (" << std::setprecision(2) << throughput_gbs(mean) << " GB/s avg)\n";
    std::cout << "  Min   add time : " << std::setw(8) << std::setprecision(3) << minimum * 1000.0 << " ms\n";
    std::cout << "  Max   add time : " << std::setw(8) << maximum * 1000.0 << " ms\n";
    std::cout << sep62 << "\n";

    return 0;
}
