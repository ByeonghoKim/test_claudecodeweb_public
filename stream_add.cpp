/*
 * stream_add.cpp — Stream 64 MB float vectors and time just the add step.
 *
 * Layout
 * ------
 *  VectorPair   – RAII struct holding two 64 MB float arrays (64-byte aligned)
 *  StreamSource – generates VectorPairs lazily (one at a time)
 *  timed_add()  – adds two vectors; timer wraps ONLY the add loop
 *  main()       – drives the stream, prints per-pair stats and a summary
 *
 * ISA used: AVX-512F  (_mm512_add_ps — 16 floats per instruction)
 * Fallback : scalar loop for any trailing elements (< 16)
 *
 * Build:
 *   g++ -O2 -std=c++17 -mavx512f -o stream_add stream_add.cpp
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
#include <immintrin.h>   // AVX-512 intrinsics

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr std::size_t MB              = 1024ULL * 1024;
static constexpr std::size_t VECTOR_BYTES    = 64 * MB;              // 64 MB
static constexpr std::size_t ELEMENTS        = VECTOR_BYTES / sizeof(float); // 16 777 216
static constexpr int         NUM_PAIRS       = 8;

// ---------------------------------------------------------------------------
// 0. Aligned allocator (64-byte alignment for AVX-512 loads/stores)
// ---------------------------------------------------------------------------
template<typename T, std::size_t Align = 64>
struct AlignedAllocator {
    using value_type = T;
    T* allocate(std::size_t n) {
        void* p = nullptr;
        if (posix_memalign(&p, Align, n * sizeof(T)) != 0) throw std::bad_alloc{};
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) { free(p); }
    template<typename U> struct rebind { using other = AlignedAllocator<U, Align>; };
    bool operator==(const AlignedAllocator&) const { return true; }
    bool operator!=(const AlignedAllocator&) const { return false; }
};

using AlignedVec = std::vector<float, AlignedAllocator<float>>;

// ---------------------------------------------------------------------------
// 1. Data types
// ---------------------------------------------------------------------------
struct VectorPair {
    int        index;
    AlignedVec a;
    AlignedVec b;
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
// 3. Timed add — isolates *just* the addition loop (AVX-512)
// ---------------------------------------------------------------------------
struct AddResult {
    AlignedVec  result;
    double      elapsed_sec;
};

AddResult timed_add(const AlignedVec& a, const AlignedVec& b)
{
    const std::size_t n       = a.size();
    const std::size_t n16     = n & ~std::size_t{15};   // largest multiple of 16 <= n
    AlignedVec result(n);

    const float* __restrict__ pa = a.data();
    const float* __restrict__ pb = b.data();
    float*       __restrict__ pr = result.data();

    // ── just the add (AVX-512: 16 floats / instruction) ───────────────────
    auto t0 = std::chrono::high_resolution_clock::now();

    // Main AVX-512 loop — processes 16 floats per iteration
    for (std::size_t i = 0; i < n16; i += 16) {
        __m512 va = _mm512_load_ps(pa + i);   // aligned load (64-byte boundary)
        __m512 vb = _mm512_load_ps(pb + i);
        __m512 vc = _mm512_add_ps(va, vb);    // C = A + B  (16 floats at once)
        _mm512_store_ps(pr + i, vc);           // aligned store
    }
    // Scalar tail for any remaining elements (< 16)
    for (std::size_t i = n16; i < n; ++i)
        pr[i] = pa[i] + pb[i];

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
    std::cout << "  Stream-add benchmark (C++ / AVX-512)\n";
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
