#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cusolverDn.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include "cusolver_utils.h"

// Error checking macro for CUDA calls
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Error checking macro for cuSOLVER calls
#define CUSOLVER_CHECK(call)                                                   \
    do {                                                                       \
        cusolverStatus_t status = call;                                        \
        if (status != CUSOLVER_STATUS_SUCCESS) {                               \
            std::cerr << "cuSOLVER error: " << status << " at "                \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Error checking macro for cuRAND calls
#define CURAND_CHECK(call)                                                     \
    do {                                                                       \
        curandStatus_t status = call;                                          \
        if (status != CURAND_STATUS_SUCCESS) {                                 \
            std::cerr << "cuRAND error: " << status << " at "                  \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

//references
//https://github.com/accelerated-computing-class/lab6
//https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/gesvd/cusolver_gesvd_example.cu
//https://github.com/ROCm/rocm-examples/tree/f9d4e5e78325c36b319d91ec37c6410b2b6e12fb/Libraries/hipSOLVER/gesvd

constexpr int32_t __host__ __device__ ceil_div_static(int32_t a, int32_t b) { return (a + b - 1) / b; }


template <typename Reset, typename F>
double
benchmark_ms(double target_time_ms, int32_t num_iters_inner, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    int k=0;
    while (elapsed_ms < target_time_ms || k<2) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
        k++;
    }
    return best_time_ms;
}

struct BenchmarkConfig {
    int32_t size_i;
};

struct TestData {
    std::map<int32_t, float*> input;
    std::map<int32_t, float*> singvals;
};

struct BenchmarkResults {
    std::map<int32_t, double> elapsed_ms;
};

enum class Phase {
    TEST,
    WARMUP,
    BENCHMARK,
};

void run_config( Phase phase,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size_i = config.size_in;
 
    curandGenerator_t curandGen;
    CURAND_CHECK(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curandGen, 12345ULL));

    float *a_gpu;
    float *svdout;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_in * size_in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&svdout, size_in * sizeof(float)));
    
    cusolverDnHandle_t cusolverH = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    int *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize( cusolverH,  size_in,    size_in,   &lwork   ));
    float *d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));

    double elapsed_ms = benchmark_ms(
        200.0,
        2,
        [&]() {
            CURAND_CHECK(curandGenerateUniform(curandGen, a_gpu, size_in*size_in)); 
        },
        [&]() {
            CUSOLVER_CHECK(cusolverDnSgesvd(
                cusolverH,  'N',  'N',  size_in,   size_in,  a_goy, size_in, svdout, nullptr, 
                size_in,   nullptr,  size_in,   d_work,  lwork,   nullptr,  d_info  ));
        });

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(svdout));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CURAND_CHECK(curandDestroyGenerator(curandGen));

    results.elapsed_ms[size_i] = elapsed_ms;
}


BenchmarkResults run_all_configs(
    Phase phase,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults;
    if (phase == Phase::WARMUP) {
        printf("warmup\n\n");
    }else {
        printf("\n\n");
        printf(
            "  %-6s  %-9s \n",
            "size_i",
            "time (ms)");
        printf(
            "  %-6s  %-9s  \n",
            "------",
            "---------");
    }
    for (auto const &config : configs) {
        run_config( phase, config, results);
    }
    printf("\n");
    return results;
}



std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
    results.push_back(run_all_configs(phase, configs));
    return results;
}



int main(int argc, char **argv) {
    std::string test_data_dir = ".";

    #int n = std::stoi(argv[1]);
    auto configs_test = std::vector<BenchmarkConfig>{
        {{64},{128},{256},{512},{1024},{2048}, {4096}, {8192}},
    };

    run_all_impls(Phase::WARMUP,  configs_test);
    run_all_impls(Phase::BENCHMARK, configs_test);

    return 0;
}


