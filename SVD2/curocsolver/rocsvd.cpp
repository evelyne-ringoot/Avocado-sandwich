#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#include <rocsolver/rocsolver.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <hip/hip_runtime_api.h>


#define HIP_CHECK(stat)                                                        \
    do {                                                                       \
        auto err = (stat);                                                     \
        if (err != hipSuccess) {                                               \
            std::cerr << "HIP error: " << hipGetErrorString(err) << " at "    \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define ROCBLAS_CHECK(stat)                                                    \
    do {                                                                       \
        auto err = (stat);                                                     \
        if (err != rocblas_status_success) {                                   \
            std::cerr << "rocBLAS error: " << err << " at "                    \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

    // Error checking macro for rocRAND calls
#define ROCRAND_CHECK(stat)                                                    \
do {                                                                       \
    auto err = (stat);                                                     \
    if (err != ROCRAND_STATUS_SUCCESS) {                                   \
        std::cerr << "rocRAND error: " << err << " at "                    \
                  << __FILE__ << ":" << __LINE__ << std::endl;             \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

#define HIPSOLVER_CHECK(condition)                                                            \
    {                                                                                         \
        const hipsolverStatus_t status = condition;                                           \
        if(status != HIPSOLVER_STATUS_SUCCESS)                                                \
        {                                                                                     \
            std::cerr << "hipSOLVER error encountered: \"" << hipsolverStatusToString(status) \
                      << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;                \
            std::exit(error_exit_code);                                                       \
        }                                                                                     \
    }
    
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
        HIP_CHECK(hipDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
        k++;
    }
    return best_time_ms;
}

struct BenchmarkConfig {
    int32_t size_in;
};

struct TestData {
    std::map<int32_t, float*> input;
    std::map<int32_t, float*> singvals;
};

enum class Phase {
    TEST,
    WARMUP,
    BENCHMARK,
};

void run_config( Phase phase,
    BenchmarkConfig const &config) {
    auto size_in = config.size_in;

    if (phase==Phase::BENCHMARK){
        printf("  %6d ", size_in);
    }else{
        printf("  %6d \n", size_in);
    }

    rocblas_handle rocblas_handle;
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_handle));
    rocrand_generator gen;
    ROCRAND_CHECK(rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_DEFAULT));
    ROCRAND_CHECK(rocrand_set_seed(gen, 12345));

    float *d_A, *d_S, *d_work;
    int *d_info;
    
    HIP_CHECK(hipMalloc(&d_A, size_in * size_in * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_S, size_in * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_work, size_in * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_info, sizeof(int)));

    double elapsed_ms = benchmark_ms(
        200.0,
        2,
        [&]() {
            ROCRAND_CHECK(rocrand_generate_uniform(gen, d_A, size_in*size_in));
        },
        [&]() {
            ROCBLAS_CHECK(rocsolver_sgesvd(
                rocblas_handle, rocblas_svect_none,  rocblas_svect_none, 
                size_in, size_in, d_A, size_in, d_S,  nullptr,size_in, 
                nullptr,  size_in,   d_work,  rocblas_inplace, d_info    ));
        });

        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_S));
        HIP_CHECK(hipFree(d_work));
        HIP_CHECK(hipFree(d_info));
        ROCBLAS_CHECK(rocblas_destroy_handle(rocblas_handle));
        ROCRAND_CHECK(rocrand_destroy_generator(gen));

        if (phase==Phase::BENCHMARK){
            printf("  %8.03f \n", elapsed_ms);
        }
}


void run_all_configs(
    Phase phase,
    std::vector<BenchmarkConfig> const &configs) {
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
        run_config( phase, config);
    }
    printf("\n");
}



int main(int argc, char **argv) {
    std::string test_data_dir = ".";
    std::vector<BenchmarkConfig> configs_test;
    if (argc==1){
        configs_test = std::vector<BenchmarkConfig>{
            {{64},{128},{256},{512},{1024},{2048}, {4096}},
        };
    }else{
        int n = std::stoi(argv[1]);
        configs_test = std::vector<BenchmarkConfig>{
            {n},
        };
    }

    run_all_configs(Phase::WARMUP,  configs_test);
    run_all_configs(Phase::BENCHMARK, configs_test);

    return 0;
}




    
