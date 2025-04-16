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

constexpr int32_t __host__ __device__ ceil_div_static(int32_t a, int32_t b) { return (a + b - 1) / b; }


void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES));
}
__device__ inline void cp_async1(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 4;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.ca.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES));
}

__device__ __forceinline__ void async_memcpy_waitall() {
    asm volatile("cp.async.wait_all;\n" ::);
}

////////////////////////////////////////////////////////////////////////////////




#define elperthread 128
#define numthreads 1
#define numpar 32



__global__ void reduction( 
    int size_in, int numsamples,
    float *input) {
    int g = blockIdx.x *blockDim.x + threadIdx.x;

    int startidx=g*size_in;
    if (g<numsamples){
        float res=input[startidx];

        for (int k=1;k<elperthread;k++){
            res+=input[startidx+k];
        }
        input[startidx]=res;
    }
    
}

void run_reduction(
    uint32_t size_i,
    uint32_t size_j,
    float *data /* pointer to GPU memory */
) {
    
    reduction<<<ceil_div_static(size_j,numpar), numthreads*numpar>>>(size_i, size_j, data);
}

////////////////////////////////////////////////////////////////////////////////

void print_matrix(int32_t n_row, int32_t n_col, std::vector<float> const &matrix) {
    for (int32_t i = 0; i < n_row; i++) {
        printf("    ");
        for (int32_t j = 0; j < n_col; j++) {
            printf("%10.5f ", matrix.at(i * n_col + j));
        }
        printf("\n");
    }
}

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

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
    int32_t size_j;
};

struct TestData {
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> ref;
};

TestData read_test_data(
    std::string const &test_data_dir,
    std::vector<BenchmarkConfig> const &configs) {
    auto data = TestData{};
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto path_prefix = test_data_dir + "/";

        if (data.a.find({size_i, size_j}) == data.a.end()) {
            data.a[{size_i, size_j}] = read_data(
                path_prefix + "test_a_" + std::to_string(size_i) + "_" +
                    std::to_string(size_j) + ".bin",
                size_i * size_j);
        }

        if (data.ref.find({size_i, size_j}) == data.ref.end()) {
            data.ref[{size_i, size_j}] = read_data(
                path_prefix + "ref_a_" + std::to_string(size_i) + "_" +
                    std::to_string(size_j) + ".bin",
                size_j);
        }

    }
    return data;
}

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t, int32_t>, double> elapsed_ms;
};

enum class Phase {
    TEST,
    WARMUP,
    BENCHMARK,
};

void run_config( Phase phase,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size_i = config.size_i;
    auto size_j = config.size_j;

    auto const &a = data.a.at({size_i, size_j});
    auto const &ref = data.ref.at({size_i, size_j});

    float *a_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_j * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size_i * size_j * sizeof(float),
        cudaMemcpyHostToDevice));

    if (phase==Phase::BENCHMARK){
        printf("  %6d  %6d  ", size_i, size_j);
    }else{
        printf("  %6d  %6d \n", size_i, size_j);
    }
    run_reduction(size_i,size_j,   a_gpu);

    std::vector<float> c_out_host(size_i * size_j);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        a_gpu,
        size_i * size_j * sizeof(float),
        cudaMemcpyDeviceToHost));

    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < size_j; ++i) {
            float diff = abs(c_out_host[i * size_i ]) - abs(ref[i ]);
            mse += diff * diff;
            ref_mean_square += abs(ref[i ]) * abs(ref[i]);
    }
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse; // std::sqrt(ref_mean_square);
    if (phase==Phase::BENCHMARK){
        printf("  %8.02e", rel_rmse);
    }
    

    double target_time_ms = 200.0;
    double elapsed_ms = benchmark_ms(
        target_time_ms,
        20,
        [&]() {
        },
        [&]() {
            run_reduction(size_i,size_j,   a_gpu);
        });

    results.elapsed_ms[{size_i, size_j}] = elapsed_ms;
    if (phase==Phase::BENCHMARK){
        printf("  %8.03f \n", elapsed_ms);
    }

    CUDA_CHECK(cudaFree(a_gpu));
}


BenchmarkResults run_all_configs(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{"reduction"};
    if (phase == Phase::WARMUP) {
        printf("warmup\n\n");
    }else {
        printf("\n\n");
        printf(
            "  %-6s  %-6s   %-8s  %-9s \n",
            "size_i",
            "size_j",
            "RRMSE",
            "time (ms)");
        printf(
            "  %-6s  %-6s  %-8s  %-9s  \n",
            "------",
            "------",
            "--------",
            "---------");
    }
    for (auto const &config : configs) {
        run_config( phase, data, config, results);
    }
    printf("\n");
    return results;
}



std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
    results.push_back(run_all_configs(phase, data, configs));
    return results;
}



int main(int argc, char **argv) {
    std::string test_data_dir = ".";

    auto configs_test = std::vector<BenchmarkConfig>{
        {{128,32},{128,128},{128,512},{128,2048},{128,1024*8},{128,32768},{128,32768*4}},
    };

    auto data = read_test_data(test_data_dir, configs_test);
    run_all_impls(Phase::WARMUP, data, configs_test);
    run_all_impls(Phase::BENCHMARK, data, configs_test);
  

    return 0;
}
