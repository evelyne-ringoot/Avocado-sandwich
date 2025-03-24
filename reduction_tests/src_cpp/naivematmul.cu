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




#define numparx 8
#define numpary 8


__global__ void naive_matmul( 
    int size_in,
    float *inputa, float *inputb, float *output ) {
    int g = blockIdx.x *blockDim.x + threadIdx.x;
    int h = blockIdx.y *blockDim.y + threadIdx.y;

    if (g<size_in && h<size_in){
        float res=0.0;

        for (int k=0;k<size_in;k++){
            res+=inputa[g+size_in*k]*inputb[k+size_in*h];
        }
        output[g+size_in*h]=res;
    }
}

void run_naive_matmul(
    int size_in,
    float *inputa, float *inputb, float *output  /* pointer to GPU memory */
) {
    dim3 dimBlock(numparx, numpary);
    dim3 dimGrid(ceil_div_static(size_in,numparx), ceil_div_static(size_in,numpary));
    naive_matmul<<<dimGrid,dimBlock>>>(size_in, inputa, inputb, output);
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
};

struct TestData {
    std::map<std::tuple<int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t>, std::vector<float>> b;
    std::map<std::tuple<int32_t>, std::vector<float>> ref;
};

TestData read_test_data(
    std::string const &test_data_dir,
    std::vector<BenchmarkConfig> const &configs) {
    auto data = TestData{};
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto path_prefix = test_data_dir + "/";

        if (data.a.find({size_i}) == data.a.end()) {
            data.a[{size_i}] = read_data(
                path_prefix + "test_a_" + std::to_string(size_i) + "_" +
                    std::to_string(size_i) + ".bin",
                size_i * size_i);
        }

        if (data.b.find({size_i}) == data.b.end()) {
            data.b[{size_i}] = read_data(
                path_prefix + "test_b_" + std::to_string(size_i) + "_" +
                    std::to_string(size_i) + ".bin",
                size_i * size_i);
        }

        if (data.ref.find({size_i}) == data.ref.end()) {
            data.ref[{size_i}] = read_data(
                path_prefix + "ref_c_" + std::to_string(size_i) + "_" +
                    std::to_string(size_i) + ".bin",
                size_i*size_i);
        }

    }
    return data;
}

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t>, double> elapsed_ms;
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

    auto const &a = data.a.at({size_i});
    auto const &b = data.b.at({size_i});
    auto const &ref = data.ref.at({size_i});

    float *a_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_i * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size_i * size_i * sizeof(float),
        cudaMemcpyHostToDevice));

    float *b_gpu;
    CUDA_CHECK(cudaMalloc(&b_gpu, size_i * size_i * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        b_gpu,
        b.data(),
        size_i * size_i * sizeof(float),
        cudaMemcpyHostToDevice));

            float *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_i * sizeof(float)));


    if (phase==Phase::BENCHMARK){
        printf("  %6d ", size_i);
    }else{
        printf("  %6d \n", size_i);
    }
    run_naive_matmul(size_i,   a_gpu, b_gpu, c_gpu);
    
    std::vector<float> c_out_host(size_i * size_i);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        c_gpu,
        size_i * size_i * sizeof(float),
        cudaMemcpyDeviceToHost));
    
    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < size_i*size_i; ++i) {
            float diff = abs(c_out_host[i]) - abs(ref[i ]);
            mse += diff * diff;
            ref_mean_square += abs(ref[i ]) * abs(ref[i]);
    }
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse/std::sqrt(size_i); // std::sqrt(ref_mean_square);
    if (phase==Phase::BENCHMARK){
        printf("  %8.02e", rel_rmse);
    }
    

    double target_time_ms = 200.0;
    double elapsed_ms = benchmark_ms(
        target_time_ms,
        2,
        [&]() {
        },
        [&]() {
            run_naive_matmul(size_i,   a_gpu, b_gpu, c_gpu);
        });

    results.elapsed_ms[{size_i}] = elapsed_ms;
    if (phase==Phase::BENCHMARK){
        printf("  %8.03f \n", elapsed_ms);
    }

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
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
            "  %-6s  %-8s  %-9s \n",
            "size_i",
            "RRMSE",
            "time (ms)");
        printf(
            "  %-6s  %-8s  %-9s  \n",
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
        {{32},{64},{128},{256},{512},{1024},{2048}},
    };

    auto data = read_test_data(test_data_dir, configs_test);
    run_all_impls(Phase::WARMUP, data, configs_test);
    run_all_impls(Phase::BENCHMARK, data, configs_test);
  

    return 0;
}
