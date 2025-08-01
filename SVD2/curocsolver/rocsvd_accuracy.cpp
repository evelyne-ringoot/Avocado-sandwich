#include <hip/hip_runtime.h>
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

//references
//https://github.com/accelerated-computing-class/lab6
//https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/gesvd/cusolver_gesvd_example.cu
//https://github.com/ROCm/rocm-examples/tree/f9d4e5e78325c36b319d91ec37c6410b2b6e12fb/Libraries/hipSOLVER/gesvd

constexpr int32_t __host__ __device__ ceil_div_static(int32_t a, int32_t b) { return (a + b - 1) / b; }


struct BenchmarkConfig {
    int32_t size_in;
};

struct TestData {
    std::map<int32_t, float*> input;
    std::map<int32_t, float*> singvals;
};

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

void run_config( BenchmarkConfig const &config) {
    auto size_in = config.size_in;

    
    auto path_prefix =  "./data_" + std::to_string(size_in);
    auto inputdata = read_data(path_prefix + ".bin", size_in*size_in);
    auto refsvd = read_data(path_prefix + "_svd.bin", size_in);

    rocblas_handle rocblas_handle;
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_handle));

    float *d_A, *d_S, *d_work;
    int *d_info;
    
    HIP_CHECK(hipMalloc(&d_A, size_in * size_in * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_S, size_in * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_work, size_in * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_info, sizeof(int)));

     HIP_CHECK(hipMemcpy(
            a_gpu,
            a.data(),
            size_in*size_in * sizeof(float),
            hipMemcpyHostToDevice));

    ROCBLAS_CHECK(rocsolver_sgesvd(
                rocblas_handle, rocblas_svect_none,  rocblas_svect_none, 
                size_in, size_in, d_A, size_in, d_S,  nullptr,size_in, 
                nullptr,  size_in,   d_work,  rocblas_inplace, d_info    ));

         std::vector<float> computed_svd(size_in );
        HIP_CHECK(hipMemcpy(
            computed_svd.data(),
            d_S,
            size_in * sizeof(float),
            hipMemcpyDeviceToHost));

        double mse = 0.0;
        double ref_mean_square = 0.0;
        for (int32_t i = 0; i < size_in; ++i) {
                float diff = computed_svd[i ] - refsvd[i ];
                mse += diff * diff;
                ref_mean_square += refsvd[i ] * refsvd[i ];
            
        }
        mse /= size_in ;
        ref_mean_square /= size_in ;
        float rmse = std::sqrt(mse);
        float rel_rmse = rmse / std::sqrt(ref_mean_square);

        printf("  size %4d :", size_in);
        printf("    relative error : %.04e \n", rel_rmse);

        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_S));
        HIP_CHECK(hipFree(d_work));
        HIP_CHECK(hipFree(d_info));
        ROCBLAS_CHECK(rocblas_destroy_handle(rocblas_handle));

}


void run_all_configs(
    std::vector<BenchmarkConfig> const &configs) {
        printf("\n\n");
        printf(
            "  %-6s  %-9s \n",
            "size  ",
            "RRMSE  ");
        printf(
            "  %-6s  %-9s  \n",
            "------",
            "---------");
    
    for (auto const &config : configs) {
        run_config(  config);
    }
    printf("\n");
}



int main(int argc, char **argv) {
    std::string test_data_dir = ".";
    std::vector<BenchmarkConfig> configs_test;
    
    configs_test = std::vector<BenchmarkConfig>{
        {{128},{256},{512},{1024},{2048}, {4096}, {8192}},
    };
    run_all_configs(  configs_test);

    return 0;
}




    
