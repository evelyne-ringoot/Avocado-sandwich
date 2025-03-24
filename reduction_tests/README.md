# Reduction tests
This code shows a naive CUDA.jl, KernalAbstractions.jl, and CUDA C++ kernel for reduction over an array and matrix multiply. It can also serve as an example for some of the basic optimization techniques discussed here: https://cuda.juliagpu.org/stable/tutorials/performance/. For full optimization of GPU kernels, please refer to other excellent examples as this code contains a naive version: https://github.com/JuliaGPU/KernelAbstractions.jl/blob/main/examples/performant_matmul.jl . Fastmath is not enabled in this simple benchmark.

# Results

Results are displayed up to two significant digits or up to 1us.

Matrix multiply:
| [ms] | C++   | cuda.jl | KA.jl |
|------|-------|---------|-------|
| 32   | 0.004 | 0.004   | 0.007 |
| 64   | 0.004 | 0.005   | 0.007 |
| 128  | 0.009 | 0.008   | 0.009 |
| 256  | 0.051 | 0.050   | 0.050 |
| 512  | 0.35  | 0.35    | 0.35  |
| 1024 | 2.7   | 2.7     | 2.8   |
| 2048 | 22    | 22      | 22    |

1D reduction:
| [ms]   | C++   | cuda.jl | KA.jl |
|--------|-------|---------|-------|
| 32     | 0.005 | 0.005   | 0.007 |
| 128    | 0.005 | 0.005   | 0.006 |
| 512    | 0.005 | 0.005   | 0.006 |
| 2048   | 0.007 | 0.007   | 0.008 |
| 8192   | 0.020 | 0.020   | 0.021 |
| 32768  | 0.072 | 0.073   | 0.073 |
| 131072 | 0.31  | 0.32    | 0.31  |


# Performance benchmarking

Benchmarking correctly is complex and comprehensive solutions are available (Google Benchmark, Benchmarktools.jl). The minimal benchmark used here aims to address GPU saturation on performance and provide a consistent benchmark over Julia and C++.  We use the following technique as per https://github.com/accelerated-computing-class/lab8, taking a minimum over different of numruns (2,20). 

```
function benchmark_ms(numruns,myfunc, args...;kwargs...)
    elapsed=0.0
    best=100000
    i=0
    while(elapsed<200.0 || i<2)
        CUDA.synchronize()
        start = time_ns()
        for i=1:numruns
            myfunc(args...;kwargs...)
        end
        CUDA.synchronize()
        endtime = time_ns()
        thisduration=(endtime-start)/1e6
        elapsed+=thisduration
        best = min(thisduration/numruns,best)
        i+=1
    end
    return best
end
 ```

### Windows build and run

On windows, make sure to install julia, CUDA toolkit >= 12.4.1, Visual Studio with C++ integration and then run the following code:
```
 powershell -executionpolicy bypass -File .\buildandrun.ps1
 powershell -executionpolicy bypass -File .\buildandrun_matmul.ps1
 julia --project=. .\src_julia\naivematmul.jl 
 julia --project=. .\src_julia\reduction.jl 
 ```
