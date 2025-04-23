using KernelAbstractions,GPUArrays, CUDA,Random, LinearAlgebra, Printf, BenchmarkTools

KernelAbstractions.get_backend(CUDA.zeros(1))
BLAS.set_num_threads(Threads.nthreads())
const backend=CUDABackend(false, false, true)


elty=Float32
sizes=8192 .*[2,4,8]
timings=ones(4,length(sizes))*1000000
errors=zeros(2,length(sizes))

include("../src/cusol_funcs.jl")
include("../src/KAfuncs.jl")
include("../src/qr_kernels.jl")
include("../src/brdgpu.jl")
include("../src/datacomms.jl")
include("../src/tiledalgos.jl")


function benchmark_ms( size_i, myfunc;kwargs...)
    a=CUDA.randn(elty,size_i, size_i)
    elapsed=0.0
    best=100000
    i=0
    while(elapsed<200.0 || (i<2 &&elapsed<5000.0))
        KernelAbstractions.synchronize(backend)
        start = time_ns()
        myfunc(a;kwargs...)
        KernelAbstractions.synchronize(backend)
        endtime = time_ns()
        thisduration=(endtime-start)/1e6
        elapsed+=thisduration
        best = min(thisduration/numruns,best)
        i+=1
    end
    unsafe_free!(a)
    return best
end


println( "warump ");
for (i,size_i) in enumerate(sizes)
    timings[2,i] = min( benchmark_ms(size_i,mygesvd!), timings[2,i])
end

println( "benchmark");
for (i,size_i) in enumerate(sizes)
    timings[2,i] = min( benchmark_ms(size_i,mygesvd!), timings[2,i])
end

println("GPU only SVD");
println( " size    RRMSE    time (ms)  cutime(ms) ");
println(" ------  --------  ----------  ---------- ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %8.02f  %8.02f  \n" size_i errors[1,i] timings[2,i] timings[1,i]
end  