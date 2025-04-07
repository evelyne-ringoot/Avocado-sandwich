using KernelAbstractions,GPUArrays, AMDGPU,Random, LinearAlgebra, Printf, BenchmarkTools

backend=KernelAbstractions.get_backend(AMDGPU.zeros(2))
const TILESIZE = 64
const QRSPLIT = 4
include("../src/cusol_funcs.jl")
include("../src/KAfuncs.jl")
include("../src/qr_kernels.jl")
include("../src/datacomms.jl")
include("../src/tiledalgos.jl")

function benchmark_ms( myfunc, args...;kwargs...)
        KernelAbstractions.synchronize(backend)
        start = time_ns()
        myfunc(args...;kwargs...)
        KernelAbstractions.synchronize(backend)
        endtime = time_ns()
        thisduration=(endtime-start)/1e6
	KernelAbstractions.synchronize(backend)
        start = time_ns()
        myfunc(args...;kwargs...)
        KernelAbstractions.synchronize(backend)
        endtime = time_ns()
        thisduration=min((endtime-start)/1e6,thisduration)
    return thisduration
end

elty=Float32
sizes=[parse(Int, ARGS[1])]

timings=ones(length(sizes))


println( "Benchmarking KA only");
    a=randn(Float32,sizes[1], sizes[1])
    timings[1] = benchmark_ms(OOC_SVD!,a, backend)




println("OOC");
println( " size    RRMSE    time (ms)  cutime(ms) ");
println(" ------  --------  ----------  ---------- ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %10.02f   \n" size_i 0 timings[i] 
end  

