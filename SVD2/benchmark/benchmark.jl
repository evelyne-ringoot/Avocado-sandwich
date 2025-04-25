using KernelAbstractions,GPUArrays, Random, LinearAlgebra, Printf

reducethreads=false

if (ARGS[1]=="AMD")
    include("vendorspecific/benchmark_amd.jl")
elseif (ARGS[1]=="CUDA")
    include("vendorspecific/benchmark_cuda.jl")
elseif (ARGS[1]=="ONE")
    include("vendorspecific/benchmark_oneapi.jl")
    reducethreads=true
elseif (ARGS[1]=="METAL")
    include("vendorspecific/benchmark_metal.jl")
    reducethreads=true

else
    error("specify correct params")
end
elty=nothing

if (ARGS[2]=="H")
    const TILESIZE = 64
    const QRSPLIT = reducethreads ? 4 : 8
    const BRDSPLIT = reducethreads ? 2 : 8
    const TILESIZEMUL = 32
    elty=Float16
    arty=typeof(KernelAbstractions.zeros(backend,elty,2,2))
    @inline vendorsvd!(input::arty) = svdvals!(Float32.(input),  alg=CUDA.CUSOLVER.QRAlgorithm())
elseif (ARGS[2]=="S")
    const TILESIZE = 64
    const QRSPLIT = reducethreads ? 4 : 8
    const BRDSPLIT = reducethreads ? 2 : 8
    const TILESIZEMUL = 32
    elty=Float32
elseif (ARGS[2]=="D")
    const TILESIZE = 32
    const QRSPLIT = 4
    const BRDSPLIT = reducethreads ? 2 : 4
    const TILESIZEMUL = 32
    elty=Float64
else
    error("specify correct params")
end
arty=typeof(KernelAbstractions.zeros(backend,elty,2,2))
include("includesrc.jl")
include("benchfuncs.jl")


if (ARGS[3]=="SMALL")
    sizes=[64,128,256,512,1024,2048]#, 4096,8192]
    include("benchmarkall.jl")
    include("benchmarkmataddmul.jl")
elseif (ARGS[3]=="LARGE")
    sizes=8192 .*[2,4,8]
    include("benchmarklarge.jl")
    include("benchmarkmataddmul.jl")
elseif (ARGS[3]=="SPECIFY")
    sizes=[parse(Int,ARGS[4])]
    include("benchmarklarge.jl")
    include("benchmarkmataddmul.jl")
else
    error("specify correct params")
end


