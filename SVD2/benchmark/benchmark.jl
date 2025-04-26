using KernelAbstractions,GPUArrays, Random, LinearAlgebra, Printf

if (ARGS[2]=="H")
    elty=Float16
elseif (ARGS[2]=="S")
    elty=Float32
elseif (ARGS[2]=="D")
    elty=Float64
else
    error("specify correct params")
end

if (ARGS[1]=="AMD")
    include("vendorspecific/benchmark_amd.jl")
elseif (ARGS[1]=="CUDA")
    include("vendorspecific/benchmark_cuda.jl")
elseif (ARGS[1]=="ONE")
    include("vendorspecific/benchmark_oneapi.jl")
elseif (ARGS[1]=="METAL")
    include("vendorspecific/benchmark_metal.jl")
else
    error("specify correct params")
end

arty=typeof(KernelAbstractions.zeros(backend,elty,2,2))

const TILESIZE = length(ARGS)>=5 ? parse(Int,ARGS[5]) : 64
const TILESIZEMUL =  length(ARGS)>=6 ? parse(Int,ARGS[6]) : 32
const QRSPLIT = length(ARGS)>=7 ? parse(Int,ARGS[7]) :  8
const BRDSPLIT = length(ARGS)>=7 ? parse(Int,ARGS[7]) : 8
const MINTIME = length(ARGS)>=8 ? parse(Int,ARGS[8]) : 200.0
const NUMRUMS= length(ARGS)>=9 ? parse(Int,ARGS[9]) : 12

include("includesrc.jl")
include("benchfuncs.jl")

if (ARGS[3]=="SMALL")
    sizes=[64,128,256,512,1024,2048, 4096,8192]
    include("benchmarkall.jl")
elseif (ARGS[3]=="LARGE")
    sizes=8192 .*[2,4,8]
    include("benchmarklarge.jl")
elseif (ARGS[3]=="SPECIFY")
    sizes=[parse(Int,ARGS[4])]
    include("benchmarklarge.jl")

else
    error("specify correct params")
end

include("benchmarkmataddmul.jl")


