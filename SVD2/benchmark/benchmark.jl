using StochasticRounding,Distributions, KernelAbstractions,GPUArrays, Random, LinearAlgebra, Printf, RandomMatrices, Roots, Dates,  DelimitedFiles
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
vecty=typeof(KernelAbstractions.zeros(backend,elty,2))
artyfp64=typeof(KernelAbstractions.zeros(backend,Float64,2,2))
vectyfp64=typeof(KernelAbstractions.zeros(backend,Float64,2))



const TILESIZE = length(ARGS)>=5 ? parse(Int,ARGS[5]) : 64
const TILESIZEMUL =  length(ARGS)>=6 ? parse(Int,ARGS[6]) : 32
const QRSPLIT = length(ARGS)>=7 ? parse(Int,ARGS[7]) :  8
const BRDSPLIT = length(ARGS)>=7 ? parse(Int,ARGS[7]) : 8
const MINTIME = length(ARGS)>=9 ? parse(Int,ARGS[9]) : 2000
const NUMRUMS= length(ARGS)>=10 ? parse(Int,ARGS[10]) : 20
const BANDOFFSET = 1


BLAS.set_num_threads(Threads.nthreads())
include("../src/KAfuncs.jl")
include("../src/qr_kernels.jl")
if (ARGS[3]!="QRB" && ARGS[3]=="MULQ")
    include("../src/brdgpu.jl")
    brd! = (length(ARGS)>=8 && ARGS[8]=="Y") ? brd2! : brd1!
else
    mygbbrd!(A) = nothing;
end
include("../src/tiledalgos.jl")
#include("../src/datacomms.jl")
include("benchfuncs.jl")


 @printf "-- starting with parameters TILESIZE=%4d MULSIZE=%4d QRSPLIT%4d ELMENT=%s  BRD=%1d  \n" TILESIZE TILESIZEMUL QRSPLIT elty Int(length(ARGS)>=8 && ARGS[8]=="Y")


if (ARGS[3]=="SMALL")
    sizes=[64,128,256,512,1024,2048, 4096]
    include("benchmarkall.jl")
elseif (ARGS[3]=="LARGE")
    sizes=8192 .*[1,2,4,8]
    include("benchmarklarge.jl")
elseif (ARGS[3]=="SPECIFY")
    sizes=[parse(Int,ARGS[4])]
    include("benchmarklarge.jl")
elseif (ARGS[3]=="CHECKERRORS")
    sizes=[64,128,256,512,1024,2048,2048*2,1024*8,1024*16,1024*32]
    include("benchmarkerrors.jl")
elseif (ARGS[3]=="SUBFUNC")
    include("benchmarksubfunctions.jl")
elseif (ARGS[3]=="ALL")
    functobench=mygesvd!
    ERRCHECK=true
    include("benchmarkallandlarge.jl")
elseif (ARGS[3]=="QRB")
    ERRCHECK=false
    functobench=myblockdiag_qrcalc!
    include("benchmarkallandlarge.jl")
elseif (ARGS[3]=="MULQ")
    ERRCHECK=false
    functobench=myblockdiag_applyqr!
    include("benchmarkallandlarge.jl")
else
    error("specify correct params")
end