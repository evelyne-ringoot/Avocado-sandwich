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

const TILESIZE = length(ARGS)>=3 ? parse(Int,ARGS[3]) : 64
const BRDSPLIT = 64 #  length(ARGS)>=4 ? parse(Int,ARGS[4]) : 8
const BRDTILESPERTILE = length(ARGS)>=5 ? parse(Int,ARGS[5]) : 1
const BRDSUBTILE2 = length(ARGS)>=5 ? parse(Int,ARGS[5]) : 128 # required larger than TILESIZE
const MINTIME = length(ARGS)>=6 ? parse(Int,ARGS[6]) : 2000
const NUMRUMS= length(ARGS)>=7 ? parse(Int,ARGS[7]) : 20

include("../src/brdgpu.jl")
include("benchfuncs.jl")

sizes=[ 512, 1024, 2048, 4096,8192]#,, 16384 32768, 65536] #
timings=ones(length(sizes))*1000000000
errors=zeros(length(sizes))
println( "Checking correctness")
for (i,size_i) in enumerate(sizes)
    a=tril(triu(randn!(KernelAbstractions.zeros(backend,elty,size_i,size_i))),TILESIZE)
    aref=svdvals(a, alg=CUDA.CUSOLVER.QRAlgorithm())
    mygbbrd3!(a)
    KernelAbstractions.synchronize(backend)
    aout=svdvals(a, alg=CUDA.CUSOLVER.QRAlgorithm())
    errors[i]=norm(aref-aout)/norm(aout)
end



println( "warump KA only");
for (i,size_i) in enumerate(sizes)
    #timings[i] = min( benchmark_ms(size_i,mygbbrd3!), timings[i])
end

println( "run KA only");
for (i,size_i) in enumerate(sizes)
    timings[i] = min( benchmark_ms_large(size_i,mygbbrd3!), timings[i])
end

println("BRD");
println( " size    RRMSE    time (ms)  ");
println(" ------  --------  ----------   ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %8.02f \n" size_i errors[i] timings[i] 
end  

