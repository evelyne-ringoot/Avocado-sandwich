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
const MINTIME = length(ARGS)>=9 ? parse(Int,ARGS[9]) : 2000
const NUMRUMS= length(ARGS)>=10 ? parse(Int,ARGS[10]) : 20
const BANDOFFSET = 1

BLAS.set_num_threads(Threads.nthreads())
include("../src/KAfuncs.jl")
include("../src/qr_kernels.jl")
include("../src/brdgpu.jl")
include("../src/tiledalgos.jl")
include("../src/datacomms.jl")
include("benchfuncs.jl")
brd! = (length(ARGS)>=8 && ARGS[8]=="Y") ? brd2! : brd1!


sizes=[64,128,256,512,1024,2048, 4096]
timings=ones(5,length(sizes))*1000000000
funcstobench=[myblockdiag_unfused!,myblockdiag!,myblockdiag_qrcalc!,myblockdiag_applyqr!,bidiag]

for (fi,f) in enumerate(funcstobench)
    for (i,size_i) in enumerate(sizes)
        timings[fi,i] = min( benchmark_ms(size_i,f), timings[fi,i])
    end
end

println( " size   unfused1(ms)  fused1(ms)  qrcalc(ms)  qrapply(ms)   qrapply(ms)");
println(" ------  --------    ----------  ----------   ----------   ---------- ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02f  %8.02f   %8.02f  %8.02f   %8.02f \n" size_i timings[1,i] timings[2,i] timings[3,i] timings[4,i] timings[5,i]
end  

sizes=8192 .*[1,2,4,8]
timings=ones(3,length(sizes))

try
    for (fi,f) in enumerate(funcstobench)
        for (i,size_i) in enumerate(sizes)
            timings[fi,i] = min( benchmark_ms(size_i,f), timings[fi,i])
        end
    end
catch e
    println("did not run all sizes")
end


println( " size   unfused1(ms)  fused1(ms)  qrcalc(ms)  qrapply(ms)   qrapply(ms)");
println(" ------  --------    ----------  ----------   ----------   ---------- ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02f  %8.02f   %8.02f  %8.02f   %8.02f \n" size_i timings[1,i] timings[2,i] timings[3,i] timings[4,i] timings[5,i]
end 



