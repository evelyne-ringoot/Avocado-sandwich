using KernelAbstractions,GPUArrays, CUDA,Random, LinearAlgebra, Printf, BenchmarkTools

backend=KernelAbstractions.get_backend(CUDA.zeros(2))
const TILESIZE = 64
const QRSPLIT = 4
include("../src/cusol_funcs.jl")
include("../src/KAfuncs.jl")
include("../src/qr_kernels.jl")
include("../src/datacomms.jl")
include("../src/tiledalgos.jl")

#@inline backendstream() = CUDA.stream()
#@inline setstream!(stream::CuStream) = CUDA.stream!(stream)
#@inline backendstream() = AMDGPU.HIPStream()
#@inline setstream!(s::HIPStream) = AMDGPU.stream!(stream)

BLAS.set_num_threads(Threads.nthreads())


function benchmark_ms( myfunc, args...;kwargs...)
    elapsed=0.0
    best=100000
    i=0
    numruns = 2#(size(args[1],1) < 1025) ? 20 : 1
    while(elapsed<200.0 || (i<2 &&elapsed<5000.0))
        KernelAbstractions.synchronize(backend)
        start = time_ns()
        for i=1:numruns
            myfunc(args...;kwargs...)
        end
        KernelAbstractions.synchronize(backend)
        endtime = time_ns()
        thisduration=(endtime-start)/1e6
        elapsed+=thisduration
        best = min(thisduration/numruns,best)
        i+=1
    end
    return best
end

function cusvdwithcopy(a)
    agpu=KernelAbstractions.allocate(backend,eltype(a),size(a,1),size(a,2))
    copyto!(agpu,a)
    svdvals!(agpu,  alg=CUDA.CUSOLVER.QRAlgorithm())
    KernelAbstractions.synchronize(backend)
end

elty=Float32
sizes=[64,128,256,512,1024,2048]
timings=ones(4,length(sizes))*1000000
errors=zeros(2,length(sizes))


inputs=[randn(elty,size_i, size_i) for size_i in sizes]
println( "Checking correctness OOC");
for (i,size_i) in enumerate(sizes[1:4])
    aout=OOC_SVD!(copy(inputs[i]), backend,kswitch=4)
    _,aref,_= LinearAlgebra.LAPACK.gesvd!('N','N',copy(inputs[i]))
    KernelAbstractions.synchronize(backend)
    errors[2,i]= norm((aout-aref)./aref)/sqrt(size_i)
end
println( "Benchmarking CUDA only");
for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[3,i] = min( benchmark_ms(cusvdwithcopy,a), timings[3,i])
end

for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[3,i] = min( benchmark_ms(cusvdwithcopy,a), timings[3,i])
end
println( "Benchmarking KA only");
for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[4,i] = min( benchmark_ms(OOC_SVD!,a, backend,kswitch=512), timings[4,i])
end

for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[4,i] = min( benchmark_ms(OOC_SVD!,a, backend,kswitch=512), timings[4,i])
end



inputs=[CUDA.zeros(elty,size_i, size_i) for size_i in sizes]
inputs=[rand!(x) for x in inputs]



println( "Checking correctness GPU only");
for (i,size_i) in enumerate(sizes[1:4])
    aout=mygesvd!(copy(inputs[i]))
    aref= Array(svdvals!(copy(inputs[i]),  alg=CUDA.CUSOLVER.QRAlgorithm()))
    KernelAbstractions.synchronize(backend)
    errors[1,i]= norm((aout-aref)./aref)/sqrt(size_i)
end
println( "Benchmarking CUDA only");
for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[1,i] = min( benchmark_ms(svdvals!, a,  alg=CUDA.CUSOLVER.QRAlgorithm()), timings[1,i])
end

for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[1,i] = min( benchmark_ms(svdvals!, a,  alg=CUDA.CUSOLVER.QRAlgorithm()), timings[1,i])
end
println( "Benchmarking KA only");
for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[2,i] = min( benchmark_ms(mygesvd!,a), timings[2,i])
end

for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[2,i] = min( benchmark_ms(mygesvd!,a), timings[2,i])
end



println("GPU only");
println( " size    RRMSE    time (ms)  cutime(ms) ");
println(" ------  --------  ----------  ---------- ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %8.02f  %8.02f  \n" size_i errors[1,i] timings[2,i] timings[1,i]
end  
println("OOC");
println( " size    RRMSE    time (ms)  cutime(ms) ");
println(" ------  --------  ----------  ---------- ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %8.02f  %8.02f  \n" size_i errors[2,i] timings[4,i] timings[3,i]
end  

