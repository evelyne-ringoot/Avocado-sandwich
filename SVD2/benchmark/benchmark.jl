using KernelAbstractions,GPUArrays, CUDA,Random, LinearAlgebra, Printf, BenchmarkTools

KernelAbstractions.get_backend(CUDA.zeros(1))
BLAS.set_num_threads(Threads.nthreads())
const backend=CUDABackend(false, false, true)


elty=Float32
sizes=[64,128,256,512,1024,2048, 4096,8192]
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
    numruns = (size_i < 1025) ? 12 : 2
    while(elapsed<200.0 || (i<2 &&elapsed<5000.0))
        KernelAbstractions.synchronize(backend)
        start = time_ns()
        for i=1:numruns
            myfunc(a;kwargs...)
        end
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


println( "Checking correctness GPU only")
for (i,size_i) in enumerate(sizes[1:4])
    input=CUDA.randn(elty,size_i, size_i)
    aout=mygesvd!(input)
    aref= Array(svdvals!(copy(input),  alg=CUDA.CUSOLVER.QRAlgorithm()))
    KernelAbstractions.synchronize(backend)
    errors[1,i]= norm((aout-aref)./aref)/sqrt(size_i)
end

println( "warmup CUDA only");
for (i,size_i) in enumerate(sizes)
    timings[1,i] = min( benchmark_ms(size_i,svdvals!,  alg=CUDA.CUSOLVER.QRAlgorithm()), timings[1,i])
end

println( "warump KA only");
for (i,size_i) in enumerate(sizes)
    timings[2,i] = min( benchmark_ms(size_i,mygesvd!), timings[2,i])
end

println( "run CUDA only");
for (i,size_i) in enumerate(sizes)
    timings[1,i] = min( benchmark_ms(size_i,svdvals!,  alg=CUDA.CUSOLVER.QRAlgorithm()), timings[1,i])
end

println( "run KA only");
for (i,size_i) in enumerate(sizes)
    timings[2,i] = min( benchmark_ms(size_i,mygesvd!), timings[2,i])
end

println("GPU only SVD");
println( " size    RRMSE    time (ms)  cutime(ms) ");
println(" ------  --------  ----------  ---------- ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %8.02f  %8.02f  \n" size_i errors[1,i] timings[2,i] timings[1,i]
end  
#=

function cusvdwithcopy(a)
    agpu=KernelAbstractions.allocate(backend,eltype(a),size(a,1),size(a,2))
    copyto!(agpu,a)
    svdvals!(agpu,  alg=CUDA.CUSOLVER.QRAlgorithm())
    KernelAbstractions.synchronize(backend)
end
inputs=[randn(elty,size_i, size_i) for size_i in sizes]
println( "Checking correctness OOC");
for (i,size_i) in enumerate(sizes[1:4])
    aout=OOC_SVD!(copy(inputs[i]), kswitch=4)
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
    timings[4,i] = min( benchmark_ms(OOC_SVD!,a, kswitch=512), timings[4,i])
end

for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[4,i] = min( benchmark_ms(OOC_SVD!,a, kswitch=512), timings[4,i])
end

println( "Checking correctness GPU only")
for (i,size_i) in enumerate(sizes[1:4])
    aout=mygeqrf!(copy(inputs[i]))
    aref= CUSOLVER.geqrf!(copy(inputs[i]))[1]
    KernelAbstractions.synchronize(backend)
    errors[1,i]= norm(triu!(aout-aref)./aref)/(size_i)
end
println( "Benchmarking CUDA only");
for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[1,i] = min( benchmark_ms(CUSOLVER.geqrf!, a), timings[1,i])
end

for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[1,i] = min( benchmark_ms(CUSOLVER.geqrf!, a), timings[1,i])
end
println( "Benchmarking KA only");
for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[2,i] = min( benchmark_ms(mygeqrf!,a), timings[2,i])
end

for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[2,i] = min( benchmark_ms(mygeqrf!,a), timings[2,i])
end

println("OOC");
println( " size    RRMSE    time (ms)  cutime(ms) ");
println(" ------  --------  ----------  ---------- ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %8.02f  %8.02f  \n" size_i errors[2,i] timings[4,i] timings[3,i]
end  
=#

#=
@inline backendstream() = CUDA.stream()
@inline setstream!(stream::CuStream) = CUDA.stream!(stream)
@inline function event(stream::CuStream) 
    e=CuEvent()
    record(e, stream)
    return e
end
const streamtype = CuStream
const eventtype = CuEvent
@inline synchronize(e::eventtype)=CUDA.synchronize(e)

#@inline backendstream() = AMDGPU.HIPStream()
#@inline setstream!(s::HIPStream) = AMDGPU.stream!(stream)
#@inline event(s::HIPStream) =  HIPEvent(stream::HIPStream)
=#