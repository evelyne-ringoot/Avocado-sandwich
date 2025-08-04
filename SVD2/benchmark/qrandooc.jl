
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