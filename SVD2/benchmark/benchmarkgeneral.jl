timings=ones(2,length(sizes))*1000000000
errors=zeros(2,length(sizes))
println( "Checking correctness GPU only")
for (i,size_i) in enumerate(sizes)
    input=rand!(KernelAbstractions.zeros(backend,elty,size_i, size_i))
    aout=mygesvd!(copy(input))
    aref=vendorsvd!(copy(input))
    KernelAbstractions.synchronize(backend)
    if (isnothing(aref))
        aref= Array(aref) #Array because mygesvd returns CPU Array
        errors[1,i]= norm((aout-aref)./aref)/sqrt(size_i)
    end
end

println( "warmup vendor only");
for (i,size_i) in enumerate(sizes)
    timings[1,i] = min( benchmark_ms(size_i,vendorsvd!), timings[1,i])
end

println( "warump KA only");
for (i,size_i) in enumerate(sizes)
    timings[2,i] = min( benchmark_ms(size_i,mygesvd!), timings[2,i])
end

println( "run vendor only");
for (i,size_i) in enumerate(sizes)
    timings[1,i] = min( benchmark_ms(size_i,vendorsvd!), timings[1,i])
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
