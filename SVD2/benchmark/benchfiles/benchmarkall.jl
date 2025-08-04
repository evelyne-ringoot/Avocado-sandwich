timings=ones(3,length(sizes))*1000000000
errors=zeros(2,length(sizes))
println( "Checking correctness GPU only")
for (i,size_i) in enumerate(sizes)
    input=arty(randwellbehaved(size_i,elty,svdtestscaling))
    aout=mygesvd!(copy(input))
    KernelAbstractions.synchronize(backend)
    aref=vecty(svdtestscaling(size_i))
    errors[1,i]= (sqrt(sum((aref-aout).^2)))/ (sqrt(sum((aref).^2)))

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

println( "warmup brd only");
for (i,size_i) in enumerate(sizes)
    timings[3,i] = min( benchmark_ms(size_i, mygbbrd!), timings[3,i])
end

println( "run brd only");
for (i,size_i) in enumerate(sizes)
    timings[3,i] = min( benchmark_ms(size_i,mygbbrd!), timings[3,i])
end

println("GPU only SVD");
println( " size    RRMSE    time (ms)  cutime(ms)  brd time(ms)");
println(" ------  --------  ----------  ----------  ---------- ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %8.02f  %8.02f   %8.02f \n" size_i errors[1,i] timings[2,i] timings[1,i] timings[3,i]
end  
