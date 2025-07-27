sizes=[64,128,256,512,1024,2048, 4096]
timings=ones(length(sizes))*1000000000
errors=zeros(length(sizes))
println( "Checking correctness GPU only")
for (i,size_i) in enumerate(sizes)
    input=arty(randwellbehaved(size_i,elty))
    aout=mygesvd!(copy(input))
    KernelAbstractions.synchronize(backend)
    aref=vecty(svdtestscaling(size_i,1,false))
    errors[i]= (sqrt(sum((aref-aout).^2)))/ (sqrt(sum((aref).^2)))
end

println( "warump KA only");
for (i,size_i) in enumerate(sizes)
    timings[i] = min( benchmark_ms(size_i,mygesvd!), timings[i])
end

println( "run KA only");
for (i,size_i) in enumerate(sizes)
    timings[i] = min( benchmark_ms(size_i,mygesvd!), timings[i])
end


println( " size    RRMSE    time (ms)  ");
println(" ------  --------  ----------   ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %8.02f   \n" size_i errors[i] timings[i] 
end  

sizes=1024 .*[8,16,32]

try
    for (i,size_i) in enumerate(sizes)
        timing = benchmark_ms_large(size_i,mygesvd!)
        @printf " %4d   %8.02e    %8.02f   \n" size_i 0.0 timing
    end
catch e
    println("did not run all sizes")
end



