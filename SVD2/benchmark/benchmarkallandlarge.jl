sizes=[64,128,256,512,1024,2048, 4096]
timings=ones(length(sizes))*1000000000
errors=zeros(length(sizes))

output=(zeros(Int,length(sizes),6))
output[:,1].=TILESIZE
output[:,2].=TILESIZEMUL
output[:,3].=QRSPLIT
output[:,4].= (elty==Float32 ? 2 : (elty==Float64 ? 3 : 1) )
output[:,5].= sizes



println( "Checking correctness GPU only")
for (i,size_i) in enumerate(sizes)
    input=arty(randwellbehaved(size_i,elty))
    aout=mygesvd!(copy(input))
    KernelAbstractions.synchronize(backend)
    aref=(svdtestscaling(size_i,1,false))
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
output[:,6].=round.(Int,timings.*10)
writedlm( "results"*string(elty)*"_"* string(TILESIZE)* "_"* string(TILESIZEMUL)* "_"* string(QRSPLIT)* "_small.csv",  output, ',')

sizes=1024 .*[8,16,32]
timings=ones(length(sizes))*1000000000
output=(zeros(Int,length(sizes),6))
output[:,1].=TILESIZE
output[:,2].=TILESIZEMUL
output[:,3].=QRSPLIT
output[:,4].= (elty==Float32 ? 2 : (elty==Float64 ? 3 : 1) )
output[:,5].= sizes

try
    for (i,size_i) in enumerate(sizes)
        timing = benchmark_ms_large(size_i,mygesvd!)
        @printf " %4d   %8.02e    %8.02f   \n" size_i 0.0 timing
        output[i,6].=round.(Int,timings.*10)
        writedlm( "results"*string(elty)*"_"* string(TILESIZE)* "_"* string(TILESIZEMUL)* "_"* string(QRSPLIT)* "_large.csv",  output, ',')
    end
catch e
    println("did not run all sizes")
end



