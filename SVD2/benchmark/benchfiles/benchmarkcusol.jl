

sizes=[64,128,256,512,1024,2048]

cugesvd!(a)=CUDA.CUSOLVER.gesvd!('A','A',a)
cugesvdvals!(a)=CUDA.CUSOLVER.gesvd!('N','N',a)
functionstobench=[CUDA.CUSOLVER.geqrf!,cugesvd!,mygeqrf!, myblockdiag!, cugesvdvals!]
timings=ones(length(functionstobench),length(sizes))*1000000000

for (j,f) in enumerate(functionstobench)
    println( "warump "*string(f))
    for (i,size_i) in enumerate(sizes)
        timings[j,i] = min( benchmark_ms(size_i, f), timings[j,i])
    end
end



    for (j,f) in enumerate(functionstobench)
        println( "run "*string(f))
        for (i,size_i) in enumerate(sizes)
            timings[j,i] = min( benchmark_ms(size_i, f), timings[j,i])
        end
    end
println( " size   cuqr(ms)    cusvd (ms)   kaqr(ms)   kasvd (ms)  cuvals(ms) curatio    karatio    svd/vals");
println(" ------  --------   ----------  ----------   --------    ----------  ----------  ----------   ----------   ");

    for (i,size_i) in enumerate(sizes)
        @printf " %4d   " size_i
        for (j,f) in enumerate(functionstobench)
            @printf "%8.02f   "  timings[j,i] 
        end
        @printf "%8.02f   %8.02f   %8.02f  \n" timings[2,i]/timings[1,i] timings[4,i]/timings[3,i] timings[5,i]/timings[2,i] 
    end  
    flush(stdout)



    sizes=[4,8]*1024

try
    for (i,size_i) in enumerate(sizes)
        @printf " %4d   " size_i
        for (j,f) in enumerate(functionstobench)
            timing = benchmark_ms_large(size_i,f)
            @printf "%8.02f   "  timing
        end
        @printf "%8.02f   %8.02f   %8.02f  \n" timings[2,i]/timings[1,i] timings[4,i]/timings[3,i] timings[5,i]/timings[2,i]   
        flush(stdout)
    end
catch e
    println("did not run all sizes")
end





