
if ARGS[4]=="LARGE"
    sizes=[64,128,256]
else
    sizes=[64,128,256,512,1024,2048, 4096]
end

functionstobench=[myblockdiag_qrcalc!,myblockdiag_applyqr!,mygesvd!,myblockdiag_unfused!,banddiagsvd,bidiagsvd]
timings=ones(length(functionstobench),length(sizes))*1000000000

for (j,f) in enumerate(functionstobench)
    println( "warump "*string(f))
    for (i,size_i) in enumerate(sizes)
        timings[j,i] = min( benchmark_ms(size_i, f), timings[j,i])
    end
end

println( " size   qrcalc(ms)   qrapply(ms)   svd(ms)   unfused(ms)  band(ms)   bidiag(ms)");
println(" ------  --------    ----------  ----------  ----------    ----------   ----------     ");

if ARGS[4]!="LARGE"
    for (j,f) in enumerate(functionstobench)
        println( "run "*string(f))
        for (i,size_i) in enumerate(sizes)
            timings[j,i] = min( benchmark_ms(size_i, f), timings[j,i])
        end
    end

    for (i,size_i) in enumerate(sizes)
        @printf " %4d   " size_i
        for (j,f) in enumerate(functionstobench)
            @printf "%8.02f  "  timings[j,i] 
        end
        @printf "\n" 
    end  
    flush(stdout)
end

if ARGS[4]=="LARGE"
    sizes=[16,32].*1024
else
    sizes=[8*1024]
end

try
    for (i,size_i) in enumerate(sizes)
        @printf " %4d   " size_i
        for (j,f) in enumerate(functionstobench)
            timing = benchmark_ms_large(size_i,f)
            @printf "%8.02f  "  timing
        end
        @printf "\n" 
        flush(stdout)
    end
catch e
    println("did not run all sizes")
end





