timings=ones(3,length(sizes))

println( "benchmark large sizes vendor");
try
    for (i,size_i) in enumerate(sizes)
        timings[1,i] = benchmark_ms(size_i,vendorsvd!)
    end
catch e
    println("did not run all sizes")
end


println( "benchmark large sizes KA");
try
    for (i,size_i) in enumerate(sizes)
        timings[2,i] = benchmark_ms_large(size_i,mygesvd!)
    end
catch e
    println("did not run all sizes")
end


println( "benchmark large sizes brd");
try 
    for (i,size_i) in enumerate(sizes)
        timings[3,i] = benchmark_ms_large(size_i,mygbbrd!)
    end
catch e
    println("did not run all sizes")
end



println("GPU only SVD");
println( " size     time (ms)      cutime(ms)    brdtime(ms) ");
println(" ------  ----------  ----------    ---------- ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d     %8.02f      %8.02f  %8.02f  \n" size_i timings[2,i] timings[1,i] timings[3,i] 
end  