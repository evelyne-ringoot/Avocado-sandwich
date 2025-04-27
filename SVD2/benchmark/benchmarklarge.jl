timings=ones(3,length(sizes))

println( "benchmark large sizes vendor");
for (i,size_i) in enumerate(sizes[1,2])
    timings[1,i] = benchmark_ms(size_i,vendorsvd!)
end

println( "benchmark large sizes KA");
for (i,size_i) in enumerate(sizes)
    timings[2,i] = benchmark_ms_large(size_i,mygesvd!)
end


println( "benchmark large sizes brd");
for (i,size_i) in enumerate(sizes)
    timings[3,i] = benchmark_ms_large(size_i,mygbbrd!)
end



println("GPU only SVD");
println( " size     time (ms)      cutime(ms)    brdtime(ms) ");
println(" ------  ----------  ----------    ---------- ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d     %8.02f      %8.02f  %8.02f  \n" size_i timings[2,i] timings[1,i] timings[3,i] 
end  