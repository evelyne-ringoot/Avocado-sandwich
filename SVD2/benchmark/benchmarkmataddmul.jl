timings=ones(2,length(sizes))*1000000000

@inline function vendoradd!(a::arty,b::arty,c::arty)  
    a .= b .+ c 
    return a
end

println( "warmup add");

try
    for (i,size_i) in enumerate(sizes)
        timings[1,i] = min( benchmark_ms_muladd(size_i,vendoradd!), timings[1,i])
    end
catch e
    println("did not run all sizes")
end

println( "warump mul");
try 
    for (i,size_i) in enumerate(sizes)
        timings[2,i] = min( benchmark_ms_muladd(size_i,mul!), timings[2,i])
    end
catch e
    println("did not run all sizes")
end

println( "benchmark add");
try 
    for (i,size_i) in enumerate(sizes)
        timings[1,i] = min( benchmark_ms_muladd(size_i,vendoradd!), timings[1,i])
    end
catch e
    println("did not run all sizes")
end


println( "benchmark mul");
try
    for (i,size_i) in enumerate(sizes)
        timings[2,i] = min( benchmark_ms_muladd(size_i,mul!), timings[2,i])
    end
catch e
    println("did not run all sizes")
end


println( " size  time add(ms) time mul(ms) ");
println(" ------  ----------  ---------- ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d    %8.03f  %8.03f  \n" size_i  timings[1,i] timings[2,i]
end  