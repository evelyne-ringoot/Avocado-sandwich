using KernelAbstractions,GPUArrays, Random, LinearAlgebra, Printf, ArgParse, Dates, DelimitedFiles

include("parseinputargs.jl")
include("../src/brdgpunew.jl")
include("benchfuncs.jl")

print("starting code : ")
println(Dates.format(now(), "HH:MM:SS")  )
size_i=4BW
b=tril(triu(randn!(KernelAbstractions.zeros(backend,elty,size_i,size_i))),BW)
print("matrix generated, compiling at : ")
println(Dates.format(now(), "HH:MM:SS")  )
mygbbrd!(b)
mygbbrd_packed!(b)
mygbbrd_packed_nocomm!(b)
print("first compile done at : ")
println(Dates.format(now(), "HH:MM:SS")  )

#  bw, maxblocks, brdwidth,brdmul,size,error,timing, type
output=(zeros(Int,12,8))
output[:,1].=BW
output[:,2].=MAXBLOCKS
output[:,3].=BRDWIDTH
output[:,4].=BRDMULSIZE
output[:,8].= (elty==Float32 ? 2 : (elty==Float64 ? 3 : 1) )
output[:,5].= (2 .^(7:18))





sizes=[128,256,512,1024,2048,4096 ]
timings=ones(length(sizes))*1000000000
errors=zeros(length(sizes))

print( "Checking correctness at : ")
println(Dates.format(now(), "HH:MM:SS")  )
for (i,size_i) in enumerate(sizes)
    a=tril(triu(randn!(KernelAbstractions.zeros(backend,elty,size_i,size_i))),BW)
    aref=svdvals(Float64.(a), alg=CUDA.CUSOLVER.QRAlgorithm())
    mygbbrd!(a)
    KernelAbstractions.synchronize(backend)
    aout=svdvals(Float64.(a), alg=CUDA.CUSOLVER.QRAlgorithm())
    errors[i]=norm(aref-aout)/norm(aout)
end
print("done accuracy : ")
println(Dates.format(now(), "HH:MM:SS")  )
output[1:6,6].=round.(Int,errors./eps(elty).*1000)
#=
print( "becnhmark KA at : ")
println(Dates.format(now(), "HH:MM:SS")  )
for (i,size_i) in enumerate(sizes)
    timings[i] = min( benchmark_ms(size_i,mygbbrd!), timings[i])
end
print("done at : ")
println(Dates.format(now(), "HH:MM:SS")  )

println("BRD");
println( " size    RRMSE    time (ms)  ");
println(" ------  --------  ----------   ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %8.02f \n" size_i errors[i] timings[i] 
end  


timings=ones(length(sizes))*1000000000
errors=zeros(length(sizes))

print( "Checking correctness packed at : ")
println(Dates.format(now(), "HH:MM:SS")  )
for (i,size_i) in enumerate(sizes)
    a=tril(triu(randn!(KernelAbstractions.zeros(backend,elty,size_i,size_i))),BW)
    aref=svdvals(Float64.(a), alg=CUDA.CUSOLVER.QRAlgorithm())
    mygbbrd_packed!(a)
    KernelAbstractions.synchronize(backend)
    aout=svdvals(Float64.(a), alg=CUDA.CUSOLVER.QRAlgorithm())
    errors[i]=norm(aref-aout)/norm(aout)
end
print("done accuracy packed : ")
println(Dates.format(now(), "HH:MM:SS")  )

print( "becnhmark KA packed (incl communication)  at : ")
println(Dates.format(now(), "HH:MM:SS")  )
for (i,size_i) in enumerate(sizes)
    timings[i] = min( benchmark_ms(size_i,mygbbrd_packed!), timings[i])
end
print("done packed at : ")
println(Dates.format(now(), "HH:MM:SS")  )

println("BRD packed incl communication");
println( " size    RRMSE    time (s)  ");
println(" ------  --------  ----------   ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %10.04f \n" size_i errors[i] timings[i]/1000 
end  


timings=ones(length(sizes))*1000000000
=#
print( "becnhmark KA packed (excl communication) at : ")
println(Dates.format(now(), "HH:MM:SS")  )
for (i,size_i) in enumerate(sizes)
    timings[i] = min( benchmark_ms(size_i,mygbbrd_packed_nocomm!, 3BW+1), timings[i])
end
print("done packed at : ")
println(Dates.format(now(), "HH:MM:SS")  )

println("BRD packed excl communication");
println( " size    RRMSE    time (s)  ");
println(" ------  --------  ----------   ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %10.04f \n" size_i errors[i] timings[i]/1000 
end  
output[1:6,7].=round.(Int,timings).*10
sizes=[1,2,4,8,16,32].*8192 
timings=ones(length(sizes))*1000000000


print( "becnhmarking large sizes (excl communication) at : ")
println(Dates.format(now(), "HH:MM:SS")  )
try 
    for (i,size_i) in enumerate(sizes)
        timings[i] = min( benchmark_ms_large(size_i,mygbbrd_packed_nocomm!, 3BW+1), timings[i])
    end
    print("done packed at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
catch e 
    println("!!! did not finish all sizes")
finally
    output[7:12,7].=round.(Int,timings).*10
    writedlm( "BRDresults"*string(elty)*"_"* string(BW)* "_"* string(MAXBLOCKS)* "_"* string(BRDWIDTH)* "_"* string(BRDMULSIZE)* "_" *".csv",  output, ',')
    println("BRD packed large excl communication");
    println( " size   time (s)  ");
    println(" ------   ----------   ");
    for (i,size_i) in enumerate(sizes)
        @printf " %4d   %10.04f \n" size_i  timings[i]/1000 
    end  

end



