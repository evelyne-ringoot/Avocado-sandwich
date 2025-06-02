using KernelAbstractions,GPUArrays, Random, LinearAlgebra, Printf, ArgParse, Dates

include("parseinputargs.jl")
include("../src/brdgpunew.jl")
include("benchfuncs.jl")

print("starting code : ")
println(Dates.format(now(), "HH:MM:SS")  )
sizes=[128,256,512,1024,2048,4096,8192 ]#,,  32768, 65536] # 2048, 4096,8192
timings=ones(length(sizes))*1000000000
errors=zeros(length(sizes))
size_i=4BW
b=tril(triu(randn!(KernelAbstractions.zeros(backend,elty,size_i,size_i))),BW)
print("matrix generated, compiling started at : ")
println(Dates.format(now(), "HH:MM:SS")  )
mygbbrd4!(b)
print("first compile done at : ")
println(Dates.format(now(), "HH:MM:SS")  )

#=
n=size_i
    Int(BW/TILESIZE):-1:1
        A=copy(b)
        mygbbrd4!(A)
        KernelAbstractions.synchronize(backend)
        norm(svdvals(A)-svdvals(b))/norm(svdvals(b))
        1:(n-1)
        k=1
        bwiter=2

    k=1
        brd4!(view(A,k:n,k:n),min(k,1+cld((n-k), (3TILESIZE-1))),bwiter)
                for k in 1:(n-1)
            brd4!(view(A,k:n,k:n),min(k,1+cld((n-k), (3TILESIZE*bwiter-1))),bwiter)
        end
=#


print( "Checking correctness at : ")
println(Dates.format(now(), "HH:MM:SS")  )
for (i,size_i) in enumerate(sizes)
    a=tril(triu(randn!(KernelAbstractions.zeros(backend,elty,size_i,size_i))),BW)
    aref=svdvals(Float64.(a), alg=CUDA.CUSOLVER.QRAlgorithm())
    mygbbrd4!(a)
    KernelAbstractions.synchronize(backend)
    aout=svdvals(Float64.(a), alg=CUDA.CUSOLVER.QRAlgorithm())
    errors[i]=norm(aref-aout)/norm(aout)
end
print("done accuracy : ")
println(Dates.format(now(), "HH:MM:SS")  )

#=
println( "warump KA only");
for (i,size_i) in enumerate(sizes)
    timings[i] = min( benchmark_ms(size_i,mygbbrd3!), timings[i])
end
=#
print( "becnhmark KA at : ")
println(Dates.format(now(), "HH:MM:SS")  )
for (i,size_i) in enumerate(sizes)
    timings[i] = min( benchmark_ms_large(size_i,mygbbrd4!), timings[i])
end
print("done at : ")
println(Dates.format(now(), "HH:MM:SS")  )

println("BRD");
println( " size    RRMSE    time (ms)  ");
println(" ------  --------  ----------   ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %8.02f \n" size_i errors[i] timings[i] 
end  
