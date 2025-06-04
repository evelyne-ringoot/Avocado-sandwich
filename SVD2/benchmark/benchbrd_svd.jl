using KernelAbstractions,GPUArrays, Random, LinearAlgebra, Printf, ArgParse, Dates

include("parseinputargs.jl")
include("../src/brdgpunew.jl")
include("benchfuncs.jl")
BLAS.set_num_threads(Threads.nthreads())
include("../src/KAfuncs.jl")
include("../src/qr_kernels.jl")
include("../src/tiledalgos.jl")

print("starting code : ")
println(Dates.format(now(), "HH:MM:SS")  )
size_i=4BW
b=tril(triu(randn!(KernelAbstractions.zeros(backend,elty,size_i,size_i))),BW)
print("matrix generated, compiling at : ")
println(Dates.format(now(), "HH:MM:SS")  )
mygbbrd!(b)
randn!(b)
mygesvd!(b)
print("first compile done at : ")
println(Dates.format(now(), "HH:MM:SS")  )


sizes=[128,256,512,1024,2048,4096,8192]#,16384 ]
timings=ones(length(sizes))*1000000000
errors=zeros(length(sizes))
print( "Checking correctness at : ")
println(Dates.format(now(), "HH:MM:SS")  )
for (i,size_i) in enumerate(sizes)
    input=randn!(KernelAbstractions.zeros(backend,elty,size_i, size_i))
    aout=(mygesvd!(copy(input)))
    aref=vendorsvd!(Float64.(copy(input)))
    KernelAbstractions.synchronize(backend)
    aref= Array(aref) #Array because mygesvd returns CPU Array
    errors[i]= norm((aref-aout))/norm(aref)

end
print("done at : ")
println(Dates.format(now(), "HH:MM:SS")  )


try
    print( "becnhmark KA at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
    for (i,size_i) in enumerate(sizes)
        timings[i] = min( benchmark_ms(size_i,mygesvd!), timings[i])
    end
    print("done at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
catch e
    println("did not finish all sizes")
end

println("SVD")
println( " size    RRMSE    time (ms)  ");
println(" ------  --------  ----------   ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %8.02f \n" size_i errors[i] timings[i] 
end  

