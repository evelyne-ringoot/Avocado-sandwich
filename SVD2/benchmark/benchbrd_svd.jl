using KernelAbstractions,GPUArrays, Random, LinearAlgebra, Printf, ArgParse, Dates

include("parseinputargs.jl")
include("../src/brdgpunew.jl")
include("benchfuncs.jl")
BLAS.set_num_threads(Threads.nthreads())
include("../src/KAfuncs.jl")
include("../src/qr_kernels.jl")
include("../src/tiledalgos.jl")
mygbbrd!(A)=mygbbrd_packed!(A)

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


sizes=[128,256,512,1024,2048,4096,8192 ]
timings=ones(3,length(sizes))*1000000000
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




    print( "becnhmark blockdiag  at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
    for (i,size_i) in enumerate(sizes)
        timings[1,i] = min( benchmark_ms(size_i,myblockdiag!), timings[1,i])
    end
    print("done at : ")
    println(Dates.format(now(), "HH:MM:SS")  )

    print( "becnhmark brd  at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
    for (i,size_i) in enumerate(sizes)
        timings[2,i] = min( benchmark_ms(size_i,mygbbrd!), timings[2,i])
    end
    print("done at : ")
    println(Dates.format(now(), "HH:MM:SS")  )

    print( "becnhmark svd at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
    for (i,size_i) in enumerate(sizes)
        timings[3,i] = min( benchmark_ms(size_i,mygesvd!), timings[3,i])
    end
    print("done at : ")
    println(Dates.format(now(), "HH:MM:SS")  )


println( " size    RRMSE    blockdiag (s)    brd (s)       svd (s)  ");
println(" ------  --------  ----------    ----------    ----------   ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %10.04f  %10.04f  %10.04f \n" size_i errors[i] timings[1,i]/1000 timings[2,i]/1000 timings[3,i]/1000  
end  

sizes=[8192,16384 ]
timings=ones(3,length(sizes))*1000000000


try
    print( "becnhmark blockdiag large at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
    for (i,size_i) in enumerate(sizes)
        timings[1,i] = min( benchmark_ms_large(size_i,myblockdiag!), timings[1,i])
    end
    print("done at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
    print( "becnhmark brd large  at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
    for (i,size_i) in enumerate(sizes)
        timings[2,i] = min( benchmark_ms_large(size_i,mygbbrd!), timings[2,i])
    end
    print("done at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
    print( "becnhmark svd large at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
    for (i,size_i) in enumerate(sizes)
        timings[3,i] = min( benchmark_ms_large(size_i,mygesvd!), timings[3,i])
    end
    print("done at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
catch e
    println("did not finish all sizes")
end


println( " size   blockdiag (s)    brd (s)       svd (s)  ");
println(" ------  ----------    ----------    ----------   ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d     %10.04f  %10.04f  %10.04f \n" size_i timings[1,i]/1000 timings[2,i]/1000 timings[3,i]/1000  
end  