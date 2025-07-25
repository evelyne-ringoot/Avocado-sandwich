using KernelAbstractions,GPUArrays, Random, LinearAlgebra, Printf, ArgParse, Dates, DelimitedFiles, RandomMatrices

include("parseinputargs.jl")
include("../src/brdgpunew.jl")
include("benchfuncs.jl")
BLAS.set_num_threads(Threads.nthreads())
include("../src/KAfuncs.jl")
include("../src/qr_kernels.jl")
include("../src/tiledalgos.jl")
mygbbrd!(A)=mygbbrd_packed!(A)
vecty=typeof(KernelAbstractions.zeros(backend,elty,2))
svdtestscaling(n) = (11:-10/(n-1):1)

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


#  bw, maxblocks, brdwidth,brdmul,size,error, timingbidiag,timingbrd, timingsvd
output=(zeros(Int,10,11))
output[:,1].=BW
output[:,2].=MAXBLOCKS
output[:,3].=BRDWIDTH
output[:,4].=BRDMULSIZE
output[:,5].=TILESIZE
output[:,6].=TILESIZEMUL
output[:,7].= (2 .^(7:16))


sizes=[128,256,512,1024,2048,4096 ]
timings=ones(3,length(sizes))*1000000000
errors=zeros(length(sizes))

print( "Checking correctness at : ")
println(Dates.format(now(), "HH:MM:SS")  )
for (i,size_i) in enumerate(sizes)
    input=KernelAbstractions.zeros(backend,elty,size_i, size_i)
    copyto!(input, randwellbehaved(size_i,elty,svdtestscaling))
    aout=(mygesvd!(copy(input)))
    aref=vecty(svdtestscaling(size_i))
    KernelAbstractions.synchronize(backend)
    aref= Array(aref) #Array because mygesvd returns CPU Array
    errors[i]= (sqrt(sum((aref-aout).^2)))/ (sqrt(sum((aref).^2)))
    if (isnan(errors[i]))
       errors[i]=10^6*eps(elty)
    end
end
print("done at : ")
println(Dates.format(now(), "HH:MM:SS")  )
output[1:6,8].=round.(Int,errors./eps(elty).*1000)



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

output[1:6,9:11].=round.(Int,timings').*10
sizes=8192 .*[1,2,4,8]
timings=ones(3,length(sizes))*1000000000


try
    print( "becnhmark large at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
    for (i,size_i) in enumerate(sizes)
        timings[1,i] = min( benchmark_ms_large(size_i,myblockdiag!), timings[1,i])
        timings[2,i] = min( benchmark_ms_large(size_i,mygbbrd!), timings[2,i])
        timings[3,i] = min( benchmark_ms_large(size_i,mygesvd!), timings[3,i])
        if (i>2) 
            output[7:10,9:11].=round.(Int,timings').*10
            writedlm( "SVDresults"*string(elty)*"_"* string(BW)* "_"* string(MAXBLOCKS)* "_"* string(BRDWIDTH)* "_"* string(BRDMULSIZE)* "_" * string(TILESIZE) * "_" * string(TILESIZEMUL) * "_" *".csv",  output, ',')
        end
    end
    print("done at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
catch e
    println("did not finish all sizes")
finally
    output[7:10,9:11].=round.(Int,timings').*10
    writedlm( "SVDresults"*string(elty)*"_"* string(BW)* "_"* string(MAXBLOCKS)* "_"* string(BRDWIDTH)* "_"* string(BRDMULSIZE)* "_" * string(TILESIZE) * "_" * string(TILESIZEMUL) * "_" *".csv",  output, ',')
    println( " size   blockdiag (s)    brd (s)       svd (s)  ");
    println(" ------  ----------    ----------    ----------   ");
    for (i,size_i) in enumerate(sizes)
        @printf " %4d     %10.04f  %10.04f  %10.04f \n" size_i timings[1,i]/1000 timings[2,i]/1000 timings[3,i]/1000  
    end  
end


