using KernelAbstractions,CUDA, LinearAlgebra, Printf, BenchmarkTools, GPUArrays
using KernelAbstractions.Extras: @unroll
backend=KernelAbstractions.get_backend(CUDA.zeros(2))
elty=Float32
const NUMPARY = 8
const NUMPARX = 8
const numruns = 20

function benchmark_ms( myfunc, args...;kwargs...)
    elapsed=0.0
    best=100000
    i=0
    while(elapsed<200.0 || i<2)
        CUDA.synchronize()
        start = time_ns()
        for i=1:numruns
            myfunc(args...;kwargs...)
        end
        CUDA.synchronize()
        endtime = time_ns()
        thisduration=(endtime-start)/1e6
        elapsed+=thisduration
        best = min(thisduration/numruns,best)
        i+=1
    end
    return best
end

@kernel cpu=false  inbounds=true unsafe_indices=false function naivematmulkernel!(size_i,inputa,inputb,output) 
    
   
    g = ( (@index(Group, NTuple)[1]-Int32(1)) * Int32(NUMPARX) +  @index(Local,NTuple)[1] -Int32(1))
    h = ( (@index(Group, NTuple)[2]-Int32(1)) * Int32(NUMPARY) +  @index(Local,NTuple)[2] -Int32(1))

    if (g<size_i && h<size_i)
        res=zero(eltype(output))
        k=0
        while k<size_i
            res = res + inputa[g+1,k+1] * inputb[k+1,h+1]
            k+=1
        end
        output[g+1,h+1]=res
    end
    return
end

@inline naivematmul!(size_in,inputa,inputb,output) =  naivematmulkernel!(backend, (NUMPARX,NUMPARY))(size_in,inputa,inputb,output, ndrange=(max(NUMPARX,size_in),max(NUMPARY,size_in))) 

function cunaivematmulkernel!(size_i, inputa,inputb,output) 
    g = ( ((blockIdx().x) - Int32(1)) *Int32(NUMPARX)+ (threadIdx().x-Int32(1)) ) 
    h = ( ((blockIdx().y) - Int32(1)) *Int32(NUMPARY)+ (threadIdx().y -Int32(1)) ) 
    if (g<size_i && h<size_i)
        res=zero(eltype(output))
        k=0
        while k<size_i
            @inbounds res = res + inputa[g+1,k+1] * inputb[k+1,h+1]
            k+=1
        end
        @inbounds output[g+1,h+1]=res
    end
    return
end


@inline cunaivematmul!(size_in, inputa,inputb,output)  =  @cuda threads=(NUMPARX,NUMPARY) blocks=(cld(size_in, NUMPARX),cld(size_in, NUMPARY) )  cunaivematmulkernel!(size_in, inputa,inputb,output) 

sizes=[32,64,128,256,512,1024,2048]
timings=ones(3,length(sizes))*100
errors=zeros(2,length(sizes))
inputsa=[CUDA.randn(size_i, size_i) for size_i in sizes]
inputsb=[CUDA.randn(size_i, size_i) for size_i in sizes]
outputsc=[CUDA.zeros(size_i, size_i) for size_i in sizes]


println( "verifying correctness");
for (i,size_i) in enumerate(sizes)
    c = inputsa[i]*inputsb[i]
    cout = CUDA.zeros(size_i,size_i)
    CUDA.@sync naivematmul!(size_i,inputsa[i],inputsb[i],cout)
    errors[1,i] = norm(cout.-c)/(size_i)
    CUDA.synchronize()
    cout = CUDA.zeros(size_i,size_i)
    CUDA.@sync cunaivematmul!(size_i,inputsa[i],inputsb[i],cout)
    errors[2,i] = norm(cout.-c)/(size_i)
    CUDA.synchronize()
       
end  

println( "benchmarking");

for (i,size_i) in enumerate(sizes)
    a=inputsa[i]
    b=inputsb[i]
    c=outputsc[i]
    
    timings[1,i] = min( benchmark_ms(mul!,c,a,b), timings[1,i])
end

for (i,size_i) in enumerate(sizes)
    a=inputsa[i]
    b=inputsb[i]
    c=outputsc[i]
    
    timings[1,i] = min( benchmark_ms(mul!,c,a,b), timings[1,i])
end

for (i,size_i) in enumerate(sizes)
    a=inputsa[i]
    b=inputsb[i]
    c=outputsc[i]

    timings[2,i] =  min(benchmark_ms(naivematmul!,size_i,a,b,c), timings[2,i])    
end
for (i,size_i) in enumerate(sizes)
    a=inputsa[i]
    b=inputsb[i]
    c=outputsc[i]

    timings[2,i] =  min(benchmark_ms(naivematmul!,size_i,a,b,c), timings[2,i])    
end

for (i,size_i) in enumerate(sizes)
    a=inputsa[i]
    b=inputsb[i]
    c=outputsc[i]

    timings[3,i] = min(benchmark_ms(cunaivematmul!,size_i,a,b,c) , timings[3,i])
end
for (i,size_i) in enumerate(sizes)
    a=inputsa[i]
    b=inputsb[i]
    c=outputsc[i]

    timings[3,i] = min(benchmark_ms(cunaivematmul!,size_i,a,b,c) , timings[3,i])
end


println( " size    RRMSE    RRMSE   KA time (ms) cu time (ms)  ju time (ms)");
println(" ------  -------- --------  ---------    ---------    ---------  ");
for (i,size_i) in enumerate(sizes)
    @printf " %5d   %8.02e  %8.02e  %8.03f  %8.03f  %8.03f \n" size_i errors[1,i] errors[2,i] timings[2,i] timings[3,i]  timings[1,i]
end  

