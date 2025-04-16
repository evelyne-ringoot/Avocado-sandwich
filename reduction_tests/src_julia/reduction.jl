using KernelAbstractions,CUDA, LinearAlgebra, Printf, BenchmarkTools, GPUArrays
using KernelAbstractions.Extras: @unroll
backend=KernelAbstractions.get_backend(CUDA.zeros(2))
elty=Float32
const ELPERTHREAD = 128
const NUMTHREADS = 1
const NUMPAR = 32
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

@kernel cpu=false  inbounds=true unsafe_indices=false function reductionkernel!(size_in, input) 
    g = ( (@index(Group, Linear)-Int32(1)) * Int32(NUMPAR) + @index(Local,Linear) -Int32(1))
    if (g<size_in)
        res=input[1,g+1]
        k=1
        while k<ELPERTHREAD
            res = res + input[k+1,g+1]
            k+=1
        end
        input[1,g+1]=res
    end
    return
end

reduction!(size_in, A) = reductionkernel!(backend, (NUMTHREADS*NUMPAR))(size_in, A, ndrange=(NUMTHREADS*cld(size_in,NUMPAR)*NUMPAR)) 

function cureductionkernel!(size_in, input::CuDeviceArray{T}) where T
    g = ( ((blockIdx().x) - Int32(1)) *Int32(NUMPAR)+ (threadIdx().x-Int32(1)) ) 
    if (g<size_in)
        @inbounds res=input[1,g+1]
        k=1
        while k<ELPERTHREAD
            @inbounds res = res + input[k+1,g+1]
            k+=1
        end
        @inbounds input[1,g+1]=res
    end
    return
end

cureduction!(size_in, input::AnyGPUArray{T}) where T =  @cuda threads=NUMTHREADS*NUMPAR blocks=cld(size_in, NUMPAR) cureductionkernel!(size_in, input)

sizes=[32,128,512, 2048,1024*8,1024*32,1024*128 ]
timings=zeros(3,length(sizes))
errors=zeros(2,length(sizes))
inputs=[CUDA.randn(ELPERTHREAD*NUMTHREADS, size_i) for size_i in sizes]


println( "verifying correctness");
for (i,size_i) in enumerate(sizes)
    a=CUDA.randn(ELPERTHREAD*NUMTHREADS, size_i)
    b=sum(a,dims=1)
    acpy=copy(a)
    acpy2=copy(a)
    CUDA.@sync reduction!(size_i,acpy)
    CUDA.@sync cureduction!(size_i,acpy2)
    errors[1,i] = norm((acpy[1,:]')-b)
    errors[2,i] = norm((acpy2[1,:]')-b)
       
end  

println( "benchmarking");
for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[1,i] =  benchmark_ms(sum,a,dims=1)
end  
for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[1,i] =  benchmark_ms(sum,a,dims=1)
end  
for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[2,i] =  benchmark_ms( reduction!,size_i, a)
end  
for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[2,i] =  benchmark_ms( reduction!,size_i, a)
end  
for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[3,i] =  benchmark_ms(cureduction!,size_i,a)
end  
for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[3,i] =  benchmark_ms(cureduction!,size_i,a)
end  

println( " size    RRMSE    RRMSE   KA time (ms) cu time (ms)  ju time (ms)");
println(" ------  -------- --------  ---------    ---------    ---------  ");
for (i,size_i) in enumerate(sizes)
    @printf " %5d   %8.02e  %8.02e  %8.03f  %8.03f  %8.03f \n" size_i errors[1,i] errors[2,i] timings[2,i] timings[3,i]  timings[1,i]
end  