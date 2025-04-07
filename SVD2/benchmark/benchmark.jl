using KernelAbstractions,GPUArrays, CUDA,Random, LinearAlgebra, Printf, BenchmarkTools

backend=KernelAbstractions.get_backend(CUDA.zeros(2))
const TILESIZE = 64
const QRSPLIT = 4
include("../src/cusol_funcs.jl")
include("../src/KAfuncs.jl")
include("../src/qr_kernels.jl")
include("../src/datacomms.jl")
include("../src/tiledalgos.jl")

function benchmark_ms( myfunc, args...;kwargs...)
    elapsed=0.0
    best=100000
    i=0
    numruns = 2#(size(args[1],1) < 1025) ? 20 : 1
    while(elapsed<200.0 || (i<2 &&elapsed<5000.0))
        KernelAbstractions.synchronize(backend)
        start = time_ns()
        for i=1:numruns
            myfunc(args...;kwargs...)
        end
        KernelAbstractions.synchronize(backend)
        endtime = time_ns()
        thisduration=(endtime-start)/1e6
        elapsed+=thisduration
        best = min(thisduration/numruns,best)
        i+=1
    end
    return best
end

function cusvdwithcopy(a)
    agpu=KernelAbstractions.allocate(backend,eltype(a),size(a,1),size(a,2))
    copyto!(agpu,a)
    svdvals!(agpu,  alg=CUDA.CUSOLVER.QRAlgorithm())
    KernelAbstractions.synchronize(backend)
end

elty=Float32
sizes=[64,128,256,512,1024,2048,4096,8192,16384]
timings=ones(4,length(sizes))*1000000
errors=zeros(2,length(sizes))


inputs=[randn(Float32,size_i, size_i) for size_i in sizes]
println( "Checking correctness OOC");
for (i,size_i) in enumerate(sizes[1:6])
    aout=OOC_SVD!(copy(inputs[i]), backend,kswitch=10, tilesinmem=14)
    _,aref,_= LinearAlgebra.LAPACK.gesvd!('N','N',copy(inputs[i]))
    KernelAbstractions.synchronize(backend)
    errors[2,i]= norm((aout-aref)./aref)/sqrt(size_i)
end
println( "Benchmarking CUDA only");
for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[3,i] = min( benchmark_ms(cusvdwithcopy,a), timings[3,i])
end

for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[3,i] = min( benchmark_ms(cusvdwithcopy,a), timings[3,i])
end
println( "Benchmarking KA only");
for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[4,i] = min( benchmark_ms(OOC_SVD!,a, backend), timings[4,i])
end

for (i,size_i) in enumerate(sizes)
    a=inputs[i]
    timings[4,i] = min( benchmark_ms(OOC_SVD!,a, backend), timings[4,i])
end



inputs=[CUDA.randn(Float32,size_i, size_i) for size_i in sizes]
println( "Checking correctness GPU only");
for (i,size_i) in enumerate(sizes[1:6])
    aout=mygesvd!(copy(inputs[i]))
    aref= Array(svdvals!(copy(inputs[i]),  alg=CUDA.CUSOLVER.QRAlgorithm()))
    KernelAbstractions.synchronize(backend)
    errors[1,i]= norm((aout-aref)./aref)/sqrt(size_i)
end
println( "Benchmarking CUDA only");
for (i,size_i) in enumerate(sizessizes[1:6])
    a=inputs[i]
    timings[1,i] = min( benchmark_ms(svdvals!, a,  alg=CUDA.CUSOLVER.QRAlgorithm()), timings[1,i])
end

for (i,size_i) in enumerate(sizes[1:8])
    a=inputs[i]
    timings[1,i] = min( benchmark_ms(svdvals!, a,  alg=CUDA.CUSOLVER.QRAlgorithm()), timings[1,i])
end
println( "Benchmarking KA only");
for (i,size_i) in enumerate(sizes[1:8])
    a=inputs[i]
    timings[2,i] = min( benchmark_ms(mygesvd!,a), timings[2,i])
end

for (i,size_i) in enumerate(sizes[1:8])
    a=inputs[i]
    timings[2,i] = min( benchmark_ms(mygesvd!,a), timings[2,i])
end



println("GPU only");
println( " size    RRMSE    time (ms)  cutime(ms) ");
println(" ------  --------  ----------  ---------- ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %8.02f  %8.02f  \n" size_i errors[1,i] timings[2,i] timings[1,i]
end  
println("OOC");
println( " size    RRMSE    time (ms)  cutime(ms) ");
println(" ------  --------  ----------  ---------- ");
for (i,size_i) in enumerate(sizes)
    @printf " %4d   %8.02e    %8.02f  %8.02f  \n" size_i errors[2,i] timings[4,i] timings[3,i]
end  

#=
timings=[    1.38     3.92      0.41
3.96     8.0       4.57
13.5     16.84     35.6
47.53    49.86    287.58
165.58   108.01   3185.4
689.62   284.77  26037.0
3841.68   891.77      0.0
27813.8   6717.43      0.0
205247       0         0]

xaxis=sizes
xaxist=[string.(xaxis[1:5]);"2k";"4k"; "8k";"16k";"32k"]
yaxis=[1, 10, 100, 1e3,1e4, 6e4]
yaxist=["1 ms","10ms","0.1s","1s", "10s", "1min"]

 plot(xaxis[1:8], timings[1:8,2], labels= "", lw=2, markershape=:circle, markerstrokewidth=0, markersize=3)
 plot!(xaxis[1:6], timings[1:6,3], labels="",lw=2, markershape=:circle, markerstrokewidth=0, markersize=3 )
 plot!(xaxis[1:9],timings[:,1], label= "" ,lw=2, markershape=:circle, markerstrokewidth=0, markersize=3)
 plot!(xaxis=:log2,  yaxis=:log10, xticks=(xaxis, xaxist), yticks=(yaxis,yaxist),xlabel= "matrix size nxn", ylabel= "Execution time", dpi=1000)
 plot!( size = (600, 300))
 plot!([32000,32000],[1,1], label="")
 savefig("benchmark.png")
 =#