using KernelAbstractions,GPUArrays, CUDA,Random, LinearAlgebra, Printf, BenchmarkTools

backend=KernelAbstractions.get_backend(CUDA.zeros(2))
const TILESIZE = 64
const QRSPLIT = 8
include("../src/cusol_funcs.jl")
include("../src/KAfuncs.jl")
include("../src/qr_kernels.jl")
include("../src/datacomms.jl")
include("../src/tiledalgos.jl")

#=
elty=Float32
sizes=[64,128,512,1024,2048,4096]


println( " size    RRMSE    time (ms)  vs CUSOLVER(%)  cutime(ms)  ");
println(" ------  --------  --------  ---------------  ----------  ");
for (i,size_i) in enumerate(sizes)
    nbtiles=Int(size_i/TILESIZE)
    A=CUDA.randn( elty,size_i, size_i)
    Tau=CUDA.zeros(nbtiles,size_i)
    Tau2=CUDA.zeros(size_i)
    Tauc=CUDA.zeros(nbtiles,size_i)
    Tau2c=CUDA.zeros(size_i)
    dh = CUSOLVER.dense_handle()
    buffersize= geqrf_buffersize(A)
    buffer=CUDA.zeros(elty,buffersize)
    Acpy=copy(A)
    Acpy2=copy(A)
    
    mygeqrf!(Acpy, Tauc, nbtiles)
    geqrf!(Acpy2, Tau2c, size_i, size_i, size_i,dh,buffer, buffersize)
    CUDA.synchronize()

    mymatch = norm(triu!(abs.(Acpy) - abs.(Acpy2)))/norm(triu(abs.(Acpy2)))/sqrt(size_i)
    t_cusolver = @belapsed (CUDA.@sync geqrf!($A, $Tau2, $size_i, $size_i, $size_i,$dh,$buffer, $buffersize)) gctrial=true evals=1 gcsample=false samples=1e5 seconds=2
    t_KA = @belapsed (CUDA.@sync mygeqrf!($A, $Tau, $nbtiles)) gctrial=true evals=1 gcsample=false samples=1e5 seconds=2
    t_cusolver = @belapsed (CUDA.@sync geqrf!($A, $Tau2, $size_i, $size_i, $size_i,$dh,$buffer, $buffersize)) gctrial=true evals=1 gcsample=false samples=1e5 seconds=2
    t_KA = @belapsed (CUDA.@sync mygeqrf!($A, $Tau, $nbtiles)) gctrial=true evals=1 gcsample=false samples=1e5 seconds=2
    @printf " %4d   %8.02e    %7.02f  %8.02f %% %10.02f \n" size_i mymatch t_KA*1000  t_cusolver/t_KA*100 1000*t_cusolver 
 
end  

println( " size    RRMSE    time (ms)  vs CUSOLVER(%)  cutime(ms)  ");
println(" ------  --------  --------  ---------------  ----------  ");
for (i,size_i) in enumerate(sizes)
    nbtiles=Int(size_i/TILESIZE)
    A=CUDA.randn( elty,size_i, size_i)
    dh = CUSOLVER.dense_handle()
    buffersize= geqrf_buffersize(A)
    buffer=CUDA.zeros(elty,buffersize)
    Acpy=copy(A)
    Acpy2=copy(A)
    
    aout=mygesvd!(Acpy)
    aref= svdvals!(Acpy2, alg=CUDA.CUSOLVER.QRAlgorithm())
    CUDA.synchronize()

    mymatch = norm(aout - Array(aref))/(size_i)
    t_cusolver = @belapsed (CUDA.@sync svdvals!($A, alg=CUDA.CUSOLVER.QRAlgorithm())) gctrial=true evals=1 gcsample=false samples=1e5 seconds=2
    t_KA = @belapsed (CUDA.@sync mygesvd!($A)) evals=1 gcsample=false samples=10 seconds=2
    t_cusolver = @belapsed (CUDA.@sync svdvals!($A, alg=CUDA.CUSOLVER.QRAlgorithm())) gctrial=true evals=1 gcsample=false samples=1e5 seconds=2
    t_KA = @belapsed (CUDA.@sync mygesvd!($A)) gctrial=true evals=1 gcsample=false samples=10 seconds=2
    @printf " %4d   %8.02e    %7.02f  %8.02f %% %10.02f \n" size_i mymatch t_KA*1000  t_cusolver/t_KA*100 1000*t_cusolver 
end  
=#
sizes=[64,128,256,512,1024,2048,4096,8192, 16384,32000]
println( " size    RRMSE    time (ms)  cutime(ms) LAPACK(ms) ");
println(" ------  --------  --------  ----------  ---------- ");
for (i,size_i) in enumerate(sizes)
    nbtiles=Int(size_i/TILESIZE)
    A=randn( Float32,size_i, size_i)
    Acpy=copy(A)
    Acu=CuArray(A)

    mymatch = 0
    t_lapack = 0
    t_cusolver = 0

    if size_i<4000
        aout=OOC_SVD!(copy(A), backend,kswitch=8)
        aref= svdvals!(copy(A))
        CUDA.synchronize()
        mymatch= norm(aout-aref)/size_i
        t_lapack = @belapsed svd($Acpy,alg = LinearAlgebra.QRIteration()) gctrial=true evals=1 gcsample=false samples=1e5 seconds=2
    end
    if size_i < 10000
        t_cusolver = @belapsed (CUDA.@sync svdvals!($Acu, alg=CUDA.CUSOLVER.QRAlgorithm())) gctrial=true evals=1 gcsample=false samples=1e5 seconds=2
    end
    t_KA = @belapsed (CUDA.@sync OOC_SVD!($A, $backend,kswitch=8)) gctrial=true evals=1 gcsample=false samples=100 seconds=2
    t_KA = @belapsed (CUDA.@sync OOC_SVD!($A, $backend,kswitch=8)) gctrial=true evals=1 gcsample=false samples=100 seconds=2
    @printf " %4d   %8.02e    %8.02f  %8.02f %8.02f  \n" size_i mymatch t_KA*1000  1000*t_cusolver t_lapack*1000
end  


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
 