using KernelAbstractions,CUDA,Random, LinearAlgebra, Printf, BenchmarkTools

n=[512,2048,8192]
tile=[16,32,64,128]
nblocks=n'./tile
nkernels= nblocks.*(nblocks.-1)*0.005

backend=KernelAbstractions.get_backend(CUDA.zeros(2))
include("qr_kernels.jl")
include("cusol_funcs.jl")


elty=Float32
sizes=[128,512,1024,2048,4096]

println( " size    type   RRMSE    time (ms)  vs CUSOLVER  cutime(ms)  cutime_ext  type     RRMSE     time (ms)  vs CUSOLVER    cutime(ms)  cutime_ext");
println(" ------  ----  --------  --------  ------------  ----------  ----------  ------  ---------  --------   -------------  ----------  -----------");
for (i,size_i) in enumerate(sizes)
    nbtiles=Int(size_i/TILESIZE)
    A=CUDA.randn( elty,size_i, size_i)
    C=CUDA.randn( elty,size_i, size_i)
    Tau=CUDA.zeros(nbtiles,size_i)
    Tau2=CUDA.zeros(size_i)
    dh = CUSOLVER.dense_handle()
    buffersize= geqrf_buffersize(A)
    buffer=CUDA.zeros(elty,buffersize)

    t_cusolver = @belapsed (CUDA.@sync geqrf!($A, $Tau2, $size_i, $size_i, $size_i,$dh,$buffer, $buffersize)) gctrial=true evals=1 gcsample=false samples=1e5 seconds=2
    t_cusolverfull = @belapsed (CUDA.@sync CUSOLVER.geqrf!($A, $Tau2)) gctrial=true evals=1 gcsample=false samples=1e5 seconds=2
    t_KA = @belapsed (CUDA.@sync mygeqrf!($A, $Tau, $nbtiles)) gctrial=true evals=1 gcsample=false samples=1e5 seconds=2
    
    Acpy=copy(A)
    Acpy2=copy(A)
    match = norm(triu!(abs.(mygeqrf!(Acpy, Tau, nbtiles)) - abs.(geqrf!(Acpy2, Tau2, size_i, size_i, size_i,dh,buffer, buffersize))))/norm(geqrf!(A, Tau2, size_i, size_i, size_i,dh,buffer, buffersize))
    @printf " %4d     QR    %8.02e    %7.02f  %8.02f %% %10.02f %10.2f" size_i match t_KA*1000  t_cusolver/t_KA*100 1000*t_cusolver 1000*t_cusolverfull

    match = norm((abs.(myormqr!(copy(C), Acpy, Tau, nbtiles)) - abs.(CUSOLormqr!(copy(C),Acpy2,Tau2))))/norm(CUSOLormqr!(copy(C),Acpy2,Tau2))
    buffersize= ormqr_bufferSize( A, Tau2, C)
    buffer=CUDA.zeros(buffersize)


    t_cusolver = @belapsed (CUDA.@sync ormqr!($C, $A, $Tau2, $dh, $size_i, $size_i, $size_i,$size_i,$size_i,$buffersize,$buffer)) gctrial=true evals=1 gcsample=false samples=1e5 seconds=2
    t_cusolverfull = @belapsed (CUDA.@sync CUSOLormqr!($C,$A,$Tau2)) gctrial=true evals=1 gcsample=false samples=1e5 seconds=2
    t_KA = @belapsed (CUDA.@sync myormqr!($C, $A, $Tau, $nbtiles)) gctrial=true evals=1 gcsample=false samples=1e5 seconds=2
    @printf "      Qmul  %8.02e   %7.02f  %8.02f %%    %10.02f %10.2f \n" match t_KA*1000  t_cusolver/t_KA*100 1000*t_cusolver 1000*t_cusolverfull
    
end  



#to do: @cuda always_inline=true,  @cuda fastmath=true,int32, streams
