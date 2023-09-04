pwd()
cd(raw"C:\Users\evely\OneDrive\Documents\CSE_MIT\Avocado")

include("SVD_GPU.jl")
using Plots, BenchmarkTools, Adapt, Revise, CUDA, LinearAlgebra

################################################################################################
################### identify bottlenecks ####################################################
###########################################################################################

timings=zeros(6,6)
xs=[2^9;2^9;2^10;2^10]
ys=[4;8;2;4];

for i in 1:6
    x=xs[i]
    y=ys[i]
    n=x*y
    println(n)
    A=rand(1:10,n,n);
    timings[i,1]=@elapsed Adiag=BandBidiagonal!(A,x,x,y,y, 1, true, true, true);
    timings[i,2]=@elapsed Adiag=bidiagonalize(Adiag,x);
    n = size(Adiag,1)
    elty=eltype(Adiag)
    U, Vt, C = Matrix{elty}(I, n, n), Matrix{elty}(I, n, n), Matrix{elty}(I, n, n);
    timings[i,3]=@elapsed LAPACK.bdsqr!('U', diag(Adiag), diag(Adiag,1), Vt, U, C);
    A=rand(1:10,n,n);
    timings[i,4]=@elapsed Adiag=BandBidiagonal!(A,x,x,y,y, 1, true, true, false);
    A=rand(1:10,n,n);
    timings[i,5]=@elapsed Adiag=BandBidiagonal!(A,x,x,y,y, 1, true, false, false);
    A=Float32.(rand(1:10,n,n)) |> cu;
    timings[i,6]=@elapsed svdvals!(A, alg=CUDA.CUSOLVER.QRAlgorithm());
    GC.gc(true)
    CUDA.reclaim()
end

myrange=1:2
ns=xs.*ys
markers=[:circle;  :circle;  :circle;  :diamond; :diamond; :diamond]
scatter(ns,timings[myrange,1], xlabel="", ylabel="", xaxis=:log2, #yaxis=:log10,
label="Total blockbidiagonalization" , markersize=5 ,  markerstrokewidth=0, markershape=markers)
scatter!(ns,timings[myrange,2], label= "Pure bidiagonalization", linewidth=5 , grid=false,  markerstrokewidth=0, markershape=markers)
scatter!(ns,abs.(timings[myrange,1] .-timings[myrange,4]), label= "Blockbidiagonalization - pure calculations", linewidth=5 ,  markerstrokewidth=0, markershape=markers )
scatter!(ns,timings[myrange,4].-timings[myrange,5], label= "Blockbidiagonalization - pure copies", linewidth=5 ,  markerstrokewidth=0, markershape=markers )
#scatter!(ns,timings[myrange,3], label= "LAPACK CPU SVD of bidiagonal", linewidth=5 ,  markerstrokewidth=0, markershape=markers )
scatter!(ns,timings[myrange,5], label= "Blockbidiagonalization - empty function", linewidth=5 ,  markerstrokewidth=0, markershape=markers )
scatter!(ns,timings[myrange,6], label= "CUDA benchmark", linewidth=5 ,  markerstrokewidth=0, markershape=markers )
plot!(dpi=1800)
plot!(size=(900,900))
plot!(xticks=[2048,4096, 8192])
plot!(yticks=[1,10,100,1000])


x=4
y=12
n=x*y
A=rand(1:10,n,n);
Adiag=BandBidiagonal!(A,x,x,y,y, 1, true, true, true)
block_bidiagonalize!(Adiag, n, 4,2);


################################################################################################
################### profiler stuff  ####################################################
###########################################################################################


n=2^5;
x=2^3;
y=4;
A=rand(1:10,n,n);
Acu=A|>cu;
A_svd=svdvals(Acu, alg=CUDA.CUSOLVER.QRAlgorithm());
CUDA.@profile begin
    Adiag = BandBidiagonal!(A,x,x,y,y, 1, true, true, true);
    NVTX.@mark "test this print" begin
        println("test")
    end
end


################################################################################################
################### verifying svd correctness  ####################################################
###########################################################################################

x=2^9;
y=4;
n=x*y;
A=rand(1:10,n,n);
Acu=A|>cu;
A_svd=svdvals(Acu, alg=CUDA.CUSOLVER.QRAlgorithm());

Adiag = BandBidiagonal!(A,x,x,y,y, 1, true, true, true);
Adiag_svd=svdvals!(CuArray(Adiag), alg=CUDA.CUSOLVER.QRAlgorithm());
norm(Adiag,2) ≈ norm(A,2)
Array(A_svd) ≈  Array(Adiag_svd)

Adiag2=round.(bidiagonalize(Adiag,x),digits=4);
Adiag2_svd=svdvals!(CuArray(Adiag2), alg=CUDA.CUSOLVER.QRAlgorithm());
Array(A_svd) ≈  Array(Adiag2_svd)
norm(Adiag2,2) ≈ norm(A,2)

n = size(Adiag2,1)
elty=eltype(Adiag2)
U, Vt, C = Matrix{elty}(I, n, n), Matrix{elty}(I, n, n), Matrix{elty}(I, n, n);
Asvdvals, _ = LAPACK.bdsqr!('U', diag(Adiag2), diag(Adiag2,1), Vt, U, C);
norm(Asvdvals,2) ≈ norm(A,2)
Asvdvals ≈ Array(A_svd)

################################################################################################
################### Benchmarking svd  ####################################################
###########################################################################################


function mysvd(A,x,y)
    Adiag = BandBidiagonal!(A,x,x,y,y, 1, true, true, true);
    Adiag=bidiagonalize(Adiag,x);
    GC.gc(true)
    CUDA.reclaim()
    return;
end

function cpusvd(A)
    svd(A, alg=LinearAlgebra.QRIteration());
    return;
end

function cusvd(A)
    Acu=A|>cu;
    A_svd=Array(svdvals(CuArray(Acu), alg=CUDA.CUSOLVER.QRAlgorithm()));
    GC.gc(true)
    CUDA.reclaim()
    return;
end

timings=zeros(5,3)
ns=[round(Int,2^i) for i=10:14]
for (i,n) in enumerate(ns[2])
    println(n)
    A=rand(Float32,n,n)
    timings[i+1,1]= @elapsed cusvd(A);
    timings[i+1,2]= @elapsed cpusvd(A);
    timings[i+1,3]= @elapsed mysvd(A,2^11,Int(n/2^11));
    GC.gc(true)
    CUDA.reclaim()
end

plot(ns[1:end-1],timings[1:end-1,1], xlabel="", ylabel="", xaxis=:log2, yaxis=:log10, 
label="CUDA" , linewidth=5, color=palette(:Blues)[9] )
scatter!(ns[1:end-1],timings[1:end-1,1], label="" , color=palette(:Blues)[9], markersize=5 ,  markerstrokewidth=0)
plot!(ns[1:3],timings[1:3,2], label= "CPU", linewidth=5 , grid=false,  color=palette(:Blues)[9])
scatter!(ns[1:3],timings[1:3,2], label= "", linewidth=5 , grid=false,  color=palette(:Blues)[9],  markerstrokewidth=0)
plot!(ns[1:5],timings[1:5,3], label= "OOM", linewidth=5 , grid=false,  color=palette(:Greens)[6])
scatter!(ns[1:5],timings[1:5,3], label= "", linewidth=5 , grid=false,  color=palette(:Greens)[6],  markerstrokewidth=0)
plot!(dpi=1800)
plot!(legend=false)
plot!(xticks=[1024,4096,16384])
plot!(yticks=[1,10,100,1000])
savefig("OOM.png")

################################################################################################
################### To do list  ####################################################
###########################################################################################

#sum of squares 

#make code prettier
#verify indices of bulge chasing
#overlap communication and calculation
#Switch to GPUArrays
#make this possible for any matrix size
#define block sizes ourselves
#change qr to LQ
#optimize memory usage
#add reft and left singular vectors
#parallelize/GPU-ize start of bulge chasing
#bulge chasing for non-square non-powers of two
#avoid reallocations
#non-square matrices
#reduce to bidiag
#add typechecks and errorchecks

#access streams

#bidiag to diag
#left and right singular vectors


################################################################################################
################### Random other stuff ####################################################
###########################################################################################

#benchmark qr

function myqr(X)
    X2=X|>cu
    qr(X2)
    X2'*X2
    copyto!(X,X2)
end


ns=[round(Int,2^i) for i=1:2:12]
timings=zeros(length(ns),3)

for (i,n) in enumerate(ns)
    println(n)
    A=rand(Float32,2n,n)
    B=A|>cu
    timings[i,1]= @belapsed qr($A);
    timings[i,2]= @belapsed qr($B);
    timings[i,3]= @belapsed myqr($A);
end

plot(ns,timings, xlabel="", ylabel="", grid=false,xaxis=:log2, yaxis=:log10 , linewidth=8,
label=["CPU"  "GPU" "GPU and MM and copy"] )
scatter!(ns,timings,  color=[palette(:Blues)[9] palette(:Greens)[6]] , markersize=7 ,  markerstrokewidth=0)
plot!(dpi=1800)
plot!(legend=false)
plot!(xticks=[4,128,4096])
plot!(yticks=[0.0001,0.01,1])
savefig("QRtiming.png")


#benchmark svd

timings_cpu=[]
timings_gpu=[]
n_vals=[  10,32,100,316, 1000, 3162, 5000]
for n in [5000]
    A=rand(n,n)
    B=A|>cu
    t= @belapsed svd!(A)
    push!(timings_cpu,t)
    t= CUDA.@elapsed CUDA.CUSOLVER.gesvdj!('V',1, B, tol=Float32(1e-5))
    push!(timings_gpu,t)
end

plot(n_vals,timings_cpu, xlabel="Matrix size n", ylabel="svd calc time (s)", xaxis=:log10, yaxis=:log10, labels="CPU", )
plot!(n_vals,timings_gpu, xlabel="Matrix size n", ylabel="svd calc time (s)", xaxis=:log10, yaxis=:log10, labels="GPU")


################################################################################################
################### Test with threads ####################################################
###########################################################################################

using CUDA, LinearAlgebra, KernelAbstractions

function multiQR(matriceslist)
    CUDA.synchronize()
    @sync for i in 1:length(matriceslist)
        Threads.@spawn begin
            qr!(matriceslist[i])
        end
    end
    CUDA.synchronize()
end

function nonmultiQR(matriceslist)
    CUDA.synchronize()
    for i in 1:length(matriceslist)
        qr!(matriceslist[i])
    end
    CUDA.synchronize()
end

n=1000
trials=5
inputmatrices=[CUDA.rand(n,n) for i in 1:trials];
multiQR(inputmatrices);
inputmatrices=[CUDA.rand(n,n) for i in 1:trials];
nonmultiQR(inputmatrices);
inputmatrices=[CUDA.rand(n,n) for i in 1:trials];
@time multiQR(inputmatrices)
inputmatrices=[CUDA.rand(n,n) for i in 1:trials];
@time nonmultiQR(inputmatrices)

