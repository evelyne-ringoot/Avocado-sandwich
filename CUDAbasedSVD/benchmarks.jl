


###################################################"
# TIMING SOME CODE #######################
##########################################""

ns=2 .^(5:2:13)
svd_timings=zeros(length(ns),2)
qr_timings=zeros(length(ns),2)
for ni in 1:length(ns)
    println(ni)
    n=ns[ni]
    a=randn(n,n)
    a2 =copy(a)
    if (ni!=5)
        svd_timings[ni,1] = @belapsed svd!($a, alg = LinearAlgebra.QRIteration())
    end
    svd_timings[ni,3] = @belapsed cusvd!($a2)
    #qr_timings[ni,4] = @belapsed qr!($a)
    #qr_timings[ni,5] = @belapsed cuqr!($a2)
    my_timings[ni,1] = @belapsed copytime!($a, false)
    my_timings[ni,2] = @belapsed copytime!($a, true)
    for j in 1:3
        my_timings[4,j+3] = @belapsed mysvd!($a, 512,4,j)
    end
end


function cusvd!(a)
    a2=a |>cu 
    svd!(a2, alg=CUDA.CUSOLVER.QRAlgorithm())
    copy!(a,a2)
end
function cuqr!(a)
    a2=a |>cu 
    qr!(a2)
    copy!(a,a2)
end

function mysvd!(testmatrixin, block_size, no_blocks,setting)
    testmatrix= testmatrixin |>cu
    testmatrix_banddiag = Array(BandBidiagonal!(testmatrix,block_size,block_size,no_blocks,no_blocks, 1, true, true, true));
    if (setting>=1)
        testmatrix_bidiag=bidiagonalize(testmatrix_banddiag,block_size);
        if (setting>=2)
            #elty=eltype(testmatrix_bidiag)
            #U, Vt, C = Matrix{elty}(I, n, n), Matrix{elty}(I, n, n), Matrix{elty}(I, n, n);
            LAPACK.bdsdc!('U','N', diag(testmatrix_bidiag), diag(testmatrix_bidiag,1));
            #LAPACK.bdsqr!('U', diag(testmatrix_bidiag), diag(testmatrix_bidiag,1), Vt, U, C);
        end
    else

    end
end

function copytime!(testmatrixin, back)
    testmatrix= testmatrixin |>cu
    if (back)
        Array(testmatrix);
    end
end



# benchmarks own svd function

function mysvd(A,x,y)
    Adiag = BandBidiagonal!(A,x,x,y,y, true, );
    Adiag=bidiagonalize(Adiag,x);
    return;
end

function cpusvd(A)
    svd(A, alg=LinearAlgebra.QRIteration());
    return;
end

function cusvd(A)
    Acu=A|>cu;
    A_svd=Array(svdvals(CuArray(Acu), alg=CUDA.CUSOLVER.QRAlgorithm()));
    return;
end

timings=zeros(5,3)
ns=[round(Int,2^i) for i=10:14]
for (i,n) in enumerate(ns)
    A=rand(Float32,n,n)
    timings[i+1,1]= @elapsed cusvd(A);
    timings[i+1,2]= @elapsed cpusvd(A);
    timings[i+1,3]= @elapsed mysvd(A,2^11,Int(n/2^11));
    GC.gc(true)
    CUDA.reclaim()
end

#benchmark qr

function qrwcopy(X)
    X2=X|>cu
    qr(X2)
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
    timings[i,3]= @belapsed qrwcopy($A);
end

plot(ns,timings, xlabel="", ylabel="", grid=false,xaxis=:log2, yaxis=:log10 , linewidth=8,
label=["CPU"  "GPU" "GPU and MM and copy"] )
scatter!(ns,timings,  color=[palette(:Blues)[9] palette(:Greens)[6]] , markersize=7 ,  markerstrokewidth=0)
plot!(dpi=1800, legend=false, xticks=[4,128,4096], yticks=[0.0001,0.01,1])
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

#Plots

using Plots
markers=[:circle;  :circle;  :circle;  :diamond; :diamond; :diamond]
scatter(n,timings, xlabel="", ylabel="", xaxis=:log2, yaxis=:log10, linewidth=5 , grid=false, 
label="label" , markersize=5 ,  markerstrokewidth=0, markershape=markers)
plot!(dpi=1800, (size=(900,900)), xticks=[2048,4096, 8192], yticks=[1,10,100,1000])

#CUDA profiler

x=2^3;
y=4;
A=rand(1:10,x*y,x*y);
Acu=A|>cu;
A_svd=svdvals(Acu, alg=CUDA.CUSOLVER.QRAlgorithm());
CUDA.@profile begin
    Adiag = BandBidiagonal!(A,x,x,y,y, true);
    NVTX.@mark "test this print" begin
        println("test")
    end
end