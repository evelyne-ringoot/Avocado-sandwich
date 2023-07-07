pwd()
cd(raw"C:\Users\evely\OneDrive\Documents\CSE_MIT\Avocado")

include("SVD_GPU.jl")
using Plots, BenchmarkTools, Adapt, Revise, CUDA, LinearAlgebra


x=2^8;
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
