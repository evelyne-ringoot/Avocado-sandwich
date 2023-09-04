pwd()
cd(raw"C:\Users\evely\OneDrive\Documents\CSE_MIT\Avocado")

include("SVD_GPU.jl")
using Plots, BenchmarkTools, Adapt, Revise, CUDA, LinearAlgebra

#create testmatrix
block_size=2^8; 
no_blocks=4;
mat_size=x*y;
testmatrix=rand(1:10,mat_size,mat_size);
testmatrix_gpu=testmatrix|>cu;
testmatrix_svd_ref=svdvals(testmatrix_gpu, alg=CUDA.CUSOLVER.QRAlgorithm());

#test reduction to bandbidiagonal
testmatrix_banddiag = BandBidiagonal!(testmatrix,block_size,block_size,no_blocks,no_blocks, 1, true, true, true);
testmatrix_band_svd=svdvals!(CuArray(testmatrix_banddiag), alg=CUDA.CUSOLVER.QRAlgorithm());
norm(testmatrix_banddiag,2) ≈ norm(testmatrix,2)
Array(testmatrix_band_svd) ≈  Array(testmatrix_svd_ref)

#test reduction to bidiagonal
testmatrix_bidiag=bidiagonalize(testmatrix_banddiag,block_size);
testmatrix_svd=svdvals!(CuArray(testmatrix_bidiag), alg=CUDA.CUSOLVER.QRAlgorithm());
norm(testmatrix_bidiag,2) ≈ norm(testmatrix,2)
Array(testmatrix_bidiag) ≈  Array(testmatrix_svd_ref)

#test lapack svd from bidiagonal
elty=eltype(testmatrix_bidiag)
U, Vt, C = Matrix{elty}(I, n, n), Matrix{elty}(I, n, n), Matrix{elty}(I, n, n);
testmatrix_fullsvd, _ = LAPACK.bdsqr!('U', diagtestmatrix_bidiag, diag(testmatrix_bidiag,1), Vt, U, C);
norm(testmatrix_fullsvd,2) ≈ norm(testmatrix,2)
norm(testmatrix,2) ≈ Array(testmatrix_svd_ref)
