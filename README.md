# Avocado-sandwich

CUDAbasedSVD contains an implementation of https://www.netlib.org/utk/people/JackDongarra/PAPERS/siam-svd-2018.pdf using CUDA.jl for a square matrix, with tiles of sizes of powers of 2. The block-bidiagonialization is supported on CPU and GPU. The bidiagonizalization by block-bulge chasing is a CPU-only implementation to achieve a mixed CPU-GPU SVD, and the diagonalization calls LAPACK's divide-and-conquer method.

KAbasedSVD contains the same implementation using KernelAbstractions.jl and a novel TiledMatrixAbstraction instead.

TiledTutorial contains a jupyter notebook tutorial walking through these developments, prepared for a tutorial at CSCS in July 2024.