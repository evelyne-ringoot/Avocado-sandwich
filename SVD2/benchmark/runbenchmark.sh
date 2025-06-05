#!/bin/bash

for N in 64 92 128 160 192 224; do ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --single --hardware cuda --brdwidth 32 --bandwidth 128 --brdmulsize $N --maxblocks 528 ; done
for N in 64 92 128 160 192 224; do ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --single --hardware cuda --brdwidth 32 --bandwidth 256 --brdmulsize $N --maxblocks 528 ; done
for N in 64 92 128 160 192 224; do ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --single --hardware cuda --brdwidth 32 --bandwidth 512 --brdmulsize $N --maxblocks 528 ; done

for N in 264 1056; do ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --single --hardware cuda --brdwidth 32 --bandwidth 128 --brdmulsize 128 --maxblocks $N ; done
for N in 264 1056; do ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --single --hardware cuda --brdwidth 32 --bandwidth 256 --brdmulsize 160 --maxblocks $N ; done
for N in 264 1056; do ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --single --hardware cuda --brdwidth 32 --bandwidth 512 --brdmulsize 192 --maxblocks $N ; done

for N in 16 64; do ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --single --hardware cuda --brdwidth $N --bandwidth 128 --brdmulsize 128 --maxblocks 528 ; done
for N in 16 64; do ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --single --hardware cuda --brdwidth $N --bandwidth 256 --brdmulsize 160 --maxblocks 528 ; done
for N in 16 64; do ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --single --hardware cuda --brdwidth $N --bandwidth 512 --brdmulsize 192 --maxblocks 528 ; done
for N in 16 64; do ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --single --hardware cuda --brdwidth $N --bandwidth 128 --brdmulsize 64 --maxblocks 528 ; done
for N in 16 64; do ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --single --hardware cuda --brdwidth $N --bandwidth 256 --brdmulsize 96 --maxblocks 528 ; done
for N in 16 64; do ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --single --hardware cuda --brdwidth $N --bandwidth 512 --brdmulsize 128 --maxblocks 528 ; done

 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd_svd.jl --single --hardware cuda --brdwidth 32 --bandwidth 32 --brdmulsize 64 --maxblocks 528 --tilesize 32 --tilesizemul 32
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd_svd.jl --single --hardware cuda --brdwidth 32 --bandwidth 64 --brdmulsize 96 --maxblocks 528 --tilesize 64 --tilesizemul 32
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd_svd.jl --single --hardware cuda --brdwidth 32 --bandwidth 128 --brdmulsize 128 --maxblocks 528 --tilesize 128 --tilesizemul 32
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd_svd.jl --single --hardware cuda --brdwidth 32 --bandwidth 256 --brdmulsize 160 --maxblocks 528 --tilesize 256 --tilesizemul 32
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd_svd.jl --single --hardware cuda --brdwidth 32 --bandwidth 512 --brdmulsize 192 --maxblocks 528 --tilesize 64 --tilesizemul 32
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd_svd.jl --single --hardware cuda --brdwidth 32 --bandwidth 1024 --brdmulsize 224 --maxblocks 528 --tilesize 64 --tilesizemul 32

 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --double --hardware cuda --brdwidth 16 --bandwidth 32 --brdmulsize 64 --maxblocks 528 
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --double --hardware cuda --brdwidth 16 --bandwidth 64 --brdmulsize 96 --maxblocks 528 
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --double --hardware cuda --brdwidth 16 --bandwidth 128 --brdmulsize 128 --maxblocks 528 
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --double --hardware cuda --brdwidth 16 --bandwidth 256 --brdmulsize 160 --maxblocks 528
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --double --hardware cuda --brdwidth 16 --bandwidth 512 --brdmulsize 192 --maxblocks 528
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --double --hardware cuda --brdwidth 16 --bandwidth 1024 --brdmulsize 224 --maxblocks 528 

 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --half --hardware cuda --brdwidth 32 --bandwidth 32 --brdmulsize 64 --maxblocks 528 
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --half --hardware cuda --brdwidth 64 --bandwidth 64 --brdmulsize 96 --maxblocks 528 
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --half --hardware cuda --brdwidth 64 --bandwidth 128 --brdmulsize 128 --maxblocks 528 
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --half --hardware cuda --brdwidth 64 --bandwidth 256 --brdmulsize 160 --maxblocks 528
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --half --hardware cuda --brdwidth 64 --bandwidth 512 --brdmulsize 192 --maxblocks 528
 ~/julia-1.11.5/bin/julia --project=. ./benchmark/benchbrd.jl --half --hardware cuda --brdwidth 64 --bandwidth 1024 --brdmulsize 224 --maxblocks 528 