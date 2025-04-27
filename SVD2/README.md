This sub-repository contains a julia-native GPU-accelerated bidiagonal reduction for singular values, and uses LAPACK for getting singular values from a bidiagonal matrix. To runa benchmark on your system:

```
wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.5-linux-x86_64.tar.gz
tar zxvf julia-1.11.5-linux-x86_64.tar.gz
git clone https://github.com/evelyne-ringoot/Avocado-sandwich.git
cd Avocado-sandwich/SVD2/
../../-1.11.5/bin/julia --project=. 
julia> Pkg.instantiate()
julia> exit()
../../-1.11.5/bin/julia --project=. -t auto benchmark/benchmark.jl CUDA S SMALL 0 64 32 8 N
```

Options
Param 1: CUDA/AMD/ONE/METAL
Param 2: S/D/H (precision) 
Param 3: SMALL/LARGE/SPECIFY (which sizes to run)
Param 4: (only applicable if option 3 is SPECIFY) int (size to benchmark)
Param 5: tilesize (optional)
Param 6: tilesize for multiplication with Qt (optional)
Param 7: splitk factor (optional)
Param 8: use a reduced memory BRD Y/N (optional, default N)
Param 8: Minimum runtime for benchmarking per matrix size (optional, default 2000)
Param 9: Amount of runs per benchmark between sync (optional, default 20)

