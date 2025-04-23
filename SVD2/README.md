This sub-repository contains a julia-native GPU-accelerated bidiagonal reduction for singular values, and uses LAPACK for getting singular values from a bidiagonal matrix. To runa benchmark on your system:

```
wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.5-linux-x86_64.tar.gz
tar zxvf julia-1.11.5-linux-x86_64.tar.gz
git clone https://github.com/evelyne-ringoot/Avocado-sandwich.git
cd Avocado-sandwich/SVD2/
../../-1.11.5/bin/julia --project=. 
julia> Pkg.instantiate()
julia> exit()
../../-1.11.5/bin/julia --project=. -t auto benchmark/benchmark_cuda.jl
```

Replace the last line file with the applicable GPU: `benchmark_amd.jl` for AMD, `benchmark_oneapi.jl` for Intel, and `benchmark_metal.jl` for Apple.
