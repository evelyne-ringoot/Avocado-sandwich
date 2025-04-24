using KernelAbstractions,GPUArrays, Random, LinearAlgebra, Printf
include("includesrc.jl")
include("benchfuncs.jl")


elty=Float16



sizes=[64,128,256,512,1024,2048, 4096]#,8192,8192*2]
include("benchmarkgeneral.jl")

sizes=[64,128,256,512,1024,2048, 4096]#,8192,8192,8192*2,8192*4, 8192*8]
include("benchmarkmataddmul.jl")

sizes=8192 .*[1]#,2,4,8]
include("benchmarkgenerallarge.jl")

@inline vendorsvd!(input::CuArray) = svdvals!(input,  alg=CUDA.CUSOLVER.QRAlgorithm())

elty=Float32

sizes=[64,128,256]#,512,1024,2048, 4096,8192,8192*2]
include("benchmarkgeneral.jl")

sizes=[64,128,256]#,512,1024,2048, 4096,8192,8192,8192*2,8192*4, 8192*8]
include("benchmarkmataddmul.jl")

sizes=8192 .*[1]#,2,4,8]
include("benchmarkgenerallarge.jl")