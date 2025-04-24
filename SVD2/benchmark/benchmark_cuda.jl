using KernelAbstractions,GPUArrays, Random, LinearAlgebra, Printf

#select the correct vendor
using CUDA
CUDA.versioninfo()
KernelAbstractions.get_backend(CUDA.zeros(1))
const backend=CUDABackend(false, false, true)
@inline vendorsvd!(input::CuArray) = svdvals!(input,  alg=CUDA.CUSOLVER.QRAlgorithm())

elty=Float32

include("includesrc.jl")
include("benchfuncs.jl")

sizes=[64,128,256,512,1024,2048, 4096,8192,8192*2]
include("benchmarkgeneral.jl")

sizes=[64,128,256,512,1024,2048, 4096,8192,8192,8192*2,8192*4, 8192*8]
include("benchmarkmataddmul.jl")

sizes=8192 .*[1,2,4,8]
include("benchmarkgenerallarge.jl")