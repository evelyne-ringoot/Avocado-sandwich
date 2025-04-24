using KernelAbstractions,GPUArrays, Random, LinearAlgebra, Printf

using Metal
Metal.versioninfo()
const backend=KernelAbstractions.get_backend( MtlArray([1]))
@inline vendorsvd!(input::MtlArray) = none

elty=Float32

include("includesrc.jl")
include("benchfuncs.jl")

sizes=[64,128,256,512,1024,2048, 4096,8192,8192,8192*2]
include("benchmarkgeneral.jl")

sizes=[64,128,256,512,1024,2048, 4096,8192,8192,8192*2,8192*4, 8192*8]
include("benchmarkmataddmul.jl")

sizes=8192 .*[2,4,8]
include("benchmarkgenerallarge.jl")