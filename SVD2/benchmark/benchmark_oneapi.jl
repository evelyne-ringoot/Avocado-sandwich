using KernelAbstractions,GPUArrays, Random, LinearAlgebra, Printf

using oneAPI
oneAPI.versioninfo()
const backend=KernelAbstractions.get_backend(oneArray(rand(Float32, 2,2)))
@inline vendorsvd!(input::oneArray) = oneAPI.gesvd!('N','N',input)

elty=Float32

include("includesrc.jl")
include("benchfuncs.jl")

sizes=[64,128,256,512,1024,2048, 4096,8192,8192,8192*2]
include("benchmarkgeneral.jl")

sizes=[64,128,256,512,1024,2048, 4096,8192,8192,8192*2,8192*4, 8192*8]
include("benchmarkmataddmul.jl")

sizes=8192 .*[2,4,8]
include("benchmarkgenerallarge.jl")