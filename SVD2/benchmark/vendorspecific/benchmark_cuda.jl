

#select the correct vendor
using CUDA
CUDA.versioninfo()
KernelAbstractions.get_backend(CUDA.zeros(1))
const backend=CUDABackend(false, false, true)
@inline vendorsvd!(input::CuArray) = svdvals!(input,  alg=CUDA.CUSOLVER.QRAlgorithm())
if (ARGS[2]=="H")
    @inline vendorsvd!(input::CuArray) = svdvals!(Float32.(input),  alg=CUDA.CUSOLVER.QRAlgorithm())
end