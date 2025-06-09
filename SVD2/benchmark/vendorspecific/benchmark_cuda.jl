

#select the correct vendor
using CUDA
#CUDA.versioninfo()
KernelAbstractions.get_backend(CUDA.zeros(1))
const backend=CUDABackend(false, false, true)
@inline vendorsvd!(input::CuArray) = svdvals!(input,  alg=CUDA.CUSOLVER.QRAlgorithm())
if (elty==Float16)
    @inline vendorsvd!(input::CuArray) = svdvals!(copy(Float32.(input)),  alg=CUDA.CUSOLVER.QRAlgorithm())
end