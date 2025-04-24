

#select the correct vendor
using CUDA
CUDA.versioninfo()
KernelAbstractions.get_backend(CUDA.zeros(1))
const backend=CUDABackend(false, false, true)
@inline vendorsvd!(input::CuArray) = nothing

