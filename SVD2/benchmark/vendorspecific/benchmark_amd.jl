
using AMDGPU
#AMDGPU.versioninfo()
const backend=KernelAbstractions.get_backend(AMDGPU.zeros(2))
@inline vendorsvd!(input::ROCArray) = AMDGPU.rocSOLVER.gesvd!('N','N',input)[2]
if (ARGS[2]=="H")
    @inline vendorsvd!(input::ROCArray) = AMDGPU.rocSOLVER.gesvd!('N','N',copy(Float32.(input)))[2]
end
