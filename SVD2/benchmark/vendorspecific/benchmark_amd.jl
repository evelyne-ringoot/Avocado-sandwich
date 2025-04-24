
using AMDGPU
AMDGPU.versioninfo()
const backend=KernelAbstractions.get_backend(AMDGPU.zeros(2))
@inline vendorsvd!(input::ROCArray) = AMDGPU.rocSOLVER.gesvd!('N','N',input)

