
using oneAPI
#oneAPI.versioninfo()
const backend=KernelAbstractions.get_backend(oneArray(rand(Float32, 2,2)))
@inline vendorsvd!(input::oneArray) = oneMKL.gesvd!('N','N',input)[2]
if (ARGS[2]=="H")
    @inline vendorsvd!(input::oneArray) = oneMKL.gesvd!('N','N',copy(Float32.(input)))
end
