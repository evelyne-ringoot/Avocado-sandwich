
using oneAPI
oneAPI.versioninfo()
const backend=KernelAbstractions.get_backend(oneArray(rand(Float32, 2,2)))
@inline vendorsvd!(input::oneArray) = oneAPI.gesvd!('N','N',input)
if (ARGS[2]=="H")
    @inline vendorsvd!(input::oneArray) = oneAPI.gesvd!('N','N',Float32.(input))
end

