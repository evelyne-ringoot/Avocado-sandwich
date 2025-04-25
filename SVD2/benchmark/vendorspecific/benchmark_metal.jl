
using Metal
Metal.versioninfo()
const backend=KernelAbstractions.get_backend( MtlArray([1]))
@inline vendorsvd!(input::MtlArray) = svdvals!(Array(input))
