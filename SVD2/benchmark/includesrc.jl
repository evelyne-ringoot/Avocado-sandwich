
BLAS.set_num_threads(Threads.nthreads())
include("../src/KAfuncs.jl")
include("../src/qr_kernels.jl")
include("../src/brdgpu.jl")
include("../src/datacomms.jl")
include("../src/tiledalgos.jl")
