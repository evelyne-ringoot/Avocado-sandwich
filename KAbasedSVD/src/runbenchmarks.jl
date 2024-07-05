include(joinpath("..","src","QRkernels.jl"))
include(joinpath("..","src","TiledMatrix.jl"))
include(joinpath("..","src","tiledalgos.jl"))
include(joinpath("..","src","bulgechasing.jl"))

tilesize=64
no_blocks_values=[1,2,4,8]
n_values=no_blocks_values*tilesize
timings=zeros(length(n_values), 4)

backend=KernelAbstractions.get_backend(CUDA.randn(2))

for ni in 1:length(n_values)
    println(ni)
    no_blocks=no_blocks_values[ni]
    i=1
    for T in [Float16,Float32]
        Tau=Array{CuArray}(undef, no_blocks,no_blocks)
        for i in 1:no_blocks
            for j in 1:no_blocks
                Tau[i,j]=KernelAbstractions.zeros(backend, T, tilesize)
            end
        end

        A=rand!(allocate(backend, T,tilesize*no_blocks, tilesize*no_blocks))

        testmatrix_own=copy(A)
        Atile = TiledMatrix(testmatrix_own,tilesize,tilesize)
        Blockbidiag!(Atile,Tau)
        cpucopy=Array(Float32.(testmatrix_own))
        my_CU_svdval(cpucopy, tile_size,no_blocks)

        testmatrix_own=copy(A)
        Atile = TiledMatrix(testmatrix_own,tilesize,tilesize)
        temptiming= CUDA.@elapsed Blockbidiag_nopar!(Atile,Tau)
        cpucopy=Array(Float32.(testmatrix_own))
        temptiming+=@belapsed my_CU_svdval($cpucopy, $tile_size,$no_blocks)
        timings[ni,i]=temptiming
        i+=1
    end

    A=rand!(allocate(backend, T,tilesize*no_blocks, tilesize*no_blocks))
    testmatrix_cuda=copy(A)
    timings[ni,3]= @belapsed svd!($testmatrix_cuda, alg=CUDA.CUSOLVER.QRAlgorithm());
    testmatrix_la=Array(A)
    timings[ni,4]=@belapsed svd($testmatrix_la, alg=LinearAlgebra.QRIteration());

end

BSON.@save "benchmarkingsvd.bson" timings


using Plots
timings=BSON.@load("benchmarkingsvd.bson")
plot(n_values, timings, labels=["Tiled Abstraction F16" "Tiled Abstraction F32" "CUDA" "GPU"], yaxis=:log10)
