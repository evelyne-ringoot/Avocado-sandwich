
include("SVD_GPU.jl")

@testset "SVD with elty = $elty" for elty in [ Float32] #not supported for other elementtypes

    @testset "number of blocks = $no_blocks" for no_blocks in [4,7],
        block_size in [2^4,2^7]

        testmatrix=elty.(rand(1:10,block_size*no_blocks, block_size*no_blocks));
        correct_vals=Array(svdvals!(CuArray(testmatrix), alg=CUDA.CUSOLVER.QRAlgorithm()));

        vals_cpu=my_CU_svdval(testmatrix, block_size);
        @test vals_cpu ≈ correct_vals

        testmatrix_cu= CuArray(testmatrix);
        vals_gpu=my_CU_svdval(testmatrix_cu, block_size);
        @test vals_gpu ≈ correct_vals
    end

end


