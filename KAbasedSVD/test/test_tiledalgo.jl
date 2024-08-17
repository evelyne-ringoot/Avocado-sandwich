include(joinpath("..","src","QRkernels.jl"))
TILE_SIZE1=1
TILE_SIZE2=1 
dims2d=false 
include(joinpath("..","src","QRkernelscompile.jl"))
include(joinpath("..","src","TiledMatrix.jl"))
include(joinpath("..","src","bulgechasing.jl"))
include(joinpath("..","src","tiledalgos.jl"))

@testset "Tiled algos elty = $elty and 2D=$dims2d and n=$tilesize" for 
                                                elty in [ Float32, Float64],
                                                tilesize in [4,16],
                                                 (mydims2d,tilesize,tilefactor ) in ( (false, tilesize,1), (true, tilesize, 4) )

    compile_kernels(tilesize=tilesize, tilefactor=tilefactor, tiledim2d=mydims2d)
    no_blocks=6
    backend=KernelAbstractions.get_backend(CUDA.randn(2))
    A=rand!(allocate(backend, elty,tilesize*no_blocks, tilesize*no_blocks))
    
    @testset "QR" begin
        testmatrix=copy(A)
        refmatrix=copy(A)
        Atile = TiledMatrix(testmatrix,tilesize,tilesize, tiledim2d= mydims2d ? TILE_SIZE2[2] : 0)
        tiled_QR!(Atile)
        @test triu(testmatrix) .*sign.(diag(testmatrix)) ≈  qr(refmatrix).R .* sign.(diag( qr(refmatrix).R))
    end

    @testset "SVD" begin
        testmatrix=copy(A)
        refmatrix=copy(A)
        Atile = TiledMatrix(testmatrix,tilesize,tilesize, tiledim2d= dims2d ? TILE_SIZE2[2] : 0)
        Blockbidiag!(Atile)
        @test svdvals(Atile.A) ≈ svdvals(refmatrix)
    end

end

    @testset "Large Tiled algos elty = $elty and 2D=$dims2d and n=$tilesize" for 
                        elty in [ Float32, Float64],
                        tilesize in [4,16],
                        (mydims2d,tilesize,tilefactor ) in ( (false, tilesize,1), ( true, tilesize, 4) )

        compile_kernels(tilesize=tilesize, tilefactor=tilefactor, tiledim2d=tiledim2d)
        no_blocks=3
        backend=KernelAbstractions.get_backend(CUDA.randn(2))
        A=randn(elty,tilesize*no_blocks, tilesize*no_blocks)

        @testset "QR" begin
        testmatrix=copy(A)
        refmatrix=copy(A)
        Atile = LargeTiledMatrix(testmatrix, backend, tilesize*no_blocks, tilesize, no_blocks, tiledim2d= dims2d ? TILE_SIZE2[2] : 0) 
        tiled_QR!(Atile)
        @test triu(testmatrix) .*sign.(diag(testmatrix)) ≈  qr(refmatrix).R .* sign.(diag( qr(refmatrix).R))
        end

        @testset "SVD" begin
        testmatrix=copy(A)
        refmatrix=copy(A)
        Atile = LargeTiledMatrix(testmatrix, backend, tilesize*no_blocks, tilesize, no_blocks, tiledim2d= dims2d ? TILE_SIZE2[2] : 0) 
        Blockbidiag!(Atile)
        @test svdvals(Atile.A) ≈ svdvals(refmatrix)
        end

    end


       