using Test, Random
include(joinpath("..","src","QRkernels.jl"))
TILE_SIZE1=1
TILE_SIZE2=1 
dims2d=false 
include(joinpath("..","src","QRkernelscompile.jl"))
include(joinpath("..","src","TiledMatrix.jl"))
include(joinpath("..","src","bulgechasing.jl"))
include(joinpath("..","src","tiledalgos.jl"))

@testset "Tiled algos 2D=$dims2d and n=$tilesize" for  tilesize in [4,16], 
                                                 (mydims2d,tilesize,tilefactor ) in ( (false, tilesize,1), (true, tilesize, 4) )
    
    compile_kernels(tilesize=tilesize, tilefactor=tilefactor, tiledim2d=mydims2d)
    backend=KernelAbstractions.get_backend(CUDA.randn(2))

    @testset "elty = $elty and 2D=$dims2d and n=$tilesize and noblocks=$no_blocks" for elty in [ Float32, Float64], no_blocks in [4,6]
        
        @testset "GPU- only QR and SVD" begin
            A=rand!(allocate(backend, elty,tilesize*no_blocks, tilesize*no_blocks))
            
            testmatrix=copy(A)
            refmatrix=copy(A)
            Atile = TiledMatrix(testmatrix,tilesize,tilesize, tiledim2d= mydims2d ? TILE_SIZE2[2] : 0)
            tiled_QR!(Atile)
            @test triu(testmatrix) .*sign.(diag(testmatrix)) ≈  qr(refmatrix).R .* sign.(diag( qr(refmatrix).R))
            
            testmatrix=copy(A)
            refmatrix=copy(A)
            Atile = TiledMatrix(testmatrix,tilesize,tilesize, tiledim2d= dims2d ? TILE_SIZE2[2] : 0)
            Blockbidiag!(Atile)
            @test svdvals(Atile.A) ≈ svdvals(refmatrix)
        end

        @testset "OOC-only QR and SVD" begin
            A=randn(elty,tilesize*no_blocks, tilesize*no_blocks)

            testmatrix=copy(A)
            refmatrix=copy(A)
            Atile = LargeTiledMatrix(testmatrix, backend, tilesize*no_blocks, tilesize, no_blocks, tiledim2d= mydims2d ? TILE_SIZE2[2] : 0) 
            tiled_QR!(Atile)
            @test triu(testmatrix) .*sign.(diag(testmatrix)) ≈  qr(refmatrix).R .* sign.(diag( qr(refmatrix).R))

            testmatrix=copy(A)
            refmatrix=copy(A)
            Atile = LargeTiledMatrix(testmatrix, backend, tilesize*no_blocks, tilesize, no_blocks, tiledim2d= mydims2d ? TILE_SIZE2[2] : 0) 
            Blockbidiag!(Atile)
            @test svdvals(Atile.A) ≈ svdvals(refmatrix)

        end

        @testset "OOC+GPU QR and SVD" begin
            A=randn(elty,tilesize*no_blocks, tilesize*no_blocks)
            
            testmatrix=copy(A)
            refmatrix=copy(A)
            OOC_QR!(testmatrix, kswitch=3, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor, backend=backend)
            @test triu(testmatrix) .*sign.(diag(testmatrix)) ≈  qr(refmatrix).R .* sign.(diag( qr(refmatrix).R))

            testmatrix=copy(A)
            refmatrix=copy(A)
            Atile = LargeTiledMatrix(testmatrix, backend, tilesize*no_blocks, tilesize, no_blocks, tiledim2d= mydims2d ? TILE_SIZE2[2] : 0) 
            OOC_Bidiag!(testmatrix, kswitch=3, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor, backend=backend)
            @test svdvals(Atile.A) ≈ svdvals(refmatrix)

        end
    end

end
