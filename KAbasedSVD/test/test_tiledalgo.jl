include(joinpath("..","src","QRkernels.jl"))
include(joinpath("..","src","TiledMatrix.jl"))
include(joinpath("..","src","tiledalgos.jl"))

@testset "Tiled algos elty = $elty" for elty in [ Float32]
    
    tilesize=4
    no_blocks=3
    backend=KernelAbstractions.get_backend(CUDA.randn(2))
    T=elty

    Tau=Array{CuArray}(undef, no_blocks,no_blocks)
    for i in 1:no_blocks
        for j in 1:no_blocks
            Tau[i,j]=KernelAbstractions.zeros(backend, T, tilesize)
        end
    end

    A=rand!(allocate(backend, T,tilesize*no_blocks, tilesize*no_blocks))

    @testset "QR" begin
        testmatrix=copy(A)
        refmatrix=copy(A)
        Atile = TiledMatrix(testmatrix,tilesize,tilesize)
        tiled_QR!(Atile,Tau)
        @test triu(testmatrix) .*sign.(diag(testmatrix)) ≈  qr(refmatrix).R .* sign.(diag( qr(refmatrix).R))
    end

    @testset "SVD" begin
        testmatrix=copy(A)
        refmatrix=copy(A)
        Atile = TiledMatrix(testmatrix,tilesize,tilesize)
        Blockbidiag_nonpar!(Atile,Tau)
        @test svdvals(Atile.ParentMatrix) ≈ svdvals(refmatrix)
    end

    @testset "SVD par" begin
        testmatrix=copy(A)
        refmatrix=copy(A)
        Atile = TiledMatrix(testmatrix,tilesize,tilesize)
        Blockbidiag!(Atile,Tau)
        @test svdvals(Atile.ParentMatrix) ≈ svdvals(refmatrix)
    end

end

@test "blockviews" begin
    A=CuArray(Float32.(rand(1:10,8,8)))
    println(A)
    Atile=TiledMatrix(A,2,2)
    println(hor_blocktileview(Atile, 2, 3))
    println(ver_blocktileview(Atile, 2,3))
end

tilesize=32
no_blocks=[1,4,16,32,64]
matrix_sizes=no_blocks*tilesize
backend=KernelAbstractions.get_backend(CUDA.randn(2))
T=Float32
timings=zeros(length(matrix_sizes), 2)

for (i,n) in enumerate(matrix_sizes)
    Tau=Array{CuArray}(undef, no_blocks[i],no_blocks[i])
    for l in 1:no_blocks[i]
        for j in 1:no_blocks[i]
            Tau[l,j]=KernelAbstractions.zeros(backend, T, tilesize)
        end
    end
    A=rand!(allocate(backend, T,n, n))
    testmatrix=copy(A)
    refmatrix=copy(A)
    Atile = TiledMatrix(testmatrix,tilesize,tilesize)
    timings[i,1]=@elapsed CUDA.@sync Blockbidiag_nonpar!(Atile,Tau)
    Blockbidiag_nonpar!(Atile,Tau)
    testmatrix=copy(A)
    refmatrix=copy(A)
    Atile = TiledMatrix(testmatrix,tilesize,tilesize)
    timings[i,2]=@elapsed CUDA.@sync Blockbidiag!(Atile,Tau)
end
plot(matrix_sizes, timings, labels= ["non par" "par" ], xaxis=:log10, yaxis=:log10, lw=2,
 xticks=(matrix_sizes, string.(matrix_sizes)), xlabel= "matrix size nxn", ylabel= "time(s)", legend=:bottomright)