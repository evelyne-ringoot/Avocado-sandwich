using KernelAbstractions,CUDA, BSON, BenchmarkTools, Random, LinearAlgebra
startpath=joinpath("Avocado","KAbasedSVD")
startpath=".."
include(joinpath(startpath,"src","QRkernels.jl"))
TILE_SIZE1=1
TILE_SIZE2=1 
dims2d=false 
include(joinpath(startpath,"src","QRkernelscompile.jl"))
include(joinpath(startpath,"src","TiledMatrix.jl"))
include(joinpath(startpath,"src","bulgechasing.jl"))
include(joinpath(startpath,"src","tiledalgos.jl"))

batch = parse(Int, ARGS[1])
kswitch=2^10
elty=Float32
tilesize=32
tilefactors=2 .^(0:2:5)

no_tiles_values =  2 .^(1:2:4)
matrixsizes=tilesize.*no_tiles_values
eltypes=[Float16,Float32,Float64,ComplexF32]

function QR_withcopy(A::Matrix)
    copyto!(A, qr!(CuArray(A)).factors)
end

function SVD_withcopy(A::Matrix)
    return Array(svdvals!(CuArray(A),alg=CUDA.CUSOLVER.QRAlgorithm()))
end

#####################################
#QR
#############################

###2D blocks of size 32 in function of matrix size (bidiag, rest) 
tilefactor=1
mydims2d=true
compile_kernels(tilesize=tilesize, tilefactor=tilefactor, tiledim2d=mydims2d)

if (batch==1)
    timings=zeros(length(no_tiles_values), 4)
    for (i,no_tiles) in enumerate(no_tiles_values)
        println(no_tiles)
        A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
        #try
            timings[i,1] =@belapsed CUDA.@sync QR_withcopy($A)
            timings[i,2] =@belapsed CUDA.@sync OOC_QR!($A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
            timings[i,3] =@belapsed CUDA.@sync OOC_QR!($A, kswitch=Int($no_tiles/2), mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
            timings[i,4] =@belapsed CUDA.@sync OOC_QR!($A, kswitch=0, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
        #catch e
        #    println(e)
        #end
    end
    BSON.@save "OOC_vs_incore_QR.bson" timings
end
#plot(matrixsizes, timings, labels= ["CUDA vendor-native QR" "julia-native KA-based QR on GPU" "julia-native OOC QR on GPU (mixed)" "OOC only" ], xaxis=:log2,  lw=2,yaxis=:log10,legend=:outerbottom,
# xticks=(matrixsizes, string.(matrixsizes)), xlabel= "matrix size nxn", ylabel= "time(s)")
 
if (batch==2)
    ###different element-types
    timings=zeros(length(no_tiles_values), 2*length(eltypes))
    for (i,no_tiles) in enumerate(no_tiles_values)
        println(no_tiles)
        for (j,myelty) in enumerate(eltypes)
            A=rand( myelty,tilesize*no_tiles, tilesize*no_tiles)
            try
                timings[i,1+2(j-1)] = @belapsed CUDA.@sync QR_withcopy($A)
            catch e
                println("error - cuda for elty")
            end
            try
                timings[i,2+2(j-1)] =@belapsed CUDA.@sync OOC_QR!($A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
            catch e
                println(e)
            end
        end
    end
    BSON.@save "eltypes_QR.bson" timings
end

if (batch==3)
    ### 2D blocks of other sizes
    timings=zeros(length(matrixsizes), 3+ length(tilefactors))
    for (j, tilefactor) in enumerate(tilefactors)
        compile_kernels(tilesize=tilesize, tilefactor=tilefactor, tiledim2d=mydims2d)
        for (i,no_tiles) in enumerate(no_tiles_values)
            A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
            if (j==1)
                timings[i,1] = @belapsed CUDA.@sync QR_withcopy($A)
            end
            try
                timings[i,j+1] =@belapsed CUDA.@sync OOC_QR!($A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
            catch e
                println(e)
            end
        end
    end
    compile_kernels(tilesize=tilesize, tilefactor=1, tiledim2d=false)
    for (i,no_tiles) in enumerate(no_tiles_values)
        A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
        try
            timings[i,2+ length(tilefactors)] =@belapsed CUDA.@sync OOC_QR!($A, kswitch=kswitch, mydims2d=false,tilesize=tilesize,tilefactor=1)
        catch e
            println(e)
        end
    end
    compile_kernels(tilesize=16, tilefactor=1, tiledim2d=true)
    for (i,no_tiles) in enumerate(no_tiles_values)
        A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
        try
            timings[i,3+ length(tilefactors)] =@belapsed CUDA.@sync OOC_QR!($A, kswitch=kswitch, mydims2d=true,tilesize=16,tilefactor=1)
        catch e
            println(e)
        end
    end
    BSON.@save "blocksizes_QR.bson" timings
end
#plot(matrixsizes, timings, labels= "tilefactor= ".*["1" "4" "16"], xaxis=:log2,  lw=2,yaxis=:log10,legend=:outerbottom,
# xticks=(matrixsizes, string.(matrixsizes)), xlabel= "matrix size nxn", ylabel= "time(s)")



#####################################
#SVD
#############################

tilefactor=1
mydims2d=true
compile_kernels(tilesize=tilesize, tilefactor=tilefactor, tiledim2d=mydims2d)

if (batch==4)
    timings=zeros(length(no_tiles_values), 4)
    for (i,no_tiles) in enumerate(no_tiles_values)
        println(no_tiles)
        A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
        timings[i,1] = @belapsed CUDA.@sync SVD_withcopy($A)
        timings[i,2] =@belapsed CUDA.@sync OOC_Bidiag!($A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
        if (no_tiles<2^11)
            timings[i,3] =@belapsed CUDA.@sync OOC_Bidiag!($A, kswitch=Int($no_tiles/2), mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
            timings[i,4] =@belapsed CUDA.@sync OOC_Bidiag!($A, kswitch=0, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
        end
    end
    BSON.@save "OOC_vs_incore_SVD.bson" timings
end

if (batch==5)
    timings=zeros(length(no_tiles_values), 4)
    for (i,no_tiles) in enumerate(no_tiles_values)
        println(no_tiles)
        A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
        timings[i,1] = @belapsed CUDA.@sync SVD_withcopy($A)
        timings[i,2] =@belapsed CUDA.@sync OOC_Bidiag!($A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
        timings[i,3] =@belapsed CUDA.@sync OOC_SVD!($A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
        timings[i,4] =@belapsed diag_lapack($A)
    end
    BSON.@save "OOC_bulgechasing_SVD.bson" timings
end

if (batch==6)
    ###different element-types
    timings=zeros(length(no_tiles_values), 2*length(eltypes))
    for (i,elty) in enumerate(no_tiles_values)
        println(no_tiles)
        for (j,myelty) in enumerate(eltypes)
            A=rand( myelty,tilesize*no_tiles, tilesize*no_tiles)
            try
                timings[i,1+2(j-1)] = @belapsed CUDA.@sync SVD_withcopy($A)
                timings[i,2+2(j-1)] =@belapsed CUDA.@sync OOC_Bidiag!($A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
            catch e
                println(e)
            end
        end
    end
    BSON.@save "eltypes_SVD.bson" timings
end

if (batch==7)
### 2D blocks of other sizes

    timings=zeros(length(matrixsizes), 3+ length(tilefactors))
    for (j, tilefactor) in enumerate(tilefactors)
        compile_kernels(tilesize=tilesize, tilefactor=tilefactor, tiledim2d=mydims2d)
        timings[i,1] = @belapsed CUDA.@sync SVD_withcopy($A)
        for (i,no_tiles) in enumerate(no_tiles_values)
            A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
            try
                timings[i,j+1] =@belapsed CUDA.@sync OOC_Bidiag!($A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
            catch e
                println(e)
            end
        end
    end
    compile_kernels(tilesize=tilesize, tilefactor=1, tiledim2d=false)
    for (i,no_tiles) in enumerate(no_tiles_values)
        A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
        try
            timings[i,2+ length(tilefactors)] =@belapsed CUDA.@sync OOC_Bidiag!($A, kswitch=kswitch, mydims2d=false,tilesize=tilesize,tilefactor=1)
        catch e
            println(e)
        end
    end
    compile_kernels(tilesize=16, tilefactor=1, tiledim2d=true)
    for (i,no_tiles) in enumerate(no_tiles_values)
        A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
        try
            timings[i,3+ length(tilefactors)] =@belapsed CUDA.@sync OOC_Bidiag!($A, kswitch=kswitch, mydims2d=true,tilesize=16,tilefactor=1)
        catch e
            println(e)
        end
    end
    BSON.@save "blocksizes_SVD.bson" timings

end