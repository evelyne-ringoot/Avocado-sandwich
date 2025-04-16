using KernelAbstractions,CUDA, BSON, BenchmarkTools, Random, LinearAlgebra
startpath=".."
include(joinpath(startpath,"src","QRkernels.jl"))
TILE_SIZE1=1
TILE_SIZE2=1 
dims2d=false 
include(joinpath(startpath,"src","QRkernelscompile.jl"))
include(joinpath(startpath,"src","TiledMatrix.jl"))
include(joinpath(startpath,"src","bulgechasing.jl"))
include(joinpath(startpath,"src","tiledalgos.jl"))

batch = 7# parse(Int, ARGS[1])
println(batch)
kswitch=2^7
elty=Float32
tilesize=32
tilefactors=2 .^(0:5)
batchvals=[1,1,8,1,1,1,7,1]

no_tiles_values =  2 .^(1:7)
no_tiles_values_long =  2 .^(7:10)
matrixsizes=tilesize.*no_tiles_values
eltypes=[Float16,Float32,Float64,ComplexF32]
BSONfiles=["OOC_vs_incore_","OOC_bulgechasing_","eltypes_","blocksizes_"]

function QR_withcopy(A::Matrix)
    copyto!(A, qr!(CuArray(A)).factors)
end

function SVD_withcopy(A::Matrix)
    B=CuArray(A)
    C=svdvals!(B,alg=CUDA.CUSOLVER.QRAlgorithm())
    return Array(C)
end

function mybelapsed(f, A, args...;kwargs...)
    if (size(A,1)<kswitch*32)
        CUDA.@sync f(copy(A),args...;kwargs...)
        t=0.0
        k=0
        while (t<1)
            B=copy(A)
            t+= @elapsed (CUDA.@sync f(B,args...;kwargs...))
            B=nothing
            CUDA.reclaim()
            k+=1
        end
    else
        t=@elapsed (CUDA.@sync f(A,args...;kwargs...))
        k=1
    end
    return t/k
end

#####################################
#SVD
#############################

tilefactor=1
mydims2d=true
compile_kernels(tilesize=tilesize, tilefactor=tilefactor, tiledim2d=mydims2d)

if (batch==batchvals[1])
    timings=zeros(length(no_tiles_values), 4)
        for (i,no_tiles) in enumerate([no_tiles_values[7]])
            println(no_tiles)
            A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
            
            
            if (no_tiles==kswitch)
                timings[i,3] =mybelapsed( OOC_Bidiag!,A, kswitch=Int(no_tiles/2), mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
                #timings[i,1] = mybelapsed( SVD_withcopy, A)
                timings[i,4] =mybelapsed( OOC_Bidiag!,A, kswitch=0, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
                
            else
                #timings[i,2] = mybelapsed( OOC_Bidiag!,A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
            end
            BSON.@save BSONfiles[1]*"SVD.bson" timings
        end
       
end

if (batch==batchvals[2])
    timings=zeros(length(no_tiles_values), 4)
        for (i,no_tiles) in enumerate(no_tiles_values)
            println(no_tiles)
            A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
            #timings[i,1] = mybelapsed( SVD_withcopy,A)
            
            timings[i,2] =mybelapsed(OOC_Bidiag!,A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
            
            timings[i,3] =mybelapsed(bidiagonalize,A,tilesize)
            
            timings[i,4] =mybelapsed( diag_lapack,A)

            BSON.@save BSONfiles[2]*"SVD.bson" timings
        end
end

#####################################
#QR
#############################

###2D blocks of size 32 in function of matrix size (bidiag, rest) 
tilefactor=1
mydims2d=true
compile_kernels(tilesize=tilesize, tilefactor=tilefactor, tiledim2d=mydims2d)

if (batch==batchvals[3])
    timings=zeros(length(no_tiles_values), 4)

        for (i,no_tiles) in enumerate([no_tiles_values[7]])
            println(no_tiles)
            A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
            try
                
                if (no_tiles==kswitch)
                    #timings[i,1] =mybelapsed( QR_withcopy,A)
                    timings[i,3] =mybelapsed(OOC_QR!,A, kswitch=Int(no_tiles/2), mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
                    timings[i,4] =mybelapsed(OOC_QR!,A, kswitch=0, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
                    #timings[i,2] =mybelapsed(OOC_QR!,A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
                end
            catch e
                println(e)
            end
            BSON.@save BSONfiles[1]*"QR3.bson" timings
        end  

end

 
if (batch==batchvals[4])
    ###different element-types
    timings=zeros(length(no_tiles_values), 2*length(eltypes))
        for (i,no_tiles) in enumerate(no_tiles_values)
            println(no_tiles)
            for (j,myelty) in enumerate(eltypes)
                A=rand( myelty,tilesize*no_tiles, tilesize*no_tiles)
                try
                    timings[i,1+2(j-1)] = mybelapsed( QR_withcopy,A)
                    
                catch e
                    println("error - cuda for elty")
                end
                try
                    timings[i,2+2(j-1)] =mybelapsed(OOC_QR!,A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
                    
                catch e
                    println(e)
                end
            end
            BSON.@save BSONfiles[3]*"QR.bson" timings
        end

end

if (batch==batchvals[5])
    ### 2D blocks of other sizes
    timings=zeros(length(matrixsizes), 3+ length(tilefactors))

        for (j, tilefactor) in enumerate(tilefactors)
            compile_kernels(tilesize=tilesize, tilefactor=tilefactor, tiledim2d=mydims2d)
            for (i,no_tiles) in enumerate(no_tiles_values)
                A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
                if (j==1)
                    #timings[i,1] = mybelapsed(QR_withcopy,A)
                    
                end
                try
                    timings[i,j+1] =mybelapsed(OOC_QR!,A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
                    
                catch e
                    println(e)
                end
            end
            BSON.@save BSONfiles[4]*"QR.bson" timings
        end
        compile_kernels(tilesize=tilesize, tilefactor=1, tiledim2d=false)
        for (i,no_tiles) in enumerate(no_tiles_values)
            A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
            try
                timings[i,2+ length(tilefactors)] =mybelapsed( OOC_QR!,A, kswitch=kswitch, mydims2d=false,tilesize=tilesize,tilefactor=1)
                
            catch e
                println(e)
            end
            BSON.@save BSONfiles[4]*"QR.bson" timings
        end
        compile_kernels(tilesize=16, tilefactor=1, tiledim2d=true)
        for (i,no_tiles) in enumerate(no_tiles_values)
            A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
            try
                timings[i,3+ length(tilefactors)] =mybelapsed(OOC_QR!,A, kswitch=kswitch, mydims2d=true,tilesize=16,tilefactor=1)
                
            catch e
                println(e)
            end
            BSON.@save BSONfiles[4]*"QR.bson" timings
        end

end



#####################################
#SVD
#############################

tilefactor=1
mydims2d=true
compile_kernels(tilesize=tilesize, tilefactor=tilefactor, tiledim2d=mydims2d)

if (batch==batchvals[6])
    println("yes")
    ###different element-types
    timings=zeros(length(no_tiles_values), 2*length(eltypes))
        for (i,no_tiles) in enumerate(no_tiles_values[6:7])
            println(no_tiles)
            for (j,myelty) in enumerate(eltypes)
                A=rand( myelty,tilesize*no_tiles, tilesize*no_tiles)
                try
                    timings[i+5,1+2(j-1)] = mybelapsed(SVD_withcopy,A) 
                catch e
                end
                timings[i+5,2+2(j-1)] =mybelapsed( OOC_Bidiag!,A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
            end
            BSON.@save BSONfiles[3]*"SVD3.bson" timings
        end

end

if (batch==batchvals[7])
    println(batchvals[7])
### 2D blocks of other sizes

    timings=zeros(length(matrixsizes), 3+ length(tilefactors))

        for (j, tilefactor) in enumerate(tilefactors)
            compile_kernels(tilesize=tilesize, tilefactor=tilefactor, tiledim2d=mydims2d)
            for (i,no_tiles) in enumerate(no_tiles_values)
                A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
                if (j==1)
                    #timings[i,1] = mybelapsed(SVD_withcopy,A)
                    
                end
                try
                    timings[i,j+1] =mybelapsed(OOC_Bidiag!, A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
                    
                catch e
                    println(e)
                end
            end
            BSON.@save BSONfiles[4]*"SVD.bson" timings
        end
        compile_kernels(tilesize=tilesize, tilefactor=1, tiledim2d=false)
        for (i,no_tiles) in enumerate(no_tiles_values)
            A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
            try
                timings[i,2+ length(tilefactors)] =mybelapsed( OOC_Bidiag!,A, kswitch=kswitch, mydims2d=false,tilesize=tilesize,tilefactor=1)
                
            catch e
                println(e)
            end
            BSON.@save BSONfiles[4]*"SVD.bson" timings
        end
        compile_kernels(tilesize=16, tilefactor=1, tiledim2d=true)
        for (i,no_tiles) in enumerate(no_tiles_values)
            A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
            try
                timings[i,3+ length(tilefactors)] =mybelapsed( OOC_Bidiag!,A, kswitch=kswitch, mydims2d=true,tilesize=16,tilefactor=1)
                
            catch e
                println(e)
            end
            BSON.@save BSONfiles[4]*"SVD.bson" timings
        end




end

tilefactor=4
mydims2d=true
compile_kernels(tilesize=tilesize, tilefactor=tilefactor, tiledim2d=mydims2d)
if (batch==batchvals[8])
    timings=zeros(length(no_tiles_values_long), 2)
        for (i,no_tiles) in enumerate([no_tiles_values_long[4]])
            println(no_tiles)
            A=rand( elty,tilesize*no_tiles, tilesize*no_tiles)
            B=copy(A)
            timings[i+3,1] =mybelapsed( OOC_QR!,A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
            BSON.@save "longtimings"*string(i+3)*"1.bson" timings
            timings[i+3,2] =mybelapsed( OOC_Bidiag!,B, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor)
            BSON.@save "longtimings"*string(i+3)*"2.bson" timings
        end
end