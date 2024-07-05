

include("QRkernels.jl")
include("TiledMatrix.jl")

function tiled_QR!(A::TiledMatrix, T)
    for k in 1:A.no_tiles
        QR1!(A,T,k)
        for i in k+1:A.no_tiles
            Qtapply1!(A,T,k,i)
        end
        for row in k+1:A.no_tiles
            QR2!(A,T,row,k)
            for col in k+1:A.no_tiles
                Qtapply2!(A,T,k,row,col)
            end
        end
    end    
end

function Blockbidiag_nonpar!(A,T) 

    for k in 1:A.no_tiles
        QR1!(A,T,k)                
        Qtapply1_blocks!(A, T, k)
        triu!(A,k,k)

        for row in k+1:A.no_tiles
            QR2!(A,T,row,k)
            Qtapply2_blocks!(A,T,row,k)
            A.TileViews[row,k].=0
        end

        (k==A.no_tiles) && break

        LQ1!(A,T,k) 
        LQtapply1_blocks!(A,T,k)
        tril!(A,k,k+1)

        for col in k+2:A.no_tiles 
            LQ2!(A,T,k,col)
            LQtapply2_blocks!(A, T, k, col)
            A.TileViews[k,col].=0
        end
    end

end

function Blockbidiag!(A,T) 

    for k in 1:A.no_tiles
        QR1!(A,T,k)              
        Qtapply1_blockspar!(A, T, k)
        triu!(A,k,k)

        for row in k+1:A.no_tiles
            QR2!(A,T,row,k)
            Qtapply2_blockspar!(A,T,row,k)
            A.TileViews[row,k].=0
        end

        (k==A.no_tiles) && break

        LQ1!(A,T,k) 
        LQtapply1_blockspar!(A,T,k)
        tril!(A,k,k+1)

        for col in k+2:A.no_tiles 
            LQ2!(A,T,k,col)
            LQtapply2_blockspar!(A, T, k,col)
            A.TileViews[k,col].=0
        end
    end

end


#=tilesize=3
no_blocks=3
backend=KernelAbstractions.get_backend(CUDA.randn(2))
T=Float32

Tau=Array{CuArray}(undef, no_blocks,no_blocks)
for i in 1:no_blocks
    for j in 1:no_blocks
        Tau[i,j]=KernelAbstractions.zeros(backend, T, tilesize)
    end
end

A=rand!(allocate(backend, T,tilesize*no_blocks, tilesize*no_blocks))



    testmatrix=copy(A)
    refmatrix=copy(A)
    Atile = TiledMatrix(testmatrix,tilesize,tilesize)
    
    
    #Blockbidiag!(Atile,Tau)
    #svdvals(testmatrix) ≈ svdvals(refmatrix)

    T=Tau
    A=Atile
    k=1
        QR1!(A,T,k)                
        Qtapply1_blocks!(A, T, k)
        
        
        norm(testmatrix,2) ≈ norm(refmatrix,2)

        for row in k+1:A.no_tiles
            QR2!(A,T,row,k)
            Qtapply2_blocks!(A,T,k,row)
            A.TileViews[row,k].=0
            println(norm(testmatrix,2) ≈ norm(refmatrix,2))
        end
        
        (k==A.no_tiles)

        LQ1!(A,T,k) 
        LQtapply1_blocks!(A,T,k)
        

        norm(testmatrix,2) ≈ norm(refmatrix,2)

        for col in k+2:A.no_tiles 
            LQ2!(A,T,col,k)
            LQtapply2_blocks!(A, T, col, k)
            A.TileViews[col,k].=0
            println(norm(testmatrix,2) ≈ norm(refmatrix,2))
        end
    =#

 