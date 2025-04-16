
########################################
# GPU- only ##
########################################

function tiled_QR!(A::GeneralTiledMatrix; kend=A.no_tiles)
    verify_kerneldims(A)
    for k in 1:kend
        QRandmulQt1!(A,k)
        for row in k+1:A.no_tiles
            QRandmulQt2!(A,row,k)
        end
    end
    finish_algo!(A, kend) 
end

function Blockbidiag!(A::GeneralTiledMatrix; kend=A.no_tiles) 
    verify_kerneldims(A)
    for k in 1:kend

        QRandmulQt1!(A,k, alg="SVD")
        for row in k+1:A.no_tiles
            QRandmulQt2!(A,row,k, alg="SVD")
        end

        (k==A.no_tiles) && break

        LQandmulQt1!(A,k, alg="SVD")
        for col in k+2:A.no_tiles 
            LQandmulQt2!(A,k,col, alg="SVD")
        end
    end
    finish_algo!(A, kend, alg="SVD")
end


function QRandmulQt1!(A::TiledMatrix, k::Int; alg="QR")
    QR1!(A,k)
    Qtapply1_par!(A, k)
    alg=="SVD" && triu!(A,k,k)
end

function QRandmulQt2!(A::TiledMatrix, row::Int, k::Int; alg="QR")
    QR2!(A,row,k)
    Qtapply2_par!(A,row,k)
    alg=="SVD" && ((A.TileViews[row,k]).=0)
end

function LQandmulQt1!(A::TiledMatrix, k::Int; alg="QR")
    LQ1!(A,k) 
    LQtapply1_par!(A,k)
    alg=="SVD" && tril!(A,k,k+1)
end

function LQandmulQt2!(A::TiledMatrix, k::Int, col::Int; alg="QR")
    LQ2!(A,k,col)
    LQtapply2_par!(A, k,col)
    alg=="SVD" && (A.TileViews[k,col].=0)
end

function verify_kerneldims(A::GeneralTiledMatrix)
    if A.m != get_kernel_dims(applyQR1!)[1]
        error("Kernels compiled at wrong size, re-define TILE_SIZE1 and TILE_SIZE2 and recompile kernels.")
    end
end

finish_algo!(::TiledMatrix, ::Int;alg="QR") = nothing

########################################
# including communication CPU-GPU ##
########################################



function QRandmulQt1!(A::LargeTiledMatrix, k::Int,prevR::Bool,currentR::Bool, zerofactor::Bool)
    if (k>1 && currentR)
        getandset_first!(A, k-1,k, prevR, currentR )
    elseif !currentR
        getandset_first!(A, k,k, prevR, currentR )
    end 
    CUDA.synchronize()
    @sync begin
        Threads.@spawn begin
            CUDA.@sync begin
                if (k+Int(!currentR)<A.no_tiles)
                    recycle_tilerow(A, k, k+1+Int(!currentR), currentR, false, false)
                else
                    push!(A.Rows, TileRow(currentR, k, 0, A.Rows[2].RowTile, A.Rows[2].Tau))
                end
            end
        end
            
        Threads.@spawn begin
            CUDA.@sync begin
                QR1!(A,k)
                Qtapply1_par!(A, k)
                zerofactor && triu!(A,1)
            end
        end
    end
    finish_recycle!(A)
    CUDA.synchronize()
end



QRandmulQt1!(A::LargeTiledMatrix, k::Int; alg="QR") = QRandmulQt1!(A, k, alg=="QR",true, alg=="SVD")
LQandmulQt1!(A::LargeTiledMatrix, k::Int; alg="QR") = QRandmulQt1!(A, k, true,false, alg=="SVD")


function QRandmulQt2!(A::LargeTiledMatrix, row::Int, k::Int, zerofactor::Bool) 
    @sync begin
        Threads.@spawn begin
            CUDA.@sync begin
                if (row>k+1)
                    (get_tilerow(A,k,row-1 ,true, 3 )) 
                end
            end
        end 

        Threads.@spawn begin
            CUDA.@sync begin
                if (row<A.no_tiles)
                    recycle_tilerow(A, k, row+1, true, false, false)
                else
                    push!(A.Rows, TileRow(true, k, 0, A.Rows[2].RowTile, A.Rows[2].Tau))
                end
            end
        end

        Threads.@spawn begin
            CUDA.@sync begin
                QR2!(A,row,k)
                Qtapply2_par!(A,k)
                zerofactor && (get_view_first(A,4).=0)
            end
        end
    end
    finish_recycle!(A)
    CUDA.synchronize()
end


QRandmulQt2!(A::LargeTiledMatrix, row::Int, k::Int; alg="QR") = QRandmulQt2!(A, row, k, alg=="SVD") 

function LQandmulQt2!(A::LargeTiledMatrix, k::Int, col::Int, zerofactor::Bool) 
    @sync begin
        Threads.@spawn begin
            CUDA.@sync begin
                if (col>k+2)
                    get_tilerow(A,k,col-1 ,false, 3 )
                end
            end
        end 

        Threads.@spawn begin
            CUDA.@sync begin
                if (col<A.no_tiles)
                    recycle_tilerow(A, k, col+1, false, false, false)
                else
                    push!(A.Rows, TileRow(false, k, 0, A.Rows[2].RowTile, A.Rows[2].Tau))
                end
            end
        end

        Threads.@spawn begin
            CUDA.@sync begin
                QR2!(A,k,col)
                Qtapply2_par!(A,k)
                
                zerofactor && (get_view_first(A,4).=0)
            end
        end
    end
    finish_recycle!(A)
    CUDA.synchronize()
end

LQandmulQt2!(A::LargeTiledMatrix, k::Int, col::Int; alg="QR") = LQandmulQt2!(A, k,col, alg=="SVD") 

function getandset_first!(A::LargeTiledMatrix, k::Int,k2::Int, prevR::Bool, nextR::Bool )
    temp=A.Rows[2]
    A.Rows[2]=A.Rows[1]
    A.Rows[1]=temp
    
    @sync begin
        Threads.@spawn begin
            CUDA.@sync begin 
                if (k<A.no_tiles) 
                    get_tilerow(A,k,A.no_tiles ,prevR, 3 )
                end
            end
        end
        Threads.@spawn begin
            CUDA.@sync begin
                recycle_tilerow(A, k2, k2+Int(!nextR),nextR,true, prevR!=nextR )
            end
        end
        Threads.@spawn begin
            CUDA.@sync begin
                get_tilerow(A, k, k+Int(!prevR), prevR, 2)
            end
        end
        Threads.@spawn begin
            CUDA.@sync begin
                if (prevR!=nextR)
                    mytranspose!(view(A.Rows[1].RowTile, 1:A.n, 1:A.m),view(A.Rows[2].RowTile, 1:A.n, A.m.+(1:A.m)), ndrange=(A.n,A.m))
                end
            end
        end
    end
    if (prevR!=nextR && k2<A.no_tiles) 
        mytranspose!(view(A.Rows[1].RowTile,1:A.n, A.m.*(A.no_tiles-k2).+(1:A.m) ), view(A.Rows[3 ].RowTile,1:A.n, A.m.+(1:A.m) ), ndrange=(A.n,A.m))
    elseif (prevR==nextR && k2==A.no_tiles)
        view(A.Rows[1].RowTile, 1:A.n, (1:A.m)) .= view(A.Rows[3].RowTile, 1:A.n, A.m.+(1:A.m))
    end
end

function finish_algo!(A::LargeTiledMatrix, k::Int, lastR::Bool, lasttile::Bool)
    !lasttile && CUDA.@sync get_tilerow(A, k-Int(k==A.no_tiles), A.no_tiles, lastR, 3)
    CUDA.@sync get_tilerow(A, k, k+Int(!lastR), lastR, 1)
end

finish_algo!(A::LargeTiledMatrix, k::Int; alg="QR") = finish_algo!(A::LargeTiledMatrix, k::Int, (alg=="QR" || k ==A.no_tiles), (alg=="SVD" && k ==A.no_tiles))


########################################
# combined ##
########################################


function OOC_alg!(A::Matrix, f::Function; kswitch=10, mydims2d=false,tilesize=32,tilefactor=1, backend=KernelAbstractions.get_backend(CUDA.randn(2)))
    n=size(A,1)
    no_blocks= Int(n/tilesize)
    if (n>(kswitch*tilesize))
        Atile = LargeTiledMatrix(A, backend, tilesize*no_blocks, tilesize, no_blocks, tiledim2d= mydims2d ? Int(tilesize/tilefactor) : 0) 
        f(Atile;kend=no_blocks-kswitch)
        if (kswitch>0) 
            Acu=CuArray(A[(n-kswitch*tilesize+1):n,(n-kswitch*tilesize+1):n])
        end
    else
        Acu = CuArray(A)
    end
    CUDA.synchronize()
    if (kswitch>0)
        Acutile=TiledMatrix(Acu,tilesize,tilesize, tiledim2d= mydims2d ? Int(tilesize/tilefactor) : 0)
        f(Acutile)
        copyto!(view(A,(n-min(kswitch,no_blocks)*tilesize+1):n,(n-min(kswitch,no_blocks)*tilesize+1):n), Array(Acu))    
    end  
    CUDA.synchronize()
end

OOC_QR!(A::Matrix; kswitch=10, mydims2d=false,tilesize=32,tilefactor=1, backend=KernelAbstractions.get_backend(CUDA.randn(2))) = OOC_alg!(A, tiled_QR!, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor, backend=backend)
OOC_Bidiag!(A::Matrix; kswitch=10, mydims2d=false,tilesize=32,tilefactor=1, backend=KernelAbstractions.get_backend(CUDA.randn(2))) = OOC_alg!(A, Blockbidiag!, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor, backend=backend)

function OOC_SVD!(A::Matrix,; kswitch=10, mydims2d=false,tilesize=32,tilefactor=1, backend=KernelAbstractions.get_backend(CUDA.randn(2)))
    OOC_Bidiag!(A, kswitch=kswitch, mydims2d=mydims2d,tilesize=tilesize,tilefactor=tilefactor, backend=backend)
    bidiag=bidiagonalize(A,tilesize);
    return bidiag #diag_lapack(bidiag)
end

function compile_kernels(;tilesize::Int=32, tilefactor::Int=1, tiledim2d::Bool=false)
    global TILE_SIZE1 = tiledim2d ? (tilesize,tilesize) : tilesize
    global TILE_SIZE2 = tiledim2d ? (tilesize,Int(tilesize/tilefactor)) : tilesize
    global dims2d = tiledim2d
    include(joinpath("..","src","QRkernelscompile.jl"))
end