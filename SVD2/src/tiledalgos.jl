@inline get_tileview(A, row , col, TILE_SIZEx=TILESIZE, TILE_SIZEy=TILESIZE ) = 
                view(A, (row-1)*TILE_SIZEx.+(1:TILE_SIZEx),
                (col-1)*TILE_SIZEy.+(1:TILE_SIZEy))
@inline get_rowview(A, row, startcol, TILE_SIZEx=TILESIZE, TILE_SIZEy=TILESIZE) =  
                view(A, (row-1)*TILE_SIZEx .+(1:TILE_SIZEx),
                ((startcol-1)*TILE_SIZEy +1):size(A,2))
get_kernel_dims(::KernelAbstractions.Kernel{B,S}) where {B,S} = S.parameters[1]



QR1!(A, Tau, k; koffset=0, singlerow=false) = QR_unsafe_kernel_2d!(backend, (TILESIZE))( get_tileview(A, singlerow ? 1 : k+koffset,k), 
                                    get_tileview(Tau, k, 1, 1, TILESIZE), ndrange=(TILESIZE)) 
QR2!(A, Tau, k, row,; koffset=0, singlerow=false, A2=A ) =QR_unsafe_kernel2_2d!(backend, (TILESIZE, QRSPLIT))(get_tileview(A,singlerow ? 1 : k+koffset,k), 
                                    get_tileview(A2, singlerow ? 1 : row,k), 
                                    get_tileview(Tau, row, 1, 1, TILESIZE), ndrange=(TILESIZE,QRSPLIT))

Qtapply1_par!(A, Tau, k; koffset=0, singlerow=false) = applyQorQt_unsafe_kernel_2d!(backend, (TILESIZE))(get_rowview(A, singlerow ? 1 : k+koffset, k+1), 
                                    get_tileview(A, singlerow ? 1 : k+koffset,k), 
                                    get_tileview(Tau, k,1, 1, TILESIZE), ndrange=( size(A,2)-k*TILESIZE) )
Qtapply2_par!(A, Tau, k, row; koffset=0, singlerow=false, A2=A) = applyQorQt_unsafe_kernel2_2d!(backend, (TILESIZE))(get_rowview(A,singlerow ? 1 : k+koffset, k+1), 
                                    get_rowview(A2, singlerow ? 1 : row, k+1), 
                                    get_tileview(A2, singlerow ? 1 : row,k), 
                                    get_tileview(Tau, row,1, 1, TILESIZE), ndrange=( size(A,2)-k*TILESIZE))
                          

function OOC_alg!(A::Matrix{T}, f::Function,backend::Backend; kswitch::Int=6) where {T}
    n=size(A,1)
    nb_tiles= Int(n/TILESIZE)
    Tau=KernelAbstractions.zeros(backend, T, nb_tiles, TILESIZE)
    
    if (n>kswitch*TILESIZE)
        f(LargeTiledMatrix(A, backend) ,Tau, nb_tiles;kend=kswitch)
        KernelAbstractions.synchronize(backend)
    end
    if (kswitch>0)
        Agpu = KernelAbstractions.allocate(backend, T, min(n,kswitch*TILESIZE), min(n,kswitch*TILESIZE))
        copyto!(Agpu,view(A,max(n-kswitch*TILESIZE+1,1):n,max(n-kswitch*TILESIZE+1,1):n))
        copyto!(view(A,max(n-kswitch*TILESIZE+1,1):n,max(n-kswitch*TILESIZE+1,1):n), f(Agpu, Tau,min(kswitch,nb_tiles)))    
    end
    return A  
end

OOC_QR!(A::Matrix, backend=KernelAbstractions.get_backend(CUDA.randn(2)); kswitch=6) = OOC_alg!(A, mygeqrf!, backend;kswitch=kswitch)
OOC_Bidiag!(A::Matrix, backend=KernelAbstractions.get_backend(CUDA.randn(2)); kswitch=6) = OOC_alg!(A, myblockdiag!, backend;kswitch=kswitch)
OOC_SVD!(A::Matrix, backend=KernelAbstractions.get_backend(CUDA.randn(2)); kswitch::Int=6) = banddiagsvd(OOC_Bidiag!(A,backend;kswitch=kswitch), TILESIZE)


function mygeqrf!(A, Tau, nbtiles;kend=0)
    for k in 1:(nbtiles-kend)
        QRandmulQt1!(A, Tau,k)

        for row in k+1:nbtiles
            QRandmulQt2!(A,Tau, k, row)
        end
    end
    finish_algo!(A, kend)
    return A
end

function myblockdiag!(A, Tau, nbtiles; kend=0)
    for k in 1:(nbtiles-kend)
        QRandmulQt1!(A, Tau,k;SVDalg=true)
        for row in k+1:nbtiles
            QRandmulQt2!(A, Tau,k,row)
        end
        (k==nbtiles) && break

        QRandmulQt1!(A', Tau,k; LQ=true, SVDalg=true)
        for col in k+2:nbtiles 
            QRandmulQt2!(A', Tau,k,col; LQ=true)
        end
        

    end
    finish_algo!(A, kend; SVDalg=true)
    return A
end

function banddiagsvd(A::AbstractGPUorCPUMat, bw::Int)
    d,e = gbbrd!(A, bw)
    return LAPACK.bdsdc!('U', 'N', d, e)[1]
end


function mygesvd!(A::AbstractGPUMatrix)
    nbtiles=Int(size(A,1)/TILESIZE)
    Tau=CUDA.zeros(nbtiles,size(A,2))
    myblockdiag!(A,Tau,nbtiles)
    return banddiagsvd(A,TILESIZE)
end


function QRandmulQt1!(A::AnyGPUMatrix, Tau,k; singlerow=false, LQ=false, SVDalg=false)
    QR1!(A,Tau, k;koffset=Int(LQ),singlerow=singlerow)
    Qtapply1_par!(A, Tau, k; koffset=Int(LQ), singlerow=singlerow)
    triu!(get_tileview(A, singlerow ?  1 : k+Int(LQ),k))
end 

function QRandmulQt2!(A::AnyGPUMatrix,Tau, k::Int, row::Int; singlerow=false, LQ=false, A2::AnyGPUMatrix=A )
    QR2!(A,Tau, k, row; koffset=Int(LQ), singlerow=singlerow, A2=A2)
    Qtapply2_par!(A,Tau, k,row; koffset=Int(LQ), singlerow=singlerow, A2=A2)
    fill!(get_tileview(A2, singlerow ?  1 : row,k),0)
end

function QRandmulQt1!(A::LargeTiledMatrix, Tau, k::Int; LQ=false, SVDalg=false)
    ((k>1) || LQ) && getandset_first!(SVDalg ? A' : A,k,LQ, SVDalg  )
    KernelAbstractions.synchronize(A.backend)
    @sync begin

        Threads.@spawn begin
            if (k+Int(LQ)<A.nb_tiles)
                recycle_tilerow(A, k, k+1+Int(LQ), !LQ, false, false)
            else
                push!(A.TileRows, A.TileRows[2])
            end
            KernelAbstractions.synchronize(A.backend)
        end
            
        Threads.@spawn begin
            QRandmulQt1!(A.TileRows[1], Tau,k; singlerow=true)
            KernelAbstractions.synchronize(A.backend)
        end
    end
    deleteat!(A.TileRows,2)
end

function QRandmulQt2!(A::LargeTiledMatrix,Tau,k::Int, row::Int ; LQ::Bool=false) 
    @sync begin
        Threads.@spawn begin
            (row>k+1+Int(LQ)) && (get_tilerow(A,k,row-1 ,!LQ, 3 )) 
            KernelAbstractions.synchronize(A.backend)
        end 

        Threads.@spawn begin
            if (row<A.nb_tiles)
                recycle_tilerow(A, k, row+1, !LQ, false, false)
            else
                push!(A.TileRows, A.TileRows[2])
            end
            KernelAbstractions.synchronize(A.backend)
        end

        Threads.@spawn begin
            QRandmulQt2!(A.TileRows[1], Tau,k, row; singlerow=true, LQ=LQ, A2=A.TileRows[4])
            KernelAbstractions.synchronize(A.backend)
        end
    end
    deleteat!(A.TileRows,2)
end


function getandset_first!(A::LargeTiledMatrix, k2::Int, LQ::Bool, SVDalg::Bool )
    temp=A.TileRows[2]
    A.TileRows[2]=A.TileRows[1]
    A.TileRows[1]=temp
    k= k2 -Int(!LQ&& k2>1)
    @sync begin
        Threads.@spawn begin
            (k<A.nb_tiles) &&  get_tilerow(A,k,A.nb_tiles ,!SVDalg || LQ, 3 )
            KernelAbstractions.synchronize(backend)

        end
        Threads.@spawn begin
            recycle_tilerow(SVDalg ? A' : A, k2, k2+Int(LQ),!LQ,true, SVDalg )
            KernelAbstractions.synchronize(A.backend)
        end
        Threads.@spawn begin
            get_tilerow(A, k, k+Int(SVDalg && !LQ), !SVDalg || LQ, 2)
            KernelAbstractions.synchronize(A.backend)
        end
    end
    KernelAbstractions.synchronize(A.backend)

    if (SVDalg)
        transpose!(get_tileview(A.TileRows[1], 1 , k2 ), get_tileview(A.TileRows[2], 1 , k+1 ))
        k2<A.nb_tiles && transpose!(get_tileview(A.TileRows[1], 1 , A.nb_tiles ), get_tileview(A.TileRows[3], 1 , k+1))
    elseif (k2==A.nb_tiles)
        copyto!(get_tileview(A.TileRows[1], 1 , k2 ), get_tileview(A.TileRows[3], 1 , k2 ))
    end

    KernelAbstractions.synchronize(A.backend)
end


function finish_algo!(A::LargeTiledMatrix,  kend::Int; SVDalg::Bool=false)
    lastR = !SVDalg || kend==0
    lasttile = SVDalg && kend==0
    if !lasttile
        get_tilerow(SVDalg ? A' : A, (A.nb_tiles-kend), A.nb_tiles, lastR, 3)
    end
    get_tilerow((SVDalg && kend!=0) ? A' : A, (A.nb_tiles-kend), (A.nb_tiles-kend)+Int(!lastR), lastR, 1)
    
    KernelAbstractions.synchronize(A.backend)
end

finish_algo!(A::AbstractGPUMatrix, kend::Int; SVDalg::Bool=false ) = nothing

