#TODO
#non-square and non-power of two matrices, and non-multiples of tilesize
#adding singular vectors
#include openblas gbbrd from nextLA
#remove some of the copies by fixing copies between views of GPU and CPU arrays
#verify type stability
#naming conventions

@inline get_tileview(A, row , col, TILE_SIZEx=TILESIZE, TILE_SIZEy=TILESIZE ) = 
                view(A, (row-1)*TILE_SIZEx.+(1:TILE_SIZEx),
                (col-1)*TILE_SIZEy.+(1:TILE_SIZEy))
@inline get_rowview(A, row, startcol, TILE_SIZEx=TILESIZE, TILE_SIZEy=TILESIZE; endcol=Int(size(A,2)/TILE_SIZEy)) =  
                view(A, (row-1)*TILE_SIZEx .+(1:TILE_SIZEx),
                ((startcol-1)*TILE_SIZEy +1):endcol*TILE_SIZEy)
get_kernel_dims(::KernelAbstractions.Kernel{B,S}) where {B,S} = S.parameters[1]



QR1!(A, Tau, k; koffset=0, singlerow=false, colinmem::Int=k) = QR_unsafe_kernel_2d!(backend, (TILESIZE))( get_tileview(A, singlerow ? 1 : k+koffset,colinmem), 
                                    get_tileview(Tau, k, 1, 1, TILESIZE), ndrange=(TILESIZE)) 
QR2!(A, Tau, k, row,; koffset=0, singlerow=false, A2=A, colinmem::Int=k) =QR_unsafe_kernel2_2d!(backend, (TILESIZE, QRSPLIT))(get_tileview(A,singlerow ? 1 : k+koffset,colinmem), 
                                    get_tileview(A2, singlerow ? 1 : row,colinmem), 
                                    get_tileview(Tau, row, 1, 1, TILESIZE), ndrange=(TILESIZE,QRSPLIT))

Qtapply1_par!(A, Tau, k; koffset=0, singlerow=false, colinmem::Int=k) = applyQorQt_unsafe_kernel_2d!(backend, (TILESIZE))(get_rowview(A, singlerow ? 1 : k+koffset, colinmem+1), 
                                    get_tileview(A, singlerow ? 1 : k+koffset,colinmem), 
                                    get_tileview(Tau, k,1, 1, TILESIZE), ndrange=( size(A,2)-colinmem*TILESIZE) )
Qtapply2_par!(A, Tau, k, row; koffset=0, singlerow=false, A2=A, colinmem::Int=k) = applyQorQt_unsafe_kernel2_2d!(backend, (TILESIZE))(get_rowview(A,singlerow ? 1 : k+koffset, colinmem+1), 
                                    get_rowview(A2, singlerow ? 1 : row, colinmem+1), 
                                    get_tileview(A2, singlerow ? 1 : row,colinmem), 
                                    get_tileview(Tau, row,1, 1, TILESIZE), ndrange=( size(A,2)-colinmem*TILESIZE))
                          

function OOC_alg!(A::Matrix{T}, f::Function,backend::Backend, kswitch::Int,tilesinmem::Int) where {T}
    n=size(A,1)
    nb_tiles= Int(n/TILESIZE)
    Tau=KernelAbstractions.zeros(backend, T, nb_tiles, TILESIZE)
    
    if (n>kswitch*TILESIZE)
        inputmatrix=LargeTiledMatrix(A, backend, tilesinmem)
        f(inputmatrix ,Tau, nb_tiles;kend=kswitch)
        clear!(inputmatrix)
    end
    if (kswitch>0)
        Agpu = KernelAbstractions.allocate(backend, T, min(n,kswitch*TILESIZE), min(n,kswitch*TILESIZE))
        copyto!(Agpu,A[max(n-kswitch*TILESIZE+1,1):n,max(n-kswitch*TILESIZE+1,1):n])
        f(Agpu, Tau,min(kswitch,nb_tiles))
        KernelAbstractions.synchronize(backend)
        temp=zeros(T,min(kswitch*TILESIZE,n),min(kswitch*TILESIZE,n))
        copyto!(temp,Agpu)
        copyto!(view(A,max(n-kswitch*TILESIZE+1,1):n,max(n-kswitch*TILESIZE+1,1):n), temp)  
    end
    return A  
end

OOC_QR!(A::Matrix, backend; kswitch=128, tilesinmem::Int=max(floor(Int,kswitch^2/4),2)) = OOC_alg!(A, mygeqrf!, backend,kswitch, tilesinmem)
OOC_Bidiag!(A::Matrix, backend; kswitch=128, tilesinmem::Int=max(floor(Int,kswitch^2/4),2)) = OOC_alg!(A, myblockdiag!, backend,kswitch, tilesinmem)
OOC_SVD!(A::Matrix, backend; kswitch::Int=128, tilesinmem::Int=max(floor(Int,kswitch^2/4),2)) = banddiagsvd(OOC_Bidiag!(A,backend,kswitch=kswitch, tilesinmem=tilesinmem), TILESIZE)


function mygeqrf!(A, Tau, nbtiles;kend=0)
    for k in 1:(nbtiles-kend)
        QRandmult!(A,Tau,k, nbtiles)
    end
    return A
end

function myblockdiag!(A, Tau, nbtiles; kend=0)
    for k in 1:(nbtiles-kend)
        QRandmult!(A,Tau,k, nbtiles; SVDalg=true)
        (k==nbtiles) && break
        QRandmult!(A',Tau,k, nbtiles; LQ=true, SVDalg=true)
    end
    return A
end

function banddiagsvd(A::AbstractGPUorCPUMat, bw::Int)
    d,e = gbbrd!(gbbrd_copy(A,bw), bw)
    return LAPACK.bdsdc!('U', 'N', d, e)[1]
end


function mygesvd!(A::AbstractGPUMatrix)
    nbtiles=Int(size(A,1)/TILESIZE)
    Tau=KernelAbstractions.zeros(get_backend(A),eltype(A),nbtiles,size(A,2))
    myblockdiag!(A,Tau,nbtiles)
    KernelAbstractions.synchronize(get_backend(A))
    unsafe_free!(Tau)
    return banddiagsvd(A,TILESIZE)
end

function QRandmult!(A::AnyGPUMatrix, Tau, k, nbtiles;LQ=false,SVDalg=false)  

    QR1!(A,Tau, k;koffset=Int(LQ),singlerow=false)
    Qtapply1_par!(A, Tau, k; koffset=Int(LQ), singlerow=false)
    triu!(get_tileview(A, k+Int(LQ),k))

        for row in k+1+Int(LQ):nbtiles

            QR2!(A,Tau, k, row; koffset=Int(LQ), singlerow=false)
            Qtapply2_par!(A,Tau, k,row; koffset=Int(LQ), singlerow=false)
            fill!(get_tileview(A, row,k),0)

        end
end

function QRandmult!(A::LargeTiledMatrix, Tau, k, nbtiles;LQ=false,SVDalg=false)
    nbcolgroups=(ceil(Int,(nbtiles-k)/(A.tilesinmem-1)))
    colgroupsize= ceil(Int,(nbtiles-k)/nbcolgroups)
    begincol=k+1
    endcol=min(k+colgroupsize,nbtiles)
    colinmem= A.tilesinmem-min(nbtiles-k, colgroupsize)

    while begincol<=nbtiles

        setfirst!(A,k, LQ, SVDalg,begincol,endcol,colinmem )

        QRandmulQt1!(A, Tau,k;LQ=LQ,SVDalg=SVDalg,begincol=begincol,endcol=endcol, colinmem=colinmem)
        for row in k+1+Int(LQ):nbtiles
            QRandmulQt2!(A, Tau,k,row;LQ=LQ,begincol=begincol,endcol=endcol, colinmem=colinmem)
        end
        getlast!(A, k,LQ, begincol,endcol, colinmem)

        begincol=endcol+1
        endcol=min(endcol+colgroupsize,nbtiles)
    end
end


function QRandmulQt1!(A::LargeTiledMatrix, Tau, k::Int; LQ=false, SVDalg=false, begincol::Int=k+1, endcol::Int=A.nb_tiles, colinmem::Int=k)

    @sync begin
        Threads.@spawn begin
            if (k+Int(LQ)<A.nb_tiles)
                recycle_tilerow(A, k, k+1+Int(LQ),  false, false, colbegin=begincol,endcol=endcol,colinmem=colinmem)
            else
                push!(A.TileRows, A.TileRows[2])
            end
            KernelAbstractions.synchronize(A.backend)
        end
            
        Threads.@spawn begin
            begincol==k+1 && QR1!(A.TileRows[1],Tau, k;singlerow=true,colinmem=colinmem)
            Qtapply1_par!(A.TileRows[1], Tau, k;  singlerow=true,colinmem=colinmem)
            endcol==A.nb_tiles && triu!(get_tileview(A.TileRows[1], 1,colinmem))
            KernelAbstractions.synchronize(A.backend)
        end
    end
    deleteat!(A.TileRows,2)
end

function QRandmulQt2!(A::LargeTiledMatrix,Tau,k::Int, row::Int ; LQ::Bool=false, begincol::Int=k+1, endcol::Int=A.nb_tiles, colinmem::Int=k) 
    @sync begin
        Threads.@spawn begin
            (row>k+1+Int(LQ)) && (get_tilerow(A,k,row-1 , 3 , colbegin=begincol,endcol=endcol,colinmem=colinmem)) 
            KernelAbstractions.synchronize(A.backend)
        end 

        Threads.@spawn begin
            if (row<A.nb_tiles)
                recycle_tilerow(A, k, row+1,  false, false, colbegin=begincol,endcol=endcol,colinmem=colinmem)
            else
                push!(A.TileRows, A.TileRows[2])
            end
            KernelAbstractions.synchronize(A.backend)
        end

        Threads.@spawn begin
            begincol==k+1 && QR2!(A.TileRows[1],Tau, k, row; koffset=Int(LQ), singlerow=true, A2=A.TileRows[4],colinmem=colinmem)
            Qtapply2_par!(A.TileRows[1],Tau, k,row; koffset=Int(LQ), singlerow=true, A2=A.TileRows[4],colinmem=colinmem)
            endcol==A.nb_tiles && fill!(get_tileview(A.TileRows[4], 1,colinmem),0)
            KernelAbstractions.synchronize(A.backend)
        end
    end
    deleteat!(A.TileRows,2)
end


function setfirst!(A::LargeTiledMatrix,k2::Int,  LQ::Bool, SVDalg::Bool , begincol::Int,endcol::Int, colinmem::Int)
    begincol!=k2+1 && copyto!(get_tileview(A.TileRows[ 1],1, colinmem ), get_tileview(A.TileRows[ 2],1, colinmem ))
    recycle_tilerow(A, k2, k2+Int(LQ),true, false, colbegin=begincol,endcol=endcol,colinmem=colinmem )
    KernelAbstractions.synchronize(A.backend)
end

function getlast!(A::LargeTiledMatrix, k::Int, prevLQ::Bool, begincol::Int,endcol::Int, colinmem::Int)
    temp=A.TileRows[2]
    A.TileRows[2]=A.TileRows[1]
    A.TileRows[1]=temp
    (k<A.nb_tiles) &&  get_tilerow(A ,k,A.nb_tiles , 3, colbegin=begincol,endcol=endcol,colinmem=colinmem )
    get_tilerow(A, k, k+Int(prevLQ),  2, colbegin=begincol,endcol=endcol,colinmem=colinmem)
end


