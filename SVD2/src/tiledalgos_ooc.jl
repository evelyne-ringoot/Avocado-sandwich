#TODO
#non-square and non-power of two matrices, and non-multiples of tilesize
#naming conventions


function LargeTiledMatrix(input::AbstractMatrix{T}, backend::Backend, tilesinmem::Int) where {T}
    tilesinmem = min(Int((size(input,1))/TILESIZE), tilesinmem)
    rows=Array{AbstractGPUMatrix{T}}(undef, 0)
    Row=KernelAbstractions.zeros(backend, T, TILESIZE, tilesinmem*TILESIZE)
    copyto!(Row,copy(view(input,1:TILESIZE,1:tilesinmem*TILESIZE)))
    push!(rows, Row)
    for _ in 1:3
        Row=KernelAbstractions.zeros(backend, T, TILESIZE, tilesinmem*TILESIZE)
        push!(rows, Row)
    end
    return LargeTiledMatrix(input,rows, backend, Int(size(input,1)/TILESIZE),tilesinmem,
                         [zeros(T, TILESIZE, tilesinmem*TILESIZE), zeros(T, TILESIZE, (tilesinmem-1)*TILESIZE),  
                         zeros(T, TILESIZE, TILESIZE), zeros(T,1,1),zeros(T,1,1)])
end



function OOC_alg!(A::Matrix{T}, f::Function, kswitch::Int,tilesinmem::Int) where {T}
    n=size(A,1)
    nb_tiles= Int(n/TILESIZE)
    Tau=KernelAbstractions.zeros(backend, T, TILESIZE, nb_tiles)
    
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

OOC_QR!(A::Matrix; kswitch::Int=256, tilesinmem::Int=max(floor(Int,kswitch^2/4)+1,2)) = OOC_alg!(A, mygeqrf!, kswitch, tilesinmem)
OOC_Bidiag!(A::Matrix; kswitch::Int=256, tilesinmem::Int=max(floor(Int,kswitch^2/4)+1,2)) = OOC_alg!(A, myblockdiag!, kswitch, tilesinmem)
OOC_SVD!(A::Matrix; kswitch::Int=256, tilesinmem::Int=max(floor(Int,kswitch^2/4)+1,2)) = banddiagsvd(OOC_Bidiag!(A,kswitch=kswitch, tilesinmem=tilesinmem))



function QRandmult!(A::LargeTiledMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int, nbtiles::Int;LQ::Bool=false) where {T}
    nbcolgroups=(ceil(Int,(nbtiles-k)/(A.tilesinmem-1)))
    colgroupsize= ceil(Int,(nbtiles-k)/nbcolgroups)
    resizecache(A,colgroupsize+1)
    begincol=k+1
    endcol=min(k+colgroupsize,nbtiles)
    colinmem= A.tilesinmem-min(nbtiles-k, colgroupsize)
    while begincol<=nbtiles
        addtocache(A,endcol-begincol+2)

        setfirst!(A,k, LQ ,begincol,endcol,colinmem )
        QRandmulQt1!(A, Tau,k;LQ=LQ,begincol=begincol,endcol=endcol, colinmem=colinmem)
        for row in k+1+Int(LQ):nbtiles
            QRandmulQt2!(A, Tau,k,row;LQ=LQ,begincol=begincol,endcol=endcol, colinmem=colinmem)
        end
        getlast!(A, k,LQ, begincol,endcol, colinmem)

        begincol=endcol+1
        endcol=min(endcol+colgroupsize,nbtiles)
    end
end


@inline function QRandmulQt1!(A::LargeTiledMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int; LQ::Bool=false,  begincol::Int=k+1, endcol::Int=A.nb_tiles, colinmem::Int=k) where {T}
    push!(A.TileRows,  A.TileRows[2])
    begincol==k+1 && QR1!(A.TileRows[1],Tau, k;singlerow=true,colinmem=colinmem)
    Qtapply1_par!(A.TileRows[1], Tau, k;  singlerow=true,colinmem=colinmem)
    endcol==A.nb_tiles && triu!(get_tileview(A.TileRows[1], 1,colinmem))
    #have kernels run while doing data movements

    (k+Int(LQ)<A.nb_tiles) &&  set_tilerow(A, k, k+1+Int(LQ),  2, colbegin=begincol,endcol=endcol,colinmem=colinmem)

    
    KernelAbstractions.synchronize(A.backend) #only sync all here
    deleteat!(A.TileRows,2)
end

@inline function QRandmulQt2!(A::LargeTiledMatrix,Tau::AbstractGPUMatrix{T},k::Int, row::Int ; LQ::Bool=false, begincol::Int=k+1, endcol::Int=A.nb_tiles, colinmem::Int=k) where {T}
    
    
    begincol==k+1 && QR2!(A.TileRows[1],Tau, k, row; koffset=Int(LQ), singlerow=true, A2=A.TileRows[4],colinmem=colinmem)
    Qtapply2_par!(A.TileRows[1],Tau, k,row; koffset=Int(LQ), singlerow=true, A2=A.TileRows[4],colinmem=colinmem)
    endcol==A.nb_tiles && fill!(get_tileview(A.TileRows[4], 1,colinmem),0)
    #have kernels run while doing data movements

    (row<A.nb_tiles) && set_tilerow(A, k, row+1,  2,  colbegin=begincol,endcol=endcol,colinmem=colinmem)
    (row>k+1+Int(LQ)) && (get_tilerow(A,k,row-1 , 3 , colbegin=begincol,endcol=endcol,colinmem=colinmem)) 

    push!(A.TileRows,  A.TileRows[2])
    KernelAbstractions.synchronize(A.backend) #only sync all here
    deleteat!(A.TileRows,2)
end



function setfirst!(A::LargeTiledMatrix,k2::Int,  LQ::Bool , begincol::Int,endcol::Int, colinmem::Int)
    set_tilerow(A, k2, k2+Int(LQ),1,colbegin=begincol,endcol=endcol,colinmem=colinmem )

end

function getlast!(A::LargeTiledMatrix, k::Int, prevLQ::Bool, begincol::Int,endcol::Int, colinmem::Int)
    (k<A.nb_tiles) &&  get_tilerow(A ,k,A.nb_tiles , 3, colbegin=begincol,endcol=endcol,colinmem=colinmem )
    get_tilerow(A, k, k+Int(prevLQ),  1, colbegin=begincol,endcol=endcol,colinmem=colinmem)
end




