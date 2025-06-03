struct LargeTiledMatrix{T} <: AbstractMatrix{T}
    parent::Union{Adjoint{<:AbstractMatrix{T}},AbstractMatrix{T}}
    TileRows::Vector{<:AbstractGPUMatrix{T}}
    backend::Backend
    nb_tiles::Int
    tilesinmem::Int
    cpucache::Vector{<:Array{T,2}}
end


Base.adjoint(A::LargeTiledMatrix{T}) where T = LargeTiledMatrix{T}(A.parent', A.TileRows, A.backend, A.nb_tiles, A.tilesinmem, A.cpucache)

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

@inline function resizecache(A::LargeTiledMatrix{T},n::Int) where {T}
    if (size(A.cpucache[1],2)!=n*TILESIZE)
        A.cpucache[1] = zeros(T, TILESIZE, n*TILESIZE)
        A.cpucache[2] = zeros(T, TILESIZE, (n-1)*TILESIZE)
    end
end

@inline function addtocache(A::LargeTiledMatrix{T},n::Int) where {T}
    if (size(A.cpucache[4],2)!=n*TILESIZE)
        A.cpucache[4] = zeros(T, TILESIZE, n*TILESIZE)
        A.cpucache[5] = zeros(T, TILESIZE, (n-1)*TILESIZE)
    end
end

function clear!(A::LargeTiledMatrix)
    for x in A.TileRows
        unsafe_free!(x)
    end
end


@inline function set_tilerow(src::LargeTiledMatrix, k::Int, l::Int,  idx::Int;  colbegin::Int=k+1,endcol::Int=src.nb_tiles,colinmem::Int=k)

    if (idx>1 && colbegin!=k+1)
        @inbounds copyto!(src.cpucache[3],get_tileview(src.parent,l,k)  )
        @inbounds copyto!(get_tileview(src.TileRows[idx], 1, colinmem), src.cpucache[3])
    end
    smallercache = (endcol-colbegin)+1 != size(src.cpucache[1],2)
    cacheidx= ((colbegin== k+1) ? 1 : 2) + (smallercache ? 3 : 0)
    @inbounds copyto!(src.cpucache[cacheidx],get_rowview(src.parent,l,(colbegin== k+1) ? k : colbegin,endcol=endcol) )    
    @inbounds copyto!(get_rowview(src.TileRows[idx],1, colinmem+ (colbegin!=k+1), endcol=colinmem+(endcol-colbegin)+1), src.cpucache[cacheidx ])
    
end

@inline function get_tilerow(src::LargeTiledMatrix, k::Int, l::Int, idx::Int;  colbegin::Int=k+1,endcol::Int=src.nb_tiles,colinmem::Int=k)
    inclfirst = (idx>2 || endcol ==src.nb_tiles) && (colbegin==k+1)
    
    if (idx>2 || endcol ==src.nb_tiles) && !inclfirst
        @inbounds copyto!(src.cpucache[3], get_tileview(src.TileRows[ idx],1 , colinmem))
        @inbounds copyto!( get_tileview(src.parent,l,k) , src.cpucache[3])
    end 
    smallercache = (endcol-colbegin)+1 != size(src.cpucache[1],2)
    cacheidx= (inclfirst ? 1 : 2 ) + (smallercache ? 3 : 0)
    @inbounds copyto!(src.cpucache[cacheidx] ,  ( get_rowview(src.TileRows[ idx],1 , colinmem+Int(!inclfirst), endcol=colinmem+(endcol-colbegin)+1)))
    @inbounds copyto!(get_rowview(src.parent,l,inclfirst ? k : colbegin,endcol=endcol) ,src.cpucache[cacheidx ]  )

end
