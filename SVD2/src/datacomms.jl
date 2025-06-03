



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
