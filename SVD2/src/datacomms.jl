struct LargeTiledMatrix{T} <: AbstractMatrix{T}
    parent::Union{Adjoint{<:AbstractMatrix{T}},AbstractMatrix{T}}
    TileRows::Vector{<:AbstractGPUMatrix{T}}
    backend::Backend
    nb_tiles::Int
    tilesinmem::Int
end

Base.adjoint(A::LargeTiledMatrix{T}) where T = LargeTiledMatrix{T}(A.parent', A.TileRows, A.backend, A.nb_tiles, A.tilesinmem)

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
    return LargeTiledMatrix(input,rows, backend, Int(size(input,1)/TILESIZE),tilesinmem)
end

function clear!(A::LargeTiledMatrix)
    for x in A.TileRows
        unsafe_free!(x)
    end
end


@inline function recycle_tilerow(src::LargeTiledMatrix, k::Int, l::Int,  firstrow::Bool, exclfirst::Bool;  colbegin::Int=k+1,endcol::Int=src.nb_tiles,colinmem::Int=k)
    (!(firstrow && colbegin!=k+1)) && copyto!(get_tileview(src.TileRows[firstrow ? 1 : 2], 1, colinmem+Int(exclfirst)), 
                copy( get_tileview(src.parent,l,k+ Int(exclfirst))    ))
    copyto!(get_rowview(src.TileRows[ firstrow ? 1 : 2 ],1, colinmem+1 +(Int(exclfirst)), endcol=colinmem+(endcol-colbegin)+1), 
            copy( get_rowview(src.parent,l,colbegin+ Int(exclfirst),endcol=endcol)    ))
    !firstrow &&  push!(src.TileRows,  src.TileRows[2])
end


@inline function get_tilerow(src::LargeTiledMatrix, k::Int, l::Int, idx::Int;  colbegin::Int=k+1,endcol::Int=src.nb_tiles,colinmem::Int=k)
    !(idx<3 &&endcol!=src.nb_tiles) && copyto!( get_tileview(src.parent,l,k) ,  Array( get_tileview(src.TileRows[ idx],1 , colinmem)))
    copyto!( get_rowview(src.parent,l,colbegin,endcol=endcol) ,  Array( get_rowview(src.TileRows[ idx],1 , colinmem+1, endcol=colinmem+(endcol-colbegin)+1)))
end
