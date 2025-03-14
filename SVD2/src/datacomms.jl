struct LargeTiledMatrix{T} <: AbstractMatrix{T}
    parent::Union{Adjoint{<:AbstractMatrix{T}},AbstractMatrix{T}}
    TileRows::Vector{<:AbstractGPUMatrix{T}}
    backend::Backend
    nb_tiles::Int
end

function Base.adjoint(A::LargeTiledMatrix{T}) where T
    return LargeTiledMatrix{T}(A.parent', A.TileRows, A.backend, A.nb_tiles)
end

function LargeTiledMatrix(input::AbstractMatrix{T}, backend::Backend) where {T}
    n,_=(size(input))
    nb_tiles=Int(n/TILESIZE)
    rows=Array{AbstractGPUMatrix{T}}(undef, 0)
    Row=KernelAbstractions.zeros(backend, T, TILESIZE, n)
    copyto!(Row,copy(view(input,1:TILESIZE,1:n)))
    push!(rows, Row)
    for i in 1:3
        Row=KernelAbstractions.zeros(backend, T, TILESIZE, n)
        push!(rows, Row)
    end
    return LargeTiledMatrix(input,rows, backend, nb_tiles)
end

getblockview(Rowslist, idx::Int , k::Int, exclfirst::Bool) = 
                    view(Rowslist[idx], :,  (1 + (Int(exclfirst)*TILESIZE)+((k-1)*TILESIZE)):size(Rowslist[idx],2) )

function recycle_tilerow(src::LargeTiledMatrix, k::Int, l::Int, R::Bool, first::Bool, exclfirst::Bool)
    copyto!(getblockview(src.TileRows, first ? 1 : 2 , k, exclfirst), 
                copy( R ? get_rowview(src.parent,l,k+ Int(exclfirst)) :
                        get_rowview(src.parent,l,k+ Int(exclfirst))    ))
    !first &&  push!(src.TileRows,  src.TileRows[2])
end

function get_tilerow(src::LargeTiledMatrix, k::Int, l::Int, R::Bool, idx::Int)
    
    copyto!((R ? get_rowview(src.parent,l,k) : get_rowview(src.parent,l,k) ) , Array( getblockview(src.TileRows, idx , k, false)))
end


