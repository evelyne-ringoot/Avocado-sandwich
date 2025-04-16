struct TileRow
    R::Bool
    k::Int
    l::Int
    RowTile::AbstractArray
    Tau::CuArray
end

struct LargeTiledMatrix
    A::AbstractArray
    M::Int
    N::Int
    m::Int
    n::Int
    no_tiles::Int
    myrange1::Tuple
    myrange2::Tuple
    backend
    Rows::Array{TileRow}
end

struct TiledMatrix
    TileViews::Array{SubArray}
    A::AbstractArray
    Tau::Array{CuArray}
    M::Int
    N::Int
    m::Int
    n::Int
    no_tiles::Int
    myrange1::Tuple
    myrange2::Tuple
end

GeneralTiledMatrix=Union{TiledMatrix, LargeTiledMatrix}

########################################
# GPU- only ##
########################################

function TiledMatrix(A::AbstractArray{T, 2}, blocksize_m::Int, blocksize_n::Int; tiledim2d::Int=0) where T
    if (blocksize_m!=blocksize_n) || size(A,1)!=size(A,2) || mod(size(A,1),blocksize_n)!=0 
        error("Not supported")
    end
    no_tiles=Int(size(A,1)/blocksize_n)
    TileViews=Array{SubArray}(undef, no_tiles,no_tiles)
    for i in 1:no_tiles
        for j in 1:no_tiles
            TileViews[i,j]=view(A, (i-1)*blocksize_m.+(1:blocksize_m),(j-1)*blocksize_n.+(1:blocksize_n))
        end
    end
    backend=KernelAbstractions.get_backend(A)
    Tau=Array{CuArray}(undef, no_tiles,no_tiles)
    for i in 1:no_tiles
        for j in 1:no_tiles
            Tau[i,j]=KernelAbstractions.zeros(backend, T, blocksize_m)
        end
    end
    if tiledim2d==0
        range1=(blocksize_n,1)
        range2= (blocksize_n,1)
    else
        range1=(blocksize_n,blocksize_n)
        range2= (blocksize_n,tiledim2d)
    end
    return TiledMatrix(TileViews, A, Tau, size(A,1),size(A,2), blocksize_m,blocksize_n,  no_tiles,  range1, range2)
end

function tileview(A, gi, gj, N)
    return view(A, (gi-1)*N.+(1:N),(gj-1)*N.+(1:N))
end

function tileview2(A, gi, gj, N)
    return view(A, (gi-1)*N.+(1:N),gj)
end

function hor_blocktileview(A::GeneralTiledMatrix, row, col)
    return view(A.A, ((row-1)*A.m .+(1:A.m)),((col-1)*A.n .+1):A.N)
end

function ver_blocktileview(A::GeneralTiledMatrix, row, col)
    return view(A.A, ( ((row-1)*A.m .+1):A.M ),(col-1)*A.n .+(1:A.n))
end

applyLQ1!(A,T;ndrange)=applyQR1!(A',T,ndrange=ndrange)
applyLQ2!(A,B, T;ndrange)=applyQR2!(A',B', T,ndrange=ndrange)
applyQt1L!(A,B,T;ndrange)=applyQt1!(A',B',T,ndrange=ndrange)
applyQt2L!(A,B,C,T;ndrange)=applyQt2!(A',B',C', T,ndrange=ndrange)

mytril!(A;ndrange)=mytriu!(A',ndrange=ndrange)

QR1!(A::TiledMatrix, k) = applyQR1!(A.TileViews[k,k],A.Tau[k,k], ndrange=A.myrange1) 
Qtapply1!(A::TiledMatrix, k, col) = applyQt1!(A.TileViews[k,col], A.TileViews[k,k],A.Tau[k,k], ndrange=A.myrange2 )
QR2!(A::TiledMatrix, row, k) =applyQR2!(A.TileViews[k,k], A.TileViews[row,k], A.Tau[row,k], ndrange=A.myrange1)

Qtapply2!(A::TiledMatrix, k, row,col) = applyQt2!(A.TileViews[k,col],A.TileViews[row,col], A.TileViews[row,k], A.Tau[row,k], ndrange=A.myrange2 )

LQ1!(A::TiledMatrix, k) =  applyLQ1!(A.TileViews[k,k+1],A.Tau[k,k],ndrange=A.myrange1)
LQtapply1!(A::TiledMatrix, row,k) = applyQt1L!(A.TileViews[row,k+1], A.TileViews[k,k+1],A.Tau[k,k], ndrange=A.myrange2 )
LQ2!(A::TiledMatrix, k, col)=applyLQ2!(A.TileViews[k,k+1], A.TileViews[k,col], A.Tau[col,k], ndrange=A.myrange1)

LQtapply2!(A::TiledMatrix, T, k, row,col)=applyQt2L!(A.TileViews[row,k+1],A.TileViews[row,col], A.TileViews[k,col], A.Tau[col,k], ndrange=A.myrange2 )

triu!(A::TiledMatrix,row,col) = mytriu!(A.TileViews[row,col],ndrange=(A.n,A.m))
tril!(A::TiledMatrix,row,col) = mytril!(A.TileViews[row,col],ndrange=(A.n,A.m))

Qtapply1_par!(A::TiledMatrix, k) = applyQt1!(hor_blocktileview(A, k, k+1), A.TileViews[k,k],A.Tau[k,k], ndrange=(A.m*(A.no_tiles-k),A.myrange2[2]) )
Qtapply2_par!(A::TiledMatrix, row,k) = applyQt2!(hor_blocktileview(A, k, k+1), hor_blocktileview(A, row, k+1),
        A.TileViews[row,k], A.Tau[row,k] , ndrange=(A.m*(A.no_tiles-k),A.myrange2[2]))

LQtapply1_par!(A::TiledMatrix, k)= applyQt1L!(ver_blocktileview(A, k+1, k+1), A.TileViews[k,k+1],A.Tau[k,k], ndrange=(A.m*(A.no_tiles-k),A.myrange2[2]) )
LQtapply2_par!(A::TiledMatrix,  k,col) =applyQt2L!(ver_blocktileview(A, k+1, k+1), ver_blocktileview(A, k+1, col),
        A.TileViews[k,col], A.Tau[col,k], ndrange=(A.m*(A.no_tiles-k),A.myrange2[2]))


########################################
# including communication CPU-GPU ##
########################################


function LargeTiledMatrix(A::Matrix{T}, backend, matrixsize::Int, blocksize::Int, no_blocks::Int; tiledim2d::Int=0) where T
    rows=[]
    Tau= KernelAbstractions.zeros(backend, T, blocksize)
    Row=KernelAbstractions.zeros(backend, T, blocksize, matrixsize)
    copyto!(Row,copy(view(A,1:blocksize,1:matrixsize)))
    push!(rows, TileRow(true, 1, 1, Row, Tau))
    for i in 1:3
        Tau= KernelAbstractions.zeros(backend, T, blocksize)
        Row=KernelAbstractions.zeros(backend, T, blocksize, matrixsize)
        push!(rows, TileRow(true, 1, 1, Row, Tau))
    end

    if tiledim2d==0
        range1=(blocksize,1)
        range2= (blocksize,1)
    else
        range1=(blocksize,blocksize)
        range2= (blocksize,tiledim2d)
    end
    return LargeTiledMatrix(A, matrixsize, matrixsize, blocksize, blocksize, no_blocks, range1, range2, backend, rows)
end

function recycle_tilerow(Atile::LargeTiledMatrix, k::Int, l::Int, R::Bool, first::Bool, exclfirst::Bool; excllast::Bool=false)
    Row= Atile.Rows[first ? 1 : 2].RowTile
    if R
        copyto!(getblockview(Atile, first ? 1 : 2 , k, exclfirst), copy(hor_blocktileview(Atile, l, k + Int(exclfirst)) ))
    else   
        #TODO: implement copy for transpose (note the copy is on GPU, so not a real performance bottleneck)
        copyto!(getblockview(Atile, first ? 1 : 2 , k, exclfirst), copy(ver_blocktileview(Atile,k+ Int(exclfirst),l)') )
    end
    if first
        Atile.Rows[1]= TileRow(R, k, l, Row, Atile.Rows[ 1 ].Tau)
    else
        push!(Atile.Rows, TileRow(R, k, l, Row, Atile.Rows[ 2].Tau))
    end
end

function finish_recycle!(Atile::LargeTiledMatrix)
    deleteat!(Atile.Rows,2)
end

function get_tilerow(A::LargeTiledMatrix, k::Int, l::Int, R::Bool, idx::Int)
    if R
        (hor_blocktileview(A,l,k) .= Array( getblockview(A, idx , k, false)))
    else   
         #TODO: implement copy for transpose (note the copy is on CPU, so not a real performance bottleneck)
        ver_blocktileview(A,k,l) .= Array(getblockview(A, idx , k, false)')
    end
end

function get_view_first(A::LargeTiledMatrix, idx)
    return view(A.Rows[idx].RowTile, :, 1:A.m)
end
function get_view_exclfirst(A::LargeTiledMatrix, idx::Int, k)
    return view(A.Rows[idx].RowTile, :,A.m + 1 : A.m * (A.no_tiles-k+1))
end

triu!(A::LargeTiledMatrix,idx) = mytriu!(get_view_first(A,idx),ndrange=(A.n,A.m))

QR1!(A::LargeTiledMatrix, k) = applyQR1!(A.Rows[1].RowTile, A.Rows[1].Tau, ndrange=A.myrange1)
QR2!(A::LargeTiledMatrix, row, k) =applyQR2!(A.Rows[1].RowTile, A.Rows[4].RowTile, A.Rows[4].Tau, ndrange=A.myrange1)

Qtapply1_par!(A::LargeTiledMatrix, k) = applyQt1!(get_view_exclfirst(A,1,k), A.Rows[1].RowTile, A.Rows[1].Tau, ndrange=(A.m*(A.no_tiles-k),A.myrange2[2]) )
Qtapply2_par!(A::LargeTiledMatrix, k) = applyQt2!(get_view_exclfirst(A,1,k), get_view_exclfirst(A,4,k), A.Rows[4].RowTile, A.Rows[4].Tau, ndrange=(A.m*(A.no_tiles-k),A.myrange2[2]))

getblockview(A::LargeTiledMatrix, idx::Int , k::Int, exclfirst::Bool) = view(A.Rows[idx].RowTile, :, (1 + (Int(exclfirst)*A.n)):(A.n * (A.no_tiles - k + 1) ) )