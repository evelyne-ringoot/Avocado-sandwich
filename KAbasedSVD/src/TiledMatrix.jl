
struct TiledMatrix
    TileViews::Array{SubArray}
    ParentMatrix
    M::Int
    N::Int
    m::Int
    n::Int
    no_tiles::Int
    backend
    myrange::Tuple
    QRkernel1!::KernelAbstractions.Kernel
    QRkernel2!::KernelAbstractions.Kernel
    Qapplykernel1!::KernelAbstractions.Kernel
    Qapplykernel2!::KernelAbstractions.Kernel
    Qtapplykernel1!::KernelAbstractions.Kernel
    Qtapplykernel2!::KernelAbstractions.Kernel
    Qtapplykernel1block!::KernelAbstractions.Kernel
    Qtapplykernel2block!::KernelAbstractions.Kernel
    triu!::KernelAbstractions.Kernel
end

function TiledMatrix(A::AbstractArray{T, 2}, blocksize_m::Int, blocksize_n::Int) where T
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
    return TiledMatrix(TileViews, A, size(A,1),size(A,2), blocksize_m,blocksize_n,
        no_tiles, backend, (blocksize_n,1),
        QR_unsafe_kernel!(backend,blocksize_n),
        QR_unsafe_kernel2!(backend,blocksize_n),
        applyQ_unsafe_kernel!(backend,blocksize_n),
        applyQ_unsafe_kernel2!(backend,blocksize_n),
        applyQt_unsafe_kernel!(backend,blocksize_n), 
        applyQt_unsafe_kernel2!(backend,blocksize_n),
        applyQt_unsafe_kernel_block!(backend,blocksize_n), 
        applyQt_unsafe_kernel2_block!(backend,blocksize_n),
        mytriu!(backend) 
    )
end

function tileview(A, gi, gj, N)
    return view(A, (gi-1)*N.+(1:N),(gj-1)*N.+(1:N))
end

function tileview2(A, gi, gj, N)
    return view(A, (gi-1)*N.+(1:N),gj)
end

function hor_blocktileview(A::TiledMatrix, row, col)
    return view(A.ParentMatrix, ((row-1)*A.m .+(1:A.m)),((col-1)*A.n .+1):A.N)
end

function ver_blocktileview(A::TiledMatrix, row, col)
    return view(A.ParentMatrix, ( ((row-1)*A.m .+1):A.M ),(col-1)*A.n .+(1:A.n))
end


function QR1!(A::TiledMatrix, T, k)
    A.QRkernel1!(A.TileViews[k,k],T[k,k], ndrange=A.myrange)
end

function Qtapply1!(A::TiledMatrix, T, k, col)
    A.Qtapplykernel1!(A.TileViews[k,col], A.TileViews[k,k],T[k,k], ndrange=A.myrange )
end

function QR2!(A::TiledMatrix, T, row, k)
    A.QRkernel2!(A.TileViews[k,k], A.TileViews[row,k], T[row,k], ndrange=A.myrange)
end

function Qtapply2!(A::TiledMatrix, T, k, row,col)
    A.Qtapplykernel2!(A.TileViews[k,col],A.TileViews[row,col], A.TileViews[row,k], T[row,k], ndrange=A.myrange )
end

function LQ1!(A::TiledMatrix, T, k)
    A.QRkernel1!(A.TileViews[k,k+1]',T[k,k], ndrange=A.myrange)
end

function LQtapply1!(A::TiledMatrix, T, row,k)
    A.Qtapplykernel1!(A.TileViews[row,k+1]', A.TileViews[k,k+1]',T[k,k], ndrange=A.myrange )
end

function LQ2!(A::TiledMatrix, T, k, col)
    A.QRkernel2!(A.TileViews[k,k+1]', A.TileViews[k,col]', T[col,k], ndrange=A.myrange)
end

function LQtapply2!(A::TiledMatrix, T, k, row,col)
    A.Qtapplykernel2!(A.TileViews[row,k+1]',A.TileViews[row,col]', A.TileViews[k,col], T[col,k], ndrange=A.myrange )
end

#TODO: make the below into block kernels

function Qtapply1_blocks!(A::TiledMatrix, T, k)
    for col in k+1:A.no_tiles
        A.Qtapplykernel1!(A.TileViews[k,col], A.TileViews[k,k],T[k,k], ndrange=A.myrange )
    end
end

function Qtapply2_blocks!(A::TiledMatrix, T, row,k)
    for col in k+1:A.no_tiles
        A.Qtapplykernel2!(A.TileViews[k,col],A.TileViews[row,col], A.TileViews[row,k], T[row,k] , ndrange=A.myrange)
    end
end

function LQtapply1_blocks!(A::TiledMatrix, T, k)
    for row in k+1:A.no_tiles
        A.Qtapplykernel1!(A.TileViews[row,k+1]', A.TileViews[k,k+1]',T[k,k], ndrange=A.myrange )
    end
end

function LQtapply2_blocks!(A::TiledMatrix, T, k, col)
    for row in k+1:A.no_tiles
        A.Qtapplykernel2!(A.TileViews[row,k+1]',A.TileViews[row,col]', A.TileViews[k,col]', T[col,k] , ndrange=A.myrange)
    end
end


function triu!(A::TiledMatrix,row,col)
    A.triu!(A.TileViews[row,col],ndrange=(A.n,A.m))
end


function tril!(A::TiledMatrix,row,col)
    A.triu!(A.TileViews[row,col]',ndrange=(A.n,A.m))
end

function Qtapply1_blockspar!(A::TiledMatrix, T, k)
    A.Qtapplykernel1!(hor_blocktileview(A, k, k+1), A.TileViews[k,k],T[k,k], ndrange=(A.m*(A.no_tiles-k),1) )
end

function Qtapply2_blockspar!(A::TiledMatrix, T, row,k)
    A.Qtapplykernel2!(hor_blocktileview(A, k, k+1), hor_blocktileview(A, row, k+1),
        A.TileViews[row,k], T[row,k] , ndrange=(A.m*(A.no_tiles-k),1))
end

function LQtapply1_blockspar!(A::TiledMatrix, T, k)
    A.Qtapplykernel1!(ver_blocktileview(A, k+1, k+1)', A.TileViews[k,k+1]',T[k,k], ndrange=(A.m*(A.no_tiles-k),1) )
end

function LQtapply2_blockspar!(A::TiledMatrix, T, k,col)
    A.Qtapplykernel2!(ver_blocktileview(A, k+1, k+1)', ver_blocktileview(A, k+1, col)',
        A.TileViews[k,col]', T[col,k] , ndrange=(A.m*(A.no_tiles-k),1))
end