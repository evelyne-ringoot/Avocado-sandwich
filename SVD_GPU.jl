using LinearAlgebra,  CUDA

#to do:
# move to AMD/GPU general

struct BandBidiag_properties
    block_x_size::Int64 
    block_y_size::Int64
    no_blocked_rows::Int64
    no_blocked_cols::Int64
    block_x_range::UnitRange{Int64}
    block_y_range::UnitRange{Int64}
    LT_indices
    UT_indices
    BandBidiag_properties(block_x_size, block_y_size,no_blocked_rows,no_blocked_cols) = 
        new(block_x_size, block_y_size,no_blocked_rows,no_blocked_cols, 1:block_x_size, 1:block_y_size,
        reduce(vcat, [block_x_size*(j-1).+[ (j+1):block_x_size...] for j in 1:block_y_size]),
        reduce(vcat, [block_x_size*(j-1).+[ 1:(j-1)...] for j in 2:block_y_size])
        )
end

function grab_block!(cachedblocks,A, prop::BandBidiag_properties, j, adj_x, adj_y, xpos_block, ypos_block, makecopies::Bool)
    makecopies && copyto!( view(cachedblocks[j],prop.block_x_range .+ adj_x ,prop.block_y_range .+ adj_y ), 
                            view(A,xpos_block .+ prop.block_x_range, ypos_block .+ prop.block_y_range))
    return;
end

function return_block!(cachedblocks,A, prop::BandBidiag_properties, j, adj_x, adj_y, xpos_block, ypos_block, makecopies::Bool)
    makecopies && copyto!( view(A,xpos_block .+ prop.block_x_range, ypos_block .+ prop.block_y_range), 
                            view(cachedblocks[j],prop.block_x_range .+ adj_x ,prop.block_y_range .+ adj_y ))
    return;
end

function setorreturn_block!(cachedblocks, A, diag_pivot_tile,  sweep_tile, secondpos::Bool, prop::BandBidiag_properties, grab_or_return!::Function, vert::Bool, makecopies::Bool)
    secondblock_adjustment_x=prop.block_x_size*Int(secondpos)*Int(!vert)
    secondblock_adjustment_y=prop.block_y_size*Int(secondpos)*Int(vert)
    
    for (j,block_pos) in enumerate(diag_pivot_tile:(vert ? prop.no_blocked_cols : prop.no_blocked_rows))
        xpos_block=((vert ? block_pos : sweep_tile ) -1)*prop.block_x_size
        ypos_block=((vert ? sweep_tile : block_pos ) -1)*prop.block_y_size
        grab_or_return!(cachedblocks,A,prop,j,secondblock_adjustment_x,secondblock_adjustment_y,xpos_block,ypos_block, makecopies)
    end
end


set_block_vert_firstblock!(cachedblocks, A, diag_pivot_tile,  row_sweep_tile, prop::BandBidiag_properties, makecopies::Bool) = 
        setorreturn_block!(cachedblocks, A, diag_pivot_tile,  row_sweep_tile, false, prop, grab_block!, false, makecopies)
set_block_vert_secondblock!(cachedblocks, A, diag_pivot_tile,  row_sweep_tile,  prop::BandBidiag_properties, makecopies::Bool) = 
        setorreturn_block!(cachedblocks, A, diag_pivot_tile,  row_sweep_tile, true, prop, grab_block!, false, makecopies)

return_block_vert_firstblock!(A, cachedblocks, diag_pivot_tile,  row_sweep_tile,  prop::BandBidiag_properties , makecopies::Bool)  = 
        setorreturn_block!(cachedblocks, A, diag_pivot_tile,  row_sweep_tile, false, prop, return_block!, false, makecopies)
return_block_vert_secondblock!(A, cachedblocks, diag_pivot_tile,  row_sweep_tile,  prop::BandBidiag_properties , makecopies::Bool) =
        setorreturn_block!(cachedblocks, A, diag_pivot_tile,  row_sweep_tile, true, prop, return_block!, false, makecopies)

set_block_hor_firstblock!(cachedblocks, A, diag_pivot_tile,  col_sweep_tile, prop::BandBidiag_properties , makecopies::Bool) =
        setorreturn_block!(cachedblocks, A, diag_pivot_tile,  col_sweep_tile, false, prop, grab_block!, true, makecopies)
set_block_hor_secondblock!(cachedblocks, A, diag_pivot_tile,  col_sweep_tile, prop::BandBidiag_properties , makecopies::Bool) =
        setorreturn_block!(cachedblocks, A, diag_pivot_tile,  col_sweep_tile, true, prop, grab_block!, true, makecopies)

return_block_hor_firstblock!(A, cachedblocks, diag_pivot_tile,  col_sweep_tile,  prop::BandBidiag_properties , makecopies::Bool) =
        setorreturn_block!(cachedblocks, A, diag_pivot_tile,  col_sweep_tile, false, prop, return_block!, true, makecopies)
return_block_hor_secondblock!(A, cachedblocks, diag_pivot_tile,  col_sweep_tile,  prop::BandBidiag_properties , makecopies::Bool) =
        setorreturn_block!(cachedblocks, A, diag_pivot_tile,  col_sweep_tile, true, prop, return_block!, true, makecopies)

function QR!(cachedblocks, prop::BandBidiag_properties, single::Bool, col::Bool,  docalc::Bool, makecopies::Bool)
    xrange= prop.block_x_range
    yrange= prop.block_y_range 
    if single
        if col
            zeroindices=prop.LT_indices
        else
            zeroindices=prop.UT_indices
        end
    else
        if col
            xrange = 1:(prop.block_x_size*2)
            zeroindices=reduce(vcat, [((i*prop.block_x_size*2) .+((prop.block_x_size+1):(2*prop.block_x_size))) for i in 0:(prop.block_y_size-1)])
        else
            yrange = 1:(prop.block_y_size*2)
            zeroindices=(prop.block_x_size*prop.block_y_size+1):2*prop.block_x_size*prop.block_y_size
        end
    end

    docalc && ( Qfactor = col ? (qr!(view(cachedblocks[1],xrange,yrange)).Q)' : qr!(view(cachedblocks[1],xrange,yrange)').Q )

    CUDA.synchronize()
    @sync for j in 2:length(cachedblocks)
        Threads.@spawn begin
            docalc && (col ? lmul!(Qfactor, view(cachedblocks[j],xrange,yrange)) : rmul!(view(cachedblocks[j],xrange,yrange), Qfactor))
            CUDA.synchronize()
        end
    end
    docalc && (view(cachedblocks[1],xrange,yrange)[zeroindices].=0)
    makecopies && CUDA.unsafe_free!(Qfactor)
    
    return;
end

QR_single_col!(cachedblocks, prop::BandBidiag_properties, docalc::Bool, makecopies::Bool) =
    QR!(cachedblocks, prop::BandBidiag_properties, true, true,  docalc::Bool, makecopies::Bool)
QR_single_row!(cachedblocks, prop::BandBidiag_properties, docalc::Bool, makecopies::Bool) =
    QR!(cachedblocks, prop::BandBidiag_properties, true, false,  docalc::Bool, makecopies::Bool)
QR_double_col!(cachedblocks, prop::BandBidiag_properties, docalc::Bool, makecopies::Bool) =
    QR!(cachedblocks, prop::BandBidiag_properties, false, true,  docalc::Bool, makecopies::Bool)
QR_double_row!(cachedblocks, prop::BandBidiag_properties, docalc::Bool, makecopies::Bool) =
    QR!(cachedblocks, prop::BandBidiag_properties, false, false,  docalc::Bool, makecopies::Bool)

#TO DO: dont calculate zeros lol

function BandBidiagonal!(A,block_x_size, block_y_size,no_blocked_rows,no_blocked_cols, onGPU::Bool, docalc::Bool, makecopies::Bool)
    (m,n) = size(A)
    if m != block_x_size*no_blocked_rows
        error("dimension mismatch")
    elseif n!= block_y_size*no_blocked_cols
        error("dimension mismatch")
    end

    no_sweeps= min(no_blocked_cols,no_blocked_rows)
    prop = BandBidiag_properties(block_x_size, block_y_size,no_blocked_rows,no_blocked_cols)

    if onGPU
        cachedblocks_large=[CUDA.zeros(block_x_size*2,block_y_size*2) for _ in 1:max(no_blocked_cols,no_blocked_rows)]
    else
        cachedblocks_large=[zeros(block_x_size*2,block_y_size*2) for _ in 1:max(no_blocked_cols,no_blocked_rows)]
    end
    
    for diag_pivot_tile in 1:no_sweeps
        # QR sweep
        cachedblocks=[view(cachedblocks_large[i],1:(block_x_size*2),1:block_y_size) for i in 1:(no_blocked_cols-diag_pivot_tile+1)]
        set_block_vert_firstblock!(cachedblocks, A, diag_pivot_tile, diag_pivot_tile,  prop, makecopies)
        QR_single_col!(cachedblocks, prop, docalc, makecopies)

        for row_sweep in diag_pivot_tile+1:no_blocked_rows
            set_block_vert_secondblock!(cachedblocks, A, diag_pivot_tile,  row_sweep, prop, makecopies)
            QR_double_col!(cachedblocks, prop,  docalc, makecopies)
            return_block_vert_secondblock!(A, cachedblocks, diag_pivot_tile,  row_sweep,  prop , makecopies)
        end
        return_block_vert_firstblock!(A, cachedblocks, diag_pivot_tile,  diag_pivot_tile,  prop , makecopies) 

        diag_pivot_tile==no_sweeps && break
        #LQ sweep
        cachedblocks=[view(cachedblocks_large[i],1:block_x_size,1:(block_y_size*2) ) for i in 1:(no_blocked_rows-diag_pivot_tile+1)]
        set_block_hor_firstblock!(cachedblocks, A, diag_pivot_tile, diag_pivot_tile+1, prop, makecopies)
        QR_single_row!(cachedblocks, prop, docalc, makecopies)

        for col_sweep in diag_pivot_tile+2:no_blocked_cols
            set_block_hor_secondblock!(cachedblocks, A, diag_pivot_tile, col_sweep, prop, makecopies)
            QR_double_row!(cachedblocks, prop,  docalc, makecopies)
            return_block_hor_secondblock!(A,cachedblocks,diag_pivot_tile,col_sweep, prop, makecopies)
        end

        return_block_hor_firstblock!(A,cachedblocks,diag_pivot_tile,diag_pivot_tile+1, prop, makecopies)
        onGPU && (CUDA.unsafe_free!(cachedblocks_large[1]))
        popfirst!(cachedblocks_large)
    end
    return round.(A,digits=5)
end


CUDA.allowscalar(false)
