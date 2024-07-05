using LinearAlgebra,  CUDA

struct BandBidiag_properties
    block_x_size::Int64 
    block_y_size::Int64
    no_blocked_rows::Int64
    no_blocked_cols::Int64
    block_x_range::UnitRange{Int64}
    block_y_range::UnitRange{Int64}
    BandBidiag_properties(block_x_size, block_y_size,no_blocked_rows,no_blocked_cols) = 
        new(block_x_size, block_y_size,no_blocked_rows,no_blocked_cols, 1:block_x_size, 1:block_y_size )
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

function no_parallelQR(prop::BandBidiag_properties )
    testmatrix=CUDA.randn(prop.block_x_size*2,prop.block_y_size)
    gpu_mem_stats = (CUDA.@timed (esc(qr!(testmatrix))))[6]
    blocksize=sizeof(Float64)*prop.block_x_size*prop.block_y_size*4+gpu_mem_stats
    CUDA.unsafe_free!(testmatrix)
    return (CUDA.MemoryInfo().free_bytes)*0.9/blocksize;
end


function QR!(cachedblocks, prop::BandBidiag_properties, single::Bool, col::Bool,  onGPU::Bool,  docalc::Bool, makecopies::Bool)
    xrange= prop.block_x_range
    yrange= prop.block_y_range 
    xrangeT= prop.block_x_range
    yrangeT= prop.block_y_range 
    if (!single)
        if col
            xrange = 1:(prop.block_x_size*2)
            xrangeT= xrange
            yrangeT= yrange
        else
            yrangeT=yrange
            yrange = 1:(prop.block_y_size*2)
            xrangeT = 1:(prop.block_x_size*2)
        end
    end
    if onGPU
        if col
            temp=CUDA.zeros(length(xrange),length(yrange))
        else
            temp=CUDA.zeros(length(xrangeT),length(yrangeT))
        end
        temp2=CUDA.zeros(length(xrange),length(yrange))
    else
        if col
            temp=zeros(length(xrange),length(yrange))
        else
            temp=zeros(length(xrangeT),length(yrangeT))
        end
        temp2=zeros(length(xrange),length(yrange))
    end
    if docalc
        if col
            copy!(temp, view(cachedblocks[1],xrange,yrange) )
            Qfactor = (qr!(temp).Q)
            Qfactor = Qfactor' 
            copy!(view(cachedblocks[1],xrange,yrange),temp)
        else
            tempspace=adjoint(view(cachedblocks[1],prop.block_x_range,prop.block_y_range))
            view(cachedblocks[1],prop.block_x_range,prop.block_y_range) .= tempspace
            if (!single)
                tempspace=adjoint(view(cachedblocks[1], prop.block_x_range ,prop.block_y_range .+prop.block_y_size))
                view(cachedblocks[1],prop.block_x_range .+prop.block_x_size,prop.block_y_range) .= tempspace
            end
            copy!(temp, view(cachedblocks[1],xrangeT,yrangeT))
            Qfactor= qr!(temp).Q 
            copy!(view(cachedblocks[1],xrangeT,yrangeT), temp)
        end
    end
    
    CUDA.synchronize()
    #@sync 
    for j in 2:length(cachedblocks)
        #Threads.@spawn begin
            if docalc 
                copy!(temp2,view(cachedblocks[j],xrange,yrange))
                if col 
                    lmul!(Qfactor,temp2)
                else
                    rmul!(temp2, Qfactor)
                end
                copy!(view(cachedblocks[j],xrange,yrange), temp2)
            end
            CUDA.synchronize()
        #end
    end
    CUDA.synchronize()
    if docalc
        if col
            view(cachedblocks[1],xrangeT,yrangeT).= triu(view(cachedblocks[1],xrangeT,yrangeT))
        else
            tempspace=adjoint(view(cachedblocks[1],prop.block_x_range,prop.block_y_range))
            view(cachedblocks[1],prop.block_x_range,prop.block_y_range) .= tempspace
            view(cachedblocks[1],xrangeT,yrangeT) .= tril(view(cachedblocks[1],xrangeT,yrangeT))
            view(cachedblocks[1], prop.block_x_range ,prop.block_y_range .+prop.block_y_size) .= 0
        end
    end
    if makecopies && onGPU
        if col 
            CUDA.unsafe_free!(Qfactor.Q.τ)
        else
            CUDA.unsafe_free!(Qfactor.τ)
        end
    end 
    return;
end

QR_single_col!(cachedblocks, prop::BandBidiag_properties,  onGPU::Bool, docalc::Bool, makecopies::Bool) =
    QR!(cachedblocks, prop::BandBidiag_properties, true, true,  onGPU::Bool,  docalc::Bool, makecopies::Bool)
QR_single_row!(cachedblocks, prop::BandBidiag_properties,  onGPU::Bool, docalc::Bool, makecopies::Bool) =
    QR!(cachedblocks, prop::BandBidiag_properties, true, false,  onGPU::Bool,  docalc::Bool, makecopies::Bool)
QR_double_col!(cachedblocks, prop::BandBidiag_properties,  onGPU::Bool, docalc::Bool, makecopies::Bool) =
    QR!(cachedblocks, prop::BandBidiag_properties, false, true,   onGPU::Bool, docalc::Bool, makecopies::Bool)
QR_double_row!(cachedblocks, prop::BandBidiag_properties,  onGPU::Bool, docalc::Bool, makecopies::Bool) =
    QR!(cachedblocks, prop::BandBidiag_properties, false, false,  onGPU::Bool,  docalc::Bool, makecopies::Bool)

#TO DO: dont calculate zeros lol



function BandBidiagonal!(A,block_x_size, block_y_size,no_blocked_rows,no_blocked_cols,  onGPU::Bool)
    docalc=true;
    makecopies=true;
    (m,n) = size(A)
    no_sweeps= min(no_blocked_cols,no_blocked_rows)
    prop = BandBidiag_properties(block_x_size, block_y_size,no_blocked_rows,no_blocked_cols)

    for diag_pivot_tile in 1:no_sweeps
        
        # QR sweep
        if onGPU
            cachedblocks=[CUDA.zeros(block_x_size*2,block_y_size) for _ in 1:no_blocked_cols-diag_pivot_tile+1]
        else
            cachedblocks=[zeros(block_x_size*2,block_y_size) for _ in 1:no_blocked_cols-diag_pivot_tile+1]
        end
        
        set_block_vert_firstblock!(cachedblocks, A, diag_pivot_tile, diag_pivot_tile,  prop, makecopies)
        QR_single_col!(cachedblocks, prop, onGPU, docalc, makecopies)

        for row_sweep in diag_pivot_tile+1:no_blocked_rows
            set_block_vert_secondblock!(cachedblocks, A, diag_pivot_tile,  row_sweep, prop, makecopies)
            QR_double_col!(cachedblocks, prop, onGPU, docalc, makecopies)
            return_block_vert_secondblock!(A, cachedblocks, diag_pivot_tile,  row_sweep,  prop , makecopies)
        end
        return_block_vert_firstblock!(A, cachedblocks, diag_pivot_tile,  diag_pivot_tile,  prop , makecopies) 
        diag_pivot_tile==no_sweeps && break

        #LQ sweep
        if onGPU
            cachedblocks=[CUDA.zeros(block_x_size*2,block_y_size*2) for _ in 1:no_blocked_rows-diag_pivot_tile+1]
        else
            cachedblocks=[zeros(block_x_size*2,block_y_size*2) for _ in 1:no_blocked_rows-diag_pivot_tile+1]
        end
        
        set_block_hor_firstblock!(cachedblocks, A, diag_pivot_tile, diag_pivot_tile+1, prop, makecopies)
        QR_single_row!(cachedblocks, prop,onGPU, docalc, makecopies)

        for col_sweep in diag_pivot_tile+2:no_blocked_cols
            set_block_hor_secondblock!(cachedblocks, A, diag_pivot_tile, col_sweep, prop, makecopies)
            QR_double_row!(cachedblocks, prop, onGPU,  docalc, makecopies)
            return_block_hor_secondblock!(A,cachedblocks,diag_pivot_tile,col_sweep, prop, makecopies)
        end

        return_block_hor_firstblock!(A,cachedblocks,diag_pivot_tile,diag_pivot_tile+1, prop, makecopies)

    end
    return A
end

function QR_row!(A, startindex, lastindex, indexgap)
    temp=transpose(view(A,startindex:lastindex, startindex+indexgap:lastindex))
    Qfactor=qr(temp).Q
    rmul!(view(A,startindex: lastindex, startindex+indexgap:lastindex),Qfactor)
    return;
end

function QR_col!(A, startindex, lastindex, indexgap)
    qr!(view(A,startindex:lastindex, startindex:indexgap+lastindex))
    triu!(view(A,startindex:lastindex, startindex:indexgap+lastindex))
    return;
end

function block_bidiagonalize!(A, n, bandwidth, target_bandwidth)
    for j=1:target_bandwidth:n-target_bandwidth    #bulge chasing: elimination on row j+1
        QR_row!(A,j, min(j+bandwidth+target_bandwidth-1, n) , target_bandwidth)
        
        for i=j:bandwidth:n
            lastindex=min(i+bandwidth+target_bandwidth-1, n) #index of end of block and its neighbor
            s_capped= min(bandwidth+target_bandwidth-1, max(n-i-bandwidth-target_bandwidth+1,0))
            QR_col!(A,i+target_bandwidth,lastindex,s_capped)
            
            i+target_bandwidth>(n-bandwidth) && break
            QR_row!(A,i+target_bandwidth,lastindex+s_capped,bandwidth)
        end
    end
    return A
end

function bidiagonalize(A, bandwidth)
    (m,n) = size(A)
    while bandwidth>16
        bandwidth=round(Int,bandwidth/2)
        block_bidiagonalize!(A,n,bandwidth*2,bandwidth);
    end

    block_bidiagonalize!(A, n,bandwidth,1);
    return A 
end


function my_CU_svdval(Ain::Matrix{Float32}, block_size::Int)
    println(size(a))
    A=copy(Ain)
    n=size(A,1)
    if (!ispow2(block_size)) || (n!=size(A,2)) || mod(n,block_size)!=0 || block_size<16
        error("Not implemented")
    end
    no_blocks=Int(n/block_size)
    banddiag = BandBidiagonal!(A,block_size,block_size, no_blocks,no_blocks,false);
    bidiag=bidiagonalize(banddiag,block_size);
    singvals, _, _, _, _,_= LAPACK.bdsdc!('U', 'N', diag(bidiag), diag(bidiag,1));
    println(size(singvals))
    return(singvals)
end

function my_CU_svdval(Ain::CuMatrix{Float32}, block_size::Int)
    A=copy(Ain)
    n=size(A,1)
    if (!ispow2(block_size)) || (n!=size(A,2)) || mod(n,block_size)!=0 || block_size<16
        error("Not implemented")
    end
    no_blocks=Int(n/block_size)
    banddiag = BandBidiagonal!(A,block_size,block_size,no_blocks,no_blocks,true);
    bidiag=bidiagonalize(Array(banddiag),block_size);
    singvals, _, _, _, _,_= LAPACK.bdsdc!('U', 'N', diag(bidiag), diag(bidiag,1));
    return(singvals)
end

CUDA.allowscalar(false)
