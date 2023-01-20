
using LinearAlgebra, Distributions, Random, StaticArrays, SparseArrays, Plots, StatsPlots, DelimitedFiles, Plots.Measures, CUDA, BenchmarkTools, CatViews


struct BandBidiag_properties
    block_x_size::Int64 
    block_y_size::Int64
    no_blocked_rows::Int64
    no_blocked_cols::Int64
    block_x_range::UnitRange{Int64}
    block_y_range::UnitRange{Int64}
    BandBidiag_properties(block_x_size, block_y_size,no_blocked_rows,no_blocked_cols) = new(block_x_size, block_y_size,no_blocked_rows,no_blocked_cols, 1:block_x_size, 1:block_y_size)
end

function grab_block!(cachedblocks,A, prop::BandBidiag_properties, j, adj_x, adj_y, xpos_block, ypos_block)
    cachedblocks[j][prop.block_x_range .+ adj_x ,prop.block_y_range.+ adj_y ] = A[xpos_block.+prop.block_x_range, ypos_block.+prop.block_y_range]
    return;
end

function return_block!(cachedblocks,A, prop::BandBidiag_properties, j, adj_x, adj_y, xpos_block, ypos_block)
    A[xpos_block.+prop.block_x_range, ypos_block.+prop.block_y_range] = cachedblocks[j][prop.block_x_range .+ adj_x,prop.block_y_range.+ adj_y]
end

function setorreturn_block!(cachedblocks, A, diag_pivot_tile,  sweep_tile, secondpos::Bool, prop::BandBidiag_properties, grab_or_return!::Function, vert::Bool)
    secondblock_adjustment_x=prop.block_x_size*Int(secondpos)*Int(!vert)
    secondblock_adjustment_y=prop.block_y_size*Int(secondpos)*Int(vert)
    
    for (j,block_pos) in enumerate(diag_pivot_tile:(vert ? prop.no_blocked_cols : prop.no_blocked_rows))
        xpos_block=((vert ? block_pos : sweep_tile ) -1)*prop.block_x_size
        ypos_block=((vert ? sweep_tile : block_pos ) -1)*prop.block_y_size
        grab_or_return!(cachedblocks,A,prop,j,secondblock_adjustment_x,secondblock_adjustment_y,xpos_block,ypos_block)
    end
end

function set_block_vert!(cachedblocks, A, diag_pivot_tile,  row_sweep_tile, secondpos::Bool, prop::BandBidiag_properties)
    setorreturn_block!(cachedblocks, A, diag_pivot_tile,  row_sweep_tile, secondpos, prop, grab_block!, false)
end

function return_block_vert!(A, cachedblocks, diag_pivot_tile,  row_sweep_tile, secondpos::Bool, prop::BandBidiag_properties )
    setorreturn_block!(cachedblocks, A, diag_pivot_tile,  row_sweep_tile, secondpos, prop, return_block!, false)
end

function set_block_hor!(cachedblocks, A, diag_pivot_tile,  col_sweep_tile, secondpos::Bool, prop::BandBidiag_properties)
    setorreturn_block!(cachedblocks, A, diag_pivot_tile,  col_sweep_tile, secondpos, prop, grab_block!, true)
end

function return_block_hor!(A, cachedblocks, diag_pivot_tile,  col_sweep_tile, secondpos::Bool, prop::BandBidiag_properties )
    setorreturn_block!(cachedblocks, A, diag_pivot_tile,  col_sweep_tile, secondpos, prop, return_block!, true)
end

function colQR!(cachedblocks, prop::BandBidiag_properties, single::Bool)
    xrange = (single ? prop.block_x_range : :)
    Qfactor=qr(cachedblocks[1][xrange,:]).Q'
    for j in eachindex(cachedblocks)
        cachedblocks[j][xrange,:] = Qfactor*cachedblocks[j][xrange,:]
    end
    return;
end
function rowQR!(cachedblocks, prop::BandBidiag_properties, single::Bool)
    yrange = (single ? prop.block_y_range : : )
    Qfactor=qr!(cachedblocks[1][:,yrange]').Q
    for j in eachindex(cachedblocks)
        cachedblocks[j][:,yrange] =  cachedblocks[j][:,yrange] * Qfactor
    end
    return;
end

#TO DO: dont calculate zeros lol

function BandBidiagonal_CPU!(A,block_x_size, block_y_size,no_blocked_rows,no_blocked_cols)
    (m,n) = size(A)
    if m != block_x_size*no_blocked_rows
        error("dimension mismatch")
    elseif n!= block_y_size*no_blocked_cols
        error("dimension mismatch")
    end

    no_sweeps= min(no_blocked_cols,no_blocked_rows)
    prop = BandBidiag_properties(block_x_size, block_y_size,no_blocked_rows,no_blocked_cols)

    for diag_pivot_tile in 1:no_sweeps

        #time in isolation - memory allocation
        #move allocation outside
        cachedblocks=[zeros(block_x_size*2,block_y_size) for _ in diag_pivot_tile:no_blocked_cols]
        set_block_vert!(cachedblocks, A, diag_pivot_tile, diag_pivot_tile, false, prop)
        colQR!(cachedblocks, prop, true)
        for row_sweep in diag_pivot_tile+1:no_blocked_rows
            set_block_vert!(cachedblocks, A, diag_pivot_tile,  row_sweep, true, prop)
            colQR!(cachedblocks, prop, false)
            return_block_vert!(A, cachedblocks, diag_pivot_tile,  row_sweep, true, prop ) 
            #display(Array(A))
        end
        return_block_vert!(A, cachedblocks, diag_pivot_tile,  diag_pivot_tile, false, prop ) 

        diag_pivot_tile==no_sweeps && break

        cachedblocks=[zeros(block_x_size,block_y_size*2) for _ in diag_pivot_tile:no_blocked_rows]
        set_block_hor!(cachedblocks, A, diag_pivot_tile, diag_pivot_tile+1, false, prop)
        rowQR!(cachedblocks, prop, true)
        for col_sweep in diag_pivot_tile+2:no_blocked_cols
            set_block_hor!(cachedblocks, A, diag_pivot_tile, col_sweep, true, prop)
            rowQR!(cachedblocks, prop, false)
            return_block_hor!(A,cachedblocks,diag_pivot_tile,col_sweep,true,prop)
            #display(Array(A))
        end
        return_block_hor!(A,cachedblocks,diag_pivot_tile,diag_pivot_tile+1,false,prop)
    end
    return round.(A,digits=12)
end

function BandBidiagonal!(A,block_x_size, block_y_size,no_blocked_rows,no_blocked_cols)
    (m,n) = size(A)
    if m != block_x_size*no_blocked_rows
        error("dimension mismatch")
    elseif n!= block_y_size*no_blocked_cols
        error("dimension mismatch")
    end
    no_sweeps= min(no_blocked_cols,no_blocked_rows)
    prop = BandBidiag_properties(block_x_size, block_y_size,no_blocked_rows,no_blocked_cols)

    for diag_pivot_tile in 1:no_sweeps

        cachedblocks=[CUDA.zeros(block_x_size*2,block_y_size) for _ in diag_pivot_tile:no_blocked_cols]
        set_block_vert!(cachedblocks, A, diag_pivot_tile, diag_pivot_tile, false, prop)
        colQR!(cachedblocks, prop, true)
        for row_sweep in diag_pivot_tile+1:no_blocked_rows
            set_block_vert!(cachedblocks, A, diag_pivot_tile,  row_sweep, true, prop)
            colQR!(cachedblocks, prop, false)
            return_block_vert!(A, cachedblocks, diag_pivot_tile,  row_sweep, true, prop ) 
            #display(Array(A))
        end
        return_block_vert!(A, cachedblocks, diag_pivot_tile,  diag_pivot_tile, false, prop ) 
        #display(Array(A))
        #time this
        #remove gc
        [CUDA.unsafe_free!(i) for i in cachedblocks]; GC.gc(true)

        diag_pivot_tile==no_sweeps && break

        cachedblocks=[CUDA.zeros(block_x_size,block_y_size*2) for _ in diag_pivot_tile:no_blocked_rows]
        set_block_hor!(cachedblocks, A, diag_pivot_tile, diag_pivot_tile+1, false, prop)
        rowQR!(cachedblocks, prop, true)
        for col_sweep in diag_pivot_tile+2:no_blocked_cols
            set_block_hor!(cachedblocks, A, diag_pivot_tile, col_sweep, true, prop)
            rowQR!(cachedblocks, prop, false)
            return_block_hor!(A,cachedblocks,diag_pivot_tile,col_sweep,true,prop)
            #display(Array(A))
        end
        return_block_hor!(A,cachedblocks,diag_pivot_tile,diag_pivot_tile+1,false,prop)
        #display(Array(A))
        [CUDA.unsafe_free!(i) for i in cachedblocks]; GC.gc(true)
    end
    return round.(A,digits=5)
end


n=100
x=10
A=float.(rand(1:10,n,n))
A2 = BandBidiagonal!(A,x,x,x,x)
A_svd=svdvals(A)
A2_svd=svdvals(A2)
norm(A_svd-A2_svd,Inf)

n=200
x=50
A=float.(rand(1:10,n,n))
Acu = A |> cu
A2=deepcopy(A)
A3=deepcopy(A)
t1 = CUDA.@time BandBidiagonal!(A,x,x,4,4)
t2 = @btime BandBidiagonal_CPU!(A2,x,x,4,4)
t3 = CUDA.@time svdvals!(Acu);
t4 = @btime svdvals!(A3);

################################################################################################
################### Random other stuff ####################################################
###########################################################################################

A=CUDA.randn(2000,2000)
Q=CUDA.zeros(2000,2000)
R=CUDA.zeros(2000,2000)

#CUDA profiler
t1= @CUDA.time Q1,R2=qr(A)
t2= @CUDA.time Q,R=qr!(A) ###how to make this faster

A=CUDA.ones(500,500)
A[1:10,1:10]= 2*ones(10,10)

timings_cpu=[]
timings_gpu=[]
for n in [round(Int,10^i) for i=1:0.5:3.5]
    A=rand(n,n)
    t= @belapsed qr($A)
    push!(timings_cpu,t)
    B=A|>cu
    t= @belapsed qr($B)
    push!(timings_gpu,t)
end

plot(ns,timings_cpu, xlabel="Matrix size n", ylabel="QR calc time (s)", xaxis=:log10, yaxis=:log10, labels="CPU", )
plot!(ns,timings_gpu, xlabel="Matrix size n", ylabel="QR calc time (s)", xaxis=:log10, yaxis=:log10, labels="GPU")

n=2000
a=CUDA.randn(n,n)

function assign_values_withoutalloc(a)
    for i in 1:200
        a[i.+(5:6),i*10] .= i*10 .+(5:6)
    end
end

function assign_values_withalloc(a)
    for i in 1:200
        xval=i.+(5:6)
        yval=i*10
        assignval= i*10 .+(5:6)
        a[xval,yval] .= assignval
    end
end

t1 = CUDA.@time assign_values_withoutalloc(a)
t2= CUDA.@time  assign_values_withalloc(a)

x_size=6
y_size=5


function compute_gpu(outputs,inputs)
    @sync for i in eachindex(inputs)
        Threads.@spawn begin
            do_svd!(outputs[i], inputs[i])
            CUDA.synchronize()
        end
        end
    return outputs
end

