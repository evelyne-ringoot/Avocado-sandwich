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

function grab_block!(cachedblocks,A, prop::BandBidiag_properties, j, adj_x, adj_y, xpos_block, ypos_block, makecopies::Bool)
    T=typeof(parent(first(cachedblocks)))
    #async copies
    makecopies && (cachedblockview=view(cachedblocks[j],prop.block_x_range .+ adj_x ,prop.block_y_range .+ adj_y ))
    makecopies && (Aview=view(A,xpos_block .+ prop.block_x_range, ypos_block .+ prop.block_y_range))
    makecopies && (cachedblockview .= T(Aview))
    return;
end

function return_block!(cachedblocks,A, prop::BandBidiag_properties, j, adj_x, adj_y, xpos_block, ypos_block, makecopies::Bool)
    makecopies && (cachedblockview=view(cachedblocks[j],prop.block_x_range .+ adj_x ,prop.block_y_range .+ adj_y ))
    makecopies && (Aview=view(A,xpos_block .+ prop.block_x_range, ypos_block .+ prop.block_y_range))
    makecopies && (Aview .= Array(cachedblockview))
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

function set_block_vert!(cachedblocks, A, diag_pivot_tile,  row_sweep_tile, secondpos::Bool, prop::BandBidiag_properties, makecopies::Bool)
    setorreturn_block!(cachedblocks, A, diag_pivot_tile,  row_sweep_tile, secondpos, prop, grab_block!, false, makecopies)
end

function return_block_vert!(A, cachedblocks, diag_pivot_tile,  row_sweep_tile, secondpos::Bool, prop::BandBidiag_properties , makecopies::Bool)
    setorreturn_block!(cachedblocks, A, diag_pivot_tile,  row_sweep_tile, secondpos, prop, return_block!, false, makecopies)
end

function set_block_hor!(cachedblocks, A, diag_pivot_tile,  col_sweep_tile, secondpos::Bool, prop::BandBidiag_properties , makecopies::Bool)
    setorreturn_block!(cachedblocks, A, diag_pivot_tile,  col_sweep_tile, secondpos, prop, grab_block!, true, makecopies)
end

function return_block_hor!(A, cachedblocks, diag_pivot_tile,  col_sweep_tile, secondpos::Bool, prop::BandBidiag_properties , makecopies::Bool)
    setorreturn_block!(cachedblocks, A, diag_pivot_tile,  col_sweep_tile, secondpos, prop, return_block!, true, makecopies)
end

function QR!(cachedblocks, prop::BandBidiag_properties, single::Bool, col::Bool, onGPU::Bool,  docalc::Bool, makecopies::Bool)
    xrange= prop.block_x_range
    yrange= prop.block_y_range 
    if (!single) && col
        xrange = 1:(prop.block_x_size*2)
    elseif (!single) && (!col)
        yrange = 1:(prop.block_y_size*2)
    end

    T=typeof(parent(first(cachedblocks)))
    makecopies && (cachedblockscopy=T(view(cachedblocks[1],xrange,yrange)))
    docalc && (Qfactor = col ? (qr(cachedblockscopy).Q)' : qr(cachedblockscopy').Q )

    @sync for j in eachindex(cachedblocks)
        Threads.@spawn begin
            makecopies && (cachedblockscopy=T(view(cachedblocks[j],xrange,yrange)))
            docalc && (col ? lmul!(Qfactor, cachedblockscopy) : rmul!(cachedblockscopy, Qfactor))
            makecopies && (view(cachedblocks[j],xrange,yrange).= cachedblockscopy)
            onGPU && CUDA.synchronize()
        end
    end
    return;
end

#QR, lmul, copy of non-contigous views
#TO DO: dont calculate zeros lol

function BandBidiagonal!(A,block_x_size, block_y_size,no_blocked_rows,no_blocked_cols, onGPU::Bool, docalc::Bool, makecopies::Bool)
    
    (m,n) = size(A)
    if m != block_x_size*no_blocked_rows
        error("dimension mismatch")
    elseif n!= block_y_size*no_blocked_cols
        error("dimension mismatch")
    end
    if docalc && (!makecopies)
        error("Function wont be able to do calculations without making copies")
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
        makecopies && (cachedblocks=[view(cachedblocks_large[i],1:(block_x_size*2),1:block_y_size) for i in 1:(no_blocked_cols-diag_pivot_tile+1)]) #fix the lengths
        set_block_vert!(cachedblocks, A, diag_pivot_tile, diag_pivot_tile, false, prop, makecopies)
        QR!(cachedblocks, prop, true, true, onGPU, docalc, makecopies)
        for row_sweep in diag_pivot_tile+1:no_blocked_rows
            set_block_vert!(cachedblocks, A, diag_pivot_tile,  row_sweep, true, prop, makecopies)
            QR!(cachedblocks, prop, false, true, onGPU, docalc, makecopies)
            return_block_vert!(A, cachedblocks, diag_pivot_tile,  row_sweep, true, prop , makecopies)
        end
        return_block_vert!(A, cachedblocks, diag_pivot_tile,  diag_pivot_tile, false, prop , makecopies) 

        diag_pivot_tile==no_sweeps && break

        #LQ sweep
        makecopies && (cachedblocks=[view(cachedblocks_large[i],1:block_x_size,1:(block_y_size*2) ) for i in 1:(no_blocked_rows-diag_pivot_tile+1)])
        set_block_hor!(cachedblocks, A, diag_pivot_tile, diag_pivot_tile+1, false, prop, makecopies)
        QR!(cachedblocks, prop, true, false, onGPU, docalc, makecopies)
        for col_sweep in diag_pivot_tile+2:no_blocked_cols
            set_block_hor!(cachedblocks, A, diag_pivot_tile, col_sweep, true, prop, makecopies)
            QR!(cachedblocks, prop, false, false, onGPU, docalc, makecopies)
            return_block_hor!(A,cachedblocks,diag_pivot_tile,col_sweep,true,prop, makecopies)

        end
        return_block_hor!(A,cachedblocks,diag_pivot_tile,diag_pivot_tile+1,false,prop, makecopies)

        onGPU && CUDA.unsafe_free!(cachedblocks_large[1])
        popfirst!(cachedblocks_large)
    end
    return round.(A,digits=5)
end



n=16
x=4
A=float.(rand(1:10,n,n))
A_svd=svdvals(A)
Adiag = BandBidiagonal!(A,x,x,x,x, true, true, true)
Adiag_svd=svdvals(Adiag)
norm(A_svd-Adiag_svd,Inf)

x_values=[4, 10]#, 17, 30, 50]

for x in x_values
    n=x*x
    A=float.(rand(1:10,n,n))
    A2=deepcopy(A)
    A3=deepcopy(A)
    A4=deepcopy(A)
    Adiag = BandBidiagonal!(A,x,x,x,x)
A_svd=svdvals(A)
Adiag_svd=svdvals(Adiag)
norm(A_svd-Adiag_svd,Inf)
t1 = CUDA.@elapsed BandBidiagonal!(A2,x,x,x,x);
t2 = @belapse BandBidiagonal_CPU!(A3,x,x,x,x, true);
t3 = @belapsed BandBidiagonal_CPU!(A4,x,x,x,x, false);

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

n=500
A=CUDA.randn(n,n)
A=randn(n,n)

function qr_notinplace(A)
    qr(A)
    return;
end
function qr_inplace!(A)
    qr!(A) #why is it not possible to provide QR
    return;
end

#CUDA profiler
t1= @btime qr_notinplace(A) setup=begin A=rand(5000,500) end
t2= @btime qr_inplace!(A) setup=begin A=rand(5000,500) end ###how to make this faster

A=CUDA.ones(500,500)
A[1:10,1:10]= 2*ones(10,10)

timings_cpu=[]
timings_gpu=[]
for n in [round(Int,10^i) for i=1:0.5:3.5]
    println(n)
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
            #do_svd!(outputs[i], inputs[i])
            CUDA.synchronize()
        end
        end
    return outputs
end

#benchmark svd

timings_cpu=[]
timings_gpu=[]
n_vals=[  10,32,100,316, 1000, 3162, 5000]
for n in [5000]
    A=rand(n,n)
    B=A|>cu
    t= @belapsed svd!(A)
    push!(timings_cpu,t)
    t= CUDA.@elapsed CUDA.CUSOLVER.gesvdj!('V',1, B, tol=Float32(1e-5))
    push!(timings_gpu,t)
end

plot(n_vals,timings_cpu, xlabel="Matrix size n", ylabel="svd calc time (s)", xaxis=:log10, yaxis=:log10, labels="CPU", )
plot!(n_vals,timings_gpu, xlabel="Matrix size n", ylabel="svd calc time (s)", xaxis=:log10, yaxis=:log10, labels="GPU")

#tolerance
