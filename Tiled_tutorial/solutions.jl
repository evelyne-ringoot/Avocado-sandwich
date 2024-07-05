function solution_mysvd!(A, no_tiles, block_size) 

    for k in 1:no_tiles

        qrval=qr(A[(1:block_size).+(k-1)*block_size, (1:block_size).+(k-1)*block_size])
        A[(1:block_size).+(k-1)*block_size, (1:block_size).+(k-1)*block_size].=qrval.R
    
        for col in k+1:no_tiles
            A[(1:block_size).+(k-1)*block_size, (1:block_size).+(col-1)*block_size].=lmul!(qrval.Q',A[(1:block_size).+(k-1)*block_size, (1:block_size).+(col-1)*block_size])
        end
        for row in k+1:no_tiles
            tempmatrix=[A[(1:block_size).+(k-1)*block_size, (1:block_size).+(k-1)*block_size];
                A[(1:block_size).+(row-1)*block_size, (1:block_size).+(k-1)*block_size] ]
            qrval=qr(tempmatrix)
            A[(1:block_size).+(k-1)*block_size, (1:block_size).+(k-1)*block_size] .= qrval.R
            A[(1:block_size).+(row-1)*block_size, (1:block_size).+(k-1)*block_size] .= 0
        
            for col in k+1:no_tiles
                tempmatrix=[A[(1:block_size).+(k-1)*block_size, (1:block_size).+(col-1)*block_size];
                A[(1:block_size).+(row-1)*block_size, (1:block_size).+(col-1)*block_size] ]
                lmul!(qrval.Q', tempmatrix)
                A[(1:block_size).+(k-1)*block_size, (1:block_size).+(col-1)*block_size] .= tempmatrix[1:block_size,:]
                A[(1:block_size).+(row-1)*block_size, (1:block_size).+(col-1)*block_size]  .= tempmatrix[(1:block_size).+block_size,:]
            end
        end
    
        (k==no_tiles) && break
    
        qrval=qr(A[(1:block_size).+(k-1)*block_size, (1:block_size).+(k)*block_size]')
        A[(1:block_size).+(k-1)*block_size, (1:block_size).+(k)*block_size].=qrval.R'
    
        for row in k+1:no_tiles
            A[(1:block_size).+(row-1)*block_size, (1:block_size).+(k)*block_size].=rmul!(A[(1:block_size).+(row-1)*block_size, (1:block_size).+(k)*block_size],qrval.Q)
        end
        for col in k+2:no_tiles 
            tempmatrix=[A[(1:block_size).+(k-1)*block_size, (1:block_size).+(k)*block_size]'
              A[(1:block_size).+(k-1)*block_size, (1:block_size).+(col-1)*block_size]']
            qrval=qr(tempmatrix)
            A[(1:block_size).+(k-1)*block_size, (1:block_size).+(k)*block_size].=qrval.R'
            A[(1:block_size).+(k-1)*block_size, (1:block_size).+(col-1)*block_size] .= 0
            
            for row in k+1:no_tiles
                tempmatrix=[
                    A[(1:block_size).+(row-1)*block_size, (1:block_size).+(k)*block_size]'; 
                    A[(1:block_size).+(row-1)*block_size, (1:block_size).+(col-1)*block_size]']
                    lmul!(qrval.Q',tempmatrix)
                    A[(1:block_size).+(row-1)*block_size, (1:block_size).+(k)*block_size]=  tempmatrix[1:block_size,:]'
                    A[(1:block_size).+(row-1)*block_size, (1:block_size).+(col-1)*block_size]= tempmatrix[(1:block_size).+block_size,:]'
            end
        end

    end

end


function LQsweep_v1!(A, no_tiles, block_size) 
    
        qrval=qr(A[(1:block_size).+(k-1)*block_size, (1:block_size).+(k)*block_size]')
        A[(1:block_size).+(k-1)*block_size, (1:block_size).+(k)*block_size].=qrval.R'
    
        for row in k+1:no_tiles
            A[(1:block_size).+(row-1)*block_size, (1:block_size).+(k)*block_size].=rmul!(A[(1:block_size).+(row-1)*block_size, (1:block_size).+(k)*block_size],qrval.Q)
        end
        for col in k+2:no_tiles 
            tempmatrix=[A[(1:block_size).+(k-1)*block_size, (1:block_size).+(k)*block_size]'
              A[(1:block_size).+(k-1)*block_size, (1:block_size).+(col-1)*block_size]']
            qrval=qr(tempmatrix)
            A[(1:block_size).+(k-1)*block_size, (1:block_size).+(k)*block_size].=qrval.R'
            A[(1:block_size).+(k-1)*block_size, (1:block_size).+(col-1)*block_size] .= 0
            
            for row in k+1:no_tiles
                tempmatrix=[
                    A[(1:block_size).+(row-1)*block_size, (1:block_size).+(k)*block_size]'; 
                    A[(1:block_size).+(row-1)*block_size, (1:block_size).+(col-1)*block_size]']
                    lmul!(qrval.Q',tempmatrix)
                    A[(1:block_size).+(row-1)*block_size, (1:block_size).+(k)*block_size]=  tempmatrix[1:block_size,:]'
                    A[(1:block_size).+(row-1)*block_size, (1:block_size).+(col-1)*block_size]= tempmatrix[(1:block_size).+block_size,:]'
            end
        end

end


struct myTiledMatrix
    TileViews
    ParentMatrix
    tilesize::Int
    no_tiles::Int
end
    
function myTiledMatrix(A::AbstractArray{T, 2}, tile_size::Int, no_tiles::Int) where T
    TileViews=Array{AbstractArray}(undef, no_tiles,no_tiles)
    for i in 1:no_tiles
        for j in 1:no_tiles
            TileViews[i,j]=A[ (i-1)*tile_size.+(1:tile_size),(j-1)*tile_size.+(1:tile_size)]
        end
    end
    return myTiledMatrix(TileViews, A, tile_size, no_tiles)
end

function returnfullmatrix(A::myTiledMatrix)
    out=CuOrROCVector(zeros(A.tilesize*A.no_tiles,A.tilesize*A.no_tiles))
    for i in 1:A.no_tiles
        for j in 1:A.no_tiles
            out[ (i-1)*A.tilesize.+(1:A.tilesize),(j-1)*A.tilesize.+(1:A.tilesize)] .=A.TileViews[i,j]
        end
    end
end

function qr!(A::myTiledMatrix, k)
    qr1=qr(A.TileViews[k,k])
    A.TileViews[k,k] .= qr1.R
    return qr1
end

function qr2!(A::myTiledMatrix, row,k)
    temp=[copy(A.TileViews[k,k]);copy(A.TileViews[row,k])]
    qr1=qr(temp)
    A.TileViews[k,k] .=qr1.R
    A.TileViews[row,k] .= 0
    return qr1
end

function mulq1!(A::myTiledMatrix, k,col,qr1)
    A.TileViews[k,col] .= qr1.Q' * A.TileViews[k,col]
end

function mulq2!(A::myTiledMatrix, k, row, col,qr1)
    temp=[ copy(A.TileViews[k,col]); copy(A.TileViews[row,col])]
    temp= qr1.Q' * temp
    A.TileViews[k,col] .= temp[1:A.tilesize, 1:A.tilesize]
    A.TileViews[row,col] .= temp[A.tilesize+1:2A.tilesize, 1:A.tilesize]
end


function solution_mysvd_v2!(A::myTiledMatrix)
    for k in 1:A.no_tiles
        qr1=qr!(A,k)
        for col in k+1:A.no_tiles
            mulq1!(A,k,col,qr1)
        end

        for row in k+1:A.no_tiles
            qr1=qr2!(A,row,k)
            for col in k+1:A.no_tiles
                mulq2!(A,k,row,col,qr1)
            end
        end

        if (k==A.no_tiles)
            break
        end

        lq1=qr(A.TileViews[k,k+1]')
        A.TileViews[k,k+1] .= lq1.R'
        for row in k+1:A.no_tiles
            A.TileViews[row,k+1] .= A.TileViews[row,k+1]*lq1.Q
        end

        for col in k+2:A.no_tiles
            temp=[copy(A.TileViews[k,k+1])';copy(A.TileViews[k,col])']
            lq2=qr(temp)
            A.TileViews[k,k+1] .=lq2.R'
            A.TileViews[k,col] .= 0
            for row in k+1:A.no_tiles
                temp=[copy(A.TileViews[row,k+1])';copy(A.TileViews[row,col])']
                temp= lq2.Q * temp
                A.TileViews[row,k+1] .= temp[1:A.tilesize, 1:A.tilesize]'
                A.TileViews[row,col] .= temp[A.tilesize+1:2A.tilesize, 1:A.tilesize]'
            end
        end

    end
end



function result_plot1()
    function benchmark_tiles!(timings, a, agpu,i,j, no_blocks, block_size)
        a2=copy(a)
        timings[i,(j-1)*4+1]= @belapsed solution_mysvd!($a2, $no_blocks, $block_size)
        a2=copy(agpu)
        timings[i,(j-1)*4+2]= @belapsed CUDA.@sync solution_mysvd!($a2, $no_blocks, $block_size)
        a2=myTiledMatrix(copy(a), block_size, no_blocks)
        timings[i,(j-1)*4+3]= @belapsed solution_mysvd_v2!($a2)
        a2=myTiledMatrix(copy(agpu), block_size, no_blocks)
        timings[i,(j-1)*4+4]= @belapsed CUDA.@sync solution_mysvd_v2!($a2)
    end
    
    no_blocks=[1,2]
    block_size=128
    matrix_sizes=block_size*no_blocks
    (timings, myplot) = mybenchmark(benchmark_tiles!, matrix_sizes, no_blocks, block_size, ["mysvd CPU" "mysvd GPU" "tilesvd CPU" "tilesvd GPU"])
    return myplot
end

function result_plot2()
    function benchmark_tiles!(timings, a, agpu,i,j, no_blocks, block_size)
        a2=myTiledMatrix(copy(a), block_size[i], no_blocks)
        timings[i,(j-1)*4+1]= @belapsed solution_mysvd_v2!($a2)
        a2=myTiledMatrix(copy(agpu), block_size[i], no_blocks)
        timings[i,(j-1)*4+2]= @belapsed CUDA.@sync solution_mysvd_v2!($a2)
        timings[i,(j-1)*2+3]= @belapsed svd!($a, alg=LinearAlgebra.QRIteration())
        timings[i,(j-1)*2+4]= @belapsed CUDA.@sync svd!($agpu, alg=CUDA.CUSOLVER.QRAlgorithm()) 
    end
    
    no_blocks=[1,2]
    block_size=128
    matrix_sizes=block_size*no_blocks
    (timings, myplot) = mybenchmark(benchmark_tiles!, matrix_sizes, no_blocks, block_size, ["tilesvd CPU" "tilesvd GPU" "LAPACK CPU"  "GPU vendor"])
    return myplot
end


function myTiledMatrix2(A::AbstractArray{T, 2}, tile_size::Int, no_tiles::Int) where T
    TileViews=Array{SubArray}(undef, no_tiles,no_tiles)
    for i in 1:no_tiles
        for j in 1:no_tiles
            TileViews[i,j]=view(A, (i-1)*tile_size.+(1:tile_size),(j-1)*tile_size.+(1:tile_size))
        end
    end
    return myTiledMatrix(TileViews, A, tile_size, no_tiles)
end

function qr_error!(A::myTiledMatrix, k)
    qr1=qr!(A.TileViews[k,k])
    return qr1
end

function qr2_error!(A::myTiledMatrix, row,k)
    qr1=qr!(view(A.ParentMatrix, [(k-1)*A.tile_size.+(1:A.tile_size),(row-1)*A.tile_size.+(1:A.tile_size) ],(k-1)*A.tile_size.+(1:A.tile_size)))
    return qr1
end

     
function solution_mysvd_v3!(A,T) 

    for k in 1:A.no_tiles
        QR1!(A,T,k)                
        Qtapply1_blocks!(A, T, k)
        triu!(A,k,k)

        for row in k+1:A.no_tiles
            QR2!(A,T,row,k)
            Qtapply2_blocks!(A,T,row,k)
            A.TileViews[row,k].=0
        end

        (k==A.no_tiles) && break

        LQ1!(A,T,k) 
        LQtapply1_blocks!(A,T,k)
        tril!(A,k,k+1)

        for col in k+2:A.no_tiles 
            LQ2!(A,T,k,col)
            LQtapply2_blocks!(A, T, k, col)
            A.TileViews[k,col].=0
        end
    end

end

@kernel function simple_transpose_kernel!(output, @Const(input))
    I, J = @index(Global, NTuple)
    @inbounds output[I, J] = input[J,I]
end

@kernel function lmem_transpose_kernel!(output, @Const(input))
    gi, gj = @index(Group, NTuple)
    i, j = @index(Local,  NTuple)

    N = @uniform @groupsize()[1]
    M = @uniform @groupsize()[2]
    
    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(output) (N+1, M) 

    # Manually calculate global indexes
    # Later on we need to pivot the group index
    I = (gi-1) * N + i
    J = (gj-1) * M + j

    @inbounds tile[i, j] = input[I, J]

    @synchronize

    # Pivot the group index
    I = (gj-1) * M + i
    J = (gi-1) * N + j

    @inbounds output[I, J] = tile[j, i]
end

function result_plot3()
    function benchmark_transpose!(timings, a, agpu,i,j, no_blocks, block_size)
        T=Float32
        N=no_blocks*block_size[1]
        input = CuOrROCArray(rand!(allocate(backend, T, N, N)))
        output = similar(input)
        mykernel=simple_transpose_kernel!(backend, block_size) #compile kernel
        timings[i,(j-1)*2+1]= @belapsed ($mykernel($input, $output, ndrange=size($output)); KernelAbstractions.synchronize(backend);)
        input = CuOrROCArray(rand!(allocate(backend, T, N, N)))
        output = similar(input)
        mykernel=lmem_transpose_kernel!(backend, block_size) #compile kernel
        timings[i,(j-1)*2+2]= @belapsed ($mykernel($input, $output, ndrange=size($output)); KernelAbstractions.synchronize(backend);)
    end
    
    no_blocks=[1,2,4,8,16,32]
    block_size=(32,32)
    matrix_sizes=block_size[1]*no_blocks
    (timings, myplot) = mybenchmark(benchmark_transpose!, matrix_sizes, no_blocks, block_size, ["simple" "lmem"])
    return(myplot)
end

@kernel function solution_lmem_matmul_kernel!(output, input1, input2)
	gi, gj = @index(Group, NTuple)
	i, j = @index(Local, NTuple)
    
	N = @uniform @groupsize()[1] #assuming here N==M
	M = @uniform @groupsize()[2]
    
	# +1 to avoid bank conflicts on shared memory
	tile_out = @localmem eltype(output) (N + 1, N)
    tile_in1 = @localmem eltype(input1) (N + 1, N)
    tile_in2 = @localmem eltype(input2) (N + 1, M)
    @inbounds tile_out[i,j] =0

	# Manually calculate global indexes of output
	I = (gi - 1) * N + i
	J = (gj - 1) * M + j
    
    #iterate over tiles, load into local and calculate
    for tile_index in 1:div(size(input1)[2],N) 
        @inbounds tile_in1[i,j]=input1[I,j+N*(tile_index-1)]
        @inbounds tile_in2[i,j]=input2[i+N*(tile_index-1),J]
        @synchronize
        tmp_sum = zero(eltype(output))
        for index in 1:N
            @inbounds tmp_sum += tile_in1[i, index] * tile_in2[index, j]
        end
        @inbounds tile_out[i,j]+=tmp_sum
        @synchronize
    end
    
	@inbounds output[I, J] = tile_out[i,j]
end

function result_plot4()
    function benchmark_matmul!(timings, a, agpu,i,j, no_blocks, block_size)
        #create second input matrices
        T=Float32
        N=no_blocks*block_size[1]
        input2 = CuOrROCArray(rand!(allocate(backend, T, N, N)))
        output = similar(input2)
        mykernel=solution_lmem_matmul_kernel!(backend, block_size) #compile kernel
        timings[i,(j-1)*2+2]= @belapsed CUDA.@sync $input2*$agpu
        timings[i,(j-1)*2+1]= @belapsed ($mykernel($output, $input2, $agpu, ndrange=size($output)); KernelAbstractions.synchronize(backend);)
    end
    
    no_blocks=[1,2,4,8,16,32]
    block_size=(32,32)
    matrix_sizes=block_size[1]*no_blocks
    (timings, myplot) = mybenchmark(benchmark_matmul!, matrix_sizes, no_blocks, block_size, ["own implementation" "reference"])
    return myplot
end

struct TiledTiledMatrix
    TileTileViews::Array{Array{SubArray}}
    tilesize1::Int
    tilesize2::Int
    no_tiles1::Int
    no_tiles2::Int
    no_tiles::Int
end

function TiledTiledMatrix(A::AbstractArray{T, 2}, tilesize1::Int, tilesize2::Int, no_tiles1::Int, no_tiles2::Int) where T
    no_tiles=Int(size(A,1)/blocksize_n)
    TileViews=Array{SubArray}(undef, no_tiles,no_tiles)
    for i in 1:no_tiles
        for j in 1:no_tiles
            TileViews[i,j]=view(A, (i-1)*blocksize_m.+(1:blocksize_m),(j-1)*blocksize_n.+(1:blocksize_n))
        end
    end
    return TiledTiledMatrix(TileViews, size(A,1),size(A,2), blocksize_m,blocksize_n, no_tiles)
end
    

