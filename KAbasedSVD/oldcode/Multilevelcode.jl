using KernelAbstractions, Test, Random, BenchmarkTools
using KernelAbstractions.Extras: @unroll
using CUDA, CUDA.CUDAKernels
#CUDA.allowscalar(false)


########################################################################
############# MATMUL ###############################################
#####################################################################

# Simple kernel for matrix multiplication
@kernel function matmul_kernel!(a, b, c)
	i, j = @index(Global, NTuple)

	# creating a temporary sum variable for matrix multiplication
	tmp_sum = zero(eltype(c))
	for k ∈ 1:size(a)[2]
		tmp_sum += a[i, k] * b[k, j]
	end

	c[i, j] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function matmul!(a, b, c)
	if size(a)[2] != size(b)[1]
		println("Matrix size mismatch!")
		return nothing
	end
	backend = KernelAbstractions.get_backend(a)
	kernel! = matmul_kernel!(backend)
	kernel!(a, b, c, ndrange = size(c))
end

#TODO: non-square matrices
@kernel function lmem_MM_kernel!(output, input1, input2)
    gi, gj = @index(Group, NTuple)
	i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1] #assuming here N==M
    
    tiled_MM!(output, input1, input2, i, j, gi, gj, N)
    
end
function tiled_MM!(output, input1, input2, i, j, gi, gj, N)
    tile_out = @localmem eltype(output) (N + 1, N)
    tile_in1 = @localmem eltype(input1) (N + 1, N)
    tile_in2 = @localmem eltype(input2) (N + 1, N)
    setzero!(tile_out,i,j)

    for tile_index in 1:div(size(input1)[2],N)
        @inbounds copy_tile!(tile_in1, tileview(input1, gi,tile_index,N ),i,j)
        @inbounds copy_tile!(tile_in2, tileview(input2, tile_index,gj, N ),i,j)
        @synchronize
        matmul_basic!(tile_out, tile_in1, tile_in2, i, j)
        @synchronize
    end
    copy_tile!(tileview(output,gi,gj,N), tile_out, i,j)

end

function matmul_basic!(c, a, b, i, j)
	tmp_sum = zero(eltype(c))
	for k ∈ 1:size(a)[2]
		@inbounds tmp_sum += a[i, k] * b[k, j]
	end
	@inbounds c[i, j] = tmp_sum
end

#TODO: find out why below is not working
@kernel function coalesced_matmul_kernel!(output, input1, input2)
	gi, gj = @index(Group, NTuple)
	i, j   = @index(Local, NTuple)

	N   = @uniform @groupsize()[1]
	BLOCK_ROWS = @uniform @groupsize()[2]

	# +1 to avoid bank conflicts on shared memory
    tile_out = @localmem eltype(output) (N + 1, N)
    tile_in1 = @localmem eltype(input1) (N + 1, N)
    tile_in2 = @localmem eltype(input2) (N + 1, N)
    tmp_sum = @localmem eltype(output) div(N,BLOCK_ROWS)
    @unroll for coales_offset in 0:BLOCK_ROWS:(N-1)
        @inbounds tile_out[i,j+coales_offset] =0
        if i==0 && j==0 
            @inbounds tmp_sum[coales_offset]=0
        end
    end
    
    for tile_index in 1:div(size(input1)[2],N) #TODO: make this work for tiles that arent dividable by 32
        @unroll for coales_offset in 0:BLOCK_ROWS:(N-1)
            L = (gi - 1) * N + i
            J = (gj - 1) * N + j
            @inbounds tile_in1[i,j+coales_offset]=input1[L,j+N*(tile_index-1)+coales_offset]
            @inbounds tile_in2[i,j+coales_offset]=input2[i+N*(tile_index-1),J+coales_offset]
            if i==0 && j==0 
                tmp_sum[coales_offset]=0
            end
        end
        @synchronize
        @unroll for coales_offset in 0:BLOCK_ROWS:(N-1)
            for index in 1:N
                @inbounds tmp_sum[coales_offset] += tile_in1[i, index] * tile_in2[index, j+coales_offset]
            end
            @inbounds tile_out[i,j+coales_offset]+=tmp_sum[coales_offset]
        end
        @synchronize
    end
    
	@unroll for coales_offset in 0:BLOCK_ROWS:(TILE_DIM-1)
        L = (gi - 1) * N + i
        J = (gj - 1) * N + j
		@inbounds output[L, J+coales_offset] = tile_out[i,j+coales_offset]
	end
    
end

########################################################################
############# TEST ###############################################
#####################################################################
backend = CPU()
T = Float32
N = 2048
ndrange = (N, N)
input1 = rand!(allocate(backend, T, N, N))
input2 = rand!(allocate(backend, T, N, N))
output = similar(input1)
output.=0
const TILE_DIM = 32
const BLOCK_ROWS = 8
lmem_transpose_kernel!(backend, (TILE_DIM, TILE_DIM))(input, output,  ndrange = ndrange)
matmul_kernel!(backend, (TILE_DIM, TILE_DIM))(input1, input2, output, ndrange=size(output))
isapprox(input1*input2,output)
coalesced_matmul_kernel!(backend, (TILE_DIM, BLOCK_ROWS))(output, input1, input2, ndrange=(N, div(N, div(TILE_DIM, BLOCK_ROWS))))




function benchmark(kernel;  backend = CUDABackend(), T = Float32, N = 2048, ndrange = (N, N), nreps = 100, time_max = 1)
	input = rand!(allocate(backend, T, N, N))
	output = similar(input)
	kernel(input, output,  ndrange = ndrange)
	time_used = 0.0
	i = 0
	while (time_used < time_max && i < nreps)
		input = rand!(allocate(backend, T, N, N))
		output = similar(input)
		time_used += CUDA.@elapsed begin
			KernelAbstractions.synchronize(backend)
			kernel(input, output,  ndrange = ndrange)
		end
		i += 1
	end
	KernelAbstractions.synchronize(backend)
	return (time_used / i)
end

function benchmark_base(func; T = Float32, N = 2048, nreps = 100, time_max = 1)
	input = CUDA.rand(T, N, N)
	output = similar(input)
	func(output, input)
	time_used = 0.0
	i = 0
	while (time_used < time_max && i < nreps)
		input = CUDA.rand(T, N, N)
		output = similar(input)
		time_used += CUDA.@elapsed func(output, input)
		i += 1
	end
	return (time_used / i)
end

function benchmark2(kernel;  backend = CUDABackend(), T = Float32, N = 2048, ndrange = (N, N), nreps = 100, time_max = 1)
	input1 = rand!(allocate(backend, T, N, N))
    input2 = rand!(allocate(backend, T, N, N))
	output = similar(input1)
	kernel(output, input1,input2,  ndrange = ndrange)
	time_used = 0.0
	i = 0
	while (time_used < time_max && i < nreps)
		input1 = rand!(allocate(backend, T, N, N))
        input2 = rand!(allocate(backend, T, N, N))
		output = similar(input1)
		time_used += CUDA.@elapsed begin
			KernelAbstractions.synchronize(backend)
			kernel(output, input1,input2,  ndrange = ndrange)
		end
		i += 1
	end
	KernelAbstractions.synchronize(backend)
	return (time_used / i)
end

function benchmark_base2(; T = Float32, N = 2048, nreps = 100, time_max = 1)
	input1 = CUDA.rand(T, N, N)
    input2 = CUDA.rand(T, N, N)
	output = similar(input1)
    output.=input1*input2
	time_used = 0.0
	i = 0
	while (time_used < time_max && i < nreps)
		input1 = CUDA.rand(T, N, N)
        input2 = CUDA.rand(T, N, N)
		output = similar(input1)
		time_used += CUDA.@elapsed output.=input1*input2
		i += 1
	end
	return (time_used / i)
end

const TILE_DIM = 32
const BLOCK_ROWS = 8
backend = CUDABackend()
N=2048

benchmark_base(CUDA.transpose!)
t_ref=benchmark_base(CUDA.transpose!)
benchmark(coalesced_transpose_kernel!(backend, (TILE_DIM, BLOCK_ROWS));  ndrange = (N, div(N, div(TILE_DIM, BLOCK_ROWS))))
t_co=benchmark(coalesced_transpose_kernel!(backend, (TILE_DIM, BLOCK_ROWS));  ndrange = (N, div(N, div(TILE_DIM, BLOCK_ROWS))))
benchmark(simple_transpose_kernel!(backend, (TILE_DIM, TILE_DIM)))
t_simple=benchmark(simple_transpose_kernel!(backend, (TILE_DIM, TILE_DIM)))
benchmark(lmem_transpose_kernel!(backend, (TILE_DIM, TILE_DIM)))
t_lmem=benchmark(lmem_transpose_kernel!(backend, (TILE_DIM, TILE_DIM)))
#why are these slow

t_simple/t_ref
t_lmem/t_ref
t_co/t_ref

benchmark_base2()
t_ref=benchmark_base2()
benchmark2(coalesced_matmul_kernel!(backend, (TILE_DIM, BLOCK_ROWS));  ndrange = (N, div(N, div(TILE_DIM, BLOCK_ROWS))))
t_co=benchmark2(coalesced_matmul_kernel!(backend, (TILE_DIM, BLOCK_ROWS));  ndrange = (N, div(N, div(TILE_DIM, BLOCK_ROWS))))
benchmark2(matmul_kernel!(backend, (TILE_DIM, TILE_DIM)))
t_simple=benchmark2(matmul_kernel!(backend, (TILE_DIM, TILE_DIM)))
benchmark2(lmem_matmul_kernel!(backend, (TILE_DIM, TILE_DIM)))
t_lmem=benchmark2(lmem_matmul_kernel!(backend, (TILE_DIM, TILE_DIM)))

t_simple/t_ref
t_lmem/t_ref
t_co/t_ref

########################################################################
############# TILING ABSTRACTION ###############################################
#####################################################################

#define localmem and synchronize per level
struct TiledAlgo    
    num_levels::int 
	hardware_setup::TBD
    tile_sizes_bylevel::
   
    function TiledAlgo()
    end
end

function setzero!(tile,i,j)
    @inbounds tile[i,j]=0
end

function tileview(A, gi, gj, N)
    return view(A, (gi-1)*N.+(1:N),(gj-1)*N.+(1:N))
end

function copy_tile!(A, B, i, j)
    @inbounds A[i,j] = B[i,j]
end



@kernel function MM_kernel!(c, a,b)
    i, j = @index(Global, NTuple)
    matmul_basic!(a,b,c,i,j)
end




backend = CPU()
T = Float32
N = 8
ndrange = (N, N)
input1 = rand!(allocate(backend, T, N, N))
input2 = rand!(allocate(backend, T, N, N))
output = similar(input1)
output.=0
const TILE_DIM = 4
MM_kernel!(backend, (TILE_DIM, TILE_DIM))(output,input1, input2,  ndrange=size(output))
lmem_MM_kernel!(backend, (TILE_DIM, TILE_DIM))(output,input1, input2, ndrange=size(output))
input1*input2≈output

n1=64*4
n2=64
n3=16
n4=4

A=randn(n1,n1)
B=randn(n1,n1)
C=zeros(n1,n1)

N=8

no_tiles=div(n1,n2)
for tile_index in 1:no_tiles
	tile_out = zeros(n2,n2)
    tile_in1 = @localmem eltype(input1) (N + 1, N)
    tile_in2 = @localmem eltype(input2) (N + 1, N)
    setzero!(tile_out,i,j)

    for tile_index in 1:div(size(input1)[2],N) #TODO: make this work for tiles that arent dividable by 32
        @inbounds copy_tile!(tile_in1, tileview(input1, gi,tile_index,N ),i,j)
        @inbounds copy_tile!(tile_in2, tileview(input2, tile_index,gj, N ),i,j)
        @synchronize
        matmul_basic!(tile_out, tile_in1, tile_in2, i, j)
        @synchronize
    end
    copy_tile!(tileview(output,gi,gj,N), tile_out, i,j)
end
	


