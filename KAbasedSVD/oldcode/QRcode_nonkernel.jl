
##################################
##################################
#1. simple non-kernel codes for reference
##################################
##################################

##################################
#1.1  LQ
##################################

struct MyLQ{T, AT}
    M
    tau
    m::Int
    n::Int
    MyLQ(M,tau)  = (length(tau)==size(M,1)) ? new{eltype(M), ArrayInterface.parameterless_type(M)}(M,tau,size(M,1),size(M,2)) : error("DimensionMismatch")
    MyLQ(M,tau, m::Int, n::Int)  = error("Please dont provide values for m and n")
end

function getL(myLQ::MyLQ)
    (m,n)=size(myLQ.M)
    if n>m
        return(tril(myLQ.M)[1:m,1:m])
    else
        return(tril(myLQ.M))
    end
end

function applyQ!(X, myLQ::MyLQ)
    (m,n)=size(myLQ.M)
    for k in 1:min(m,n) 
        @views X[k:n] .= X[k:n]-(myLQ.tau[k]*([1;myLQ.M[k,(k+1):n]] *([1;myLQ.M[k,(k+1):n]])'.* X[k:n]))
    end
    return X
end

function applyQ!(X, myQR::MyLQ)
    (m,n)=size(myLQ.M)
    for k in 1:min(m,n)
        @views X[k:n] .= X[k:n]-myLQ.tau[k]*[1;myLQ.M[(k+1):m,k]]*([1;myQR.M[(k+1):m,k]]'*X[k:m])
    end
    return X
end

function applyQt!(X, myLQ::MyLQ)
    (m,n)=size(myLQ.M)
    for k in min(m,n):-1:1
        @views X[k:n]=X[k:n]-(myLQ.tau[k]*(X[k:n]' .*[1;myLQ.M[k,(k+1):n]])*[1;myLQ.M[k,(k+1):n]]')'
    end
    return X
end

function getQ(myLQ::MyLQ)
    (m,n)=size(myLQ.M)
    AT=ArrayInterface.parameterless_type(myLQ.M)
    Q=AT(zeros(n,n))
    Q[diagind(Q)].=1
    for k=1:n 
        applyQ!(view(Q,:,k),myLQ)
    end
    return Q
end


applyQ(X, myLQ::MyLQ) = applyQ!(copy(X), myLQ::MyLQ)
applyQt(X, myLQ::MyLQ) = applyQt!(copy(X), myLQ::MyLQ)

function simple_LQ!(M)
    AT= ArrayInterface.parameterless_type(M)
    (m,n)=size(M)
    tau=AT(zeros(m))
    no_iter= (n>m) ? m : (n-1)
    temp_space=AT(zeros(m))
    current_col=AT(zeros(n))
    for k in 1:no_iter
        view(current_col,k:n).=view(M,k,k:n)
        view(current_col,k) .+= (view(current_col,k)  .< 0 ? -1 : 1 )*norm(view(current_col,k:n))
        view(tau,k) .=2*(view(current_col,k)/norm(view(current_col,k:n))).^2
        view(current_col,k:n) ./= view(current_col,k)
        view(temp_space,k:m) .= (view(M,k:m,k:n)*view(current_col,k:n))
        view(M,k:m,k:n).-=view(tau,k).*view(temp_space,k:m).*view(current_col,k:n)'
        view(M,k,(k+1):n).=view(current_col,(k+1):n)
    end
    return MyLQ(M, tau)
end


simple_LQ(M) = simple_LQ!(copy(M))

##################################
#1.2  QR
##################################

struct MyQR{T,AT}
    M
    tau
    m::Int
    n::Int
    MyQR(M ,tau)  = (length(tau)==size(M,2)) ? new{eltype(M), ArrayInterface.parameterless_type(M)}(M,tau,size(M,1),size(M,2)) : error("DimensionMismatch")
    MyQR(M,tau, m::Int, n::Int)  = error("Please dont provide values for m and n")
end


function getR(myQR::MyQR)
    (m,n)=size(myQR.M)
    if m>n
        return(triu(myQR.M)[1:n,1:n])
    else
        return(triu(myQR.M))
    end
end

function applyQ!(X, myQR::MyQR)
    (m,n)=size(myQR.M)
    for k in min(m,n):-1:1
        @views X[k:m] .= X[k:m]-myQR.tau[k]*[1;myQR.M[(k+1):m,k]]*([1;myQR.M[(k+1):m,k]]'*X[k:m])
    end
    return X
end

function applyQt!(X, myQR::MyQR)
    (m,n)=size(myQR.M)
    for k in 1:min(m,n)
        @views X[k:m]=X[k:m]-myQR.tau[k]*[1;myQR.M[(k+1):m,k]]*([1;myQR.M[(k+1):m,k]]'*X[k:m])
    end
    return X
end

function getQ(myQR::MyQR)
    (m,n)=size(myQR.M)
    AT=ArrayInterface.parameterless_type(myQR.M)
    Q=AT(zeros(m,m))
    Q[diagind(Q)].=1
    for k=1:m 
        applyQ!(view(Q,:,k),myQR)
    end
    return Q
end

applyQ(X, myQR::MyQR) = applyQ!(copy(X), myQR::MyQR)
applyQt(X, myQR::MyQR) = applyQt!(copy(X), myQR::MyQR)

function simple_QR!(M)
    (m,n)=size(M)
    AT=ArrayInterface.parameterless_type(M)
    tau=AT(zeros(n))
    no_iter= (m>n) ? n : (m-1)
    current_col=AT(zeros(m))
    temp_space=AT(zeros(n))
    for k in 1:no_iter
        view(current_col,k:m).=view(M,k:m,k)
        view(current_col,k).+=(view(current_col,k) .< 0 ? -1 : 1)*norm(view(current_col,k:m))
        view(tau,k) .= 2*(view(current_col,k) /norm(view(current_col,k:m))).^2
        view(current_col,k:m) ./= view(current_col,k)
        view(temp_space,k:n) .= (view(current_col,k:m)'*view(M,k:m,k:n))'
        view(M,k:m,k:n).-=view(tau,k).*view(current_col,k:m)*view(temp_space,k:n)'
        view(M,(k+1):m,k).=view(current_col,(k+1):m)
    end
    return MyQR(M,tau)
end

simple_QR(M)= simple_QR!(copy(M))

##################################
#1.3  verifying accuracy
##################################
CUDA.allowscalar(true)

matrix_dims_test=[(3,3),(3,5),(5,3)]

function verify_correctness_simpleCPU(matrix_dims_test)
    for (fname, myfname ,get_output_left, get_output_right, size_q) in ((:qr, :simple_QR, :getQ, :getR, 1 ),
                                                             (:lq, :simple_LQ, :getL, :getQ, 2)  
                                                             )
        @info fname

        @eval begin
            for (m,n) in matrix_dims_test
                a=rand(m,n)
                a_gpu= a |> cu
                b=rand((m,n)[$size_q])
                b_gpu = b |> cu
                output_left_ref,output_right_ref=$fname(a)
                my_output = $myfname(a_gpu) 
                my_output_left = $get_output_left(my_output)
                my_output_right= $get_output_right(my_output)
                q_ref=$fname(a).Q

                @info (m,n) Float32.(Array(my_output_left) ) ≈ Float32.( (output_left_ref)) 
                @info "" Float32.((output_right_ref) ) ≈  Float32.( Array(my_output_right)) 
                @info "" (q_ref'*b ≈ Array(applyQt(b_gpu,my_output))) (q_ref*b ≈ Array(applyQ(b_gpu,my_output)))
            end
        end
    end
end

verify_correctness_simpleCPU(matrix_dims_test)

##################################
# 1.4 QR multiply functions
##################################

#minimal working example of QR according to Trefethen, Bau Numerical Linear Algebra
function simple_QR!(M)
    (m,n)=size(M)
    tau=zeros(n)
    no_iter= (m>n) ? n : (m-1)
    current_col=zeros(m)
    temp_space=zeros(n)
    for k in 1:no_iter
        view(current_col,k:m).=view(M,k:m,k)
        view(current_col,k).+=(view(current_col,k) .< 0 ? -1 : 1)*norm(view(current_col,k:m))
        view(tau,k) .= 2*(view(current_col,k) /norm(view(current_col,k:m))).^2
        view(current_col,k:m) ./= view(current_col,k)
        view(temp_space,k:n) .= (view(current_col,k:m)'*view(M,k:m,k:n))'
        view(M,k:m,k:n).-=view(tau,k).*view(current_col,k:m)*view(temp_space,k:n)'
        view(M,(k+1):m,k).=view(current_col,(k+1):m)
    end
    return M,tau
end


function applyQt_vec!(X, M, tau, m, n)
    for k in 1:min(m,n)
        @views X[k:m]=X[k:m]-tau[k]*[1;M[(k+1):m,k]]*([1;M[(k+1):m,k]]'*X[k:m])
    end
    return X
end

function applyQt_mat!(A, M,tau)
    (m,n)=size(M)
    for k=1:size(A,2) 
        applyQt_vec!(view(A,:,k),M, tau, m,n)
    end
    return A
end

function getQt(M,tau)
    (m,n)=size(M)
    Q=zeros(m,m)
    Q[diagind(Q)].=1
    applyQt_mat!(Q, M,tau)
    return Q
end


function getQt_block(M,tau)
    (m,_)=size(M)
    Q=zeros(2m,2m)+I
    applyQt_mat_block!(Q, M,tau)
    return Q
end
function applyQt_vec_block!(X, M, tau, m, n, m2)
    for k in 1:min(m2,n)
        @views X[k:m]=X[k:m]-tau[k]*[1;zeros(m-k-m2);M[:,k]]*([1;zeros(m-k-m2);M[:,k]]'*X[k:m])
    end
    for k in min(m2,n)+1:min(m,n)
        @views X[k:m]=X[k:m]-tau[k]*[1;M[(k+1):m,k]]*([1;M[(k+1):m,k]]'*X[k:m])
    end
    return X
end

function applyQt_mat_block!(A, M,tau)
    (m2,n)=size(M)
    m=m2*2
    for k=1:size(A,2) 
        applyQt_vec_block!(view(A,:,k),M, tau, m,n, m2)
    end
    return A
end


##################################
##################################
#2. Kernel code --- extremely slow
##################################
##################################

function simple_QR_GPU_ref!(M::CuArray)
    (m,n)=size(M)
    
    no_iter= (m>n) ? n : (m-1)
    tau=CUDA.zeros(n)
    cache_mem=CUDA.zeros(m,n)
    cache_mem2=CUDA.zeros(5)

    no_threads=min.((m,n),32)
    no_blocks=cld.((m,n), no_threads)
    @cuda cooperative=true blocks=cld.((m,n), no_threads) threads=no_threads simple_QR_kernel!(M, tau, cache_mem,cache_mem2, no_iter, m,n)
    return M, tau
end

function simple_QR_kernel!(M, tau, cache_mem, cache_mem2, no_iter, m,n)
    grid_handle = this_grid()
    current_index_x = (blockIdx().x-1) * blockDim().x + threadIdx().x
    current_index_y = (blockIdx().y-1) * blockDim().y + threadIdx().y

    for k in 1:no_iter
        j=current_index_x+(k-1)
        l=current_index_y+(k-1)
        if (k<j<=m) && (l==k)
            cache_mem[j,l]=(M[j,l])^2
        end
        sync_grid(grid_handle)
        if (k==j && l==k)
            sum!(view(cache_mem2,2),view(cache_mem,(k+1):m,k))
            cache_mem2[3]=sign(M[k,k])*sqrt(cache_mem2[2]+(M[k,k])^2)+M[k,k] 
        end
        sync_grid(grid_handle)
        one_val=cache_mem2[3]
        if (k==j && l==k)
            cache_mem2[4]=sqrt(cache_mem2[2]+(one_val)^2)
            tau[k]=2*(one_val/cache_mem2[4])^2
        elseif (k<j<=m) && (k==l)
            cache_mem[j,l]=M[j,k]/one_val*M[j,l]
        elseif (k<j<=m) && (k<l<=n)
            cache_mem[j,l]=M[j,k]/one_val*M[j,l]
        end
        sync_grid(grid_handle)
        if (j==k) && (k<=l<=n)
            sum!(view(cache_mem,k,l),view(cache_mem,(k+1):m,l))
            cache_mem[j,l]+=M[j,l]
        end
        sync_grid(grid_handle)
        if (j==k) && (k<=l<=n)
            M[j,l]=M[j,l]-tau[k]*cache_mem[j,l]
        elseif (k<j<=m) && (k<l<=n) 
            M[j,l]=M[j,l]-tau[k]*M[j,k]/one_val*cache_mem[k,l]
        end
        sync_grid(grid_handle)
        if (k<j<=m) && (k==l)
            M[j,l]=M[j,l]/one_val
        end

    end
    return;
end

function simple_QR_GPU!(M::CuArray)
    (m,n)=size(M)
    
    no_iter= (m>n) ? n : (m-1)
    tau=CUDA.zeros(n)
    cache_mem=CUDA.zeros(m,n)
    cache_mem2=CUDA.zeros(5)

    no_threads=min.((m,n),32)
    no_blocks=cld.((m,n), no_threads)
    for k in 1:no_iter
        @cuda blocks=cld.((m,n), no_threads) threads=no_threads QR_kernela!(M, tau, cache_mem,cache_mem2, no_iter, m,n,k)
        @cuda blocks=cld.((m,n), no_threads) threads=no_threads QR_kernelb!(M, tau, cache_mem,cache_mem2, no_iter, m,n,k)
        @cuda blocks=cld.((m,n), no_threads) threads=no_threads QR_kernelc!(M, tau, cache_mem,cache_mem2, no_iter, m,n,k)
    end
    @cuda blocks=cld.((m,n), no_threads) threads=no_threads QR_kernelend!(M, tau, cache_mem,cache_mem2, no_iter, m,n,no_iter)
    return M, tau
end



function QR_kernela!(M, tau, cache_mem, cache_mem2, no_iter, m,n,k)
    
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x
    l = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if (k<j<=m) && (l==k)
        CUDA.@atomic cache_mem2[2]+=(M[j,l])^2
    elseif (j==k) && (k<=l<=n)
        cache_mem[k,l]=0.0
    end
    if ((k-1)<j<=m) && (l==(k-1)) && k>1
        M[j,l]=M[j,l]/cache_mem2[1]
    end
    return;
end
function QR_kernelb!(M, tau, cache_mem, cache_mem2, no_iter, m,n,k)
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x
    l = (blockIdx().y-1) * blockDim().y + threadIdx().y
    
    one_val=sign(M[k,k])*sqrt(cache_mem2[2]+(M[k,k])^2)+M[k,k]
    if (k==j && l==k)
        cache_mem2[1]=one_val
        tau[k]=2*(one_val/sqrt(cache_mem2[2]+(one_val)^2))^2
        cache_mem2[2]=0.0
        CUDA.@atomic cache_mem[k,l]+=M[j,l]
    elseif (k<j<=m) && (k==l)
        cache_mem[j,l]=M[j,k]/one_val*M[j,l]
        CUDA.@atomic cache_mem[k,l]+=cache_mem[j,l]
    elseif (k<j<=m) && (k<l<=n)
        cache_mem[j,l]=M[j,k]./one_val*M[j,l]
        CUDA.@atomic cache_mem[k,l]+=cache_mem[j,l]
    elseif (j==k) && (k<l<=n)
        CUDA.@atomic cache_mem[k,l]+=M[j,l]
    end
    return;
end

function QR_kernelc!(M, tau, cache_mem, cache_mem2, no_iter, m,n,k)
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x
    l = (blockIdx().y-1) * blockDim().y + threadIdx().y
    one_val=cache_mem2[1]
    if (j==k) && (k<=l<=n)
        M[j,l]=M[j,l]-tau[k]*cache_mem[j,l]
    elseif (k<j<=m) && (k<l<=n) 
        M[j,l]=M[j,l]-tau[k]*M[j,k]/one_val*cache_mem[k,l]
    end
    return;
end

function QR_kernelend!(M, tau, cache_mem, cache_mem2, no_iter, m,n,k)
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x
    l = (blockIdx().y-1) * blockDim().y + threadIdx().y
    one_val=cache_mem2[1]
    if (k<j<=m) && (k==l)
        M[j,l]=M[j,l]/one_val
    end
    return;
end

function applyQ!(X::CuArray, M::CuArray, tau::CuArray)
    (m,n)=size(M)
    workspace1=CUDA.zeros(m)
    for k in min(m,n):-1:1 
        view(workspace1,k:m) .= view(X,k:m)
        view(workspace1,(k+1):m) .*= view(M,(k+1):m,k)
        alpha = sum(view(workspace1,k:m)) * tau[k]
        view(X,k) .-= alpha
        view(X,(k+1):m) .-= view(M,(k+1):m,k) .* alpha
    end
    return X
end


function applyQ!(X,myQR::MyQR)
    M=myQR.M
    tau=myQR.tau
    (m,n)=size(M)
    V=similar(M)
    Xcu= CuArray(X)
    @cuda threads=min(m,n) applyQ!(V,Xcu,M,tau)
    X=Array(Xcu)
end

function applyQ1a!(V, X, M, tau, k, m, n)
    current_index = (blockIdx().x-1) * blockDim().x + threadIdx().x

            j=current_index+(k-1)
            if ((k+1)<=j<=m)
                V[k,j]=M[j,k]*X[j]
            end
            return nothing
end

function applyQ1b!(V, X, M, tau, k, m, n)
    current_index = (blockIdx().x-1) * blockDim().x + threadIdx().x

            j=current_index+(k-1)
            
            if j==k
                sum!(view(V,k,k),view(V,k,(k+1):m))
            end
            return nothing
end
function applyQ1c!(V, X, M, tau, k, m, n)
    current_index = (blockIdx().x-1) * blockDim().x + threadIdx().x

            j=current_index+(k-1)
            
            alpha=(V[k,k]+X[k])*tau[k]
            if (k<=j<=m)
                if j==k
                    HH_el=alpha
                else
                    HH_el=alpha* M[j,k]
                end
                X[j]=X[j]-HH_el
            end
            return nothing
end



function applyQ1!(X::CuArray,myQR::MyQR)
    M=myQR.M
    tau=myQR.tau
    V=myQR.V
    (m,n)=size(M)
    no_threads=min(768,min(m,n))
    no_blocks=cld(min(m,n),no_threads)
    for k in min(m,n):-1:1
        @cuda blocks = no_blocks threads=no_threads applyQ1a!(V,X,M,tau,k,m,n) 
        @cuda blocks = no_blocks threads=no_threads applyQ1b!(V,X,M,tau,k,m,n) 
        @cuda blocks = no_blocks threads=no_threads applyQ1c!(V,X,M,tau,k,m,n) 
    end
    return X
end
applyQ1(X, myQR::MyQR) = applyQ1!(copy(X), myQR::MyQR)


##################################
##################################
#9. Determining optimal configuration
##################################
##################################

kernel = @cuda launch=false myfunc(args...)
config = launch_configuration(kernel.fun)
threads = min(N, config.threads)
blocks = cld(N, threads)

n=1024
a=CUDA.randn(n,n)
qr_out=simple_QR(Array(a))
b=CUDA.randn(n)
(k1a,k1b,k1c)=applyQ1_calc(b,qr_out)
k2=applyQ2_calc(b,qr_out)
config = launch_configuration(k2.fun)


function applyQ1_calc(Y::CuArray,myQR::MyQR)
    X=copy(Y)
    M=CuArray(myQR.M)
    tau=CuArray(myQR.tau)
    (m,n)=size(M)
    V=similar(M)
    a=@cuda launch=false applyQ1a!(V,X,M,tau,1,m,n)
    b=@cuda launch=false applyQ1b!(V,X,M,tau,1,m,n)
    c=@cuda launch=false applyQ1c!(V,X,M,tau,1,m,n)
    return (a,b,c)
end


##################################
##################################
#02. Tring out some kernel functions
##################################
##################################




function copycontents!(to,from)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds to[i]=from[i]
    return nothing
end

function mul_matrix_vec!(out,M,v, workspace)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @cuda threads=size(M,2) dynamic=true mul_vec_vec!(view(workspace,i,:),view(M,i,:),v)
    @inbounds out[i] = sum(view(workspace,i,:))
    return nothing
end

function mul_vec_matrix!(out,v,M, workspace)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @cuda threads=size2 dynamic=true mul_vec_vec!(view(workspace,:,i),view(M,:,i),v)
    @inbounds out[i] = sum(view(workspace,:,i))
    return nothing
end

function mul_vec_vec!(out,v1,v2) #comparison of python equivalent
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds out[i]=v1[i]*v2[i]
    return nothing
end

function operation_vec!(v, operation)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds v[i] = operation(v[i])
    return nothing
end

function Householderreflector!(M,v,vtM) 
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    @inbounds M[i,j] = M[i,j] - 2*v[i]*vtM[j]
    return nothing
end

function checkdims_HH(M, v, vtM)
    return ( size(M,1) == length(v) ) && ( size(M,2) == length(vtM) )
end


function multi_execute!(size_error, outputs,inputs1,inputs2, f, checkdims_f, ref_size )
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    if (size(outputs[i]) != ref_size) || !checkdims_f(outputs[i], inputs1[i], inputs2[i])
        size_error[i]=true
    else
        @inbounds @cuda threads=ref_size dynamic=true f(outputs[i], inputs1[i], inputs2[i])
    end

    return nothing
end

function batched_function!(outputs, f,checkdims_f,  inputs1,inputs2)
    if ( length(outputs) != length(inputs1) ) || ( length(outputs) != length(inputs2) )
        throw( DimensionMismatch( "Dimensions of output vector " * string(length(outputs)) * " do not match input vectors sizes " * string(length(inputs1)) * " and "*string(length(inputs2)) ) )
    end
    ref_size=size(outputs[1])
    size_error=CUDA.zeros(Bool,length(outputs))
    @cuda threads=length(outputs) multi_execute!(size_error, outputs,inputs1,inputs2, f, checkdims_f, ref_size)
    if (findfirst(size_error) != nothing)
        i=findfirst(size_error)
        throw( DimensionMismatch(" At least one matrix does not match expected sized, got sizes " * string(size(outputs[i])) *", " * string(size(inputs1[i])) *" and " * string(size(inputs2[i])) * ", expected size " * string(ref_size)) )
    end
    return outputs
end


n=3
x=2
M=CUDA.randn(n,x)
M2=[view(M,:,i) for i in 1:x]
N=CUDA.randn(n,x)
N2=[view(N,:,i) for i in 1:x]
X=CUDA.zeros(n,n,x)
X2=[view(X,:,:,i) for i in 1:x]

a = tuple(M2...)
b = tuple(N2...)
c = tuple(X2...)

batched_function!(c, Householderreflector!, checkdims_HH,a,b )


function mm_cuda!(c, a,b)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    val=0.0
    for k in 1:size(a,2)
        val+=a[i,k]*b[k,j]
    end

    @inbounds c[i, j] = val
    nothing 
end

T=Float32
n=6
shape=(n,n)
a = CUDA.rand(n,n)
b = CUDA.rand(n,n)
c = CUDA.zeros(n,n)

@cuda threads=size(c) mm_cuda!(c, a,b)

