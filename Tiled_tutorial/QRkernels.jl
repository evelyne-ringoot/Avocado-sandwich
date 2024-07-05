
@kernel function QR_unsafe_kernel!(input, tau)
    i,_ = @index(Local,  NTuple)
    N = @uniform @groupsize()[1]
    
    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(input) (N+1, N)
    cache = @localmem eltype(input) (N+1)
    tau_iter= @localmem eltype(input) (1)
    corrvalue = @localmem eltype(input) (1)

    @synchronize
    for j in 1:N
        @inbounds tile[i, j] = input[i,j]
    end

    for iter in 1:N-1
        if (i>iter)
            cache[i]= tile[i,iter]^2
        end
        @synchronize
        if (i==iter)
            tmp_sum = zero(eltype(input))
            for j in iter+1:N
                tmp_sum+=cache[j]
            end
            tmp_sum2=sqrt(tmp_sum+tile[i,i]^2)
            if (tile[i,i]<0)
                newvalue=tile[i,i]-tmp_sum2
            else
                newvalue=tile[i,i]+tmp_sum2
            end
            tmp_sum2=sqrt(tmp_sum+newvalue^2)
            tau_iter[1]=2*(newvalue/tmp_sum2)^2
            corrvalue[1]=newvalue
            tau[iter]=tau_iter[1]
        end
        @synchronize
        if (i>=iter)
            tmp_sum =  corrvalue[1] *tile[iter,i]
            for j = iter+1:N
                tmp_sum += tile[j,iter]*tile[j,i]
            end
            cache[i]=tmp_sum
        end
        @synchronize
        if (i>iter)
            for j in N:-1:iter+1
                tile[i, j] = tile[i, j] - tile[i,iter]*cache[j] *tau_iter[1] /corrvalue[1]^2
            end
            tile[i,iter] = tile[i,iter] /corrvalue[1]
        elseif (i==iter)
            for j in N:-1:iter
                tile[iter,j] = tile[iter,j] - cache[j] *tau_iter[1] /corrvalue[1]
            end
        end
        @synchronize
    end
    @synchronize
    for j in 1:N
        @inbounds  input[i,j] = tile[i, j] 
    end
    @synchronize

 end

 @kernel function QR_unsafe_kernel2!(input,input2, tau)
    i,_ = @index(Local,  NTuple)
    N = @uniform @groupsize()[1]
    
    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(input) (2N+1, N)
    cache = @localmem eltype(input) (2N+1)
    tau_iter= @localmem eltype(input) (1)
    corrvalue = @localmem eltype(input) (1)

    
    for j in 1:N
        @inbounds tile[N+i, j] = input2[i,j]
        if (j>=i)
            @inbounds tile[i, j] = input[i,j]
        else
            @inbounds tile[i, j] = 0
        end
    end

    @synchronize
    for iter in 1:N
        if (i > iter)
            cache[i] = tile[i, iter]^2
        end
        cache[i+N] = tile[i+N, iter]^2
        @synchronize
        if (i==iter)
            tmp_sum = zero(eltype(input))
            for j in iter+1:2N
                tmp_sum+=cache[j]
            end
            tmp_sum2=sqrt(tmp_sum+tile[i,i]^2)
            if (tile[i,i]<0)
                newvalue=tile[i,i]-tmp_sum2
            else
                newvalue=tile[i,i]+tmp_sum2
            end
            tmp_sum2=sqrt(tmp_sum+newvalue^2)
            tau_iter[1]=2*(newvalue/tmp_sum2)^2
            corrvalue[1]=newvalue
            tau[iter]=tau_iter[1]
        end
        @synchronize
        if (i >= iter)
            tmp_sum = corrvalue[1] * tile[iter, i]
            for j = iter+1:2N
                tmp_sum += tile[j, iter] * tile[j, i]
            end
            cache[i] = tmp_sum
        end
        @synchronize
        if (i > iter)
            for j in N:-1:iter+1
                tile[i, j] = tile[i, j] - tile[i, iter] * cache[j] * tau_iter[1] / corrvalue[1]^2
            end
            tile[i, iter] = tile[i, iter] / corrvalue[1]
        elseif (i == iter)
            for j in N:-1:iter
                tile[iter, j] = tile[iter, j] - cache[j] * tau_iter[1] / corrvalue[1]
            end
        end
        for j in N:-1:iter+1
            tile[i+N, j] = tile[i+N, j] - tile[i+N, iter] * cache[j] * tau_iter[1] / corrvalue[1]^2
        end
        tile[i+N, iter] = tile[i+N, iter] / corrvalue[1]
        @synchronize
    end
    @synchronize
    for j in 1:N
        @inbounds  input2[i,j] = tile[N+i, j] 
    end
    for j in i:N
        @inbounds input[i,j] = tile[i, j] 
    end
    @synchronize
    

 end

 @kernel function applyQ_unsafe_kernel!(A, @Const(Min), @Const(tau))
    i,_ = @index(Local,  NTuple)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(A) (N+1, N)
    M= @localmem eltype(A) (N+1,N)
    
    for j in 1:N
        @inbounds tile[i, j] = A[i,j]
        @inbounds M[i, j] = Min[i,j]
    end

    @synchronize
    for k in N-1:-1:1
        tmp_sum=tile[k,i]
        for j in k+1:N
            tmp_sum+=M[j,k]*tile[j,i]
        end
        tmp_sum=tmp_sum*tau[k]
        for j in k+1:N
            tile[j,i]=tile[j,i]-tmp_sum*M[j,k]
        end
        tile[k,i]=tile[k,i]-tmp_sum
    end
    @synchronize
    for j in 1:N
        @inbounds  A[i,j] = tile[i, j] 
    end
end
@kernel function applyQt_unsafe_kernel!(A, @Const(Min), @Const(tau))
    i,_ = @index(Local,  NTuple)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(A) (N+1, N)
    M= @localmem eltype(A) (N+1,N)

    for j in 1:N
        @inbounds tile[i, j] = A[i,j]
        @inbounds M[i, j] = Min[i,j]
    end
    @synchronize
    for k in 1:N-1
        tmp_sum=tile[k,i]
        for j in k+1:N
            tmp_sum+=M[j,k]*tile[j,i]
        end
        tmp_sum=tmp_sum*tau[k]
        for j in k+1:N
            tile[j,i]=tile[j,i]-tmp_sum*M[j,k]
        end
        tile[k,i]=tile[k,i]-tmp_sum
    end
    @synchronize
    for j in 1:N
        @inbounds  A[i,j] = tile[i, j] 
    end
end
@kernel function applyQ_unsafe_kernel2!(A, B, @Const(Min), @Const(tau))
    i,_ = @index(Local,  NTuple)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(A) (2N+1, N)
    M= @localmem eltype(A) (N+1,N)
    
    for j in 1:N
        @inbounds tile[i, j] = A[i,j]
        @inbounds tile[i+N, j] = B[i,j]
        @inbounds M[i, j] = Min[i,j]
    end

    @synchronize
    for k in N:-1:1
        tmp_sum=tile[k,i]
        for j in 1:N
            tmp_sum+=M[j,k]*tile[j+N,i]
        end
        tmp_sum=tmp_sum*tau[k]
        tile[k,i]=tile[k,i]-tmp_sum
        for j in 1:N
            tile[j+N,i]=tile[j+N,i]-tmp_sum*M[j,k]
        end
    end
    @synchronize
    for j in 1:N
        @inbounds  A[i,j] = tile[i, j] 
        @inbounds B[i,j]=tile[i+N, j]
    end
end

@kernel function applyQt_unsafe_kernel2!(A, B, @Const(Min), @Const(tau))
    i,_ = @index(Local,  NTuple)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(A) (2N+1, N)
    M= @localmem eltype(A) (N+1,N)
    
    for j in 1:N
        @inbounds tile[i, j] = A[i,j]
        @inbounds tile[i+N, j] = B[i,j]
        @inbounds M[i, j] = Min[i,j]
    end

    @synchronize
    for k in 1:N
        tmp_sum=tile[k,i]
        for j in 1:N
            tmp_sum+=M[j,k]*tile[j+N,i]
        end
        tmp_sum=tmp_sum*tau[k]
        tile[k,i]=tile[k,i]-tmp_sum
        for j in 1:N
            tile[j+N,i]=tile[j+N,i]-tmp_sum*M[j,k]
        end
    end
    @synchronize
    for j in 1:N
        @inbounds  A[i,j] = tile[i, j] 
        @inbounds B[i,j]=tile[i+N, j]
    end
end


@kernel function mytriu!(input)
    i,j = @index(Local,  NTuple)
    if (i>j)
        input[i,j]=0
    end
end

@kernel function applyQt_unsafe_kernel_block!(A, @Const(Min), @Const(tau))
    g,_ = @index(Group,  NTuple)
    i,_ = @index(Local,  NTuple)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(A) (N+1, N)
    M= @localmem eltype(A) (N+1,N)

    for j in 1:N
        @inbounds tile[i, j] = A[i,j+(g-1)*N]
        @inbounds M[i, j] = Min[i,j]
    end
    @synchronize
    for k in 1:N-1
        tmp_sum=tile[k,i]
        for j in k+1:N
            tmp_sum+=M[j,k]*tile[j,i]
        end
        tmp_sum=tmp_sum*tau[k]
        for j in k+1:N
            tile[j,i]=tile[j,i]-tmp_sum*M[j,k]
        end
        tile[k,i]=tile[k,i]-tmp_sum
    end
    @synchronize
    for j in 1:N
        @inbounds  A[i,j+(g-1)*N] = tile[i, j] 
    end
end

@kernel function applyQt_unsafe_kernel2_block!(A, B, @Const(Min), @Const(tau))
    g,_ = @index(Group,  NTuple)
    i,_ = @index(Local,  NTuple)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(A) (2N+1, N)
    M= @localmem eltype(A) (N+1,N)
    
    for j in 1:N
        @inbounds tile[i, j] = A[i,j+(g-1)*N]
        @inbounds tile[i+N, j] = B[i,j+(g-1)*N]
        @inbounds M[i, j] = Min[i,j]
    end

    @synchronize
    for k in 1:N
        tmp_sum=tile[k,i]
        for j in 1:N
            tmp_sum+=M[j,k]*tile[j+N,i]
        end
        tmp_sum=tmp_sum*tau[k]
        tile[k,i]=tile[k,i]-tmp_sum
        for j in 1:N
            tile[j+N,i]=tile[j+N,i]-tmp_sum*M[j,k]
        end
    end
    @synchronize
    for j in 1:N
        @inbounds  A[i,j+(g-1)*N] = tile[i, j] 
        @inbounds B[i,j+(g-1)*N]=tile[i+N, j]
    end
end

