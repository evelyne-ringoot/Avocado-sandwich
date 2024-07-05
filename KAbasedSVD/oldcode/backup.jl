


function QR_unsafe_kernel!(input, tau,i, N, tile, cache, tau_iter, corrvalue )
    @synchronize
    for j in 1:N
        tile[i, j] = input[i,j]
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
          input[i,j] = tile[i, j] 
    end
    @synchronize

 end

function QR_unsafe_kernel2!(input,input2, tau, i, N, tile, cache, tau_iter, corrvalue )
    for j in 1:N
        tile[N+i, j] = input2[N,j]
    end
    for j in i:N
        tile[i, j] = input[i,j]
    end
    @synchronize
    for iter in 1:N
        if (i>iter)
            cache[i]= tile[i,iter]^2
        end
        cache[i+N]= tile[i+N,iter]^2
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
        if (i>=iter)
            tmp_sum =  corrvalue[1] *tile[iter,i]
            for j = iter+1:2N
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
        for j in N:-1:iter+1
            tile[i+N, j] = tile[i+N, j] - tile[i+N,iter]*cache[j] *tau_iter[1] /corrvalue[1]^2
        end
        tile[i+N,iter] = tile[i+N,iter] /corrvalue[1]
        @synchronize
    end
    @synchronize
    for j in 1:N
         input2[i,j] = tile[N+i, j] 
    end
    for j in i:N
        input[i,j] = tile[i, j] 
    end
    @synchronize

 end

 @kernel function QR_unsafe!(  input_args...)
    i,_ = @index(Local,  NTuple)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(input_args[1]) (N+1, N)
    cache = @localmem eltype(input_args[1]) (N+1)
    tau_iter= @localmem eltype(input_args[1]) (1)
    corrvalue = @localmem eltype(input_args[1]) (1)
    @inbounds QR_unsafe_kernel!(input_args..., i, N, tile, cache, tau_iter, corrvalue)
end
@kernel function QR_unsafe2!(  input_args...)
    i,_ = @index(Local,  NTuple)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(input_args[1]) (2N+1, N)
    cache = @localmem eltype(input_args[1]) (2N+1)
    tau_iter= @localmem eltype(input_args[1]) (1)
    corrvalue = @localmem eltype(input_args[1]) (1)
    @inbounds QR_unsafe_kernel!(input_args..., i, N, tile, cache, tau_iter, corrvalue)
end




function applyQ_unsafe_kernel!(A, Min, tau, i, N, tile, M)
    for j in 1:N
        tile[i, j] = A[i,j]
        M[i, j] = Min[i,j]
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
         A[i,j] = tile[i, j] 
    end
end

function applyQt_unsafe_kernel!(A, Min, tau, i, N, tile, M)
    for j in 1:N
        tile[i, j] = A[i,j]
        M[i, j] = Min[i,j]
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
         A[i,j] = tile[i, j] 
    end
end
function applyQ_unsafe_kernel2!(A, B, Min, tau, i, N, tile, M)
    for j in 1:N
        tile[i, j] = A[i,j]
        tile[i+N, j] = B[i,j]
        M[i, j] = Min[i,j]
    end
    @synchronize
    for k in N:-1:1
        tmp_sum=tile[k,i]
        for j in 1:N
            tmp_sum+=M[j,k]*tile[j+N,i]
        end
        tmp_sum=tmp_sum*tau[k]
        tile[k,i]=tile[k,i]-tmp_sum
        for j in N+1:2N
            tile[j,i]=tile[j,i]-tmp_sum*M[j,k]
        end
    end
    @synchronize
    for j in 1:N
         A[i,j] = tile[i, j] 
        B[i,j]=tile[i+N, j]
    end
end

function applyQt_unsafe_kernel2!(A, B, Min, tau, i, N, tile, M)
    for j in 1:N
        tile[i, j] = A[i,j]
        tile[i+N, j] = B[i,j]
        M[i, j] = Min[i,j]
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
         A[i,j] = tile[i, j] 
        B[i,j]=tile[i+N, j]
    end
end


@kernel function unsafe_applyQ!(input_args...)
    i,_ = @index(Local,  NTuple)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(A) (N+1, N)
    M= @localmem eltype(A) (N+1,N)
    @inbounds applyQ_unsafe_kernel!(input_args..., i, N, tile, M)
end
@kernel function unsafe_applyQ!(input_args...)
    i,_ = @index(Local,  NTuple)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(A) (N+1, N)
    M= @localmem eltype(A) (N+1,N)
    @inbounds applyQt_unsafe_kernel!(input_args..., i, N, tile, M)
end
@kernel function unsafe_applyQ!(input_args...)
    i,_ = @index(Local,  NTuple)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(A) (2N+1, N)
    M= @localmem eltype(A) (N+1,N)
    @inbounds applyQ_unsafe_kernel!(input_args..., i, N, tile, M)
end
@kernel function unsafe_applyQ!(input_args...)
    i,_ = @index(Local,  NTuple)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(A) (2N+1, N)
    M= @localmem eltype(A) (N+1,N)
    @inbounds applyQt_unsafe_kernel2!(input_args..., i, N, tile, M)
end

n=8
backend=KernelAbstractions.get_backend(randn(2))
T=Float32

    a=rand!(allocate(backend, T,n, n))
    b=triu(rand!(allocate(backend, T,n, n)))
    c=rand!(allocate(backend, T,n, n))
    t2=qr!(c).τ
    t= KernelAbstractions.zeros(backend, T, n)
    myrange=(n,1)
    d=KernelAbstractions.zeros(backend, T, 2n,n)
    view(d,1:n,1:n) .= triu(rand!(allocate(backend, T,n, n)))
    view(d,n+1:2n,1:n) .= rand!(allocate(backend, T,n, n))
    dq=view(qr(d).factors,n+1:2n,1:n)


    QR_unsafe!(backend,n)((a,t), ndrange=myrange)
    QR_unsafe_kernel2!(backend,n)(b, a, t ; ndrange=myrange)
    applyQt_unsafe_kernel!(backend,n)(a, c,t2; ndrange=myrange )
    applyQt_unsafe_kernel2!(backend,n)(b,a, dq, qr(d).τ ; ndrange=myrange)

    @kernel function QR_unsafe_kernel_!(input, tau, i, N, tile, cache, tau_iter, corrvalue)
        @synchronize
        for j in 1:N
            tile[i, j] = input[i,j]
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
              input[i,j] = tile[i, j] 
        end
        @synchronize
    
     end

    @kernel function QR_unsafe_kernel!(input, tau)
        i,_ = @index(Local,  NTuple)
        N = @uniform @groupsize()[1]
        tile = @localmem eltype(input) (N+1, N)
        cache = @localmem eltype(input) (N+1)
        tau_iter= @localmem eltype(input) (1)
        corrvalue = @localmem eltype(input) (1)
        @synchronize
        for j in 1:N
            tile[i, j] = input[i,j]
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
              input[i,j] = tile[i, j] 
        end
        @synchronize
     
    end
    
    @kernel function wrap_kernel_QR!( input_args...)
        i,_ = @index(Local,  NTuple)
        N = @uniform @groupsize()[1]
        tile = @localmem eltype(input_args[1]) (N+1, N)
        cache = @localmem eltype(input_args[1]) (N+1)
        tau_iter= @localmem eltype(input_args[1]) (1)
        corrvalue = @localmem eltype(input_args[1]) (1)
        @inbounds QR_unsafe_kernel_!(input_args..., i, N, tile, cache, tau_iter, corrvalue)
    end



    QRkernel1! = QR_unsafe_kernel!(backend,n)
    QRkernel1!(a,t, ndrange=myrange)

    QRkernel1! = wrap_kernel_QR!(backend,n)
    QRkernel1!(a,t, ndrange=myrange)