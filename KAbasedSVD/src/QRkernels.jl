using KernelAbstractions.Extras: @unroll

@kernel function QR_unsafe_kernel!(input, tau) 
    i, _ = @index(Local, NTuple)
    N = @uniform @groupsize()[1]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(input) (N + 1, N)
    cache = @localmem eltype(input) (N + 1)
    tau_iter = @localmem eltype(input) (1)
    corrvalue = @localmem eltype(input) (1)

    @unroll for j in 1:N
        @inbounds tile[i, j] = input[i, j]
    end

    for iter in 1:N-1
        if (i > iter)
            cache[i] = tile[i, iter]^2
        end
        @synchronize
        if (i == iter)
            tmp_sum = zero(eltype(input))
            for j in iter+1:N
                tmp_sum += cache[j]
            end
            tmp_sum2 = sqrt(tmp_sum + tile[i, i]^2)
            newvalue = tile[i, i] + sign(tile[i,i]) *tmp_sum2
            tmp_sum2 = sqrt(tmp_sum + newvalue^2)
            tau_iter[1] = 2 * (newvalue / tmp_sum2)^2
            corrvalue[1] = newvalue
            tau[iter] = tau_iter[1]
        end
        @synchronize
        taucorr=tau_iter[1] / corrvalue[1]
        if (i >= iter)
            tmp_sum = corrvalue[1] * tile[iter, i]
            for j = iter+1:N
                tmp_sum += tile[j, iter] * tile[j, i]
            end
            cache[i] = tmp_sum
        end
        tileiiter=taucorr
        if (i>iter)
            tileiiter= tile[i, iter] *taucorr / corrvalue[1]
        end
        @synchronize
        if (i>=iter)
            for j in iter+1:N
                tile[i, j] = tile[i, j] -  tileiiter* cache[j] 
            end
        end
        if (i > iter)
            tile[i, iter] = tileiiter / taucorr
        elseif (i == iter)
            tile[i, iter] = tile[i, iter] - cache[iter] * taucorr
        end
        @synchronize
    end
    @synchronize
    @unroll for j in 1:N
        @inbounds input[i, j] = tile[i, j]
    end
    @synchronize

end




@kernel function QR_unsafe_kernel2!(input, input2, tau)
    i, _ = @index(Local, NTuple)
    N = @uniform @groupsize()[1]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(input) (2N + 1, N)
    cache = @localmem eltype(input) (2N + 1)
    tau_iter = @localmem eltype(input) (1)
    corrvalue = @localmem eltype(input) (1)

    @unroll for j in 1:N
        @inbounds tile[N+i, j] = input2[i, j]
        @inbounds tile[i, j] = input[i, j]
    end

    @synchronize
    for iter in 1:N
        cache[i] = tile[i+N, iter]^2
        @synchronize
        if (i == iter)
            tmp_sum = zero(eltype(input))
            for j in 1:N
                tmp_sum += cache[j]
            end
            tmp_sum2 = sqrt(tmp_sum + tile[i, i]^2)
            newvalue = tile[i, i] + sign(tile[i,i]) *tmp_sum2
            tmp_sum2 = sqrt(tmp_sum + newvalue^2)
            tau_iter[1] = 2 * (newvalue / tmp_sum2)^2
            corrvalue[1] = newvalue
            tau[iter] = tau_iter[1]
        end
        @synchronize
        if (i >= iter)
            tmp_sum = corrvalue[1] * tile[iter, i]
            for j = N+1:2N
                tmp_sum += tile[j, iter] * tile[j, i]
            end
            cache[i] = tmp_sum
        end
        taucorr=tau_iter[1] / corrvalue[1]
        tileiNiter= tile[i+N, iter] / corrvalue[1] *taucorr
        @synchronize
        if (i >= iter)
            tile[iter, i] = tile[iter, i] - cache[i] * taucorr
        end
        for j in iter+1:N
            tile[i+N, j] = tile[i+N, j] - tileiNiter * cache[j] 
        end
        tile[i+N, iter] = tileiNiter / taucorr
        @synchronize
    end
    
    @unroll for j in 1:N
        @inbounds input2[i, j] = tile[N+i, j]
        @inbounds input[i, j] = tile[i, j]
    end
    @synchronize

end

@kernel function applyQorQt_unsafe_kernel!(A, @Const(Min), @Const(tau), applyt::Bool)
    g, _ = @index(Group, NTuple)
    i, _ = @index(Local, NTuple)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(A) (N + 1, N)
    M = @localmem eltype(A) (N + 1, N)

    @unroll for j in 1:N
        @inbounds tile[i, j] = A[i, j+(g-1)*N]
        @inbounds M[i, j] = Min[i, j]
    end

    applyrange = applyt ? (1:N-1) : (N-1:-1:1)

    @synchronize
    for k in applyrange
        tmp_sum = tile[k, i]
        for j in k+1:N
            tmp_sum += M[j, k] * tile[j, i]
        end
        tmp_sum = tmp_sum * tau[k]
        for j in k+1:N
            tile[j, i] = tile[j, i] - tmp_sum * M[j, k]
        end
        tile[k, i] = tile[k, i] - tmp_sum
    end
    @synchronize
    @unroll for j in 1:N
        @inbounds A[i, j+(g-1)*N]  = tile[i, j]
    end
end



@kernel function applyQorQt_unsafe_kernel2!(A, B, @Const(Min), @Const(tau), applyt::Bool)
    g, _ = @index(Group, NTuple)
    i, _ = @index(Local, NTuple)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(A) (2N + 1, N)
    M = @localmem eltype(A) (N + 1, N)

    @unroll for j in 1:N
        @inbounds tile[i, j] = A[i, j+(g-1)*N]
        @inbounds tile[i+N, j] = B[i, j+(g-1)*N]
        @inbounds M[i, j] = Min[i, j]
    end

    applyrange = applyt ? (1:N) : (N:-1:1)

    @synchronize
    for k in applyrange
        tmp_sum = tile[k, i]
        for j in 1:N
            tmp_sum += M[j, k] * tile[j+N, i]
        end
        tmp_sum = tmp_sum * tau[k]
        tile[k, i] = tile[k, i] - tmp_sum
        for j in 1:N
            tile[j+N, i] = tile[j+N, i] - tmp_sum * M[j, k]
        end
    end
    @synchronize
    @unroll for j in 1:N
        @inbounds A[i, j+(g-1)*N] = tile[i, j]
        @inbounds B[i, j+(g-1)*N] = tile[i+N, j]
    end
end



@kernel function mytriukernel!(input) #TODO : optimize performance
    I, J = @index(Global, NTuple)
    if (J<I)
        @inbounds input[I,J] = 0
    end
end

@kernel function QR_unsafe_kernel_2d!(input, tau) 
    i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(input) (N + 1, N)
    cache = @localmem eltype(input) (N + 1)
    tau_iter = @localmem eltype(input) (1)
    corrvalue = @localmem eltype(input) (1)

    @inbounds tile[i, j] = input[i, j]


    for iter in 1:N-1
        if (i > iter) && (j == iter)
            cache[i] = tile[i, iter]^2
        end
        @synchronize
        if (i == 1) && (j == 1)
            tmp_sum = zero(eltype(input))
            for l in iter+1:N
                tmp_sum += cache[l]
            end
            tmp_sum2 = sqrt(tmp_sum + tile[iter, iter]^2)
            newvalue = tile[iter, iter] + sign(tile[iter, iter]) * tmp_sum2
            tmp_sum2 = sqrt(tmp_sum + newvalue^2)
            tau_iter[1] = 2 * (newvalue / tmp_sum2)^2
            corrvalue[1] = newvalue
            tau[iter] = tau_iter[1]
        end
        if (j >= iter) && (i >= iter)
            tmp_sum = zero(eltype(input))
            for k = iter+1:N
                tmp_sum += tile[k, iter]  * tile[k, j]
            end
        end
        tileiterj=tile[iter, j]
        tileiiter = tile[i, iter] 
        @synchronize
        if (j >= iter) && (i >= iter)
            corrvalue1 = corrvalue[1]
            tmp_sum = (tmp_sum / corrvalue1+ tileiterj)*tau_iter[1] 
            tileiiter = tileiiter / corrvalue1

            if (j==iter) && (i > iter) 
                tile[i, j] = tileiiter 
            elseif (i>iter)
                tile[i, j] = tile[i, j] - tileiiter* tmp_sum  
            else
                tile[i, j] = tile[i, j] - tmp_sum 
            end
        end
        @synchronize
    end
    @inbounds input[i, j] = tile[i, j]
    @synchronize

end


@kernel function QR_unsafe_kernel2_2d!(input, input2, tau)
    i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(input) (2N + 1, N)
    cache = @localmem eltype(input) (2N + 1)
    tau_iter = @localmem eltype(input) (1)
    corrvalue = @localmem eltype(input) (1)

    @inbounds tile[N+i, j] = input2[i, j]
    @inbounds tile[i, j] = input[i, j]
    
    @synchronize
    for iter in 1:N
        if (j==iter)
            cache[i] = tile[i+N, iter]^2
        end
        @synchronize
        if (i == 1) && (j==1)
            tmp_sum = zero(eltype(input))
            for l in 1:N
                tmp_sum += cache[l]
            end
            tmp_sum2 = sqrt(tmp_sum + tile[iter, iter]^2)
            newvalue = tile[iter, iter] + sign(tile[iter,iter]) *tmp_sum2
            tmp_sum2 = sqrt(tmp_sum + newvalue^2)
            tau_iter[1] = 2 * (newvalue / tmp_sum2)^2
            corrvalue[1] = newvalue
            tau[iter] = tau_iter[1]
        end
        tileiNiter= tile[i+N, iter]
        tileiterj=tile[iter, j]
        if (j>=iter)
            tmp_sum = zero(eltype(input))
            for l = N+1:2N
                tmp_sum += tile[l, iter] * tile[l, j]
            end
        end
        @synchronize
        taucorr=tau_iter[1] / corrvalue[1]
        corrvalue1 = corrvalue[1]
        if (j >= iter) 
            tmp_sum += corrvalue1 * tileiterj
            if (i==iter)
                tile[i, j] = tile[i,j] - tmp_sum * taucorr
            end
            if (j>iter)
                tile[i+N, j] = tile[i+N, j] - tileiNiter * tmp_sum *taucorr / corrvalue1
            end
        end
        if (j==1)
            tile[i+N, iter] = tileiNiter / corrvalue1
        end
        @synchronize
    end
    
    @inbounds input2[i, j] = tile[N+i, j]
    @inbounds input[i, j] = tile[i, j]
    @synchronize

end

@kernel function applyQorQt_unsafe_kernel_2d!(A, @Const(Min), @Const(tau), applyt::Bool)
    g, _ = @index(Group, NTuple)
    i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1]
    K = @uniform @groupsize()[2]
    tile = @localmem eltype(A) (N + 1, N)
    M = @localmem eltype(A) (N + 1, N)
    cache = @localmem eltype(A) (N + 1,K)
    
    @unroll for l in j:K:N
        @inbounds tile[i, l] = A[i, l+(g-1)*N]
        @inbounds M[i, l] = Min[i, l]
    end

    applyrange = applyt ? (1:N-1) : (N-1:-1:1)
    
    @synchronize
    for k in applyrange
        tmp_sum = zero(eltype(A))
        for l in k+j:K:N
            tmp_sum += M[l, k] * tile[l, i]
        end
        cache[i,j]=tmp_sum
        @synchronize
        tmp_sum = tile[k, i] 
        for l in 1:K
            tmp_sum+=cache[i,l]
        end
        tmp_sum = tmp_sum * tau[k]
        for l in k+j:K:N
            tile[l, i] = tile[l, i] - tmp_sum * M[l, k]
        end
        if (j==1)
            tile[k, i] = tile[k, i] - tmp_sum
        end
        @synchronize
    end
    @unroll for l in j:K:N
        @inbounds A[i, l+(g-1)*N]  = tile[i, l]
    end
end

@kernel function applyQorQt_unsafe_kernel2_2d!(A, B, @Const(Min), @Const(tau), applyt::Bool)
    g, _ = @index(Group, NTuple)
    i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1]
    K = @uniform @groupsize()[2]
    tile = @localmem eltype(A) (2N + 1, N)
    M = @localmem eltype(A) (N + 1, N)
    cache = @localmem eltype(A) (N+1, K)
    
    @unroll for l in j:K:N
        @inbounds tile[i, l] = A[i, l+(g-1)*N]
        @inbounds tile[i+N, l] = B[i, l+(g-1)*N]
        @inbounds M[i, l] = Min[i, l]
    end

    applyrange = applyt ? (1:N) : (N:-1:1)

    @synchronize
    for k in applyrange
        tmp_sum= zero(eltype(A))       
        for j in j:K:N
            tmp_sum += M[j, k] * tile[j+N, i]
        end
        cache[i,j]=tmp_sum
        @synchronize
        tmp_sum = tile[k, i]
        for l in 1:K
            tmp_sum+=cache[i,l]
        end
        tmp_sum = tmp_sum * tau[k]
        if (j==1)
            tile[k, i] = tile[k, i] - tmp_sum
        end
        for l in j:K:N
            tile[l+N, i] = tile[l+N, i] - tmp_sum * M[l, k]
        end
        @synchronize
    end
    @unroll for l in j:K:N
        @inbounds A[i, l+(g-1)*N] = tile[i, l]
        @inbounds B[i, l+(g-1)*N] = tile[i+N, l]
    end
end


@kernel function coalesced_transpose_kernel!( output, @Const(input)) 
    I,J = @index(Global, NTuple)
    @inbounds output[I, J ] = input[J,I]
end

### below some old slower kernels, for later benchmarking/research on kernel optimization
#=
@kernel function QR_unsafe_kernel!(input, tau) #correct outcome
    i, _ = @index(Local, NTuple)
    N = @uniform @groupsize()[1]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(input) (N + 1, N)
    cache = @localmem eltype(input) (N + 1)
    tau_iter = @localmem eltype(input) (1)
    corrvalue = @localmem eltype(input) (1)

    @synchronize
    for j in 1:N
        @inbounds tile[i, j] = input[i, j]
    end

    for iter in 1:N-1
        if (i > iter)
            cache[i] = tile[i, iter]^2
        end
        @synchronize
        if (i == iter)
            tmp_sum = zero(eltype(input))
            for j in iter+1:N
                tmp_sum += cache[j]
            end
            tmp_sum2 = sqrt(tmp_sum + tile[i, i]^2)
            newvalue = tile[i, i] + sign(tile[i,i]) *tmp_sum2
            tmp_sum2 = sqrt(tmp_sum + newvalue^2)
            tau_iter[1] = 2 * (newvalue / tmp_sum2)^2
            corrvalue[1] = newvalue
            tau[iter] = tau_iter[1]
        end
        @synchronize
        if (i >= iter)
            tmp_sum = corrvalue[1] * tile[iter, i]
            for j = iter+1:N
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
        @synchronize
    end
    @synchronize
    for j in 1:N
        @inbounds input[i, j] = tile[i, j]
    end
    @synchronize

end

@kernel function QR_unsafe_kernel2!(input, input2, tau)
    i, _ = @index(Local, NTuple)
    N = @uniform @groupsize()[1]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(input) (2N + 1, N)
    cache = @localmem eltype(input) (2N + 1)
    tau_iter = @localmem eltype(input) (1)
    corrvalue = @localmem eltype(input) (1)

    

    for j in 1:N
        @inbounds tile[N+i, j] = input2[i, j]
        if (j >= i)
            @inbounds tile[i, j] = input[i, j]
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
        if (i == iter)
            tmp_sum = zero(eltype(input))
            for j in iter+1:2N
                tmp_sum += cache[j]
            end
            tmp_sum2 = sqrt(tmp_sum + tile[i, i]^2)
            newvalue = tile[i, i] + sign(tile[i,i]) *tmp_sum2
            tmp_sum2 = sqrt(tmp_sum + newvalue^2)
            tau_iter[1] = 2 * (newvalue / tmp_sum2)^2
            corrvalue[1] = newvalue
            tau[iter] = tau_iter[1]
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
        @inbounds input2[i, j] = tile[N+i, j]
        @inbounds input[i, j] = tile[i, j]
    end
    @synchronize


end

@kernel function QR_unsafe_kernel_2dalt!(input, tau) #SLOW!!
    i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(input) (N + 1, N)
    cache = @localmem eltype(input) (N + 1, N)
    tau_iter = @localmem eltype(input) (1)
    corrvalue = @localmem eltype(input) (1)

    @synchronize
    @inbounds tile[i, j] = input[i, j]


    for iter in 1:N-1
        if (i==iter) && (j>iter)
            cache[j,1]= tile[j, iter]^2
        elseif (j==iter) && (i==iter)
            cache[N+1,1] =0
        end
        @synchronize
        if (i==iter) && (j==iter || j==iter+1)
            tmp_sum = cache[j+1,1]
            for k in j+3:2:N
                tmp_sum += cache[k,1]
            end
            cache[j+1,1]=tmp_sum
        end
        @synchronize
        if (i == iter) && (j == iter)
            tmp_sum= cache[iter+2,1] + cache[iter+1,1]
            tmp_sum2 = sqrt(tmp_sum + tile[i, i]^2)
            newvalue = tile[i, i] + sign(tile[i, i]) * tmp_sum2
            tmp_sum2 = sqrt(tmp_sum + newvalue^2)
            tau_iter[1] = 2 * (newvalue / tmp_sum2)^2
            corrvalue[1] = newvalue
            tau[iter] = tau_iter[1]
        end
        @synchronize
        corrvalue1 = corrvalue[1]
        tauiter = tau_iter[1]
        tileiiter = tile[i, iter]
        if (j>=iter)
            if (i>iter)
                cache[i,j] = tile[i, iter] * tile[i, j]
            elseif (i==iter)
                cache[i,j]= corrvalue1 * tile[i, j]
            end
        end
        @synchronize
        if (j >= iter) && (i==iter || i==iter+1)
            tmp_sum = cache[i,j]
            for k = i+2:2:N
                tmp_sum += cache[k,j]
            end
            cache[i,j]=tmp_sum
        end

        @synchronize
        tmp_sum = cache[iter,j]+cache[iter+1,j]
        if (i > iter)
            if (j > iter)
                tile[i, j] = tile[i, j] - tileiiter * tmp_sum * tauiter / corrvalue1^2
            elseif (j==iter)
                tile[i, j] = tileiiter / corrvalue1
            end
        elseif (i==iter)
            tile[i, j] = tile[i, j] - tmp_sum * tauiter / corrvalue1
        end
        @synchronize
    end

    @inbounds input[i, j] = tile[i, j]
    @synchronize

end

=#