using KernelAbstractions.Extras: @unroll
using KernelAbstractions


@kernel cpu=false  inbounds=true unsafe_indices=false function QR_unsafe_kernel_2d!(input, tau) 
    i = @index(Local,Linear)

    tilecol = @private eltype(input) (TILESIZE)
    cache = @localmem eltype(input) (TILESIZE)
    tau_iter = @private eltype(input) (1)
    sharedvalue = @localmem eltype(input) (1)
    
    for j=1:TILESIZE
        tilecol[j] = input[j,i]
    end

    for iter in 1:TILESIZE-1
        tmp_sum= zero(eltype(input))
        if (i==iter)
            for j in iter+1:TILESIZE
                cache[j] = tilecol[j]
                tmp_sum+=tilecol[j]*tilecol[j]
            end
            cache[iter]=tilecol[iter]
            sharedvalue[1]=tmp_sum
        end
        @synchronize
        if (i>=iter)       
            if (i>iter)
                for j in iter+1:TILESIZE
                    tmp_sum+=cache[j]*tilecol[j]
                end
            end
            newvalue = cache[iter] + sign(cache[iter]) * sqrt(sharedvalue[1] + cache[iter]*cache[iter])
            taucurrent = 2 / (sharedvalue[1]/(newvalue*newvalue) + 1)
            tmp_sum2 = (tmp_sum/newvalue + tilecol[iter])*taucurrent
            
            if (i==iter)
                tau_iter[1]=taucurrent
            else
                for j in iter+1:TILESIZE
                    tilecol[j]= tilecol[j]*newvalue-cache[j]*tmp_sum2
                end
            end
            for j in iter+1:TILESIZE
                tilecol[j]/=newvalue
            end
            tilecol[iter]-=tmp_sum2

        end
        input[iter,i] = tilecol[iter]
        @synchronize
    end
    input[TILESIZE,i] = tilecol[TILESIZE]
    tau[i]=tau_iter[1]
    @synchronize
end


@kernel cpu=false  inbounds=true unsafe_indices=false function QR_unsafe_kernel2_2d!(input, input2, tau)
    i,k = @index(Local, NTuple)

    tilecol = @private eltype(input) (Int(TILESIZE/QRSPLIT))
    cache = @localmem eltype(input) (TILESIZE)
    cache2 = @localmem eltype(input) (TILESIZE, QRSPLIT)
    tau_iter = @private eltype(input) (1)
    sharedvalue = @localmem eltype(input) (2)

    for j in 1:Int(TILESIZE/QRSPLIT)
        tilecol[j] = input2[(j-1)*QRSPLIT+k, i]
    end

    for iter in 1:TILESIZE
        tmp_sum = zero(eltype(input))
        tileiter= input[iter, i]
        if (i==iter)
            for j in 1:Int(TILESIZE/QRSPLIT)
                cache[(j-1)*QRSPLIT+k] = tilecol[j]
            end
        end
        @synchronize

        if (i>=iter)
            for j in 1:Int(TILESIZE/QRSPLIT)
                tmp_sum+=tilecol[j]*cache[(j-1)*QRSPLIT+k]
            end
            if (i==iter && k==1)
                sharedvalue[2]=tileiter
            end
            cache2[i,k]=tmp_sum
        end
        
        @synchronize

        if (i>=iter)
            tmpsumiter = zero(eltype(input))
            tmp_sum = zero(eltype(input))
            for j = 1:QRSPLIT
                tmpsumiter+= cache2[iter,j]
                tmp_sum += cache2[i,j]
            end

            newvalue = sharedvalue[2] + sign(sharedvalue[2]) *sqrt(tmpsumiter+ sharedvalue[2]*sharedvalue[2])
            taucurrent = 2 / (tmpsumiter / (newvalue*newvalue)+1)
            tmp_sum2 = (tmp_sum/newvalue + tileiter)*taucurrent
            if (i==iter && k==1)
                tau_iter[1] = taucurrent
            elseif (i>iter)
                for j in 1:Int(TILESIZE/QRSPLIT)
                    temp = tilecol[j]
                    tilecol[j]*=newvalue
                    tilecol[j]-=cache[(j-1)*QRSPLIT+k]*tmp_sum2
                end
            end
            for j in 1:Int(TILESIZE/QRSPLIT)
                tilecol[j]/=newvalue
            end
            if (k==1)
                input[iter, i]-=tmp_sum2
            end
        end
        @synchronize
    end
    
    for j in 1:Int(TILESIZE/QRSPLIT)
        input2[(j-1)*QRSPLIT+k, i]=tilecol[j] 
    end
    if (k==1)
        tau[i]=tau_iter[1]
    end
    

end

@kernel cpu=false  inbounds=true unsafe_indices=false  function applyQorQt_unsafe_kernel_2d!(A, @Const(Min), @Const(tau))
    g = @index(Group, Linear)
    i = @index(Local, Linear)
    tilecol = @private eltype(A) (TILESIZE)
    M = @localmem eltype(A) (TILESIZE)
    
    for l in 1:TILESIZE
         tilecol[l] = A[l, (g-1)*TILESIZE+i]
    end

    for k in 1:TILESIZE-1
         M[i] = Min[i, k]
        @synchronize
        tmp_sum = zero(eltype(A))
        for l in k+1:TILESIZE
             tmp_sum += M[l] * tilecol[l]
        end
         tmp_sum+=tilecol[k]
         tmp_sum*=tau[k]

        for l in k+1:TILESIZE
             tilecol[l] -= tmp_sum * M[l]
        end
         tilecol[k]-=tmp_sum
        @synchronize
    end

    for l in 1:TILESIZE
         A[l, (g-1)*TILESIZE+i]=tilecol[l]
    end
end

@kernel cpu=false  inbounds=true unsafe_indices=false  function applyQorQt_unsafe_kernel2_2d!(A, B, @Const(Min), @Const(tau))
    g = @index(Group, Linear)
    i = @index(Local, Linear)
    tilecol = @private eltype(A) (TILESIZE)
    
    for l in 1:TILESIZE
         tilecol[l] = B[l, i+(g-1)*TILESIZE] 
    end

    for k in 1:TILESIZE
        tmp_sum= zero(eltype(A))       
        for j in 1:TILESIZE
             tmp_sum += Min[j, k] * tilecol[j]
        end
         tmp_sum+= A[k, i+(g-1)*TILESIZE]
         tmp_sum *= tau[k]
         A[k, i+(g-1)*TILESIZE] -= tmp_sum

        for l in 1:TILESIZE
             tilecol[l] -= tmp_sum * Min[l, k]
        end
    end
    for l in 1:TILESIZE
         B[l, i+(g-1)*TILESIZE] = tilecol[l]
    end
end




