using KernelAbstractions.Extras: @unroll
using KernelAbstractions


@kernel function QR_unsafe_kernel_2d!(input, tau) 
    i = @index(Local,Linear)

    tilecol = @private eltype(input) (TILESIZE)
    cache = @localmem eltype(input) (TILESIZE)
    tau_iter = @private eltype(input) (1)
    sharedvalue = @localmem eltype(input) (1)
    
    @unroll for j=1:TILESIZE
        @inbounds tilecol[j] = input[j,i]
    end

    for iter in 1:TILESIZE-1
        tmp_sum= zero(eltype(input))
        if (i==iter)
            @unroll for j in iter+1:TILESIZE
                @inbounds cache[j] = tilecol[j]
                @inbounds tmp_sum+=tilecol[j]*tilecol[j]
            end
            @inbounds cache[iter]=tilecol[iter]
            @inbounds sharedvalue[1]=tmp_sum
        end
        @synchronize
        if (i>=iter)       
            if (i>iter)
                @unroll for j in iter+1:TILESIZE
                    @inbounds tmp_sum+=cache[j]*tilecol[j]
                end
            end
            @inbounds newvalue = cache[iter] + sign(cache[iter]) * sqrt(sharedvalue[1] + cache[iter]*cache[iter])
            @inbounds taucurrent = 2 / (sharedvalue[1]/(newvalue*newvalue) + 1)
            @inbounds tmp_sum2 = (tmp_sum/newvalue + tilecol[iter])*taucurrent
            
            if (i==iter)
                @inbounds tau_iter[1]=taucurrent
            else
                @unroll for j in iter+1:TILESIZE
                    @inbounds tilecol[j]= tilecol[j]*newvalue-cache[j]*tmp_sum2
                end
            end
            @unroll for j in iter+1:TILESIZE
                @inbounds tilecol[j]/=newvalue
            end
            @inbounds tilecol[iter]-=tmp_sum2

        end
        @inbounds input[iter,i] = tilecol[iter]
        @synchronize
    end
    @inbounds input[TILESIZE,i] = tilecol[TILESIZE]
    tau[i]=tau_iter[1]
    @synchronize
end


@kernel function QR_unsafe_kernel2_2d!(input, input2, tau)
    i,k = @index(Local, NTuple)

    tilecol = @private eltype(input) (Int(TILESIZE/QRSPLIT))
    cache = @localmem eltype(input) (TILESIZE)
    cache2 = @localmem eltype(input) (TILESIZE, QRSPLIT)
    tau_iter = @private eltype(input) (1)
    sharedvalue = @localmem eltype(input) (2)

    @unroll for j in 1:Int(TILESIZE/QRSPLIT)
        @inbounds tilecol[j] = input2[(j-1)*QRSPLIT+k, i]
    end

    for iter in 1:TILESIZE
        tmp_sum = zero(eltype(input))
        @inbounds tileiter= input[iter, i]
        if (i==iter)
            @unroll for j in 1:Int(TILESIZE/QRSPLIT)
                @inbounds cache[(j-1)*QRSPLIT+k] = tilecol[j]
            end
        end
        @synchronize

        if (i>=iter)
            @unroll for j in 1:Int(TILESIZE/QRSPLIT)
                @inbounds tmp_sum+=tilecol[j]*cache[(j-1)*QRSPLIT+k]
            end
            if (i==iter && k==1)
                @inbounds sharedvalue[2]=tileiter
            end
            @inbounds cache2[i,k]=tmp_sum
        end
        
        @synchronize

        if (i>=iter)
            tmpsumiter = zero(eltype(input))
            tmp_sum = zero(eltype(input))
            @unroll for j = 1:QRSPLIT
                @inbounds tmpsumiter+= cache2[iter,j]
                @inbounds tmp_sum += cache2[i,j]
            end

            @inbounds newvalue = sharedvalue[2] + sign(sharedvalue[2]) *sqrt(tmpsumiter+ sharedvalue[2]*sharedvalue[2])
            @inbounds taucurrent = 2 / (tmpsumiter / (newvalue*newvalue)+1)
            @inbounds tmp_sum2 = (tmp_sum/newvalue + tileiter)*taucurrent
            if (i==iter && k==1)
                @inbounds tau_iter[1] = taucurrent
            elseif (i>iter)
                @unroll for j in 1:Int(TILESIZE/QRSPLIT)
                    temp = tilecol[j]
                    @inbounds tilecol[j]*=newvalue
                    #@cushow tilecol[j]+100*k*sign(tilecol[j]) temp+(100*k+1000*iter+10000*i)*sign(temp)
                    @inbounds tilecol[j]-=cache[(j-1)*QRSPLIT+k]*tmp_sum2
                    #@cushow tilecol[j]+100*k*sign(tilecol[j]) temp+(100*k+1000*iter+10000*i)*sign(temp) 
                end
            end
            @unroll for j in 1:Int(TILESIZE/QRSPLIT)
                @inbounds tilecol[j]/=newvalue
            end
            if (k==1)
                @inbounds input[iter, i]-=tmp_sum2
            end
        end
        @synchronize
    end
    
    @unroll for j in 1:Int(TILESIZE/QRSPLIT)
        #@cushow tilecol[j]+100*k*sign(tilecol[j])
        @inbounds input2[(j-1)*QRSPLIT+k, i]=tilecol[j] 
    end
    if (k==1)
        @inbounds tau[i]=tau_iter[1]
    end
    

end

@kernel inbounds=true function applyQorQt_unsafe_kernel_2d!(A, @Const(Min), @Const(tau))
    g = @index(Group, Linear)
    i = @index(Local, Linear)
    tilecol = @private eltype(A) (TILESIZE)
    M = @localmem eltype(A) (TILESIZE)
    
    @unroll for l in 1:TILESIZE
         tilecol[l] = A[l, (g-1)*TILESIZE+i]
    end

    for k in 1:TILESIZE-1
         M[i] = Min[i, k]
        @synchronize
        tmp_sum = zero(eltype(A))
        @unroll for l in k+1:TILESIZE
             tmp_sum += M[l] * tilecol[l]
        end
         tmp_sum+=tilecol[k]
         tmp_sum*=tau[k]

        @unroll for l in k+1:TILESIZE
             tilecol[l] -= tmp_sum * M[l]
        end
         tilecol[k]-=tmp_sum
        @synchronize
    end

    @unroll for l in 1:TILESIZE
         A[l, (g-1)*TILESIZE+i]=tilecol[l]
    end
end

@kernel inbounds=true function applyQorQt_unsafe_kernel2_2d!(A, B, @Const(Min), @Const(tau))
    g = @index(Group, Linear)
    i = @index(Local, Linear)
    tilecol = @private eltype(A) (TILESIZE)
    
    @unroll for l in 1:TILESIZE
         tilecol[l] = B[l, i+(g-1)*TILESIZE] 
    end

    for k in 1:TILESIZE
        tmp_sum= zero(eltype(A))       
        @unroll for j in 1:TILESIZE
             tmp_sum += Min[j, k] * tilecol[j]
        end
         tmp_sum+= A[k, i+(g-1)*TILESIZE]
         tmp_sum *= tau[k]
         A[k, i+(g-1)*TILESIZE] -= tmp_sum

        @unroll for l in 1:TILESIZE
             tilecol[l] -= tmp_sum * Min[l, k]
        end
    end
    @unroll for l in 1:TILESIZE
         B[l, i+(g-1)*TILESIZE] = tilecol[l]
    end
end




