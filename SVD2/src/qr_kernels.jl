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
    k,i = @index(Local, NTuple)

    tilecol = @private eltype(input) (FACTORQR)
    cache = @localmem eltype(input) (TILESIZE)
    cache2 = @localmem eltype(input) (QRSPLIT,TILESIZE)
    tau_iter = zero(eltype(input))
    sharedvalue = @localmem eltype(input) (2)
    
    for j in 1:FACTORQR
        tilecol[j] = input2[j+(k-1)*FACTORQR, i]
    end

    for iter in 1:TILESIZE
        
        tileiter= input[iter, i]
        if (i==iter)
            for j in 1:FACTORQR
                cache[j+(k-1)*FACTORQR] = tilecol[j]
            end
        end
        @synchronize
        
        if (i>=iter)
            tmp_sum = zero(eltype(input))
            for j in 1:FACTORQR
                tmp_sum+=tilecol[j]*cache[j+(k-1)*FACTORQR]
            end
            cache2[k,i]=tmp_sum
            if (i==iter && k==1)
                sharedvalue[2]=tileiter
            end
            
        end
        
        @synchronize

        if (i>=iter)
            tmpsumiter = zero(eltype(input))
            tmp_sum = zero(eltype(input))
            for j = 1:QRSPLIT
                tmpsumiter+= cache2[j,iter]
                tmp_sum += cache2[j,i]
            end
            
            newvalue = sharedvalue[2] + sign(sharedvalue[2]) *sqrt(tmpsumiter+ sharedvalue[2]*sharedvalue[2])
            taucurrent = 2 / (tmpsumiter / (newvalue*newvalue)+1)
            tmp_sum2 = (tmp_sum/newvalue + tileiter)*taucurrent
            tau_iter = i==iter ? taucurrent : tau_iter

            if (i>iter)
                for j in 1:FACTORQR
                    tilecol[j]*=newvalue
                    tilecol[j]-=cache[j+(k-1)*FACTORQR]*tmp_sum2
                end
            end
            
            for j in 1:FACTORQR
                tilecol[j]/=newvalue
            end
            
            if (k==1)
                input[iter, i]-=tmp_sum2
            end
            
        end
        @synchronize
    end
    


    for j in 1:FACTORQR
        input2[j+(k-1)*FACTORQR, i]=tilecol[j] 
    end
    if (k==1)
        tau[i]=tau_iter
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
    Mcurr= @localmem eltype(A) (TILESIZE)

        for l in 1:TILESIZE
            tilecol[l] = B[l, i+(g-1)*TILESIZEMUL] 
        end

        for k in 1:TILESIZE
            tmp_sum= zero(eltype(A))
            for j in 0:FACTORMUL-1
                Mcurr[j*TILESIZEMUL+i]=Min[j*TILESIZEMUL+i,k]
            end 
            @synchronize      
            for l in 1:TILESIZE
                tmp_sum += Mcurr[l] * tilecol[l]
            end
            tmp_sum+= A[k, i+(g-1)*TILESIZEMUL]
            tmp_sum *= tau[k]
            A[k, i+(g-1)*TILESIZEMUL] -= tmp_sum

            for l in 1:TILESIZE
                tilecol[l] -= tmp_sum * Mcurr[l]
            end
            @synchronize  
        end
        for l in 1:TILESIZE
            B[l, i+(g-1)*TILESIZEMUL] = tilecol[l]
        end
    
end

#=
const MULSPLIT = 1
const SIZESPLIT = Int(TILESIZE/MULSPLIT)
@kernel cpu=false  inbounds=true unsafe_indices=false  function applyQorQt_unsafe_kernel2_2d!(A, B, @Const(Min), @Const(tau))
    g = @index(Group, Linear)
    i,j = @index(Local, NTuple)
    tilecol = @private eltype(A) (SIZESPLIT)
    Mcurr= @localmem eltype(A) (TILESIZE)
    cache = @localmem eltype(A) (TILESIZEMUL,MULSPLIT)

        for l in 1:SIZESPLIT
            tilecol[l] = B[l+(j-1)*SIZESPLIT, i+(g-1)*TILESIZEMUL] 
        end

        for k in 1:TILESIZE
            if (j==1)
                for l in 0:FACTORMUL-1
                    Mcurr[l*TILESIZEMUL+i]=Min[l*TILESIZEMUL+i,k]
                end 
            end
            @synchronize     
            tmp_sum= zero(eltype(A)) 
            for l in 1:SIZESPLIT
                tmp_sum += Mcurr[l+(j-1)*SIZESPLIT] * tilecol[l]
            end
            cache[i,j]=tmp_sum
            @synchronize
            tmp_sum= zero(eltype(A))
            for l in 1:MULSPLIT
                tmp_sum+=cache[i,l]
            end
            tmp_sum+= A[k, i+(g-1)*TILESIZEMUL]
            tmp_sum *= tau[k]
            for l in 1:SIZESPLIT
                tilecol[l] -= tmp_sum * Mcurr[l+(j-1)*SIZESPLIT]
            end
            if (j==1)
                A[k, i+(g-1)*TILESIZEMUL] -= tmp_sum
            end

            @synchronize  
        end
        for l in 1:SIZESPLIT
            B[l+(j-1)*SIZESPLIT, i+(g-1)*TILESIZEMUL] = tilecol[l]
        end
    
end
=#


