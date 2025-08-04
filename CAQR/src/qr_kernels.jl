using KernelAbstractions.Extras: @unroll

const NUMILPQR = 2
const TILESIZE =64
const QRSPLIT =8 


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
            @inbounds taucurrent = 2 * (tmpsumiter / (newvalue*newvalue)+1)
            @inbounds tmp_sum2 = (tmp_sum/newvalue + tileiter)*taucurrent
            if (i==iter && k==1)
                @inbounds tau_iter[1] = taucurrent
            else
                @unroll for j in 1:Int(TILESIZE/QRSPLIT)
                    @inbounds tilecol[j]*=newvalue
                    @inbounds tilecol[j]-=cache[(j-1)*QRSPLIT+k]*tmp_sum2
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
        @inbounds input2[(j-1)*QRSPLIT+k, i]=tilecol[j] 
    end
    if (k==1)
        @inbounds tau[i]=tau_iter[1]
    end
    

end

@kernel function applyQorQt_unsafe_kernel_2d!(A, @Const(Min), @Const(tau))
    g = @index(Group, Linear)
    i = @index(Local, Linear)
    tilecol = @private eltype(A) (TILESIZE)
    M = @localmem eltype(A) (TILESIZE)
    
    @unroll for l in 1:TILESIZE
        @inbounds tilecol[l] = A[l, (g-1)*TILESIZE+i]
    end

    for k in 1:TILESIZE-1
        @inbounds M[i] = Min[i, k]
        @synchronize
        tmp_sum = zero(eltype(A))
        @unroll for l in k+1:TILESIZE
            @inbounds tmp_sum += M[l] * tilecol[l]
        end
        @inbounds tmp_sum+=tilecol[k]
        @inbounds tmp_sum*=tau[k]

        @unroll for l in k+1:TILESIZE
            @inbounds tilecol[l] -= tmp_sum * M[l]
        end
        @inbounds tilecol[k]-=tmp_sum
        @synchronize
    end

    @unroll for l in 1:TILESIZE
        @inbounds A[l, (g-1)*TILESIZE+i]=tilecol[l]
    end
end

@kernel function applyQorQt_unsafe_kernel2_2d!(A, B, @Const(Min), @Const(tau))
    g = @index(Group, Linear)
    i = @index(Local, Linear)
    tilecol = @private eltype(A) (TILESIZE)
    M = @localmem eltype(A) (TILESIZE)
    
    @unroll for l in 1:TILESIZE
        @inbounds tilecol[l] = B[l, i+(g-1)*TILESIZE] 
    end

    for k in 1:TILESIZE
        @inbounds M[i] = Min[i, k]
        @synchronize
        tmp_sum= zero(eltype(A))       
        @unroll for j in 1:TILESIZE
            @inbounds tmp_sum += M[j] * tilecol[j]
        end
        @inbounds tmp_sum+= A[k, i+(g-1)*TILESIZE]
        @inbounds tmp_sum *= tau[k]
        @inbounds A[k, i+(g-1)*TILESIZE] -= tmp_sum

        @unroll for l in 1:TILESIZE
            @inbounds tilecol[l] -= tmp_sum * M[l]
        end
        @synchronize
    end
    @unroll for l in 1:TILESIZE
        @inbounds B[l, i+(g-1)*TILESIZE] = tilecol[l]
    end
end



get_tileview(A, row , col, TILE_SIZEx, TILE_SIZEy ) = view(A, (row-1)*TILE_SIZEx.+(1:TILE_SIZEx),(col-1)*TILE_SIZEy.+(1:TILE_SIZEy))
get_rowview(A, row, startcol, TILE_SIZEx, TILE_SIZEy) =  view(A, (row-1)*TILE_SIZEx .+(1:TILE_SIZEx),((startcol-1)*TILE_SIZEy +1):size(A,2))
get_kernel_dims(::KernelAbstractions.Kernel{B,S}) where {B,S} = S.parameters[1]

QR1!(A, Tau, k) = QR_unsafe_kernel_2d!(backend, (TILESIZE))( get_tileview(A, k,k, TILESIZE, TILESIZE), 
                                    get_tileview(Tau, k,k, 1, TILESIZE), ndrange=(TILESIZE)) 
QR2!(A, Tau, row, k) =QR_unsafe_kernel2_2d!(backend, (TILESIZE, QRSPLIT))(get_tileview(A, k,k, TILESIZE, TILESIZE), 
                                    get_tileview(A, row,k, TILESIZE, TILESIZE), 
                                    get_tileview(Tau, row,k, 1, TILESIZE), ndrange=(TILESIZE,QRSPLIT))

Qtapply1_par!(A, Tau, k) = applyQorQt_unsafe_kernel_2d!(backend, (TILESIZE))(get_rowview(A, k, k+1, TILESIZE, TILESIZE), 
                                    get_tileview(A, k,k, TILESIZE, TILESIZE), 
                                    get_tileview(Tau, k,k, 1, TILESIZE), ndrange=( size(A,2)-k*TILESIZE) )
Qtapply2_par!(A, Tau, row,k) = applyQorQt_unsafe_kernel2_2d!(backend, (TILESIZE))(get_rowview(A, k, k+1, TILESIZE, TILESIZE), 
                                    get_rowview(A, row, k+1, TILESIZE, TILESIZE), 
                                    get_tileview(A, row,k, TILESIZE, TILESIZE), 
                                    get_tileview(Tau, row,k, 1, TILESIZE), ndrange=( size(A,2)-k*TILESIZE))

Qtapply1_par_full!(B, A, Tau, k) = applyQorQt_unsafe_kernel_2d!(backend, (TILESIZE))(view(B, (1:TILESIZE).+(k-1)*TILESIZE,: ), 
                                    get_tileview(A, k,k, TILESIZE, TILESIZE), 
                                    get_tileview(Tau, k,k, 1, TILESIZE), ndrange=( size(B,2)) )
Qtapply2_par_full!(B, A, Tau, row,k) = applyQorQt_unsafe_kernel2_2d!(backend, (TILESIZE))(view(B, (1:TILESIZE).+(k-1)*TILESIZE,: ), 
                                    view(B, (1:TILESIZE).+(row-1)*TILESIZE,: ), 
                                    get_tileview(A, row,k, TILESIZE, TILESIZE), 
                                    get_tileview(Tau, row,k, 1, TILESIZE), ndrange=( size(B,2)))

#Threads.@spawn begin, CUDA.@sync begin
 #=   
function mygeqrf!(A, Tau, nbtiles)
    QR1!(A,Tau, 1)
    for k in 1:(nbtiles-1)
        @sync begin
            @async QR2!(A,Tau, k+1,k)
            @async Qtapply1_par!(A, Tau, k)
        end
        for row in k+1:nbtiles
            @sync begin
                if (row<nbtiles)
                    @async QR2!(A,Tau, row+1,k)
                elseif (k<nbtiles-1)
                    @async QR1!(A,Tau, k+1)
                end
                @async Qtapply2_par!(A,Tau, row,k)
            end
        end
    end
    if(nbtiles>1)
        QR1!(A,Tau, nbtiles)
    end
    return A
end
=#


function myormqr!(B, A, Tau, nbtiles)
    for k in 1:(nbtiles)
        Qtapply1_par_full!(B,A, Tau, k)
        for row in k+1:nbtiles
            Qtapply2_par_full!(B,A,Tau, row,k)
        end
    end
    return B
end


function mygeqrf!(A, Tau, nbtiles)
    for k in 1:(nbtiles-1)
        QR1!(A,Tau, k)
        Qtapply1_par!(A, Tau, k)
        for row in k+1:nbtiles
            QR2!(A,Tau, row,k)
            Qtapply2_par!(A,Tau, row,k)
        end
    end
    QR1!(A,Tau, nbtiles)
    return A
end
