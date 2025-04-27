
using KernelAbstractions.Extras: @unroll
const BRDSPLITFACTOR = Int(TILESIZE/BRDSPLIT)

@kernel cpu=false inbounds=true unsafe_indices=false function brdkernel!(input, nbrows, secondsweep)
    k,i,l = @index(Local, NTuple)
    g = @index(Group, Linear)
    rowidx= -TILESIZE+1+Int(secondsweep)*2TILESIZE +(g-1)*(4TILESIZE-1)
    colidx= rowidx + TILESIZE
    cache = @localmem eltype(input) (TILESIZE+1, TILESIZE,2)
    cache2 = @localmem eltype(input) (BRDSPLIT+1,TILESIZE,2)

    idxcurr = l-1 
    idxiter=0
    fullblock= (g>1 || l==2 || secondsweep)
    
    if (l==1)
        if (fullblock)
            loadblockintocache(view(cache,:,:,1),input, k,i,rowidx,colidx, nbrows)
        else
            for j in 1:BRDSPLITFACTOR
                cache[(k-1)*BRDSPLITFACTOR+j,i,1] = zero(eltype(input))
            end
            if (k==1 && colidx+i <=nbrows)
                cache[1,i,1] =  input[1,colidx+i] 
            end
        end
    end
    @synchronize

    @unroll for idx in 0:1
        @unroll for dolq in 0:1
        idxiter= idxiter % 2 +1
        idxcurr = idxcurr % 2 +1
        
        if (l==2)
            loadblockintocache(view(cache,:,:,idxcurr),input, k,i,rowidx+TILESIZE*(1+idx),colidx+(idx+dolq)*TILESIZE, nbrows)
        end

        @synchronize
        mulvecvec_split_exclfirst((dolq==0) ? view(cache,1,:,idxiter) : view(cache,:,1,idxiter),
                                (dolq==0) ? view(cache,i,:,idxcurr) : view(cache,:,i,idxcurr),view(cache2,:,i,idxcurr), k,0)
 
        @synchronize
        
        tmpsumiter = zero(eltype(input))
        tmp_sum = zero(eltype(input))
        for j = 1:BRDSPLIT
            tmpsumiter+= cache2[j,1,idxiter]
            tmp_sum += cache2[j,i,idxcurr]
        end
        
        newvalue, factor = calc_factor(cache[1,1,idxiter], tmpsumiter, tmp_sum, dolq==0 ? cache[i,1,idxcurr] : cache[1,i,idxcurr] )
        if (fullblock && (i>1||l==2))
                updatevector((dolq==0) ? view(cache, i,:,idxcurr) : view(cache, :,i,idxcurr) , 
                            (dolq==0) ? view(cache,1,:,idxiter) : view(cache,:,1,idxiter) , factor,k)
        end
        
        @synchronize
        idx_x = dolq==0 ? 1 : i
        idx_y = dolq==0 ? i : 1
        (k==1 && i>1 &&l==1) && (cache[idx_x,idx_y,idxiter]= zero(eltype(input)))
        (k==1) &&  (cache[idx_y,idx_x,idxcurr]-=factor*newvalue)

        @synchronize

        if (l==1) 
            if (fullblock)
                sendblockbacktomem(input, view(cache,:,:,idxiter), k, i, rowidx+TILESIZE*(idx+dolq), colidx+idx*TILESIZE, nbrows)
            elseif (k==1 && colidx+i<=nbrows)
                input[1,colidx+i] = cache[1,i,1] 
            end
        end
        fullblock=true
        @synchronize

    end
    end

    if (l==2)
        sendblockbacktomem(input, view(cache,:,:,idxcurr), k, i, rowidx+TILESIZE*2, colidx+2*TILESIZE, nbrows)
    end

end


@kernel cpu=false inbounds=true unsafe_indices=false function brdkernel_lowmem!(input, nbrows, secondsweep)
    k,i = @index(Local, NTuple)
    g = @index(Group, Linear)
    rowidx= -TILESIZE+1+Int(secondsweep)*2TILESIZE +(g-1)*(4TILESIZE-1)
    colidx= rowidx + TILESIZE
    cache = @localmem eltype(input) (TILESIZE+1, TILESIZE)
    cache2 = @localmem eltype(input) (BRDSPLIT+1,TILESIZE)
    tilecol_cache = @localmem eltype(input) (TILESIZE)


    fullblock= (g>1|| secondsweep)
    
        if (fullblock)
            loadblockintocache(cache,input, k,i,rowidx,colidx, nbrows)
        else
            for j in 1:BRDSPLITFACTOR
                cache[(k-1)*BRDSPLITFACTOR+j,i] = zero(eltype(input))
            end
            if (k==1 && colidx+i <=nbrows)
                cache[1,i] =  input[1,colidx+i] 
            end
        end

    @synchronize

    @unroll for idx in 0:1
        @unroll for dolq in 0:1
        
        mulvecvec_split_exclfirst((dolq==0) ? view(cache,1,:) : view(cache,:,1),
                                (dolq==0) ? view(cache,i,:) : view(cache,:,i),view(cache2,:,i), k,0)
 
        @synchronize
        
        tmpsumiter = zero(eltype(input))
        tmp_sum = zero(eltype(input))
        for j = 1:BRDSPLIT
            tmpsumiter+= cache2[j,1]
            tmp_sum += cache2[j,i]
        end
        
        newvalue, factor = calc_factor(cache[1,1], tmpsumiter, tmp_sum, dolq==0 ? cache[i,1] : cache[1,i] )
        if (fullblock && (i>1))
                updatevector((dolq==0) ? view(cache, i,:) : view(cache, :,i) , 
                            (dolq==0) ? view(cache,1,:) : view(cache,:,1) , factor,k)
        end
        if (k+BRDSPLIT*(i-1) <= TILESIZE) 
            tilecol_cache[k+BRDSPLIT*(i-1)]= (dolq == 0) ?  cache[1,k+BRDSPLIT*(i-1)] : cache[k+BRDSPLIT*(i-1),1]
        end
        @synchronize
        idx_x = dolq==0 ? 1 : i
        idx_y = dolq==0 ? i : 1
        (k==1 && i>1) && (cache[idx_x,idx_y]= zero(eltype(input)))
        (k==1) &&  (cache[idx_y,idx_x]-=factor*newvalue)

        @synchronize


        if (fullblock)
            sendblockbacktomem(input, view(cache,:,:), k, i, rowidx+TILESIZE*(idx+dolq), colidx+idx*TILESIZE, nbrows)
        elseif (k==1 && colidx+i<=nbrows)
            input[1,colidx+i] = cache[1,i,1] 
        end

        
        if (i==1 && k==1)
            tilecol_cache[1]=newvalue
        end
        @synchronize

        loadblockintocache(view(cache,:,:),input, k,i,rowidx+TILESIZE*(1+idx),colidx+(idx+dolq)*TILESIZE, nbrows)

        @synchronize

        mulvecvec_split_exclfirst(tilecol_cache,  (dolq==0) ? view(cache,i,:) : view(cache,:,i),view(cache2,:,i), k,1)

        @synchronize

        tmp_sum = zero(eltype(input))
        for j = 1:BRDSPLIT
            tmp_sum += cache2[j,i]
        end
        factor = calc_factor2(newvalue, tmpsumiter, tmp_sum)

        updatevector((dolq==0) ? view(cache, i,:) : view(cache, :,i) , tilecol_cache, factor,k)
        @synchronize
        (k==1) &&  (cache[idx_y,idx_x]-=factor*newvalue)
        @synchronize

        fullblock=true
        @synchronize

    end
    end

    sendblockbacktomem(input, view(cache,:,:), k, i, rowidx+TILESIZE*2, colidx+2*TILESIZE, nbrows)
end

@inline function loadblockintocache(cache, input, k::Int, i::Int, rowidx::Int, colidx::Int, nbrows::Int)
        for j in 1:BRDSPLITFACTOR
            cache[(k-1)*BRDSPLITFACTOR+j,i] = 
                    (rowidx+(k-1)*BRDSPLITFACTOR+j<=nbrows && colidx+i<=nbrows) ? 
                    input[rowidx+(k-1)*BRDSPLITFACTOR+j,colidx+i] : zero(eltype(input))
        end
end

@inline function mulvecvec_split_exclfirst(vec1, vec2,cache2, k::Int, nonfirst::Int )
    tmp_sum = zero(eltype(vec1))
    for j in 1:BRDSPLITFACTOR
        tmp_sum+= ((k+j+nonfirst==2) ? zero(eltype(vec1)) : vec1[(k-1)*BRDSPLITFACTOR+j]*vec2[(k-1)*BRDSPLITFACTOR+j])
    end
    cache2[k]=tmp_sum
end


@inline function sendblockbacktomem(input, cache, k::Int, i::Int, rowidx::Int, colidx::Int, nbrows::Int)
        for j in 1:BRDSPLITFACTOR
            if (rowidx+(k-1)*BRDSPLITFACTOR+j<=nbrows && colidx+i<=nbrows)
                input[rowidx+(k-1)*BRDSPLITFACTOR+j,colidx+i] = cache[(k-1)*BRDSPLITFACTOR+j,i]
            end
        end
end


@inline function calc_factor(u1::T, unorm::T, uv::T, v1::T) where {T<:Number}
    newvalue = u1 + sign(u1) *sqrt(u1*u1+unorm)
    factor = (uv +newvalue*v1)*2/ (unorm + newvalue*newvalue)
    if ( isinf(factor))
        factor = (uv/(newvalue*newvalue)  +v1/newvalue)*2/ (unorm/(newvalue*newvalue) + 1)
    end
    return newvalue, factor
end

@inline function calc_factor2(newvalue::T, unorm::T, uv::T) where {T<:Number}
    factor = uv*2/ (unorm + newvalue*newvalue)
    if ( isinf(factor))
        factor = uv/(newvalue*newvalue) *2/ (unorm/(newvalue*newvalue) + 1)
    end
    return factor
end

@inline function updatevector(vec1, vec2, factor::Number,k::Int)
    for j in 1:BRDSPLITFACTOR
        vec1[(k-1)*BRDSPLITFACTOR+j]-=((k+j==2) ?  zero(eltype(vec1)) : factor* vec2[(k-1)*BRDSPLITFACTOR+j])
    end
end



#=

@kernel cpu=false inbounds=true unsafe_indices=false function brdkernel!(input, nbrows, secondsweep)
    k,i,l = @index(Local, NTuple)
    g = @index(Group, Linear)
    rowidx= -TILESIZE+1+Int(secondsweep)*2TILESIZE +(g-1)*(4TILESIZE-1)
    colidx= rowidx + TILESIZE
    cache = @localmem eltype(input) (TILESIZE+1, TILESIZE,2)
    cache2 = @localmem eltype(input) (BRDSPLIT+1,TILESIZE,2)


    fullblock= (g>1 || l==2 || secondsweep)
    
    if (l==1)
        if (fullblock)
            for j in 1:BRDSPLITFACTOR
                cache[(k-1)*BRDSPLITFACTOR+j,i,1] = 
                        (rowidx+(k-1)*BRDSPLITFACTOR+j<=nbrows && colidx+i<=nbrows) ? 
                        input[rowidx+(k-1)*BRDSPLITFACTOR+j,colidx+i] : zero(eltype(input))
            end
        else
            for j in 1:BRDSPLITFACTOR
                cache[(k-1)*BRDSPLITFACTOR+j,i,1] = zero(eltype(input))
            end
            if (k==1)
                cache[1,i,1] = (colidx+i <=nbrows) ?  input[1,colidx+i] : zero(eltype(input))
            end
        end
    end

    idxiter = 0
    idxcurr = l-1 

    for idx in 0:1
        idxiter+=1
        if (idxiter>2)
            idxiter=1
        end
        idxcurr+=1
        if (idxcurr>2)
            idxcurr=1
        end
        
        if (l==2)
            for j in 1:BRDSPLITFACTOR
                cache[(k-1)*BRDSPLITFACTOR+j,i,idxcurr] = 
                            (rowidx+(k-1)*BRDSPLITFACTOR+j+TILESIZE*(1+idx)<= nbrows && colidx+i+idx*TILESIZE<=nbrows) ? 
                            input[rowidx+(k-1)*BRDSPLITFACTOR+j+TILESIZE*(1+idx),colidx+i+idx*TILESIZE] : zero(eltype(input))
            end
        end
        

       @synchronize
        tmp_sum = zero(eltype(input))
        for j in 1:BRDSPLITFACTOR
            tmp_sum+= ((k+j==2) ? zero(eltype(input)) : cache[1,(k-1)*BRDSPLITFACTOR+j,idxiter]*cache[i,(k-1)*BRDSPLITFACTOR+j,idxcurr])
        end
        cache2[k,i,idxcurr]=tmp_sum

        @synchronize
        
        tmpsumiter = zero(eltype(input))
        tmp_sum = zero(eltype(input))
        for j = 1:BRDSPLIT
            tmpsumiter+= cache2[j,1,idxiter]
            tmp_sum += cache2[j,i,idxcurr]
        end
        
        newvalue = cache[1,1,idxiter] +  sign(cache[1,1,idxiter])*sqrt(tmpsumiter+ cache[1,1,idxiter]*cache[1,1,idxiter])
        factor = (tmp_sum+newvalue*cache[i,1,idxcurr]) *2/ (tmpsumiter+newvalue*newvalue)
        if (isinf(factor))
            factor = (tmp_sum/(newvalue*newvalue)+cache[i,1,idxcurr]/newvalue) *2/ (tmpsumiter/(newvalue*newvalue)+1)
        end
        
        if (fullblock && (i>1||l==2))
            for j in 1:BRDSPLITFACTOR
                cache[i,(k-1)*BRDSPLITFACTOR+j,idxcurr]-=((k+j==2) ?  zero(eltype(input)) : factor* cache[1,(k-1)*BRDSPLITFACTOR+j,idxiter])
            end
        end
        
        @synchronize

        if (k==1 && i>1 &&l==1)
            cache[1,i,idxiter]= zero(eltype(input))
        end
        if (k==1)
            cache[i,1,idxcurr]-=factor*newvalue
        end

        @synchronize

        if(l==1) 
            if (fullblock)
                for j in 1:BRDSPLITFACTOR
                    if (rowidx+(k-1)*BRDSPLITFACTOR+j+(l-1+idx)*TILESIZE<=nbrows && colidx+i+idx*TILESIZE<=nbrows)
                        input[rowidx+(k-1)*BRDSPLITFACTOR+j+(l-1+idx)*TILESIZE,colidx+i+idx*TILESIZE] = cache[(k-1)*BRDSPLITFACTOR+j,i,idxiter]
                    end
                end
            elseif (k==1)
                if (colidx+i<=nbrows)
                    input[1,colidx+i] = cache[1,i,idxiter] 
                end
            end
        end


        ##########################################################################


        idxiter+=1
        if (idxiter>2)
            idxiter=1
        end
        idxcurr+=1
        if (idxcurr>2)
            idxcurr=1
        end
        @synchronize
        if (l==2) 
            for j in 1:BRDSPLITFACTOR
                cache[(k-1)*BRDSPLITFACTOR+j,i,idxcurr] = 
                        (rowidx+(k-1)*BRDSPLITFACTOR+j+TILESIZE*(1+idx)<=nbrows && colidx+TILESIZE*(1+idx)+i<=nbrows) ? 
                        input[rowidx+(k-1)*BRDSPLITFACTOR+j+TILESIZE*(1+idx),colidx+TILESIZE*(1+idx)+i] : zero(eltype(input))
            end
        end
        @synchronize
        tmp_sum = zero(eltype(input))
        for j in 1:BRDSPLITFACTOR
            tmp_sum+= ((k+j==2) ? zero(eltype(input)) : cache[(k-1)*BRDSPLITFACTOR+j,1,idxiter]*cache[(k-1)*BRDSPLITFACTOR+j,i,idxcurr])
        end
        cache2[k,i,idxcurr]=tmp_sum

        @synchronize
  

        tmpsumiter = zero(eltype(input))
        tmp_sum = zero(eltype(input))
        for j = 1:BRDSPLIT
            tmpsumiter+= cache2[j,1,idxiter]
            tmp_sum += cache2[j,i,idxcurr]
        end
        
        newvalue = cache[1,1,idxiter] +  sign(cache[1,1,idxiter])*sqrt(tmpsumiter+ cache[1,1,idxiter]*cache[1,1,idxiter])

        factor = (tmp_sum+cache[1,i,idxcurr]*newvalue) *2/ (tmpsumiter+newvalue*newvalue)
        if(isinf(factor))
            factor = (tmp_sum/(newvalue*newvalue)+cache[i,1,idxcurr]/newvalue) *2/ (tmpsumiter/(newvalue*newvalue)+1)
        end

        @synchronize
        
        if ((i>1 || l==2))
            for j in 1:BRDSPLITFACTOR
                cache[(k-1)*BRDSPLITFACTOR+j,i,idxcurr]-=((k+j==2) ?  zero(eltype(input)) : factor* cache[(k-1)*BRDSPLITFACTOR+j,1,idxiter])
            end
        end


        @synchronize
        
        if (k==1 && i>1 &&l==1)
            cache[i,1,idxiter]= zero(eltype(input))
        end
        if (k==1)
            cache[1,i,idxcurr]-=factor*newvalue
        end
        @synchronize

        if (l==1)
            for j in 1:BRDSPLITFACTOR
                if (rowidx+(k-1)*BRDSPLITFACTOR+j+TILESIZE*(1+idx)<=nbrows && colidx+i+idx*TILESIZE<= nbrows)
                    input[rowidx+(k-1)*BRDSPLITFACTOR+j+TILESIZE*(1+idx),colidx+i+idx*TILESIZE]=cache[(k-1)*BRDSPLITFACTOR+j,i,idxiter]
                end
            end
        end

        @synchronize

        fullblock=true
    end

    if (l==2)
        for j in 1:BRDSPLITFACTOR
            if (rowidx+(k-1)*BRDSPLITFACTOR+j+2TILESIZE<= nbrows && colidx+i+2TILESIZE<= nbrows)
                input[rowidx+(k-1)*BRDSPLITFACTOR+j+2TILESIZE,colidx+i+2TILESIZE]=cache[(k-1)*BRDSPLITFACTOR+j,i,idxcurr]
            end
        end
    end
    


end
=#
