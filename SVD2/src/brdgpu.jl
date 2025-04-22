#using KernelAbstractions,GPUArrays, CUDA,Random, LinearAlgebra, Printf, BenchmarkTools

#const backend=KernelAbstractions.get_backend(CUDA.zeros(2))
#const TILESIZE = 9
const BRDSPLIT = 8
const BRDSPLITFACTOR = Int(TILESIZE/BRDSPLIT)

@kernel cpu=false inbounds=true unsafe_indices=false function brdkernel!(input, nbrows, secondsweep)
    k,i,l = @index(Local, NTuple)
    g = @index(Group, Linear)
    rowidx= -TILESIZE+1+Int(secondsweep)*2TILESIZE +(g-1)*(4TILESIZE-1)
    colidx= rowidx + TILESIZE
    cache = @localmem eltype(input) (TILESIZE+1, TILESIZE,2)
    cache2 = @localmem eltype(input) (BRDSPLIT+1,TILESIZE,2)
    
    #TODO: end cases

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
        for j in 1:BRDSPLIT
            tmp_sum+= ((k+j==2) ? zero(eltype(input)) : cache[1,(k-1)*BRDSPLITFACTOR+j,idxiter]*cache[i,(k-1)*BRDSPLITFACTOR+j,idxcurr])
        end
        cache2[k,i,3-l]=tmp_sum

        @synchronize
        
        tmpsumiter = zero(eltype(input))
        tmp_sum = zero(eltype(input))
        for j = 1:BRDSPLIT
            tmpsumiter+= cache2[j,1,2]
            tmp_sum += cache2[j,i,3-l]
        end
        
        newvalue = cache[1,1,idxiter] +  sign(cache[1,1,idxiter])*sqrt(tmpsumiter+ cache[1,1,idxiter]*cache[1,1,idxiter])
        factor = (tmp_sum+newvalue*cache[i,1,idxcurr]) *2/ (tmpsumiter+newvalue*newvalue)
        
        if (fullblock && (i>1||l==2))
            for j in 1:BRDSPLITFACTOR
                cache[i,(k-1)*BRDSPLITFACTOR+j,idxcurr]-=((k+j==2) ?  zero(eltype(input)) : factor* cache[1,(k-1)*BRDSPLITFACTOR+j,idxiter])
            end
        end
        
        @synchronize

        if (k==1 && i>1 &&l==1)
            cache[1,i,idxiter]= zero(eltype(input))
        elseif (k==2)
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

        tmp_sum = zero(eltype(input))
        for j in 1:BRDSPLITFACTOR
            tmp_sum+= ((k+j==2) ? zero(eltype(input)) : cache[(k-1)*BRDSPLITFACTOR+j,1,idxiter]*cache[(k-1)*BRDSPLITFACTOR+j,i,idxcurr])
        end
        cache2[k,i,l]=tmp_sum

        @synchronize
  

        tmpsumiter = zero(eltype(input))
        tmp_sum = zero(eltype(input))
        for j = 1:BRDSPLIT
            tmpsumiter+= cache2[j,1,1]
            tmp_sum += cache2[j,i,l]
        end
        
        newvalue = cache[1,1,idxiter] +  sign(cache[1,1,idxiter])*sqrt(tmpsumiter+ cache[1,1,idxiter]*cache[1,1,idxiter])
        factor = (tmp_sum+cache[1,i,idxcurr]*newvalue) *2/ (tmpsumiter+newvalue*newvalue)
        
        if ((i>1 || l==2))
            for j in 1:BRDSPLITFACTOR
                cache[(k-1)*BRDSPLITFACTOR+j,i,idxcurr]-=((k+j==2) ?  zero(eltype(input)) : factor* cache[(k-1)*BRDSPLITFACTOR+j,1,idxiter])
            end
        end

        @synchronize
        if (k==1 && i>1 &&l==1)
            cache[i,1,idxiter]= zero(eltype(input))
        elseif (k==2)
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


#A = triu(tril(rand(Float32,n,n),bw))
#norm(min.(abs.(((Array(svdvals(B))-svdvals(A))./svdvals(A))),abs.(svdvals(A))))/n

