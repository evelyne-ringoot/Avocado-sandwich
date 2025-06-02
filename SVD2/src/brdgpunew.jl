
using KernelAbstractions.Extras: @unroll
const SUBTILEFACTOR = (BRDSUBTILE2 >= TILESIZE ? 1 : Int(TILESIZE/BRDSUBTILE2))

@inline packedrowidx(rowidx::Int,colidx::Int, packed::Bool) = (packed ? (2BW+1-(rowidx-colidx)) : rowidx) 

struct PackedBandGPUMatrix{T} 
    data::AbstractGPUMatrix{T}     # diagonals plus bufferspace above and below
    bw::Int
    function PackedBandGPUMatrix{T}(d::AbstractGPUMatrix{T},bw::Int) where {T}
        new{T}(d,bw)
    end
end
function PackedBandGPUMatrix(input::AbstractGPUMatrix{T},bw::Int) where {T}
    n=size(input,1)
    d=KernelAbstractions.zeros(backend,T,bw*3+1,size(input,2))
    for i in 0:bw
        d[2bw+1-i,i+1:(end)].=input[i*n+1:n+1:end]
    end
    return PackedBandGPUMatrix{T}(d,bw)
end
function PackedBandGPUMatrix(input::Matrix{T},workspace::AbstractGPUMatrix{T},bw::Int) where {T}
    n=size(input,1)
    zero.(workspace)
    for i in 0:bw
        workspace[2bw+1-i,1:(end-i)].=input[i*n+1:n+1:end]
    end
    return PackedBandGPUMatrix{T}(d,bw)
end

function randPackedBandGPUMatrix(n::Int,bw::Int,T::DataType)
    d=KernelAbstractions.zeros(backend,T,bw*3+1,n)
    rand!(d)
    return PackedBandGPUMatrix{T}(d,bw)
end

function Matrix_frombidiag!(output::Matrix{T},input::PackedBandGPUMatrix{T}) where T
    zero.(output)
    n=size(output,1)
    bw=input.bw
    output[2bw+1,1:(end-i)].=input[1:n+1:end]
    output[2bw,1:(end-1)].=input[n+1:n+1:end]
    return output
end


function brd4!(A::AnyGPUMatrix{T}, noblocks::Int,bwiter::Int,packed::Bool) where T 
    CURRTILE=min(BRDSUBTILE2, TILESIZE*(bwiter+1))
        brdkernel_large_v2!(backend, (CURRTILE))(A,size(A,2), 0, bwiter, cld(noblocks,MAXBLOCKS), packed, ndrange=(CURRTILE*min(noblocks,MAXBLOCKS)))
        brdkernel_large_v2!(backend, (CURRTILE))(A,size(A,2), 1, bwiter, cld(noblocks,MAXBLOCKS), packed, ndrange=(CURRTILE*min(noblocks,MAXBLOCKS)))
        brdkernel_large_v2!(backend, (CURRTILE))(A,size(A,2), 2, bwiter, cld(noblocks,MAXBLOCKS), packed, ndrange=(CURRTILE*min(noblocks,MAXBLOCKS)))
end

function mygbbrd4!(A::AnyGPUMatrix{T}) where T 
    bw=BW
    n=size(A,1)
    for bwiter in Int(bw/TILESIZE):-1:1
        for k in 1:(n-1)
            brd4!(view(A,k:n,k:n),min(k,1+cld((n-k), (3TILESIZE*bwiter-1))),bwiter,false)
        end
    end
end 

function mygbbrd4_packed!(A::AbstractGPUMatrix{T}) where T 
    bw=BW
    n=size(A,1)
    Apacked=PackedBandGPUMatrix(A,BW)

    for bwiter in Int(bw/TILESIZE):-1:1
        for k in 1:(n-1)
            brd4!(view(Apacked,:,k:n),min(k,1+cld((n-k), (3TILESIZE*bwiter-1))),bwiter,true)
        end
    end
    return Matrix_frombidiag!(A,Apacked)
end 

@kernel cpu=false inbounds=true unsafe_indices=false function brdkernel_large_v2!(input, nbrows::Int, offset::Int,bwiter::Int, nbblocks::Int, packed::Bool)
    i = @index(Local, Linear)
    g = @index(Group, Linear)
    tilecol = @private eltype(input) (TILESIZE+1)
    tilecol_cache = @localmem eltype(input) (TILESIZE+1)

    for nbrun in 0:(nbblocks-1)
        
        rowidx= -TILESIZE+(bwiter==1)+offset*TILESIZE*bwiter +(g-1+nbrun*MAXBLOCKS)*(3TILESIZE*bwiter-1)
        colidx= rowidx + TILESIZE *bwiter
        fullblock= (g>1||nbrun>0|| offset>0 || i>TILESIZE)
        
        (i==1) && (tilecol_cache[TILESIZE+1]=zero(eltype(input)))
        tilecol[TILESIZE+1]=zero(eltype(input))

            @unroll for dolq in 0:1
                currrowidx=rowidx+TILESIZE*bwiter*dolq
                currcolidx=colidx
                i_corr=i+Int(bwiter>1)

                if (i<=TILESIZE)
                    @unroll for k in 1:(SUBTILEFACTOR)
                        idx_x = dolq==1 ? k+SUBTILEFACTOR*(i-1) : 1
                        idx_y = dolq==1 ? 1 : k+SUBTILEFACTOR*(i-1)
                        row= (fullblock ? currrowidx+idx_x : 1 )
                        col= currcolidx+idx_y
                        tilecol_cache[k+SUBTILEFACTOR*(i-1)] =  (row <=nbrows && col <=nbrows) ? 
                                input[row,col] : zero(eltype(input))
                    end
                end
                if (i==1 && bwiter>1)
                    idx_x = dolq==1 ? TILESIZE+1 : 1
                    idx_y = dolq==1 ? 1 : TILESIZE+1
                    row =(fullblock ? currrowidx+idx_x : 1 )
                    col = currcolidx+idx_y
                    tilecol_cache[TILESIZE+1]= ( row <=nbrows && col <=nbrows) ? 
                        input[row,col] : zero(eltype(input))
                end
            if ((fullblock && (i>1|| bwiter>1 )))
                @unroll for j in 1:TILESIZE
                    idx_x = dolq==1 ? j : i_corr
                    idx_y = dolq==1 ? i_corr : j
                    tilecol[j] =   (currrowidx+idx_x<=nbrows && currcolidx+idx_y<=nbrows) ? 
                            input[currrowidx+idx_x,currcolidx+idx_y] : zero(eltype(input))
                end
                idx_x = dolq==1 ? TILESIZE+1 : i_corr
                idx_y = dolq==1 ? i_corr : TILESIZE+1
                tilecol[TILESIZE+1] = (bwiter>1 && currrowidx+idx_x<=nbrows && currcolidx+idx_y<=nbrows) ?
                    input[ currrowidx+idx_x ,currcolidx+idx_y] : zero(eltype(input))
            end
            
            @synchronize
            tmp_sum = mulvecvec_nosplit(tilecol, tilecol_cache,0)
            tmpsumiter = mulvecvec_nosplit(tilecol_cache, tilecol_cache,0)
            
            pivotel=tilecol_cache[1]
            
            newvalue, factor,execiter = calc_factor(pivotel, tmpsumiter, tmp_sum, tilecol[1] )

            (i==1)  &&  (tilecol_cache[1]=newvalue)
            @synchronize
            updatevector(tilecol , tilecol_cache , factor ,3, execiter && fullblock && (i>1|| bwiter>1))
            
            if ((fullblock && (i>1|| bwiter>1)))
                @unroll for j in 1:TILESIZE
                    idx_x = dolq==1 ? j : i_corr
                    idx_y = dolq==1 ? i_corr : j
                    if (currrowidx+idx_x<=nbrows && currcolidx+idx_y<=nbrows)
                        input[currrowidx+idx_x,currcolidx+idx_y] = tilecol[j]
                    end
                end
                idx_x = dolq==1 ? TILESIZE+1 : i_corr
                idx_y = dolq==1 ? i_corr : TILESIZE+1
                if (bwiter>1 && currrowidx+idx_x<=nbrows && currcolidx+idx_y<=nbrows)
                        input[currrowidx+idx_x,currcolidx+idx_y] = tilecol[TILESIZE+1]
                end
            end
            if (i<=TILESIZE)
                @unroll for k in 1:(SUBTILEFACTOR)
                        idx_x = dolq==1 ? k+SUBTILEFACTOR*(i-1) : 1
                        idx_y = dolq==1 ? 1 : k+SUBTILEFACTOR*(i-1)
                        row= (fullblock ? currrowidx+idx_x : 1 )
                        col= currcolidx+idx_y
                    if ( row<=nbrows && col<=nbrows)
                        input[row,col]  = (k+i==2) ? pivotel-Int(abs(tmpsumiter)>2*floatmin(eltype(input)))*newvalue : zero(eltype(input))
                    end
                end
            end
            idx_x = dolq==1 ? TILESIZE+1 : 1
            idx_y = dolq==1 ? 1 : TILESIZE+1
            row= (fullblock ? currrowidx+idx_x : 1 )
            col= currcolidx+idx_y
            if (i==1 && bwiter>1 && row<=nbrows && col<=nbrows)
                    input[row,col] = zero(eltype(input))
            end
            
            
                @unroll for muliter in 1:7
                    if (((1+bwiter)*TILESIZE)>=(BRDSUBTILE2*(muliter)+i) && (fullblock || muliter>=SUBTILEFACTOR))
                        
                        currrowidx=rowidx+TILESIZE*bwiter*dolq+(1-dolq)*(BRDSUBTILE2*muliter+Int(bwiter>1))
                        currcolidx=colidx+(dolq)*(BRDSUBTILE2*muliter+Int(bwiter>1))     
                        #@print muliter " " i " " currcolidx " " currrowidx " \n"
                        
                        @unroll for j in 1:TILESIZE
                            idx_x = dolq==1 ? j : i
                            idx_y = dolq==1 ? i : j
                            tilecol[j] = (currrowidx+idx_x<=nbrows && currcolidx+idx_y<=nbrows) ? 
                                    input[currrowidx+idx_x,currcolidx+idx_y] : zero(eltype(input))
                        end
                        idx_x = dolq==1 ? TILESIZE+1 : i
                        idx_y = dolq==1 ? i : TILESIZE+1
                        tilecol[TILESIZE+1] = (bwiter>1 && currrowidx+idx_x<=nbrows && currcolidx+idx_y<=nbrows ) ?
                            input[ currrowidx+idx_x ,currcolidx+idx_y] : zero(eltype(input))

                        tmp_sum = mulvecvec_nosplit(tilecol, tilecol_cache,1)
                        factor = calc_factor2(pivotel, tmpsumiter, tmp_sum,tilecol[1] ,newvalue)
                        updatevector(tilecol , tilecol_cache, factor,3, execiter)

                        
                        @unroll for j in 1:TILESIZE
                            idx_x = dolq==1 ? j : i
                            idx_y = dolq==1 ? i : j
                            if (currrowidx+idx_x<=nbrows && currcolidx+idx_y<=nbrows)
                                input[currrowidx+idx_x,currcolidx+idx_y] =tilecol[j] 
                            end
                        end
                        idx_x = dolq==1 ? TILESIZE+1 : i
                        idx_y = dolq==1 ? i : TILESIZE+1
                        if (bwiter>1 && currrowidx+idx_x<=nbrows && currcolidx+idx_y<=nbrows)
                                input[currrowidx+idx_x,currcolidx+idx_y] = tilecol[TILESIZE+1]
                        end
                        
                    end
                
                end
            
            fullblock=true
            @synchronize
        end
        end


    
end



@inline function mulvecvec_nosplit(vec1, vec2, nonfirst::Int )
    tmp_sum = zero(eltype(vec1))
    @unroll for j in 1:TILESIZE+1
        tmp_sum+= ((j+nonfirst==1) ? zero(eltype(vec1)) : vec1[j]*vec2[j])
    end
    return tmp_sum
end


@inline function calc_factor(u1::T, unorm::T, uv::T, v1::T) where {T<:Number}
    execiter = !(abs(unorm)<2*floatmin(T)) 
    newvalue = u1 + sign(u1) *sqrt(u1*u1+unorm)
    if ( abs(unorm)<2*floatmin(T) && abs(uv)<2*floatmin(T) )
        return 2u1, (v1)/ ( u1), execiter
    end
    factor = (uv +newvalue*v1)*2/ (unorm + newvalue*newvalue)
    if ( isinf(factor))
        factor = (uv/(newvalue*newvalue)  +v1/newvalue)*2/ (unorm/(newvalue*newvalue) + 1)
    end
    
    return newvalue, factor, execiter
end

@inline function calc_factor2( u1::T, unorm::T, uv::T, v1::T, newvalue::T) where {T<:Number}
    if ( abs(unorm)<2*floatmin(T) && abs(newvalue*newvalue)<2*floatmin(T) )
        return (v1)/ ( u1)
    end
    factor = uv*2/ (unorm + newvalue*newvalue)
    if ( isinf(factor))
        factor = uv/(newvalue*newvalue) *2/ (unorm/(newvalue*newvalue) + 1)
    end
    return factor
end

@inline function updatevector(vec1, vec2, factor::Number,k::Int,execute::Bool)
    if (execute)
        @unroll for j in 1:TILESIZE+1
            vec1[j]-=((k+j==2) ?  zero(eltype(vec1)) : factor* vec2[j])
        end
    end
end
