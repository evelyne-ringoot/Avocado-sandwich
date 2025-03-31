IndexGPUArray{T} = Union{AbstractGPUArray{T},
                               SubArray{T, <:Any, <:AbstractGPUArray},
                               LinearAlgebra.Adjoint{T}, 
                               SubArray{T, <:Any, <:LinearAlgebra.Adjoint{T, <:AbstractGPUArray }}}


function LinearAlgebra.triu!(A::IndexGPUArray{T}, d::Integer = 0) where T
    @kernel cpu=false  inbounds=true unsafe_indices=false function triu_kernel!(_A, _d)
      I = @index(Global, Cartesian)
      i, j = Tuple(I)
      if j < i + _d
        _A[i, j] = zero(T)
      end
    end
    triu_kernel!(get_backend(A))(A, d; ndrange = size(A))
    return A
  end

  function Base.fill!(A::IndexGPUArray{T}, x) where T
    isempty(A) && return A

    @kernel cpu=false  inbounds=true unsafe_indices=false function fill_kernel!(a, val)
        idx = @index(Global, Linear)
        a[idx] = val
    end

    # ndims check for 0D support
    kernel = fill_kernel!(get_backend(A))
    kernel(A, x; ndrange = ndims(A) > 0 ? size(A) : (1,))
    A
end



for (destType,srcType) in ((CUDA.StridedSubCuArray,SubArray) , (SubArray, CUDA.StridedSubCuArray), 
                            (CUDA.StridedSubCuArray, CUDA.StridedSubCuArray),
                            (CUDA.StridedSubCuArray, Array) ,  (Array, CUDA.StridedSubCuArray), 
                            (AbstractGPUArray, CUDA.StridedSubCuArray) , ( CUDA.StridedSubCuArray, AbstractGPUArray),
                            (AbstractGPUArray, SubArray) , (SubArray, AbstractGPUArray) )
  @eval begin
    function Base.copyto!(dest::$destType{T,2},src::$srcType{T,2}, Copy2D::Bool=false) where {T} 
      if (dest isa CUDA.StridedSubCuArray) || (dest isa SubArray)
        dest_index1=findfirst((typeof.(dest.indices) .<: Int).==0)
        dest_index2=findnext((typeof.(dest.indices) .<: Int).==0, dest_index1+1)
        dest_step_x=step(dest.indices[dest_index1])
        dest_step_height=step(dest.indices[dest_index2])
        dest_parent_size=size(parent(dest))
      else
        dest_index1=1
        dest_index2=2
        dest_step_x=1
        dest_step_height=1
        dest_parent_size=size(dest)
      end
      if (src isa CUDA.StridedSubCuArray) || (src isa SubArray)
        src_index1=findfirst((typeof.(src.indices) .<: Int).==0)
        src_index2=findnext((typeof.(src.indices) .<: Int).==0, src_index1+1)
        src_step_x=step(src.indices[src_index1])
        src_step_height=step(src.indices[src_index2])
        src_parent_size=size(parent(src)) 
      else
        src_index1=1
        src_index2=2
        src_step_x=1
        src_step_height=1
        src_parent_size=size(src) 
      end

      dest_pitch1= (dest_index1==1) ? 1 :  prod(dest_parent_size[1:(dest_index1-1)])
      dest_pitch2=  prod(dest_parent_size[dest_index1:(dest_index2-1)])
      src_pitch1= (src_index1==1) ? 1 :  prod(src_parent_size[1:(src_index1-1)])
      src_pitch2= prod(src_parent_size[src_index1:(src_index2-1)])
      destLocation= ((dest isa CUDA.StridedSubCuArray) || (dest isa CuArray)) ? Mem.Device : Mem.Host
      srcLocation= ((src isa CUDA.StridedSubCuArray) || (src isa CuArray)) ? Mem.Device : Mem.Host
      @boundscheck checkbounds(1:size(dest, 1), 1:size(src,1))
      @boundscheck checkbounds(1:size(dest, 2), 1:size(src,2))

      if (size(dest,1)==size(src,1) || (Copy2D))
      #Non-contigous views can be accomodated by copy3d in certain cases
        if isinteger(src_pitch2*src_step_height/src_step_x/src_pitch1) && isinteger(dest_pitch2*dest_step_height/dest_step_x/dest_pitch1) 
          CUDA.unsafe_copy3d!(pointer(dest), destLocation, pointer(src), srcLocation,
                                    1, size(src,1), size(src,2);
                                    srcPos=(1,1,1), dstPos=(1,1,1),
                                    srcPitch=src_step_x*sizeof(T)*src_pitch1,srcHeight=Int(src_pitch2*src_step_height/src_step_x/src_pitch1),
                                    dstPitch=dest_step_x*sizeof(T)*dest_pitch1, dstHeight=Int(dest_pitch2*dest_step_height/dest_step_x/dest_pitch1))
        #In other cases, use parallel threads
        else
          for col in 1:length(src.indices[src_index2])
              CUDA.unsafe_copy3d!(pointer(view(dest,:,col)),destLocation, pointer(view(src,:,col)),  srcLocation,
                                  1, 1, size(src,1);
                                  srcPos=(1,1,1), dstPos=(1,1,1),
                                  srcPitch=sizeof(T)*src_step_x*src_pitch1,srcHeight=1,
                                  dstPitch=sizeof(T)*dest_step_x*dest_pitch1, dstHeight=1)

          end
        end
      else  #Ensure same behavior as Base copying from smaller to bigger matrix if copy2D is false
        start_indices=(1:size(src,1):size(src,1)*(size(src,2)+1))
        dest_col=div.(start_indices.-1,size(dest,1)).+1
        start_indices=mod.(start_indices,size(dest,1))
        replace!(start_indices,0=>size(dest,1))
        split_col=start_indices[1:end-1].>start_indices[2:end]

        for col in 1:length(src.indices[src_index2])
            n= split_col[col] ? (size(dest,1)-start_indices[col]+1) : size(src,1)
            CUDA.unsafe_copy3d!(pointer(view(dest,:,dest_col[col])),destLocation, pointer(view(src,:,col)),  srcLocation,
                                1, 1, n;
                                srcPos=(1,1,1), dstPos=(1,1,start_indices[col]),
                                srcPitch=sizeof(T)*src_step_x*src_pitch1,srcHeight=1,
                                dstPitch=sizeof(T)*dest_step_x*dest_pitch1, dstHeight=1)
            if split_col[col]
              CUDA.unsafe_copy3d!(pointer(view(dest,:,dest_col[col]+1)),destLocation, pointer(view(src,:,col)),  srcLocation,
                                1, 1, size(src,1)-n;
                                srcPos=(1,1,n+1), dstPos=(1,1,1),
                                srcPitch=sizeof(T)*src_step_x*src_pitch1,srcHeight=1,
                                dstPitch=sizeof(T)*dest_step_x*dest_pitch1, dstHeight=1)
          end
        end
      end

      return dest
    end

    function Base.copyto!(dest::$destType{T,1},doffs::Integer,src::$srcType{T,1},  soffs::Integer,
                                  n::Integer) where {T} 
      n==0 && return dest
      @boundscheck checkbounds(dest, doffs)
      @boundscheck checkbounds(dest, doffs+n-1)
      @boundscheck checkbounds(src, soffs)
      @boundscheck checkbounds(src, soffs+n-1)
      if (dest isa CUDA.StridedSubCuArray) || (dest isa SubArray)
        dest_index=findfirst((typeof.(dest.indices) .<: Int).==0)
        dest_step=step(dest.indices[dest_index])
        dest_pitch=(dest_index==1) ? 1 : prod(size(parent(dest))[1:(dest_index-1)])
      else
        dest_index=1
        dest_step=1
        dest_pitch=1
      end

      if (src isa CUDA.StridedSubCuArray) || (src isa SubArray)
        src_index=findfirst((typeof.(src.indices) .<: Int).==0)
        src_step=step(src.indices[src_index])
        src_pitch= (src_index==1) ? 1 : prod(size(parent(src))[1:(src_index-1)])
      else
        src_index=1
        src_step=1
        src_pitch=1
      end
      destLocation= ((dest isa CUDA.StridedSubCuArray) || (dest isa CuArray)) ? Mem.Device : Mem.Host
      srcLocation= ((src isa CUDA.StridedSubCuArray) || (src isa CuArray)) ? Mem.Device : Mem.Host

      CUDA.unsafe_copy3d!(pointer(dest), destLocation, pointer(src), srcLocation,
                                1, 1, n;
                                srcPos=(1,1,soffs), dstPos=(1,1,doffs),
                                srcPitch=src_step*sizeof(T)*src_pitch,srcHeight=1,
                                dstPitch=dest_step*sizeof(T)*dest_pitch, dstHeight=1)
      return dest
    end



    Base.copyto!(dest::$destType{T}, src::$srcType{T}) where {T} =
      copyto!(dest, 1, src, 1, length(src))

  end
end