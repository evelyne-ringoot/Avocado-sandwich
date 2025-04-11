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
