using LinearAlgebra.LAPACK: chkargsok, chklapackerror, chktrans, chkside, chkdiag, chkuplo
using LinearAlgebra: BlasInt, checksquare


const AbstractGPUorCPUMat{T} = Union{AbstractGPUArray{T, 2}, AbstractMatrix{T}, Adjoint{<:AbstractMatrix{T}}, Adjoint{<:AbstractGPUArray{T, 2}}}
const AbstractGPUorCPUArray{T} = Union{AbstractGPUArray{T}, AbstractArray{T}}

using ..LinearAlgebra.BLAS: @blasfunc, chkuplo

using ..LinearAlgebra: libblastrampoline, BlasFloat, BlasInt, LAPACKException, DimensionMismatch,
    SingularException, PosDefException, chkstride1, checksquare, triu, tril, dot

using Base: iszero, require_one_based_indexing

const liblapack = libblastrampoline

function diagcopyto!(dest::AbstractGPUorCPUArray,  src::AbstractGPUorCPUArray, bw::Int)
    @kernel function diag_copy_kernel!(dest, src,bw)
        i,j = @index(Global, NTuple)
        if i+j>=bw+2
            @inbounds dest[i,j] = src[i+j-bw-1,j]
        end
    end
    kernel = diag_copy_kernel!(get_backend(dest))
    kernel(dest,  src, bw; ndrange=((bw+1,size(src,2))))
    KernelAbstractions.synchronize(get_backend(dest))
    return dest
end



for (gebrd, elty) in
    ((:sgbbrd_, :Float32), (:dgbbrd_, :Float64),)
@eval begin
function gbbrd!(AB::AbstractMatrix{$elty}, bandwidth::Int) 
    m  = size(AB,2)
    k  =m
    n = m
    d     = ones( $elty, k)
    e     = ones( $elty, k-1)
    work  = Vector{$elty}(undef, 2*max(m,n))
    tempvar1  = Vector{$elty}(undef, 0)
    tempvar2  = Vector{$elty}(undef, 0)
    tempvar3  = Vector{$elty}(undef, 0)
    info  = Ref{BlasInt}(0)
    
        ccall((@blasfunc($gebrd), libblastrampoline), Cvoid, 
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},  Ptr{$elty}, Ref{BlasInt},
                    Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{$elty}, Ptr{BlasInt}),
                    'N', m, n, Ref{BlasInt}(0),Ref{BlasInt}(0), bandwidth,  AB, bandwidth+1, 
                    d, e, tempvar1, m,tempvar2,n,tempvar3,m,
                    work,  info)
    
    #chklapackerror(info[])
    return d,e

end
end
end

function gbbrd_copy(A::AbstractMatrix, bandwidth::Int)
    AB=similar(A,bandwidth+1,size(A,2))
    diagcopyto!(AB,A,bandwidth)
    return AB
end

gbbrd!(AB::AbstractMatrix{Float16}, bandwidth::Int) = gbbrd!(Float32.(AB), bandwidth) 


