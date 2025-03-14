using LinearAlgebra.LAPACK: chkargsok, chklapackerror, chktrans, chkside, chkdiag, chkuplo
using LinearAlgebra: BlasInt, checksquare

for (bname, fname,elty) in ((:cusolverDnSgeqrf_bufferSize, :cusolverDnSgeqrf, :Float32),
    (:cusolverDnSgeqrf_bufferSize, :cusolverDnSgeqrf, :Float32))
    @eval begin
        function geqrf!(A::StridedCuMatrix{$elty}, tau::CuVector{$elty}, m,n,lda,dh,buffer, buffersize )
            CUSOLVER.$fname(dh, m, n, A, lda, tau, buffer, buffersize, dh.info)
            info = CUDA.@allowscalar dh.info[1]
            chkargsok(BlasInt(info))
            A
        end
        function geqrf_buffersize(A::StridedCuMatrix{$elty})
            out = Ref{Cint}(0)
            CUSOLVER.$bname(CUSOLVER.dense_handle(), size(A,1), size(A,2), A, max(1, stride(A, 2)), out)
            return out[] * sizeof($elty)
        end
    end
end

for (bname, fname, elty) in ((:cusolverDnSormqr_bufferSize, :cusolverDnSormqr, :Float32),
(:cusolverDnDormqr_bufferSize, :cusolverDnDormqr, :Float64))
    @eval begin
        function ormqr!(C::StridedCuVecOrMat{$elty},
            A::StridedCuMatrix{$elty},
            tau::CuVector{$elty},
             dh, m,n,k,lda,ldc,buffersize,buffer)
            CUSOLVER.$fname(dh, 'L', 'T', m, n, k, A, lda, tau, C, ldc, buffer, buffersize, dh.info)
            info = CUDA.@allowscalar dh.info[1]
            chkargsok(BlasInt(info))
            return C
        end
        function ormqr_bufferSize( A::StridedCuVecOrMat{$elty}, tau::CuVector{$elty}, C::StridedCuVecOrMat{$elty})
            out = Ref{Cint}(0)
            CUSOLVER.$bname(CUSOLVER.dense_handle(), 'L', 'T', size(C,1), size(C,2), length(tau), A, max(1, stride(A, 2)), tau, C, max(1, stride(C, 2)), out)
            return out[] * sizeof($elty)
        end



    end
end



using ..LinearAlgebra.BLAS: @blasfunc, chkuplo

using ..LinearAlgebra: libblastrampoline, BlasFloat, BlasInt, LAPACKException, DimensionMismatch,
    SingularException, PosDefException, chkstride1, checksquare, triu, tril, dot

using Base: iszero, require_one_based_indexing

const liblapack = libblastrampoline

function diagcopyto!(dest::Array,  src::Array, bw::Int)
    @kernel function diag_copy_kernel!(dest, src,bw)
        i,j = @index(Global, NTuple)
        if i+j>=bw+2
            @inbounds dest[i,j] = src[i+j-bw-1,j]
        end
    end
    kernel = diag_copy_kernel!(get_backend(dest))
    kernel(dest,  src, bw; ndrange=((bw+1,size(src,1))))
end

const AbstractGPUorCPUMat{T} = Union{AbstractGPUArray{T, 2}, AbstractMatrix{T}}

for (gebrd, elty) in
    ((:sgbbrd_, :Float32), (:sgbbrd_, :Float32),)
@eval begin
function gbbrd!(A::AbstractGPUorCPUMat{T}, bandwidth::Int) where T
    m, n  = size(A)
    k     = min(m, n)
    d     = zeros($elty, k)
    e     = zeros( $elty, k-1)
    tauq  = zeros( $elty, k)
    taup  = zeros( $elty, k)
    work  = Vector{$elty}(undef, 2*max(m,n))
    tempvar  = Vector{$elty}(undef, m)
    info  = Ref{BlasInt}()
    
    AB=zeros($elty,bandwidth+1,n)
    ABGPU=KernelAbstractions.zeros(get_backend(A),$elty,bandwidth+1,m)
    diagcopyto!(ABGPU,A,bandwidth)
    if (get_backend(AB)== get_backend(A)) 
        AB=ABGPU
    else
        copyto!(AB,ABGPU)
    end
    KernelAbstractions.synchronize(get_backend(ABGPU))

    ccall((@blasfunc($gebrd), libblastrampoline), Cvoid, 
            (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},  Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ptr{BlasInt}),
                'N', m, n, 1,0, bandwidth,  AB, bandwidth+1, 
                d, e, tempvar, 1,tempvar,1,tempvar,m,
                work,  info)
    return d,e
end
end
end



