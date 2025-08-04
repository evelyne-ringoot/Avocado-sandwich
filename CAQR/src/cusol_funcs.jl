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

@inline CUSOLormqr!(C,A,Tau2)=CUSOLVER.ormqr!('L','T',A,Tau2,C);

function mybelapsed(f,A, args...)
    f(copy(A),args...)
    t=0.0
    k=0
    tmin=1000
    Acpy=copy(A)
    if((k<1000 && t<0.2) || k<5)
        synchronize()
        curt= @elapsed (CUDA.@sync f(Acpy,args...);)
        CUDA.synchronize()
        t+=curt
        tmin=min(tmin,curt)
        Acpy.=(A)
        k+=1
     end
     return tmin
end