using KernelAbstractions,CUDA,  BenchmarkTools, Random, LinearAlgebra

####################################
# LAPACK WRAPPER
####################################
using ..LinearAlgebra.BLAS: @blasfunc, chkuplo

using ..LinearAlgebra: libblastrampoline, BlasFloat, BlasInt, LAPACKException, DimensionMismatch,
    SingularException, PosDefException, chkstride1, checksquare, triu, tril, dot

using Base: iszero, require_one_based_indexing

const liblapack = libblastrampoline
function chklapackerror(ret::BlasInt, f...)
    if ret == 0
        return
    elseif ret < 0
        throw(ArgumentError(lazy"invalid argument #$(-ret) to LAPACK call"))
    else # ret > 0
        chklapackerror_positive(ret, f...)
    end
end

for (gebrd, elty) in
    ((:dgbbrd_, :Float64), (:dgbbrd_, :Float64),)
@eval begin
function gbbrd!(A::AbstractMatrix{Float64}, bandwidth::Int)
    gebrd=
    elty=:Float64
    m, n  = size(A)
    k     = min(m, n)
    d     = similar(A, $elty, k)
    e     = similar(A, $elty, k)
    tauq  = similar(A, $elty, k)
    taup  = similar(A, $elty, k)
    work  = Vector{$elty}(undef, 2*max(m,n))
    tempvar  = Vector{$elty}(undef, 1)
    info  = Ref{BlasInt}()
    AB=zeros(bandwidth+1,m)
    for b in 0:bandwidth
        AB[b+1,bandwidth+1-b:end]=diag(A,bandwidth-b)
    end
    ccall((@blasfunc($gebrd), libblastrampoline), Cvoid, 
            (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},  Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ptr{BlasInt}),
                'N', m, n, 0,0, bandwidth,  AB, bandwidth+1, 
                d, e, tempvar, 1,tempvar,1,tempvar,1,
                work,  info)
    #chklapackerror(info[])
    return d,e
end
end
end

####################################
# BRD functions
####################################
function QR_row!(A, startindex, lastindex, indexgap)
    qr!(view(A,startindex:lastindex, startindex+indexgap:lastindex)')
    tempcopy=view(A,startindex:lastindex, startindex+indexgap:lastindex)'*I
    tempcopy=triu(tempcopy)
    view(A, startindex:lastindex, startindex+indexgap:lastindex) .= tempcopy'
    return;
end

function QR_col!(A, startindex, lastindex, indexgap)
    qr!(view(A,startindex:lastindex, startindex:indexgap+lastindex))
    view(A,startindex:lastindex, startindex:indexgap+lastindex).=triu(view(A,startindex:lastindex, startindex:indexgap+lastindex))
    return;
end

function block_bidiagonalize!(A, n, bandwidth, target_bandwidth)
    for j=1:target_bandwidth:n-target_bandwidth    #bulge chasing: elimination on row j+1
        QR_row!(A,j, min(j+bandwidth+target_bandwidth-1, n) , target_bandwidth)
        
        for i=j:bandwidth:n
            lastindex=min(i+bandwidth+target_bandwidth-1, n) #index of end of block and its neighbor
            s_capped= min(bandwidth+target_bandwidth-1, max(n-i-bandwidth-target_bandwidth+1,0))
            QR_col!(A,i+target_bandwidth,lastindex,s_capped)
            
            i+target_bandwidth>(n-bandwidth) && break
            QR_row!(A,i+target_bandwidth,lastindex+s_capped,bandwidth)
        end
    end
    return;
end

function bidiagonalizerecold(A, bandwidth)
    (m,n) = size(A)
    while bandwidth>1
        bandwidth=round(Int,bandwidth/2)
        block_bidiagonalize!(A,n,bandwidth*2,bandwidth);
    end
    return;
end
function bidiagonalizeold(A, bandwidth)
    (m,n) = size(A)
    block_bidiagonalize!(A,n,bandwidth,1);
    return;
end

function QR_QMUL(A,idx1, idx2, n,ts)
    if (idx2.stop>n)
        idx2=idx2.start:n
    end
    if (idx1.stop>n)
        idx1=idx1.start:n
    end
    q,_ =qr!(view(A,idx1,idx2.start:min(idx2.start+ts-1,n)))
    view(A,idx1, idx2.start+ts:idx2.stop) .= q' * view(A,idx1, idx2.start+ts:idx2.stop)
    triu!(view(A,idx1,idx2.start:min(idx2.start+ts-1,n)))
    return;
end

function LQ_QMUL(A,idx1, idx2, n, ts)
    if (idx2.stop>n)
        idx2=idx2.start:n
    end
    if (idx1.stop>n)
        idx1=idx1.start:n
    end
    _, q =lq!(view(A,idx1.start:min(idx1.start+ts-1,n),idx2))
    view(A,idx1.start+ts:idx1.stop, idx2) .=   view(A,idx1.start+ts:idx1.stop, idx2) * q'
    tril!(view(A,idx1.start:min(idx1.start+ts-1,n), idx2))
    return;
end

function bidiagonalizerec(A, bandwidth)
    (m,n) = size(A)
    tbw=bandwidth
    cbw=bandwidth

    while tbw>1
        tbw=round(Int,tbw/2)
        ts=tbw
        for row=1:ts:n-1-ts
            LQ_QMUL(A,row:row+cbw+ts-1, row+tbw:row+cbw+ts-1, n,ts)
            QR_QMUL(A, row+ts:row+ts+cbw,row+tbw:row+2*cbw+ts,n,ts)
            LQ_QMUL(A, row+ts:row+2*cbw+ts-1,row+ts+cbw:row+2*cbw+ts-1,n,ts)
        end
        cbw=tbw
    end
    return;
end
function bidiagonalize(A, bandwidth)
    (m,n) = size(A)
    tbw=1
    cbw=bandwidth
    ts=1
    for row=1:ts:n-1-ts
        LQ_QMUL(A,row:row+cbw+ts-1, row+tbw:row+cbw+ts-1, n,ts)
        QR_QMUL(A, row+ts:row+ts+cbw,row+tbw:row+2*cbw+ts,n,ts)
        LQ_QMUL(A, row+ts:row+2*cbw+ts-1,row+ts+cbw:row+2*cbw+ts-1,n,ts)
    end
    return;
end

bsdc(A)=LAPACK.bdsdc!('U', 'N', diag(A), diag(A,1));
bdsqr(A, Vt, U, C)=LAPACK.bdsqr!('U', diag(A), diag(A,1), Vt, U, C);
cusvd(A)=svdvals!(A,alg=CUDA.CUSOLVER.QRAlgorithm())

####################################
#timing functions
####################################"
function mycubelapsed(f, A, args...;kwargs...)
        CUDA.@sync f(copy(A),args...;kwargs...)
        t=0.0
        k=0
        while (t<1)
            B=copy(A)
            CUDA.synchronize()
            t+= @elapsed (CUDA.@sync f(B,args...;kwargs...))
            CUDA.synchronize()
            k+=1
        end
    return t/k
end
function mybelapsed(f, A, args...;kwargs...)
    f(copy(A),args...;kwargs...)
    t=0.0
    k=0
    while (t<1)
        B=copy(A)
        t+= @elapsed (f(B,args...;kwargs...))
        k+=1
    end
return t/k
end



####################################
#benchmarking
####################################"
n_values=[1024,2048,4096,8192]
bw_values=[4,16,64,256]
timings=zeros(length(n_values),length(bw_values),5)
timings_bidiag=zeros(length(n_values))

for (iter_n,n) in enumerate(n_values)
    for (iter_bw,bw) in enumerate(bw_values)
        println((n,bw))
        A = triu(tril(rand(n,n),bw))
        B=Float32.(CuArray(A))
        #timings[iter_n, iter_bw, 1] = mybelapsed(bidiagonalizeold, A, bw)
        #timings[iter_n, iter_bw, 2] = mybelapsed(bidiagonalizerecold, A, bw)
        #timings[iter_n, iter_bw, 3] = mybelapsed(bidiagonalize, A, bw)
        timings[iter_n, iter_bw, 4] = mybelapsed(bidiagonalizerec, A, bw)
        timings[iter_n, iter_bw, 5] = mybelapsed(gbbrd!,A, bw)
    end
    A=Float32.(CUDA.randn(n,n))
    timings_bidiag[iter_n]=mycubelapsed(cusvd,A)
end


using Plots

iter=4
titles=["BRD QR bulgechasing - naive elementwise"; "BRD QR bulgechasing - blocked";
"BRD QR bulgechasing - cache efficient"; "BRD QR bulgechasing - cache efficient blocked" ]
xaxis=n_values
xaxist=string.(n_values)
yaxis=[0.001, 0.01, 0.1, 1,10, 60, 300 ]
yaxist=["1 ms","10ms","0.1s","1s", "10s", "1min", "5min"]
plot(n_values, timings[:,:,iter], labels= "bandwith ".*["4" "16" "64" "256" "1024"] , lw=2, markershape=:circle, markerstrokewidth=0, markersize=3, color=[1 2 3 4 5])
plot!(n_values,timings[:,:,5], labels= "", lw=1, linestyle=:dash, markerstrokewidth=0, markersize=0, color=[1 2 3 4 5])
plot!([128,128],[1,1], labels= "LAPACK GBBRD", lw=1, linestyle=:dash, markerstrokewidth=0, markersize=0, color="black")
plot!(n_values,timings_bidiag, labels= "CUSOLVER SVD total time", lw=5, markerstrokewidth=0, markersize=0, color="grey")
plot!(xaxis=:log2,  yaxis=:log10,legend=:topleft,  yticks=(yaxis,yaxist), xticks=(xaxis,xaxist), xlabel= "matrix size nxn", ylabel= "Execution time", title=titles[iter], dpi=1000)


#####################################
#=
    #U, Vt, C = Matrix{Float64}(I, n, n), Matrix{Float64}(I, n, n), Matrix{Float64}(I, n, n)
    #timings_bidiag[iter_n,1]=mybelapsed(bsdc,A)
    #timings_bidiag[iter_n,2]=mybelapsed(bdsqr,A, Vt, U, C)

=#