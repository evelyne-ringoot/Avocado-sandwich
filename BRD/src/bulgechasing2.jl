using LinearAlgebra, BenchmarkTools, Base.@Threads


function QR_QMUL(A,idx1, idx2, n,ts)
    idx2=max(1,idx2.start):min(n,idx2.stop)
    idx1=max(1,idx1.start):min(n,idx1.stop)
    q,_ =qr!(view(A,idx1,idx2.start:min(idx2.start+ts-1,n)))
    view(A,idx1, idx2.start+ts:idx2.stop) .= q' * view(A,idx1, idx2.start+ts:idx2.stop)
    triu!(view(A,idx1,idx2.start:min(idx2.start+ts-1,n)))
    return;
end

function LQ_QMUL(A,idx1, idx2, n, ts)
    idx2=max(1,idx2.start):min(n,idx2.stop)
    idx1=max(1,idx1.start):min(n,idx1.stop)
    _, q =lq!(view(A,idx1.start:min(idx1.start+ts-1,n),idx2))
    view(A,idx1.start+ts:idx1.stop, idx2) .=   view(A,idx1.start+ts:idx1.stop, idx2) * q'
    tril!(view(A,idx1.start:min(idx1.start+ts-1,n), idx2))
    return;
end

function bidiag!(A, bandwidth;factor=4, kernelsizeinit=32)
    (m,n) = size(A)
    cbw=bandwidth
    while cbw>1
        tbw=max(1,div(cbw,factor))
        ts=tbw
        kernelsize=max(4,ceil(Int,kernelsizeinit/tbw))
        for sweeprow=1:kernelsize*ts:n-1-ts #row to cancel
            for iter=0: n #propagation idx
                for kernelidx=1:kernelsize
                    row=sweeprow + (kernelidx)*ts+cbw*(iter+1-2kernelidx) 
                    if ((iter>=(2kernelidx-2)) && row<n)
                        rowstart= (iter==(2kernelidx-2)) ? row+cbw-tbw : row
                        LQ_QMUL(A,rowstart:row+2cbw-1, row+cbw:row+2cbw-1, n,ts)
                        QR_QMUL(A, row+cbw:row+2cbw-1,row+cbw:row+3cbw-1,n,ts)
                    end
                end
            end
            
        end
        cbw=tbw
    end
    return A;
end

function bidiag_async!(A, bandwidth;factor=4, kernelsizeinit=32)
    (m,n) = size(A)
    cbw=bandwidth
    while cbw>1
        tbw=max(1,div(cbw,factor))
        ts=tbw
        kernelsize=max(4,ceil(Int,kernelsizeinit/tbw))
        for sweeprow=1:kernelsize*ts:n-1-ts #row to cancel
            for iter=0: n #propagation idx
                @sync begin
                    Threads.@spawn for kernelidx=1:kernelsize
                        row=sweeprow + (kernelidx)*ts+cbw*(iter+1-2kernelidx) 
                        if ((iter>=(2kernelidx-2)) && row<n)
                            rowstart= (iter==(2kernelidx-2)) ? row+cbw-tbw : row
                            LQ_QMUL(A,rowstart:row+2cbw-1, row+cbw:row+2cbw-1, n,ts)
                            QR_QMUL(A, row+cbw:row+2cbw-1,row+cbw:row+3cbw-1,n,ts)
                        end
                    end
                end
            end
            
        end
        cbw=tbw
    end
    return A;
end




#testing for correctness

function verify_bidiag(A)
    Acopy=copy(A)
    (m,n) = size(A)
    Acopy[1:m+1:end].=0
    Acopy[m+1:m+1:end].=0
    return maximum(Acopy)
end

for n in [64,512,1024]
  for bw in [4,16,64]
    A = triu(tril(rand(n,n),bw))
    sol=bidiag_async!(copy(A),bw)
    print(verify_bidiag(sol) ≈ 0 )
    print(svdvals(copy(A)) ≈ svdvals(sol))
  end
end




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
# old
####################################
#=
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

function bidiagonalizerec2(A, bandwidth, factor)
    (m,n) = size(A)
    tbw=bandwidth
    cbw=bandwidth

    while tbw>1
        tbw=round(Int,tbw/factor)
        ts=bw
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



=#
####################################
#benchmarking
####################################"
n_values=[1024,2048,4096]
bw_values=[2,4,16,64]
timings=zeros(length(n_values),length(bw_values),3)


for (iter_n,n) in enumerate(n_values)
    for (iter_bw,bw) in enumerate(bw_values)
        println((n,bw))
        A = triu(tril(rand(n,n),bw))
        timings[iter_n, iter_bw,1] = @belapsed bidiag!( $A, $bw)
        A = triu(tril(rand(n,n),bw))
        timings[iter_n, iter_bw, 2] = @belapsed gbbrd!($A, $bw)
        A = triu(tril(rand(n,n),bw))
        timings[iter_n, iter_bw,3] = @belapsed bidiag_async!( $A, $bw)
    end
end


using Plots

titles=["BRD QR bulgechasing - hiearchical (sync)";  "LAPACK GBBRD" ]
xaxis=n_values
xaxist=string.(n_values)
yaxis=[0.001, 0.01, 0.1, 1,10, 60, 300 ]
yaxist=["1 ms","10ms","0.1s","1s", "10s", "1min", "5min"]
plot(n_values, timings[:,:,1], labels= "bandwith ".*string.(bw_values') , lw=2, markershape=:circle, markerstrokewidth=0, markersize=3, color=[1 2 3 4])
plot!(n_values,timings[:,:,2], labels= "", lw=1, linestyle=:dash, markerstrokewidth=0, markersize=0, color=[1 2 3 4])
plot!([128,128],[1,1], labels= titles[2], lw=1, linestyle=:dash, markerstrokewidth=0, markersize=0, color="black")
plot!([128,128],[1,1], labels= titles[1],  lw=2, markershape=:circle, markerstrokewidth=0, markersize=3, color="black")
plot!(xaxis=:log2,  yaxis=:log10,legend=:topleft,  xlabel= "matrix size nxn", ylabel= "Execution time", dpi=1000)


####################################
#memory barriers
####################################"

function LinearAlgebra.qr!(A::AbstractMatrix{T}, τ::AbstractVector{T}) where {T}
    require_one_based_indexing(A)
    m, n = size(A)
    for k = 1:min(m - 1 + !(T<:Real), n)
        x = view(A, k:m, k)
        τk = LinearAlgebra.reflector!(x)
        τ[k] = τk
        LinearAlgebra.reflectorApply!(x, τk, view(A, k:m, k + 1:n))
    end
    QR(A, τ)
end

function tasksync(channellist, tasknb,n)
    if tasknb>1
        take!(channellist[tasknb-1])
        printf("yes"*string(task))
    end
    put!(channellist[tasknb],true)
    if tasknb==1
        take!(channellist[n])
    else
        take!(channellist[tasknb-1])
    end
    if tasknb<n
        put!(channellist[tasknb],true)
    end
end

create_channellist(n) = [Channel{Bool}(1) for i=1:n]
n=100000
mylist=[rand(10,10) for i=1:n]
mylist2=[zeros(10) for i=1:n]


using Polyester
@benchmark begin
    for i=1:n
        qr!($mylist[i],$mylist2[i])
    end
end

@benchmark begin
    qr!()
end

@benchmark begin
    @sync begin
        Threads.@spawn for i=1:n
            qr!($mylist[i],$mylist2[i])
        end
    end
end


@benchmark begin
    for i=1:8:n
        @sync begin
            Threads.@spawn for j=0:7
                qr!($mylist[i+j],$mylist2[i+j])
            end
        end
    end
end

@benchmark begin
    @sync begin
        Threads.@spawn for j=0:7
            for i=1:8:n
                qr!($mylist[i+j],$mylist2[i+j])
            end
        end
    end
end



@benchmark begin
    for j in 1:8:n
        @batch for i=j:j+7
            qr!($mylist[i],$mylist2[i])
        end
    end
end
