

function benchmark_ms( size_i, myfunc, dim2::Int=size_i)
    a=arty(randn!(zeros( elty,dim2, size_i)))
    b=arty(randn!(zeros( elty,dim2, size_i)))
    elapsed=0.0
    best=10000000000
    i=0
    numruns = (size_i < 1025) ? NUMRUMS : 2
    while(elapsed<MINTIME || (i<2 &&elapsed<5000.0))
        KernelAbstractions.synchronize(backend)
        start = time_ns()
        for i=1:numruns
            copyto!(a,b)
            myfunc(a)
        end
        KernelAbstractions.synchronize(backend)
        endtime = time_ns()
        thisduration=(endtime-start)/1e6
        elapsed+=thisduration
        best = min(thisduration/numruns,best)
        i+=1
    end
    unsafe_free!(a)
    return best
end



function benchmark_ms_large( size_i, myfunc, dim2::Int=size_i)
    a=arty(randn!(zeros( elty,dim2, size_i)))
    b=arty(randn!(zeros( elty,dim2, size_i)))
    elapsed=0.0
    best=1000000000
    i=0
    while((i<2))
        KernelAbstractions.synchronize(backend)
        start = time_ns()
        copyto!(a,b)
        myfunc(a)
        KernelAbstractions.synchronize(backend)
        endtime = time_ns()
        thisduration=(endtime-start)/1e6
        elapsed+=thisduration
        best = min(thisduration,best)
        i+=1
    end
    unsafe_free!(a)
    return best
end

function benchmark_ms_muladd( size_i, myfunc)
    a=arty(rand!(zeros( elty,size_i, size_i)))
    b=arty(rand!(zeros( elty,size_i, size_i)))
    c=arty(rand!(zeros( elty,size_i, size_i)))
    elapsed=0.0
    best=100000000
    i=0
    numruns = (size_i < 1024*16+1) ? NUMRUMS : 2
    while(elapsed<MINTIME || (i<2 &&elapsed<5000.0))
        KernelAbstractions.synchronize(backend)
        start = time_ns()
        for i=1:numruns
            myfunc(a,b,c)
        end
        KernelAbstractions.synchronize(backend)
        endtime = time_ns()
        thisduration=(endtime-start)/1e6
        elapsed+=thisduration
        best = min(thisduration/numruns,best)
        i+=1
    end
    unsafe_free!(a)
    return best
end

function svdtestscaling(n, type, outlier::Bool)
    if type==1
       out= (1:-1/(n-1):0)
    elseif (type==2)
        out= 10 .^((0:1/(n-1):1).*log10(10eps(elty)))

    elseif (type==3)
        semicriclecdf(x)= (x>1 ? 1+eps(elty) : (x<0 ? 0-eps(elty) : (2x/π)*sqrt(1-x^2)+(2/π)*asin(x) ))
        out = zeros(n)
        for i in 2:n
            f(x)=semicriclecdf(x)-(i-1)/(n-1)
            out[i] = find_zero(f ,(0,1))
        end
        out= reverse(out)
    end
    if outlier
        out=[out...]
        out[1]=sqrt(n)
    end
    return out
end

function randtestmatrix(n,type,outlier,elty)
    svdvals= (vectyfp64(svdtestscaling(n,type,outlier)))
    a=KernelAbstractions.zeros(backend,Float64,n,n)
    a[1:n+1:end].=svdvals
    unit1= (qr!(artyfp64(randn(n,n))).Q)
    unit2= qr!(artyfp64(randn(n,n))).Q
    return elty.( unit1*a*unit2)
end

function randwellbehaved(n,elty)
    svdvals= (vectyfp64(svdtestscaling(n,1,false)))
    a=KernelAbstractions.zeros(backend,Float64,n,n)
    a[1:n+1:end].=svdvals
    unit1= (qr!(artyfp64(randn(n,n))).Q)
    unit2= qr!(artyfp64(randn(n,n))).Q
    return elty.( unit1*a*unit2)
end

