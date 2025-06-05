

function benchmark_ms( size_i, myfunc, dim2::Int=size_i)
    a=arty(randn!(zeros(backend, elty,dim2, size_i)))
    b=arty(randn!(zeros(backend, elty,dim2, size_i)))
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
    a=arty(randn!(zeros(backend, elty,dim2, size_i)))
    b=arty(randn!(zeros(backend, elty,dim2, size_i)))
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
    a=arty(rand!(zeros(backend, elty,size_i, size_i)))
    b=arty(rand!(zeros(backend, elty,size_i, size_i)))
    c=arty(rand!(zeros(backend, elty,size_i, size_i)))
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