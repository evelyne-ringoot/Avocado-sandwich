using KernelAbstractions,CUDA, BSON, BenchmarkTools, Random, LinearAlgebra, Dagger, Profile, PProf


function QR_row!(A, startindex, lastindex, indexgap)
    lq!(view(A,startindex:lastindex, startindex+indexgap:lastindex))
    tril!(view(A,startindex:lastindex, startindex+indexgap:lastindex))
    return;
end

function QR_col!(A, startindex, lastindex, indexgap)
    qr!(view(A,startindex:lastindex, startindex:indexgap+lastindex))
    triu!(view(A,startindex:lastindex, startindex:indexgap+lastindex))
    return;
end


function bidiagonalize(A, bandwidth; bwswitch=16)
    (m,n) = size(A)
    while bandwidth>1
        if bandwidth>bwswitch
            target_bandwidth=round(Int,bandwidth/2)
        else
            target_bandwidth=1
        end
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
            bandwidth=target_bandwidth
    end
    return A 
end

function QR_row1!(A)
    lq!(A)
    tril!(A)
    return;
end

function QR_col1!(A)
    qr!(A)
    triu!(A)
    return;
end


function bidiagonalize1(A, bandwidth; bwswitch=16)
    (m,n) = size(A)
    while bandwidth>1
        if bandwidth>bwswitch
            target_bandwidth=round(Int,bandwidth/2)
        else
            target_bandwidth=1
        end
        Dagger.spawn_datadeps() do
            for j=1:target_bandwidth:n-target_bandwidth    #bulge chasing: elimination on row j+1
                Dagger.@spawn QR_row1!(InOut(view(A,startindex:lastindex, startindex:indexgap+lastindex)))
                for i=j:bandwidth:n
                    lastindex=min(i+bandwidth+target_bandwidth-1, n) #index of end of block and its neighbor
                    s_capped= min(bandwidth+target_bandwidth-1, max(n-i-bandwidth-target_bandwidth+1,0))
                    Dagger.@spawn QR_col1!(InOut(view(A,startindex:lastindex, startindex:indexgap+lastindex)))
                    
                    i+target_bandwidth>(n-bandwidth) && break
                    Dagger.@spawn QR_row1!(InOut(view(A,startindex:lastindex, startindex:indexgap+lastindex)))
                end
            end
            bandwidth=target_bandwidth
        end
    end
    return A 
end

timings=zeros(2,6)
b=32
#n=512
#A = triu(tril(rand(n,n),b));
#Profile.@profile bidiagonalize1(A, b)
for (i,n) in enumerate(2 .^(6:10))
    println("executing size "*string(n));
    A = triu(tril(rand(n,n),b));
    BLAS.set_num_threads(1)
    timings[1,i]=@belapsed bidiagonalize1($A, b);
    BLAS.set_num_threads(4)
    timings[2,i]=@belapsed bidiagonalize($A, b);
    BSON.@save "timings.bson" timings
end


