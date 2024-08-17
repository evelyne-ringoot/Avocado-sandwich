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
    return A
end

function bidiagonalize(A, bandwidth)
    (m,n) = size(A)
    while bandwidth>16
        bandwidth=round(Int,bandwidth/2)
        block_bidiagonalize!(A,n,bandwidth*2,bandwidth);
    end

    block_bidiagonalize!(A, n,bandwidth,1);
    return A 
end


function my_CU_svdval(A::Matrix, block_size)
    bidiag=bidiagonalize(A,block_size);
    return diag_lapack(bidiag) 
end

function diag_lapack(A)
    singvals, _, _, _, _,_= LAPACK.bdsdc!('U', 'N', diag(A), diag(A,1));
    return singvals
end

