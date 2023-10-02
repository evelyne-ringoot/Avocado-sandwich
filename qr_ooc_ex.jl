using LinearAlgebra, Random

# WARNING: not tested for non-square matrices, does not work with weird matrices containing zeros etc

#minimal working example of QR according to Trefethen, Bau Numerical Linear Algebra
function simple_QR!(M)
    (m,n)=size(M)
    tau=zeros(n)
    no_iter= (m>n) ? n : (m-1)
    current_col=zeros(m)
    temp_space=zeros(n)
    for k in 1:no_iter
        view(current_col,k:m).=view(M,k:m,k)
        view(current_col,k).+=(view(current_col,k) .< 0 ? -1 : 1)*norm(view(current_col,k:m))
        view(tau,k) .= 2*(view(current_col,k) /norm(view(current_col,k:m))).^2
        view(current_col,k:m) ./= view(current_col,k)
        view(temp_space,k:n) .= (view(current_col,k:m)'*view(M,k:m,k:n))'
        view(M,k:m,k:n).-=view(tau,k).*view(current_col,k:m)*view(temp_space,k:n)'
        view(M,(k+1):m,k).=view(current_col,(k+1):m)
    end
    return M,tau
end


function applyQt_vec!(X, M, tau, m, n)
    for k in 1:min(m,n)
        @views X[k:m]=X[k:m]-tau[k]*[1;M[(k+1):m,k]]*([1;M[(k+1):m,k]]'*X[k:m])
    end
    return X
end

function applyQt_mat!(A, M,tau)
    (m,n)=size(M)
    for k=1:size(A,2) 
        applyQt_vec!(view(A,:,k),M, tau, m,n)
    end
    return A
end

function getQt(M,tau)
    (m,n)=size(M)
    Q=zeros(m,m)
    Q[diagind(Q)].=1
    applyQt_mat!(Q, M,tau)
    return Q
end

function applyQt_vec_block!(X, M, tau, m, n, m2)
    for k in 1:min(m2,n)
        X[k:m]=X[k:m]-tau[k]*[1;zeros(m-k-m2);M[:,k]]*([1;zeros(m-k-m2);M[:,k]]'*X[k:m])
    end
    for k in min(m2,n)+1:min(m,n)
        @views X[k:m]=X[k:m]-tau[k]*[1;M[(k+1):m,k]]*([1;M[(k+1):m,k]]'*X[k:m])
    end
    return X
end

function applyQt_mat_block!(A, M,tau)
    (m2,n)=size(M)
    m=m2*2
    for k=1:size(A,2) 
        applyQt_vec_block!(view(A,:,k),M, tau, m,n, m2)
    end
    return A
end

function getQt_block(M,tau)
    (m,_)=size(M)
    Q=zeros(2m,2m)+I
    applyQt_mat_block!(Q, M,tau)
    return Q
end


#minimal working example of OOC QR algorithm according to https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.1301
tile_size_hor=3 #start with n=2
tile_size_ver=3 #start with n=2
number_of_tiles_hor=3
number_of_tiles_ver=3
M_test=Float64.(rand(1:4, (tile_size_ver*number_of_tiles_ver,tile_size_hor*number_of_tiles_hor)))
QR_ref=qr(copy(M_test))

no_pivots=min(number_of_tiles_hor,number_of_tiles_ver)
size_pivot=min(tile_size_hor,tile_size_ver)
tau=zeros(Float64,number_of_tiles_ver,size_pivot*number_of_tiles_hor )

#for debugging purposes Q_results=Matrix{Any}(undef, number_of_tiles_hor,number_of_tiles_ver) 

for pivot_element in 1:no_pivots
    index_range_hor= (pivot_element-1)*tile_size_hor+1:pivot_element*tile_size_hor
    index_range_ver= (pivot_element-1)*tile_size_ver+1:pivot_element*tile_size_ver
    
    #DGEQT2
    QR_pivot=  qr(view(M_test,index_range_ver, index_range_hor)) 
    view(M_test,index_range_ver, index_range_hor) .= QR_pivot.factors
    Q_tile_t=QR_pivot.Q'
    #for debugging purposes Q_results[pivot_element,pivot_element]=Q_tile_t
    view(tau,pivot_element,(pivot_element-1)*size_pivot.+(1:size_pivot)) .= diag(QR_pivot.T)
    #notice that this is the same as the line below
    #I-(I+tril(view(M_test,1:tile_size,1:tile_size),-1))*T11*(I+tril(view(M_test,1:tile_size,1:tile_size),-1))'

    #in-place version
    #QR11 = qr!(view(M_test,index_range,index_range)) #replaces view index_range,index_range of M_test with factors, saves pointer of the factorization to QR11 where factors now points back to view index_range,index_range of M_test
    #Q_tile_t=QR11.Q'

    #DLARFB
    for col in (pivot_element+1):number_of_tiles_hor
        index_shift=(col-pivot_element)*tile_size_hor
        view(M_test,index_range_ver , index_range_hor.+index_shift) .= Q_tile_t*view(M_test,index_range_ver , index_range_hor.+index_shift) 
    end

    #DTSQT2
    for row in (pivot_element+1):number_of_tiles_ver
        index_shift_row=(row-pivot_element)*tile_size_ver
        QR_row=qr(vcat(triu(view(M_test,index_range_ver, index_range_hor)), view(M_test,index_range_ver.+index_shift_row, index_range_hor)))
        view(M_test,index_range_ver, index_range_hor) .= tril(view(M_test,index_range_ver, index_range_hor),-1) +triu(view(QR_row.factors, 1:tile_size_ver,1:tile_size_hor))
        view(M_test,index_range_ver.+index_shift_row, index_range_hor) .= view(QR_row.factors, tile_size_ver+1:2tile_size_ver,1:tile_size_hor) 
        V_row= view(M_test,index_range_ver.+index_shift_row, index_range_hor)   # pointers to make reading the next lines easier
        Q_row_t= QR_row.Q' #vcat( hcat(I-T_row_t,  -T_row_t*V_row'),hcat(-V_row*T_row_t,I-V_row*T_row_t*V_row' ) ) #equal to I - (I;V)*T*(I T')
        #for debugging purposes Q_results[row,pivot_element]=Q_row_t
        view(tau,row,(pivot_element-1)*size_pivot.+(1:size_pivot)) .= view(diag(QR_row.T),1:size_pivot)

        #DSSRFB
        for col in (pivot_element+1):number_of_tiles_hor
            index_shift_col=(col-pivot_element)*tile_size_hor
            result_merged=  Q_row_t * vcat(view(M_test, index_range_ver, index_range_hor.+index_shift_col),view(M_test,index_range_ver.+index_shift_row, index_range_hor.+index_shift_col)) #temporary copy
            view(M_test, index_range_ver, index_range_hor.+index_shift_col) .= view(result_merged, 1:tile_size_ver,1:tile_size_hor)
            view(M_test,index_range_ver.+index_shift_row, index_range_hor.+index_shift_col) .= view(result_merged, tile_size_ver+1:2tile_size_ver,1:tile_size_hor)
        end
    end
end



triu(M_test) 
triu(QR_ref.factors)
negative= diag(M_test) .≈ -1 .* diag(triu(QR_ref.factors))
triu(M_test.* ((negative .*-2).+1)) ≈ triu(QR_ref.factors) #the sign of the rows are different, this is a convention thing and is fine

#notice that the bottom part of the resulting matrix where householder reflections are stored are different, however we can recompose Q by combining those in a block-manner
Q_t=zeros(tile_size_ver*number_of_tiles_ver,tile_size_ver*number_of_tiles_ver)+I

for pivot_element in 1:no_pivots
    index_range_hor= (pivot_element-1)*tile_size_hor+1:pivot_element*tile_size_hor
    index_range_ver= (pivot_element-1)*tile_size_ver+1:pivot_element*tile_size_ver

    #for debugging purposes @info pivot_element getQt(M_test[index_range_ver,index_range_hor],view(tau,pivot_element,(pivot_element-1)*size_pivot.+(1:size_pivot))) ≈ Q_results[pivot_element,pivot_element]

    applyQt_mat!(view(Q_t, index_range_ver, :), view(M_test, index_range_ver, index_range_hor), view(tau,pivot_element,(pivot_element-1)*size_pivot.+(1:size_pivot)))

    for row in (pivot_element+1):number_of_tiles_ver
        index_shift_row=(row-pivot_element)*tile_size_ver
        #notice that the below could be split into first calculating the top column and then botom
        Qtemp=[view(Q_t,index_range_ver,:); view(Q_t,index_range_ver.+index_shift_row,:) ] #concatenations of views dont exist yet (they create a copy instead)
        
        #for debugging purposes @info (row,pivot_element) getQt_block(view(M_test,index_range_ver.+index_shift_row,index_range_hor),tau[row, index_range_ver]) ≈ Q_results[row,pivot_element]
        
        applyQt_mat_block!(Qtemp, view(M_test,index_range_ver.+index_shift_row,index_range_hor),  tau[row, index_range_ver])
        view(Q_t,index_range_ver,:).= view(Qtemp,1:tile_size_ver,:)
        view(Q_t,index_range_ver.+index_shift_row,:) .= view(Qtemp,(1:tile_size_ver).+tile_size_ver,:)
    end
end

QR_ref.Q'.* ((negative .*-2).+1) ≈ Q_t 

