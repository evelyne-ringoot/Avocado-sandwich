using LinearAlgebra, Random


function BandBidiagonal_tile!(A,n, N)
    for k in 1:1
        qr1=qr(A[k,k])
        A[k,k] .= qr1.R
        for col in k+1:n
            A[k,col] .= qr1.Q' * A[k,col]
        end


        for row in k+1:n
            temp=[copy(A[k,k]);copy(A[row,k])]
            println(temp)
            qr2=qr(temp)
            A[k,k] .=qr2.R
            A[row,k] .= 0
            for col in k+1:n
                temp=[ copy(A[k,col]); copy(A[row,col])]
                temp= qr2.Q' * temp
                println(temp)
                A[k,col] .= temp[1:N, 1:N]
                A[row,col] .= temp[N+1:2N, 1:N]
            end
        end

        if (k==n)
            break
        end

        lq1=qr(A[k,k+1]')
        A[k,k+1] .= lq1.R'
        for row in k+1:n
            A[row,k+1] .= A[row,k+1]*lq1.Q
        end

        for col in k+2:n
            temp=[copy(A[k,k+1])';copy(A[k,col])']
            lq2=qr(temp)
            A[k,k] .=lq2.R'
            A[k,col] .= 0
            for row in k+1:n
                temp=[copy(A[row,k+1])';copy(A[row,col])']
                temp= lq2.Q * temp
                A[row,k+1] .= temp[1:N,N+1:2N]'
                A[row,col] .= temp[1:N, N+1:2N]'
            end
        end

    end
end

##################################
#tiled algos
##################################

#minimal working example of OOC QR algorithm according to https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.1301
#works with assymetrical number of tiles, not with assymetrical blocks
tile_size_hor=3 #start with n=2
tile_size_ver=3 #start with n=2
number_of_tiles_hor=3
number_of_tiles_ver=3
M_test=Float64.(rand(1:4, (tile_size_ver*number_of_tiles_ver,tile_size_hor*number_of_tiles_hor)))
QR_ref=qr(copy(M_test))

no_pivots=min(number_of_tiles_hor,number_of_tiles_ver)
size_pivot=min(tile_size_hor,tile_size_ver)
tau=zeros(Float64,number_of_tiles_ver,size_pivot*number_of_tiles_hor )

#for debugging purposes 
Q_results=Matrix{Any}(undef, number_of_tiles_ver, number_of_tiles_ver) 

for pivot_element in 1:no_pivots
    index_range_hor= (pivot_element-1)*tile_size_hor+1:pivot_element*tile_size_hor
    index_range_ver= (pivot_element-1)*tile_size_ver+1:pivot_element*tile_size_ver
    
    #DGEQT2
    QR_pivot=  qr(view(M_test,index_range_ver, index_range_hor)) 
    view(M_test,index_range_ver, index_range_hor) .= QR_pivot.factors
    Q_tile_t=QR_pivot.Q'
    #for debugging purposes 
    Q_results[pivot_element,pivot_element]=Q_tile_t
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
        Q_row_t= QR_row.Q' 
        T_row_t=QR_row.T'
 
        @info (row,pivot_element) Q_row_t ≈ vcat( hcat(I-T_row_t,  -T_row_t*V_row'),hcat(-V_row*T_row_t,I-V_row*T_row_t*V_row' ) )
        
        #for debugging purposes 
        Q_results[row,pivot_element]=Q_row_t
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
negative=ones(tile_size_ver*number_of_tiles_ver)
view(negative,1:min(size(M_test)...))[diag(M_test) .≈ -1 .* diag(triu(QR_ref.factors))] .= -1
triu(M_test.* negative) ≈ triu(QR_ref.factors) #the sign of the rows are different, this is a convention thing and is fine

#notice that the bottom part of the resulting matrix where householder reflections are stored are different, however we can recompose Q by combining those in a block-manner
Q_t=zeros(tile_size_ver*number_of_tiles_ver,tile_size_ver*number_of_tiles_ver)+I

for pivot_element in 1:no_pivots
    index_range_hor= (pivot_element-1)*tile_size_hor+1:pivot_element*tile_size_hor
    index_range_ver= (pivot_element-1)*tile_size_ver+1:pivot_element*tile_size_ver

    #for debugging purposes
    @info pivot_element getQt(M_test[index_range_ver,index_range_hor],view(tau,pivot_element,(pivot_element-1)*size_pivot.+(1:size_pivot))) ≈ Q_results[pivot_element,pivot_element]

    applyQt_mat!(view(Q_t, index_range_ver, :), view(M_test, index_range_ver, index_range_hor), view(tau,pivot_element,(pivot_element-1)*size_pivot.+(1:size_pivot)))

    for row in (pivot_element+1):number_of_tiles_ver
        index_shift_row=(row-pivot_element)*tile_size_ver
        #notice that the below could be split into first calculating the top column and then botom
        Qtemp=[view(Q_t,index_range_ver,:); view(Q_t,index_range_ver.+index_shift_row,:) ] #concatenations of views dont exist yet (they create a copy instead)
        
        #for debugging purposes 
        @info (row,pivot_element) getQt_block(view(M_test,index_range_ver.+index_shift_row,index_range_hor),tau[row, index_range_ver]) ≈ Q_results[row,pivot_element]
        
        applyQt_mat_block!(Qtemp, view(M_test,index_range_ver.+index_shift_row,index_range_hor),  tau[row, index_range_ver])
        view(Q_t,index_range_ver,:).= view(Qtemp,1:tile_size_ver,:)
        view(Q_t,index_range_ver.+index_shift_row,:) .= view(Qtemp,(1:tile_size_ver).+tile_size_ver,:)
    end
end

QR_ref.Q'.* negative ≈ Q_t 


###################################################"
# OLD CODE JUST IN CASE #######################
##########################################""

#QR kernel
for k in 1:n
    q1, r1 = qr(A[k,k])
    A[k,k].=r1
    for i in k+1:n
        A[k,i].=q1'*A[k,i]
    end
    for row in k+1:n
        temp=zeros(2n,n)
        temp[1:n,1:n].=A[k,k]
        temp[n+1:2n,1:n]=A[row,k]
        q2,r2=qr(temp)
        A[k,k].=r2
        for col in k+1:n
            temp[1:n,1:n].=A[k,col]
            temp[n+1:2n,1:n]=A[row,col]
            temp=q2'*temp
            A[k,col] .=temp[1:n,1:n]
            A[row,col] .= temp[n+1:2n,1:n]
        end
    end
end