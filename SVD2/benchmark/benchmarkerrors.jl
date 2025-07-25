errors=zeros(6,2,length(sizes))
print( "Checking correctness GPU only at : ")
println(Dates.format(now(), "HH:MM:SS")  )

    for (i,size_i) in enumerate(sizes)
        for matrixtype in 1:3
            for outlier in [true, false]
                input=arty(randtestmatrix(size_i,matrixtype,outlier,elty))
                #testing vendorsvd
                avendor=Array(vendorsvd!(copy(input)))

                #testing KA
                mysvdres = mygesvd!(copy(input))
        
                #reference
                aref=(svdtestscaling(size_i,matrixtype,outlier))
                KernelAbstractions.synchronize(backend)
    
                errors[(matrixtype-1)*2+1+outlier,1,i]= norm((mysvdres-aref))/norm(aref)
                errors[(matrixtype-1)*2+1+outlier,2,i]= norm((avendor-aref))/norm(aref)
            end
        end
    end
    print("Finished at : ")
    println(Dates.format(now(), "HH:MM:SS")  )

    println( " size   testmatrix   RRMSE KA   RRMSE vendor  ");
    println(" ------  --------    ----------  ----------  ");
    for type in 1:6
        for (i,size_i) in enumerate(sizes)
            @printf " %4d       %2d       %8.02e     %8.02e     \n" size_i type errors[type,1,i] errors[type,2,i] 
        end
    end  
