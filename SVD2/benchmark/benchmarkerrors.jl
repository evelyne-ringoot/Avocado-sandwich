errors=zeros(6,4,length(sizes))
print( "Checking correctness GPU only at : ")
println(Dates.format(now(), "HH:MM:SS")  )
    println( " size   testmatrix   ERR KA max  ER vend max   ERR KA avg  ER vend avg ");
    println(" ------  --------    ----------  ----------   ----------  ----------  ");
    for (i,size_i) in enumerate(sizes)
        for matrixtype in 1:3
            for outlier in [true, false]
                maxerroka=0
                toterrorka=0
                maxerrorcu=0
                toterrorcu=0
                for numtest in 1:10

                    input=arty(randtestmatrix(size_i,matrixtype,outlier,elty))
                    #testing vendorsvd
                    avendor=Array(vendorsvd!(copy(input)))

                    #testing KA
                    mysvdres = mygesvd!(copy(input))
            
                    #reference
                    aref=(svdtestscaling(size_i,matrixtype,outlier))
                    KernelAbstractions.synchronize(backend)
                    kaerror=norm((mysvdres-aref))/norm(aref)
                    cuerror=norm((avendor-aref))/norm(aref)
                    maxerroka = maxerroka> kaerror ? maxerroka : kaerror
                    maxerrorcu = maxerrorcu > cuerror ? maxerrorcu : cuerror
                    toterrorcu+=cuerror
                    toterrorka+=kaerror
                end
    
                errors[(matrixtype-1)*2+1+outlier,1,i]= toterrorka/10
                errors[(matrixtype-1)*2+1+outlier,2,i]= toterrorcu/10
                errors[(matrixtype-1)*2+1+outlier,3,i]= maxerroka
                errors[(matrixtype-1)*2+1+outlier,4,i]= maxerrorcu
                @printf " %4d       %2d       %8.02e     %8.02e     %8.02e     %8.02e    \n" size_i (matrixtype-1)*2+1+outlier errors[(matrixtype-1)*2+1+outlier,3,i] errors[(matrixtype-1)*2+1+outlier,4,i] errors[(matrixtype-1)*2+1+outlier,1,i] errors[(matrixtype-1)*2+1+outlier,2,i] 
            end
            
        end
        flush(STDOUT)
    end
    print("Finished at : ")
    println(Dates.format(now(), "HH:MM:SS")  )

