
using Dates 
errors=zeros(4,length(sizes))
print( "Checking correctness GPU only at : ")
println(Dates.format(now(), "HH:MM:SS")  )
try
    for (i,size_i) in enumerate(sizes)
        input=arty(randn!(zeros(elty,size_i, size_i)))
        aband=copy(input)
        myblockdiag!(aband)
        KernelAbstractions.synchronize(get_backend(input))
        abandfp64=Float64.(copy(aband))
        KernelAbstractions.synchronize(get_backend(input))
        abandsvd= vendorsvd!(abandfp64)
        mygbbrd!(aband)
        KernelAbstractions.synchronize(get_backend(input))
        abidiagsvd= vendorsvd!(Float64.(copy(aband)))
        aout=mygesvd!(copy(input))
        aref=vendorsvd!(copy(input))
        areffp64=vendorsvd!(Float64.(copy(input)))
        KernelAbstractions.synchronize(backend)
        if (!isnothing(aref))
            output=KernelAbstractions.zeros(backend,elty,size_i)
            copyto!(output,aout) #Array because mygesvd returns CPU Array
            errors[1,i]= norm((output-areffp64))/norm(areffp64)
            errors[2,i]= norm((abandsvd-areffp64))/norm(areffp64)
            errors[3,i]= norm((abidiagsvd-areffp64))/norm(areffp64)
            errors[4,i]= norm((aref-areffp64))/norm(areffp64)
        end
    end
    print("Finished at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
catch e
    print("unfinished at : ")
    println(Dates.format(now(), "HH:MM:SS")  )
finally
    println( " size    RRMSE svd  RRMSE band  RRMSE BRD  RRMSE CUSOLVER");
    println(" ------  --------  ----------  ----------  ---------- ");
    for (i,size_i) in enumerate(sizes)
        @printf " %4d   %8.02e    %8.02e  %8.02e   %8.02e \n" size_i errors[1,i] errors[2,i] errors[3,i] errors[4,i] 
    end  
end