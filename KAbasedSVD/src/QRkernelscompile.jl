##DEFINE TILE_SIZE1,TILE_SIZE2,dims2d
if !(@isdefined TILE_SIZE1) || !(@isdefined TILE_SIZE2) || !(@isdefined dims2d) 
    error(" Please define kernel parameters")
end
backend=KernelAbstractions.get_backend(CUDA.randn(2))
println("compiling kernels for tilesize ", TILE_SIZE1, " and ", TILE_SIZE2, " and for ", dims2d ? "2 dimensions" : " single dimension", " and backend ", backend )


if !dims2d
    (f1,f2,f3,f4) = (QR_unsafe_kernel!,QR_unsafe_kernel2!,applyQorQt_unsafe_kernel!, applyQorQt_unsafe_kernel2!)
else
    (f1,f2,f3,f4) = (QR_unsafe_kernel_2d!,QR_unsafe_kernel2_2d!,applyQorQt_unsafe_kernel_2d!, applyQorQt_unsafe_kernel2_2d!)
end

    for (fname, kernel, blocksize) in ( ("applyQR1!", f1 , TILE_SIZE1),
        ("applyQR2!", f2, TILE_SIZE1 ) ,
        ("applyQorQt1!", f3, TILE_SIZE2 ),
        ("applyQorQt2!", f4 , TILE_SIZE2),
        ("mytriu!", mytriukernel!, TILE_SIZE1),
        ("mytranspose!", coalesced_transpose_kernel!, TILE_SIZE1) )
    @eval begin
        $(Symbol(fname))= $kernel(backend, $blocksize)  
    end
end

for (fname, kernelf, applyt) in ( ("applyQ1!", applyQorQt1!, false),
            ("applyQt1!", applyQorQt1!, true),
            ("applyQ2!", applyQorQt2!, false),
            ("applyQt2!", applyQorQt2!, true))
    @eval begin
        $(Symbol(fname))(args...;ndrange) = $kernelf(args...,$applyt,ndrange=ndrange)
    end    
end

get_kernel_dims(::KernelAbstractions.Kernel{B,S}) where {B,S} = S.parameters[1]

