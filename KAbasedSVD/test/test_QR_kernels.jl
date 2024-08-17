include(joinpath("..","src","QRkernels.jl"))

@testset "QRkernels 1D with elty = $elty" for elty in [ Float32, Float16, Float64] 
    n=32
    backend=KernelAbstractions.get_backend(CUDA.randn(2))
    T=elty
    myrange=(n,1)
    t= KernelAbstractions.zeros(backend, T, n)
    
    @testset "single tile QR" begin
    
        a=rand!(allocate(backend, T,n, n))
        acopy=copy(a)
        
        QR_unsafe_kernel!(backend,n)(acopy,t, ndrange=myrange)
        @test Array(acopy) ≈ qr(Array(a)).factors

        b=rand!(allocate(backend, T,n, n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel!(backend,n)(bcopy, acopy,t, true, ndrange=myrange )
        @test Array(bcopy) ≈ qr(Array(a)).Q'*Array(b)

        bcopy=copy(b)
        applyQorQt_unsafe_kernel!(backend,n)(bcopy, acopy,t, false, ndrange=myrange )
        @test Array(bcopy) ≈ qr(Array(a)).Q*Array(b)

        bcopy=copy(b)
        applyQorQt_unsafe_kernel!(backend,n)(bcopy', acopy,t, true, ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array(a)).Q

        bcopy=copy(b)
        applyQorQt_unsafe_kernel!(backend,n)(bcopy', acopy,t, false, ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array(a)).Q'

    end

    @testset "double tile QR" begin 
        a=rand!(allocate(backend, T,2n, n))
        acopy=copy(a)
        QR_unsafe_kernel2!(backend,n)(view(acopy,1:n,1:n),view(acopy,n+1:2n,1:n), t , ndrange=myrange)
        tril(view(acopy,1:n,1:n),-1) ≈ tril(view(a,1:n,1:n),-1)
        @test  [triu(Array(view(acopy,1:n,1:n)));Array(view(acopy,n+1:2n,1:n))] ≈ qr(Array([triu(view(a,1:n,1:n));view(a,n+1:2n,1:n)])).factors
        
        b=rand!(allocate(backend, T,2n, n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2!(backend,n)(view(bcopy,1:n,1:n),view(bcopy,n+1:2n,1:n),view(acopy,n+1:2n,1:n) , t,true, ndrange=myrange )
        @test Array(bcopy) ≈ qr(Array([triu(Array(view(a,1:n,1:n)));Array(view(a,n+1:2n,1:n))])).Q'*Array(b)

        b=rand!(allocate(backend, T,2n, n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2!(backend,n)(view(bcopy,1:n,1:n),view(bcopy,n+1:2n,1:n),view(acopy,n+1:2n,1:n) , t,false,ndrange=myrange )
        @test Array(bcopy) ≈ qr([triu(Array(view(a,1:n,1:n)));Array(view(a,n+1:2n,1:n))]).Q*Array(b)

        b=rand!(allocate(backend, T,n, 2n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2!(backend,n)(view(bcopy,1:n,1:n)',view(bcopy,1:n, n+1:2n)',view(acopy,n+1:2n,1:n) , t,true,ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array([triu(view(a,1:n,1:n));view(a,n+1:2n,1:n)])).Q

        b=rand!(allocate(backend, T,n, 2n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2!(backend,n)(view(bcopy,1:n,1:n)',view(bcopy,1:n, n+1:2n)',view(acopy,n+1:2n,1:n) , t,false,ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array([triu(view(a,1:n,1:n));view(a,n+1:2n,1:n)])).Q'
   
    end

    @testset "block QR" begin 
        
        a=rand!(allocate(backend, T,n, n))
        acopy=copy(a)
        t= KernelAbstractions.zeros(backend, T, n)
        QR_unsafe_kernel!(backend,n)(acopy,t, ndrange=myrange)

        b=rand!(allocate(backend, T,n, 5n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel!(backend,n)(bcopy, acopy,t, true, ndrange=(5n,1) )
        @test Array(bcopy) ≈ qr(Array(a)).Q'*Array(b)

        a=rand!(allocate(backend, T,2n, n))
        acopy=copy(a)
        QR_unsafe_kernel2!(backend,n)(view(acopy,1:n,1:n),view(acopy,n+1:2n,1:n), t , ndrange=myrange)
   
        b=rand!(allocate(backend, T,2n, 5n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2!(backend,n)(view(bcopy,1:n,1:n),view(bcopy,n+1:2n,1:n),view(acopy,n+1:2n,1:n) , t,true,ndrange=(5n,1) )
        @test Array(bcopy) ≈ qr(Array([triu(Array(view(a,1:n,1:n)));Array(view(a,n+1:2n,1:n))])).Q'*Array(b)

    end
end


@testset "QRkernels 2D with elty = $elty" for elty in [ Float32, Float16, Float64] 
    n=32
    backend=KernelAbstractions.get_backend(CUDA.randn(2))
    T=elty
    myrange=(n,n)
    t= KernelAbstractions.zeros(backend, T, n)
    
    @testset "single tile QR" begin
    
        a=rand!(allocate(backend, T,n, n))
        acopy=copy(a)
        
        QR_unsafe_kernel_2d!(backend,myrange)(acopy,t, ndrange=myrange)
        @test Array(acopy) ≈ qr(Array(a)).factors

        b=rand!(allocate(backend, T,n, n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel_2d!(backend,myrange)(bcopy, acopy,t, true, ndrange=myrange )
        @test Array(bcopy) ≈ qr(Array(a)).Q'*Array(b)

        bcopy=copy(b)
        applyQorQt_unsafe_kernel_2d!(backend,myrange)(bcopy, acopy,t, false, ndrange=myrange )
        @test Array(bcopy) ≈ qr(Array(a)).Q*Array(b)

        bcopy=copy(b)
        applyQorQt_unsafe_kernel_2d!(backend,myrange)(bcopy', acopy,t, true, ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array(a)).Q

        bcopy=copy(b)
        applyQorQt_unsafe_kernel_2d!(backend,myrange)(bcopy', acopy,t, false, ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array(a)).Q'

    end

    @testset "double tile QR" begin 
        a=rand!(allocate(backend, T,2n, n))
        acopy=copy(a)
        QR_unsafe_kernel2_2d!(backend,myrange)(view(acopy,1:n,1:n),view(acopy,n+1:2n,1:n), t , ndrange=myrange)
        tril(view(acopy,1:n,1:n),-1) ≈ tril(view(a,1:n,1:n),-1)
        @test  [triu(Array(view(acopy,1:n,1:n)));Array(view(acopy,n+1:2n,1:n))] ≈ qr(Array([triu(view(a,1:n,1:n));view(a,n+1:2n,1:n)])).factors
        
        b=rand!(allocate(backend, T,2n, n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2_2d!(backend,myrange)(view(bcopy,1:n,1:n),view(bcopy,n+1:2n,1:n),view(acopy,n+1:2n,1:n) , t,true, ndrange=myrange )
        @test Array(bcopy) ≈ qr(Array([triu(Array(view(a,1:n,1:n)));Array(view(a,n+1:2n,1:n))])).Q'*Array(b)

        b=rand!(allocate(backend, T,2n, n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2_2d!(backend,myrange)(view(bcopy,1:n,1:n),view(bcopy,n+1:2n,1:n),view(acopy,n+1:2n,1:n) , t,false,ndrange=myrange )
        @test Array(bcopy) ≈ qr([triu(Array(view(a,1:n,1:n)));Array(view(a,n+1:2n,1:n))]).Q*Array(b)

        b=rand!(allocate(backend, T,n, 2n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2_2d!(backend,myrange)(view(bcopy,1:n,1:n)',view(bcopy,1:n, n+1:2n)',view(acopy,n+1:2n,1:n) , t,true,ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array([triu(view(a,1:n,1:n));view(a,n+1:2n,1:n)])).Q

        b=rand!(allocate(backend, T,n, 2n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2_2d!(backend,myrange)(view(bcopy,1:n,1:n)',view(bcopy,1:n, n+1:2n)',view(acopy,n+1:2n,1:n) , t,false,ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array([triu(view(a,1:n,1:n));view(a,n+1:2n,1:n)])).Q'
   
    end

    @testset "block QR" begin 
        myredrange=(n,Int(n/2))
        myextrange=(5n,Int(n/2))

        a=rand!(allocate(backend, T,n, n))
        acopy=copy(a)
        t= KernelAbstractions.zeros(backend, T, n)
        QR_unsafe_kernel_2d!(backend,myrange)(acopy,t, ndrange=myrange)

        b=rand!(allocate(backend, T,n, 5n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel_2d!(backend,myredrange)(bcopy, acopy,t, true, ndrange=myextrange )
        @test Array(bcopy) ≈ qr(Array(a)).Q'*Array(b)

        a=rand!(allocate(backend, T,2n, n))
        acopy=copy(a)
        QR_unsafe_kernel2_2d!(backend,myrange)(view(acopy,1:n,1:n),view(acopy,n+1:2n,1:n), t , ndrange=myrange)
   
        b=rand!(allocate(backend, T,2n, 5n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2_2d!(backend,myredrange)(view(bcopy,1:n,1:n),view(bcopy,n+1:2n,1:n),view(acopy,n+1:2n,1:n) , t,true,ndrange=myextrange)
        @test Array(bcopy) ≈ qr(Array([triu(Array(view(a,1:n,1:n)));Array(view(a,n+1:2n,1:n))])).Q'*Array(b)

    end

end

