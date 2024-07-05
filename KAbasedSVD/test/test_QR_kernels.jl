include(joinpath("..","src","QRkernels.jl"))

@testset "QRkernels with elty = $elty" for elty in [ Float32, Float16] #not supported for other elementtypes
    n=4
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
        applyQt_unsafe_kernel!(backend,n)(bcopy, acopy,t, ndrange=myrange )
        @test Array(bcopy) ≈ qr(Array(a)).Q'*Array(b)

        bcopy=copy(b)
        applyQ_unsafe_kernel!(backend,n)(bcopy, acopy,t, ndrange=myrange )
        @test Array(bcopy) ≈ qr(Array(a)).Q*Array(b)

        bcopy=copy(b)
        applyQt_unsafe_kernel!(backend,n)(bcopy', acopy,t, ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array(a)).Q

        bcopy=copy(b)
        applyQ_unsafe_kernel!(backend,n)(bcopy', acopy,t, ndrange=myrange )
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
        applyQt_unsafe_kernel2!(backend,n)(view(bcopy,1:n,1:n),view(bcopy,n+1:2n,1:n),view(acopy,n+1:2n,1:n) , t,ndrange=myrange )
        @test Array(bcopy) ≈ qr(Array([triu(Array(view(a,1:n,1:n)));Array(view(a,n+1:2n,1:n))])).Q'*Array(b)

        b=rand!(allocate(backend, T,2n, n))
        bcopy=copy(b)
        applyQ_unsafe_kernel2!(backend,n)(view(bcopy,1:n,1:n),view(bcopy,n+1:2n,1:n),view(acopy,n+1:2n,1:n) , t,ndrange=myrange )
        @test Array(bcopy) ≈ qr([triu(Array(view(a,1:n,1:n)));Array(view(a,n+1:2n,1:n))]).Q*Array(b)

        b=rand!(allocate(backend, T,n, 2n))
        bcopy=copy(b)
        applyQt_unsafe_kernel2!(backend,n)(view(bcopy,1:n,1:n)',view(bcopy,1:n, n+1:2n)',view(acopy,n+1:2n,1:n) , t,ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array([triu(view(a,1:n,1:n));view(a,n+1:2n,1:n)])).Q

        b=rand!(allocate(backend, T,n, 2n))
        bcopy=copy(b)
        applyQ_unsafe_kernel2!(backend,n)(view(bcopy,1:n,1:n)',view(bcopy,1:n, n+1:2n)',view(acopy,n+1:2n,1:n) , t,ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array([triu(view(a,1:n,1:n));view(a,n+1:2n,1:n)])).Q'
   
    end

    @testset "block QR" begin 
        
        a=rand!(allocate(backend, T,n, n))
        acopy=copy(a)
        t= KernelAbstractions.zeros(backend, T, n)
        QR_unsafe_kernel!(backend,n)(acopy,t, ndrange=myrange)

        b=rand!(allocate(backend, T,n, 5n))
        bcopy=copy(b)
        applyQt_unsafe_kernel_block!(backend,n)(bcopy, acopy,t, ndrange=(5n,1) )
        @test Array(bcopy) ≈ qr(Array(a)).Q'*Array(b)

        a=rand!(allocate(backend, T,2n, n))
        acopy=copy(a)
        QR_unsafe_kernel2!(backend,n)(view(acopy,1:n,1:n),view(acopy,n+1:2n,1:n), t , ndrange=myrange)
   
        b=rand!(allocate(backend, T,2n, 5n))
        bcopy=copy(b)
        applyQt_unsafe_kernel2_block!(backend,n)(view(bcopy,1:n,1:n),view(bcopy,n+1:2n,1:n),view(acopy,n+1:2n,1:n) , t,ndrange=(5n,1) )
        @test Array(bcopy) ≈ qr(Array([triu(Array(view(a,1:n,1:n)));Array(view(a,n+1:2n,1:n))])).Q'*Array(b)


    end
end

