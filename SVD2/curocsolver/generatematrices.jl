using LinearAlgebra

sizes = [128,256,512,1024,2048,4096,8192]
for n in sizes
    svdvals= [((1:-1/(n-1):0))...]
    unit1= qr!(randn(Float32,n,n)).Q
    unit2= qr!(randn(Float32,n,n)).Q
    a= unit1*diagm(svdvals)*unit2
    open("data_"*string(n)*".bin", "w") do f
        write(f, a)
    end
    open("data_"*string(n)*"_svd.bin", "w") do f
        write(f, svdvals)
    end
end