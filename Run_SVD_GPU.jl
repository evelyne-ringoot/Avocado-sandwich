pwd()
cd(raw"C:\Users\evely\OneDrive\Documents\CSE_MIT\Avocado")

include("SVD_GPU.jl")

using Plots, BenchmarkTools
n=6
x=2
y=3
A=float.(rand(1:10,n,n))
A_svd=svdvals(A)
Adiag = BandBidiagonal!(A,x,x,y,y, true, true, true)
Adiag_svd=svdvals(Adiag)
norm(A_svd-Adiag_svd,Inf)

x_values=[4, 10, 17, 30, 50]

for x in x_values
    n=x*x
    A=float.(rand(1:10,n,n))
    A2=deepcopy(A)
    A3=deepcopy(A)
    A4=deepcopy(A)
    Adiag = BandBidiagonal!(A,x,x,x,x)
end
A_svd=svdvals(A)
Adiag_svd=svdvals(Adiag)
norm(A_svd-Adiag_svd,Inf)
t1 = CUDA.@elapsed BandBidiagonal!(A2,x,x,x,x);
t2 = @belapse BandBidiagonal_CPU!(A3,x,x,x,x, true);
t3 = @belapsed BandBidiagonal_CPU!(A4,x,x,x,x, false);

n=200
x=50
A=float.(rand(1:10,n,n))
Acu = A |> cu
A2=deepcopy(A)
A3=deepcopy(A)
t1 = CUDA.@time BandBidiagonal!(A,x,x,4,4)
t2 = @btime BandBidiagonal_CPU!(A2,x,x,4,4)
t3 = CUDA.@time svdvals!(Acu);
t4 = @btime svdvals!(A3);

################################################################################################
################### Random other stuff ####################################################
###########################################################################################

n=500
A=CUDA.randn(n,n)
A=randn(n,n)

function qr_notinplace(A)
    qr(A)
    return;
end
function qr_inplace!(A)
    qr!(A) #why is it not possible to provide QR
    return;
end

#CUDA profiler
t1= @btime qr_notinplace(A) setup=begin A=rand(5000,500) end
t2= @btime qr_inplace!(A) setup=begin A=rand(5000,500) end ###how to make this faster

A=CUDA.ones(500,500)
A[1:10,1:10]= 2*ones(10,10)

timings_cpu=[]
timings_gpu=[]
for n in [round(Int,10^i) for i=1:0.5:3.5]
    println(n)
    A=rand(n,n)
    t= @belapsed qr($A)
    push!(timings_cpu,t)
    B=A|>cu
    t= @belapsed qr($B)
    push!(timings_gpu,t)
end

plot(ns,timings_cpu, xlabel="Matrix size n", ylabel="QR calc time (s)", xaxis=:log10, yaxis=:log10, labels="CPU", )
plot!(ns,timings_gpu, xlabel="Matrix size n", ylabel="QR calc time (s)", xaxis=:log10, yaxis=:log10, labels="GPU")

n=2000
a=CUDA.randn(n,n)

function assign_values_withoutalloc(a)
    for i in 1:200
        a[i.+(5:6),i*10] .= i*10 .+(5:6)
    end
end

function assign_values_withalloc(a)
    for i in 1:200
        xval=i.+(5:6)
        yval=i*10
        assignval= i*10 .+(5:6)
        a[xval,yval] .= assignval
    end
end

t1 = CUDA.@time assign_values_withoutalloc(a)
t2= CUDA.@time  assign_values_withalloc(a)

x_size=6
y_size=5


function compute_gpu(outputs,inputs)
    @sync for i in eachindex(inputs)
        Threads.@spawn begin
            #do_svd!(outputs[i], inputs[i])
            CUDA.synchronize()
        end
        end
    return outputs
end

#benchmark svd

timings_cpu=[]
timings_gpu=[]
n_vals=[  10,32,100,316, 1000, 3162, 5000]
for n in [5000]
    A=rand(n,n)
    B=A|>cu
    t= @belapsed svd!(A)
    push!(timings_cpu,t)
    t= CUDA.@elapsed CUDA.CUSOLVER.gesvdj!('V',1, B, tol=Float32(1e-5))
    push!(timings_gpu,t)
end

plot(n_vals,timings_cpu, xlabel="Matrix size n", ylabel="svd calc time (s)", xaxis=:log10, yaxis=:log10, labels="CPU", )
plot!(n_vals,timings_gpu, xlabel="Matrix size n", ylabel="svd calc time (s)", xaxis=:log10, yaxis=:log10, labels="GPU")

1+1