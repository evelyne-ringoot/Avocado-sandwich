using LinearAlgebra, Statistics, Plots, BSON

function generate(n)
    mymatrix=randn(n,n)
    return (mymatrix+mymatrix')
    #return mymatrix
end    

function generate_MOUprocess_covar(numtrials, matrixsize, numtimestep, timelength, kthsvdval)
    n=matrixsize
    nt=numtimestep
    val=kthsvdval
    deltat=timelength/nt
    t=numtrials
    matrix_val=zeros(nt, t)
    for trial in 1:t
        mymatrices= [generate(n) for i=1:nt];
        mymatrices[2:end].*=sqrt(deltat);
        mymatrices[1]./=sqrt(2);
        mynewmatrices=[zeros(n,n) for i=1:nt];
        mynewmatrices[1]=mymatrices[1];
        matrixcopy=copy(mynewmatrices[1]);
        _, d, e, _, _ = LAPACK.gebrd!(matrixcopy)
        matrix_val[1,trial]=d[val]
        for i in 2:nt
            mynewmatrices[i]=mynewmatrices[i-1]*(1-deltat)+mymatrices[i]
            matrixcopy=copy(mynewmatrices[i])
            _, d, e, _, _ = LAPACK.gebrd!(matrixcopy)
            matrix_val[i,trial]=d[val]
        end
    end
    matrix_val=matrix_val.^2
    matrix_val2=(matrix_val .- (n-val+1))./sqrt(n-val+1)
    covar_val=matrix_val2.*matrix_val2[1,:]'
    return mean(matrix_val, dims=2), mean(covar_val, dims=2), 0:deltat:(nt-1)*deltat
end

#average over trials, shows evolution over time

avals=[]
bvals=[]
cvals=[]

numtrials=[100,10,1000,100,100,100,100,100,100]
matrixsizes=[100,100,100,25,400,100,100,100,100 ]
numtimestep=[100,100,100, 100,100,1000,10, 100,100]
kthsvdval2=[10,10,10,10,10,10,10]
kthsvdval=[1,1,1,1,1,1,1]
labels=["t=100,n=100, nt=100", "t=10", "t=1000", "n=25", "n=400", "nt=1000", "nt=10", "k=1", "k=10"]

for i in 1:7
    a,b,c = generate_MOUprocess_covar(numtrials[i],matrixsizes[i],numtimestep[i],0.05,kthsvdval[i])
    push!(avals, a)
    push!(bvals, b)
    push!(cvals, c)
end

avalsold2=copy(avals)
bvalsold2=copy(bvals)
cvalsold2=copy(cvals)

f(x) = 2* exp(-2x)
f2(x) = 2 * exp(-100x)
avalsold, bvalsold, cvalsold = BSON.load("rmt1data.bson")[:a1]
avalsold2, bvalsold2, cvalsold2 = BSON.load("rmt2data.bson")[:a2]

a1=[avalsold, bvalsold, cvalsold]
a2=[avalsold2, bvalsold2, cvalsold2]



p=plot()
for i in [1 6 7]
    p=plot!(cvalsold[i],avalsold[i].-(matrixsizes[i]-kthsvdval[i]+1),label=labels[i])
end

p=plot!(title="Expectation value of a_k^2 - (n-k+1) ")
display(p)

p=plot()
for i in [1 4 5]
    p=plot!(cvalsold2[i],bvalsold2[i],label=labels[i])
end


p=plot!(title="Covariance value of a_k^2(t) and  a_k^2(0), \n normalized -(n-k+1)/sqrt(n-k+1)  ")
plot!(cvalsold2[1],f2.(cvalsold2[1]), label="Exponential approx (2e^-100x)")
display(p)
plot!(legend=:top, xlims=(0, 0.02))

#

a=randn(5,5)
_, d, e, _, _ = LAPACK.gebrd!(copy(a))
LinearAlgebra.Bidiagonal(d,e[1:end-1], :U)
_, d, e, _, _ = LAPACK.gebrd!(copy(a'));
LinearAlgebra.Bidiagonal(d,e[1:end-1], :U)
