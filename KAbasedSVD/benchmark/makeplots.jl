
using Plots, BSON
nvals=2 .^(6:9)
timings=[  0.172319  0.0014909
0.715925  0.0070021
3.12396   0.0297446
21.6781    0.135321]

xaxis=nvals
xaxist=string.(nvals)
yaxis=[0.001, 0.01, 0.1, 1,10, 60 ]
yaxist=["1 ms","10ms","0.1s","1s", "10s", "1min"]
plot(nvals, timings, labels= ["with Dagger" " without Dagger"] , lw=2, markershape=:circle, markerstrokewidth=0, markersize=3)
plot!(xaxis=:log2,  yaxis=:log10,legend=:topleft,  yticks=(yaxis,yaxist), xticks=(xaxis,xaxist), xlabel= "matrix size nxn", ylabel= "Execution time", title="Bulge chasing for bandwith 32", dpi=1000)

tilesize=32
if true
    BSON.@load "finaldata_hpc.bson" timings_elty_QR timings_elty_SVD timings_blocksize_QR timings_blocksize_SVD timings_OOC_QR timings_OOC_SVD timings_bulge timings_ext
    BSON.@load "lapack_hpc.bson" timings
    timings_OOC_QR[8,1]=2.231719351
    timings_blocksize_QR[8,1]=2.231719351
    
    timings_OOC_SVD[8,1]=6.49913549
    timings_bulge[8,1]=6.49913549
    timings_blocksize_SVD[8,1]=6.49913549
    
    timings_ext[9,:]*=3.2
    title="Supercomputer"
    hpc="hpc"
    no_tiles_values =  2 .^(1:8)
    yaxis=10. .^(-4:1)
    yaxist=["0.1 ms", "1 ms", "10 ms", "0.1s", "1s", "10s"]
    cutimings=[timings_OOC_QR[:,1] timings_OOC_SVD[:,1];5.928973353 151.425932425 ]  
    timings_ext[9,:]=reverse(timings_ext[9,:])
else
    BSON.@load "finaldata.bson" timings_elty_QR timings_elty_SVD timings_blocksize_QR timings_blocksize_SVD timings_OOC_QR timings_OOC_SVD timings_bulge timings_ext
    BSON.@load "longtimings42.bson" timings
    timings_ext[end,:]=timings[4,:]
    BSON.@load "lapack6.bson" timings
    timings_bulge[4,3]=0.48
    timings_ext[8,:]*=3.2
    timings[7,1]/=10
    timings_OOC_QR[7,1]=0.127749
    timings_blocksize_QR[7,1]=0.127749
    timings_OOC_SVD[7,1]=4.0356196
    timings_bulge[7,1]=4.0356196
    timings_blocksize_SVD[7,1]=4.0356196
    title="Consumer laptop"
    hpc=""
    no_tiles_values =  2 .^(1:7)
    yaxis=[10. .^(-4:1);60]
    yaxist=["0.1 ms", "1 ms", "10 ms", "0.1s", "1s", "10s","1min"]
    timings[7,:] .= [   2.16009; 767.681 ]
    timings=[timings; 13.9672463 6354.0205557 ]
    cutimings=[timings_OOC_QR[:,1] timings_OOC_SVD[:,1] ; 0.7923836 14.9090194; ]
end
no_tiles_values_long =  2 .^(1:11)
matrixsizes=tilesize.*no_tiles_values
eltypes=["Float16" "Float32" "Float64" "ComplexF32"]

yaxis=[10. .^(-4:1);60]
yaxist=["0.1 ms", "1 ms", "10 ms", "0.1s", "1s", "10s","1min"]
plot(matrixsizes, timings_OOC_SVD[:,4], labels= "KA-native pure-OOC banddiag" , lw=2, markershape=:circle, markerstrokewidth=0, markersize=3, color=["lightskyblue"])
plot!(matrixsizes, timings_bulge[:,[2;3;4]], labels= ["KA-native in-GPU banddiag" "Bulgechasing on CPU" "Diagonalization on CPU"  ], lw=2, markershape=:circle, markerstrokewidth=0, markersize=3, color=[1 2 3])
plot!(xaxis=:log2,  yaxis=:log10,legend=:topleft,  yticks=(yaxis,yaxist), xticks=(matrixsizes, string.(matrixsizes)), xlabel= "matrix size nxn", ylabel= "Execution time", dpi=1000)
savefig("SVD_comp"*hpc*".png")

xaxis=(no_tiles_values_long*tilesize)
xaxist=[string.(xaxis[1:5]);"2k";"4k"; "8k"; "16k";"32k";"64k"]
yaxis=[0.001, 0.01, 0.1, 1,10, 60, 600, 3600, ]
yaxist=["1 ms","10ms","0.1s","1s", "10s", "1min", "10min", "1h"]

 plot(xaxis[1:size(cutimings,1)], cutimings[:,1], labels= "CUSOLVER SVD", lw=2, markershape=:circle, markerstrokewidth=0, markersize=3)
 plot!(xaxis[1:size(timings,1)], timings[:,2], labels="LAPACK CPU SVD",lw=2, markershape=:circle, markerstrokewidth=0, markersize=3 )
 plot!(xaxis[1:size(timings_ext, 1)],timings_ext[:,2], label= "KA-native OOC banddiag" ,lw=2, markershape=:circle, markerstrokewidth=0, markersize=3)
 plot!(xaxis=:log2,  yaxis=:log10,legend=:topleft, xticks=(xaxis, xaxist), yticks=(yaxis,yaxist),xlabel= "matrix size nxn", ylabel= "Execution time", dpi=1000)
 savefig("SVD_ooc"*hpc*".png")

 plot(xaxis[1:size(cutimings,1)], cutimings[:,2], labels= "CUSOLVER QR" , lw=2, markershape=:circle, markerstrokewidth=0, markersize=3)
 plot!(xaxis[1:size(timings,1)], timings[:,1], labels="LAPACK CPU QR",lw=2, markershape=:circle, markerstrokewidth=0, markersize=3 )
 plot!(xaxis[1:size(timings_ext, 1)],timings_ext[:,1], label= "KA-native OOC QR" ,lw=2, markershape=:circle, markerstrokewidth=0, markersize=3)
 plot!(xaxis=:log2,  yaxis=:log10,legend=:topleft, xticks=(xaxis, xaxist), yticks=(yaxis,yaxist),xlabel= "matrix size nxn", ylabel= "Execution time", dpi=1000)
 savefig("QR_ooc"*hpc*".png")

plot(xaxis,timings_ext[:,1], label= "KA-native OOC QR" ,lw=2, markershape=:circle, markerstrokewidth=0, markersize=3)
plot!(xaxis,timings_ext[:,2], label= "KA-native OOC banddiag" ,lw=2, markershape=:circle, markerstrokewidth=0, markersize=3)
plot!(xaxis=:log2,  yaxis=:log10,legend=:topleft, xticks=(xaxis, xaxist), yticks=(yaxis,yaxist),xlabel= "matrix size nxn", ylabel= "Execution time", dpi=1000)
savefig("QRSVD_ooc"*hpc*".png")


 yaxis=10. .^(-3:1)
 yaxist=[ "1 ms", "10 ms", "0.1s", "1s", "10s"]
plot(matrixsizes, timings_elty_QR[:,2:2:end], labels= eltypes.*" KA-native OOC QR", lw=2, color=[1 2 3 4] )
plot!(matrixsizes, timings_elty_QR[:,3:2:end], labels= (eltypes.*" CUSOLVER QR")[:,2:4], lw=2,  color=[2 3 4] , linestyle=:dash)
 plot!(xaxis=:log2,  yaxis=:log10,legend=:topleft, xticks=(xaxis, string.(xaxis)), yticks=(yaxis,yaxist),xlabel= "matrix size nxn", ylabel= "Execution time", dpi=1000)
 savefig("QR_eltypes"*hpc*".png")

plot(matrixsizes, timings_elty_SVD[:,2:2:end], labels= eltypes.*" KA-native OOC banddiag", lw=2,  color=[1 2 3 4] )
plot!(matrixsizes, timings_elty_SVD[:,3:2:end], labels= (eltypes.*" CUSOLVER SVD")[:,2:4], lw=2, color=[2 3 4] , linestyle=:dash)
 plot!(xaxis=:log2,  yaxis=:log10,legend=:topleft, xticks=(xaxis, string.(xaxis)), yticks=(yaxis,yaxist),xlabel= "matrix size nxn", ylabel= "Execution time", dpi=1000)
 savefig("SVD_eltypes"*hpc*".png")

 tiletests= ("thread-block size ".*["(32, 32)" "(32,16)" "(32, 8)" "(32,4)" "(32, 2)" "(32,1)" "(16,16)"])
 yaxis=10. .^(-3:1)
 yaxist=[ "1 ms", "10 ms", "0.1s", "1s", "10s"]
plot(matrixsizes, timings_blocksize_SVD[:,[2:6;8;9]], labels= tiletests, lw=1.5)
plot!(xaxis=:log2,  yaxis=:log10,legend=:topleft, xticks=(xaxis, string.(xaxis)), yticks=(yaxis,yaxist),xlabel= "matrix size nxn", ylabel= "Execution time", title="SVD", dpi=1000)
 savefig("SVD_blocksize"*hpc*".png")


plot(matrixsizes, timings_blocksize_QR[:,[2:6;8;9]], labels= tiletests, lw=1.5)
plot!(xaxis=:log2,  yaxis=:log10,legend=:topleft, xticks=(xaxis, string.(xaxis)), yticks=(yaxis,yaxist),xlabel= "matrix size nxn", ylabel= "Execution time", title="QR", dpi=1000)
 savefig("QR_blocksize"*hpc*".png")