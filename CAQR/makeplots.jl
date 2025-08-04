using Plots, StatsPlots
nvals=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
timings=[ 5.16E-07	2.12E-06	8.61E-06
1.03E-06	4.24E-06	1.72E-05
2.07E-06	8.48E-06	3.44E-05
4.13E-06	1.70E-05	6.88E-05
8.26E-06	3.39E-05	1.38E-04
1.65E-05	6.79E-05	2.75E-04
3.30E-05	1.36E-04	5.51E-04
6.61E-05	2.71E-04	1.10E-03
1.32E-04	5.43E-04	2.20E-03

]
timings_tc=[4.93E-07	4.42E-06	4.48E-05
5.32E-07	4.57E-06	4.54E-05
6.11E-07	4.89E-06	4.66E-05
7.68E-07	5.52E-06	4.91E-05
1.08E-06	6.77E-06	5.42E-05
1.71E-06	9.28E-06	6.42E-05
2.97E-06	1.43E-05	8.43E-05
5.48E-06	2.44E-05	1.24E-04
1.05E-05	4.44E-05	2.05E-04
]
#timings./=sum(timings,dims=2)
xaxis=nvals
xaxist=string.(nvals)
yaxis=[1e-6,1e-5,1e-4, 1e-3]
yaxist=["1us","10us","100 us","1ms"]
plot(nvals, timings, labels= ["tilesize 16" "tilesize 32" "tilesize 64"] , lw=2, markershape=:circle, markerstrokewidth=0, markersize=3, color=[1 2 3])#, linestyle=:dash)
plot!(xaxis=:log2,  yaxis=:log10,legend=:topleft,  yticks=(yaxis,yaxist), xticks=(xaxis,xaxist), xlabel= "matrix length n", ylabel= "Theoretical computation time", title="", dpi=1000)
plot!(nvals, timings_tc, labels= "" , lw=2, markershape=:circle, markerstrokewidth=0, markersize=3, linestyle=:dash, color=[1 2 3])
plot!([64,64],[1e-4,1e-4], lw=2, color=:black, label="non-tensor core")
plot!([64,64],[1e-4,1e-4], lw=2, color=:black, label="tensor core", linestyle=:dash)
savefig("tctime.png")

#groupedbar(string.(nvals), timings, bar_position = :stack, yticks=([0,0.5,1],["0%","50%","100%"]), bar_width=0.7,labels= ["kernel launches" "QR1" "QR2" "Qmul1" "Qmul2"] , xlabel= "matrix size nxn", ylabel= "Execution time")
