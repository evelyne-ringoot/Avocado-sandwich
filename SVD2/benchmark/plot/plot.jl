using Plots, Colors, StatsPlots


green1=colorant"#74B800" #RTX4060
green2=colorant"#48994D" #V100
green3=colorant"#0B8462" #A100
green4=colorant"#426155" #H100
red=colorant"#A30000" #MI250
grey=colorant"#B3BBBC" #MI250
blue=colorant"#0068B5" #MI250
#2,7,9
#12,13,14

#PLOT 1

xaxis=[64,128,256,512,1024,2048,4096,1024*8,1024*16,1024*32,1024*64]
xaxist=[string.(xaxis[1:5]);"2k";"4k"; "8k";"16k";"32k";"65k"]
yaxis=[1e-3,1,1000]
yaxist=["1 ms","1s","1000s" ]
yaxis=[0.1,0.33,1,3,10]
yaxist=(["SLATE\n10 times\nfaster" "" "equal\nruntime" "" "Unified API\n10 times\nfaster" ])


kaslate=[8.67	2.48	13.42	21.82
28.06	1.32	6.87	9.06
64.40	0.83	2.50	3.79
784.46	1.06	1.73	2.25
1490.60	1.74	1.95	2.22
2224.13	3.17	3.47	2.85
1129.80	3.84	2.42	2.79
617.89	5.10	1.73	2.21
384.35	5.65	1.62	1.84
0.0001 5.54	1.86	1.72
0.0001 3.69	1.83	1.40]

p1=groupedbar(
    xaxist,
    kaslate,
    bar_width = 0.8,
    label = ["NVIDIA RTX4060" "NVIDIA A100" "NVIDIA H100" "AMD MI250"],
    color = [green1 green3 green4 red],
    title = "SLATE \n", ylims=(0.1,30), yaxis=:log10,
    xlabel = "\n Matrix Size (nxn)\n",
    size = (600, 300), titlefontsize=18,
    lw=0, bottom_margin = 10Plots.px, top_margin = 30Plots.px,
    legend = false,
    yticks=(yaxis,yaxist),  legend_columns = 4, 
    legend_foreground_color = :transparent, legendfontsize = 12, dpi=1200
)

p1=hline!([1 1], linestyle = :dash, color = :gray, label="", lw=2)


yaxis=[0.1,0.33,1,3,10]
yaxist=["vendor \n 10 times\n faster", "", "equal \n runtime", "","Unified API\n 10 times\n faster"]

kacuroc=[1.55	0.67	0.85	1.58	0.11
1.01	0.65	0.86	2.07	0.11
1.33	0.56	0.75	2.93	0.11
1.32	0.49	0.64	4.30	0.20
1.11	0.46	0.59	6.53	0.72
0.97	0.45	0.59	9.64	1.30
1.15	0.49	0.59	12.77	1.75
2.52	0.62	0.63	15.42	6.07
4.18	0.80	0.88	16.49	9.78]
p2= groupedbar(
    xaxist[1:9],
    kacuroc,
    bar_width = 0.8,
    label = ["NVIDIA RTX4060" "NVIDIA A100"  "NVIDIA H100" "AMD MI250" "Intel PVC"],
    color = [green1  green3 green4 red blue],
    xlabel = " Matrix Size (nxn)",title = "\n Ratio of Singular Value Runtime of \n Unified API to cuSOLVER/rocSOLVER/oneMKL",
     ylims=(0.1,30),legend=:outerbottom,
    lw=0,titlefontsize=14, size=(600,400),dpi=1200,
    yticks=(yaxis,yaxist),  yaxis=:log10, legend_columns = 3, legend_foreground_color = :transparent, legendfontsize = 10
)

p2 = hline!([1 1], linestyle = :dash, color = :gray, label="", lw=2)
savefig("cusolroc.png")
yaxis=[0.1,0.33,1,3,10]
yaxist=["MAGMA\n10 times\nfaster", "","equal\nruntime", "","Unified API\n10 times\nfaster"]

kamagma=[0.50	0.59	0.86	0.45
0.26	0.49	0.54	0.18
2.08	12.78	0.90	0.87
2.71	3.18	0.97	0.52
3.02	1.95	1.05	0.86
2.92	1.75	1.25	0.96
2.23	1.46	1.30	1.16
3.58	1.80	1.62	1.43
6.07	4.76	6.02	5.46
7.06	3.19	9.27	4.93
0.00001	1.48	6.40	4.28]
p3= groupedbar(
    xaxist,
    kamagma,
    bar_width = 0.8,
    label = ["NVIDIA RTX4060" "NVIDIA A100"  "NVIDIA H100" "AMD MI250"],
    color = [green1 green3 green4 red],
    title = "MAGMA \n",ylims=(0.1,30),
    xlabel = "\n Matrix Size (nxn)\n",
    size = (600, 300),
    lw=0, titlefontsize=18,
    legend = false, bottom_margin = 10Plots.px, top_margin = 30Plots.px,
    yticks=(yaxis,yaxist),  yaxis=:log10, legend_columns = 5, legend_foreground_color = :transparent, legendfontsize = 10
)

p3 = hline!([1 1], linestyle = :dash, color = :gray, label="", lw=2)


title = plot(title = "\n Ratio of Singular Value Runtime of Unified API to", grid = false, showaxis = false, 
    bottom_margin = -40Plots.px,titlefontsize=18, size=(1200,80))

mylegend = groupedbar(zeros(2,4), grid = false, showaxis = false, bottom_margin = -10Plots.px,legend_foreground_color = :transparent, 
legend = :outertop , label = ["NVIDIA RTX4060" "NVIDIA A100"  "NVIDIA H100" "AMD MI250"], lw=0,top_margin = 0Plots.px,
legend_columns = 4, legendfontsize = 12, size=(1200,60), 
color = [green1 green3 green4 red], framestyle = :none)
plot(title, p3, p1,mylegend,size=(1200,450),dpi=1200,bottom_margin = 30Plots.px, layout = @layout [ a{0.01h}; grid(1,2);b{0.05h} ])
 
savefig("slatemagma.png")

#PLOT3
xaxis=[64,128,256,512,1024,2048,4096,1024*8,1024*16,1024*32,1024*65,1024*128]
xaxist=[string.(xaxis[1:5]);"2k";"4k"; "8k";"16k";"32k";"65k";"132k"]

curesults=[0.00223	0.00243	0.00487
0.00505	0.00553	0.01086
0.01276	0.01437	0.0268
0.0335	0.03866	0.06951
0.09063	0.10308	0.18338
0.25654	0.27939	0.48795
0.82569	0.86393	1.51757
2.97485	3.10821	5.65647
11.28604	10.429	17.51061
45.40763	41.43568	71.3509
202.8105	195.2511	350.1136
1048.88277	0 0]
amdresults=[0.0066	0.00489
0.01766	0.01084
0.04199	0.0257
0.0991	0.06751
0.23891	0.20166
0.61818	0.68725
1.77104	2.72621
5.69218	10.80348
20.68813	53.63026
87.53255	345.19551
455.721	926.266]
appleresults=[0.13012	0.15812
0.26254	0.35032
0.53493	0.75913
1.10839	1.5295
2.26939	3.04918
4.39103	6.1153
9.67935	11.72467
28.25715	29.6767
120.17775	113.85385
859.51259	0]
intelresults=[0.00971
0.02574
0.06509
0.15425
0.36833
0.92227
2.62558
7.87507
27.91424]

xaxis=[64,128,256,512,1024,2048,4096,1024*8,1024*16, 1024*32,1024*64,1024*128]
xaxist=[string.(xaxis[1:5]);"2k";"4k"; "8k";"16k";"32k";"65k";"131k"]
yaxis=[1e-2,1e-1,1,10,100]
yaxist=["10ms","100ms","1s","10s","100s"]

logrange(x1, x2, n) = (10^y for y in range(log10(x1), log10(x2), length=n))
function createdots(x,y,n)
    outx=zeros((length(x)-1)*n+1)
    outy=zeros((length(x)-1)*n+1)
    for i in 0:(length(x)-2)
        startx=x[i+1]
        starty=y[i+1]
        endx=x[i+2]
        endy=y[i+2]
            outx[i*n+1:(i+1)*n+1 ].=logrange(startx,endx,n+1)
            outy[i*n+1:(i+1)*n+1 ].=logrange(starty,endy,n+1)

    end
    return outx,outy
end



p1=plot(xaxis[1:11], curesults[1:11,[2,3]], legend=false , lw=2, color=green1, line=[:solid :dash], title="                    H100\n",
xaxis=:log2,  yaxis=:log10, xticks=(xaxis[2:2:end], xaxist[2:2:end]), yticks=(yaxis,yaxist),xlabel= "matrix size nxn",  ylims=(2e-3,1050),
size = (350, 380),top_margin = 30Plots.px, bottom_margin = 20Plots.px, titlefontsize=16)
p1=scatter!(createdots(xaxis, curesults[:,1],3)..., markerstrokewidth=0,color=green1, markersize=3, lw=0)


p2=plot(xaxis[1:11], amdresults, legend=false , lw=2, color=red, line=[:solid :dash], title="               MI250\n",  bottom_margin = 20Plots.px,ylims=(2e-3,1050), titlefontsize=16,
xaxis=:log2,  yaxis=:log10, xticks=(xaxis[2:2:11], xaxist[2:2:11]), yticks=(yaxis,yaxist),xlabel= "matrix size nxn", size = (350, 380),  top_margin = 30Plots.px,)

p3=plot(xaxis[1:9], appleresults[1:9,2], legend=false , lw=2, color=grey, line=[:solid], title="            M1\n", bottom_margin = 20Plots.px, titlefontsize=16,
xaxis=:log2,  yaxis=:log10, xticks=(xaxis[2:2:end], xaxist[2:2:end]), yticks=(yaxis,yaxist),xlabel= "matrix size nxn",  ylims=(2e-3,1050),
size = (350, 380),top_margin = 30Plots.px, xlims=(64,32768))
p3=scatter!(createdots(xaxis[1:10], appleresults[:,1],4)..., markerstrokewidth=0,color=grey, markersize=3, lw=0)
p4=plot(xaxis[1:9], intelresults[1:9], legend=false , lw=2, color=blue, line=[:solid], title="         PVC\n", bottom_margin = 20Plots.px, titlefontsize=16,
xaxis=:log2,  yaxis=:log10, xticks=(xaxis[2:2:end], xaxist[2:2:end]), yticks=(yaxis,yaxist),xlabel= "matrix size nxn",  ylims=(2e-3,1050),
size = (350, 380),top_margin = 30Plots.px)

title = plot(title = "\n Runtime Unified API for Singular Values", grid = false, showaxis = false, bottom_margin = -20Plots.px)
mylegend = plot(zeros(2,2), legend = :top , label = [ "FP32" "FP64"],color=:black, ylims=(1,1), legend_columns = 3, legend_foreground_color = :transparent, legendfontsize = 12, size=(1800,100), 
 framestyle = :none, line=[:solid :dash], bottom_margin = -20Plots.px, lw=3)
 mylegend = scatter!(zeros(2,1), legend = :top , label =  "FP16",color=:black,markerstrokewidth=0, markersize=3, lw=0, legendfontsize=10)

 plot(title,  p1,p2, p3,p4,mylegend,size=(1400,400), dpi=1200,layout = @layout [ a{0.2h};  grid(1,4);b{0.2h}  ])
 savefig("crosshardware.png")

label=["FP16 Unified" "FP32 Unified" "FP64 Unified"]



mulcuda=[0.0	-0.4
-0.2	-0.4
-0.2	-0.3
-0.2	-0.2
-0.3	-0.3
-0.5	-0.4
-0.8	-0.5
-0.2	0.1
2.1	1.2
14.1	12.7
20.1	18.9]

mulamd=[-0.8	0.6
0.9	0.8
0.4	0.2
0.2	0.4
0.1	0.2
0.0	0.5
1.5	1.8
6.7	6.2
15.2	11.5
37.3	27.5
58.4	43.6]

tilevarcu=[-60.9	-6.6
-45.1	-6.0
-37.9	-3.6
-23.4	3.2
1.1	10.9
16.5	14.0
31.6	17.8
45.9	23.5
42.7	20.9
47.6	25.0
49.4	31.1]
tilevaramd=[-50.7	-35.6
-74.7	-36.8
-79.0	-34.8
-64.3	-25.9
-33.7	-11.8
-3.3	0.9
20.5	11.4
37.1	19.5
49.2	26.0
59.0	35.5
74.3	53.0]
xaxis=[64,128,256,512,1024,2048,4096,1024*8,1024*16, 1024*32,1024*64]
xaxist=[string.(xaxis[1:5]);"2k";"4k"; "8k";"16k";"32k";"65k"]
xaxist2=xaxist
xaxist2[2:2:end].=""
yaxis=[0,20,40]
yaxist=["0%", "20%", "40%"]

p1=groupedbar(1:11, mulcuda,legend=false, color=[4 9], bar_width = 0.8, lw=0, ylims=(-3,50), xticks=(1:11,xaxist2))
p1=plot!( xlabel= "matrix size nxn",  dpi=1000, size = (350, 350), bottom_margin = 5Plots.px,  yticks=(yaxis,yaxist))
mylegend = groupedbar(zeros(2,2), title="Relative to MULSIZE=16", lw=0,legend = :top , label = ["MULSIZE 64" "MULSIZE 32" ],color=[4 9], legend_columns = 2, legendfontsize = 12, size=(600,100), 
 framestyle = :none, bottom_margin = -30Plots.px, legend_foreground_color = :transparent)

 p2=groupedbar(1:11, mulamd,legend=false, color=[4 9], bar_width = 0.8, lw=0, ylims=(-3,50), xticks=(1:11,xaxist2))
 p2=plot!( xlabel= "matrix size nxn",  dpi=1000, size = (350, 350), bottom_margin = 5Plots.px,   yticks=(yaxis,yaxist))
title = plot(title = "\n Performance Improvement due to Parameter Variation \n \n ", grid = false, dpi=1000,
showaxis = false, bottom_margin = -20Plots.px, titlefontsize=12)

yaxis=[-50, 0,50]
yaxist=["-50%","0%", "50%"]

p3=groupedbar(1:11,xticks=(1:11,xaxist2), yticks=(yaxis,yaxist), tilevarcu,legend=false, color=[4 9], bar_width = 0.8, lw=0, ylims=(-80,80))
p3=plot!( xlabel= "matrix size nxn",  dpi=1000, size = (350, 350),bottom_margin = 5Plots.px)
mylegend2 = groupedbar(zeros(2,2), title="Relative to TILESIZE=16", lw=0,legend = :top , label = ["TILESIZE 64" "TILESIZE 32" ],color=[4 9], legend_columns = 2, legendfontsize = 12, size=(600,100), 
 framestyle = :none, bottom_margin = -30Plots.px, legend_foreground_color = :transparent)

p4=groupedbar(1:11,xticks=(1:11,xaxist2), yticks=(yaxis,yaxist), tilevaramd,legend=false, color=[4 9], bar_width = 0.8, lw=0, ylims=(-80,80))
p4=plot!( xlabel= "matrix size nxn",  dpi=1000, size = (350, 350),bottom_margin = 5Plots.px)

 plot(title, mylegend,p1, p2,mylegend2, p3,p4,size=(600,850), dpi=1200, bottom_margin = 20Plots.px, layout = @layout [ a{0.01h}; b{0.01h}; grid(1,2);c{0.01h};grid(1,2)   ])
 
 savefig("params.png")
 savefig("title.png")

