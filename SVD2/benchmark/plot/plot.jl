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

myplotchars() = plot!( dpi=1200,xgrid=false, gridalpha=0.5)


xaxis=[64,128,256,512,1024,2048,4096,1024*8,1024*16,1024*32,1024*64]
xaxist=[string.(xaxis[1:5]);"2k";"4k"; "8k";"16k";"32k";"65k"]
yaxis=[1e-3,1,1000]
yaxist=["1 ms","1s","1000s" ]
yaxis=[0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10]
yaxist=(["0.1" "" "" "" "" "1" "" "" "" "" "10" ])


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

#make gridlines darker

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
    legend_foreground_color = :transparent, legendfontsize = 12
)

p1=hline!([1 1], linestyle = :dash, color = :black, label="", lw=2)
p1=myplotchars()

#yaxis=[0.1,0.33,1,3,10]
#yaxist=["vendor \n 10 times\n faster", "", "equal \n runtime", "","Unified API\n 10 times\n faster"]
#add padding at bottom
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
    xlabel = " Matrix Size (nxn)",title = "",
     ylims=(0.1,30),legend=:outerright, bottom_margin = 20Plots.px,
    lw=0,titlefontsize=14, size=(800,270),dpi=1200,
    yticks=(yaxis,yaxist),  yaxis=:log10, legend_columns = 1, legend_foreground_color = :transparent, legendfontsize = 10
)
p2=myplotchars()
p2 = hline!([1 1], linestyle = :dash, color = :black, label="", lw=2,bottom_margin = 20Plots.px)
savefig("cusolroc.png")
#yaxis=[0.1,0.33,1,3,10]
#yaxist=["MAGMA\n10 times\nfaster", "","equal\nruntime", "","Unified API\n10 times\nfaster"]

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

p3 = hline!([1 1], linestyle = :dash, color = :black, label="", lw=2)
p3=myplotchars()

title = plot(title = "", grid = false, showaxis = false, 
    bottom_margin = -40Plots.px,titlefontsize=18, size=(1200,80))

mylegend = groupedbar(zeros(2,4), grid = false, showaxis = false, bottom_margin = -10Plots.px,legend_foreground_color = :transparent, 
legend = :outertop , label = ["NVIDIA RTX4060" "NVIDIA A100"  "NVIDIA H100" "AMD MI250"], lw=0,top_margin = 0Plots.px,
legend_columns = 1, legendfontsize = 12, size=(200,100), 
color = [green1 green3 green4 red], framestyle = :none)
plot(title, p3, p1,mylegend,size=(1200,400),dpi=1200,bottom_margin = 30Plots.px, layout = @layout [ a{0.01h}; grid(1,2){0.88w} b{0.12w} ])
 
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
p1=myplotchars()

p2=plot(xaxis[1:11], amdresults, legend=false , lw=2, color=red, line=[:solid :dash], title="               MI250\n",  bottom_margin = 20Plots.px,ylims=(2e-3,1050), titlefontsize=16,
xaxis=:log2,  yaxis=:log10, xticks=(xaxis[2:2:11], xaxist[2:2:11]), yticks=(yaxis,yaxist),xlabel= "matrix size nxn", size = (350, 380),  top_margin = 30Plots.px,)
p2=myplotchars()

p3=plot(xaxis[1:9], appleresults[1:9,2], legend=false , lw=2, color=grey, line=[:solid], title="            M1\n", bottom_margin = 20Plots.px, titlefontsize=16,
xaxis=:log2,  yaxis=:log10, xticks=(xaxis[2:2:end], xaxist[2:2:end]), yticks=(yaxis,yaxist),xlabel= "matrix size nxn",  ylims=(2e-3,1050),
size = (350, 380),top_margin = 30Plots.px, xlims=(64,32768))
p3=scatter!(createdots(xaxis[1:10], appleresults[:,1],4)..., markerstrokewidth=0,color=grey, markersize=3, lw=0)
p4=plot(xaxis[1:9], intelresults[1:9], legend=false , lw=2, color=blue, line=[:solid], title="         PVC\n", bottom_margin = 20Plots.px, titlefontsize=16,
xaxis=:log2,  yaxis=:log10, xticks=(xaxis[2:2:end], xaxist[2:2:end]), yticks=(yaxis,yaxist),xlabel= "matrix size nxn",  ylims=(2e-3,1050),
size = (350, 380),top_margin = 30Plots.px)
p3=myplotchars()

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
yaxis=[0,10,20,30,40,50]
yaxist=["0%"," ", "20%"," ", "40%"," "]

p1=groupedbar(1:11, mulcuda,legend=false, color=[4 9], bar_width = 0.8, lw=0, ylims=(-3,50), xticks=(1:11,xaxist2))
p1=plot!( xlabel= "matrix size nxn",  dpi=1000, size = (350, 350), bottom_margin = 5Plots.px,  yticks=(yaxis,yaxist))
mylegend = groupedbar(zeros(2,2), title="Relative to COLPERBLOCK=16", lw=0,legend = :top , label = ["COLPERBLOCK 64" "COLPERBLOCK 32" ],color=[4 9], legend_columns = 2, legendfontsize = 12, size=(600,100), 
 framestyle = :none, bottom_margin = -30Plots.px, legend_foreground_color = :transparent)
p1=myplotchars()

 p2=groupedbar(1:11, mulamd,legend=false, color=[4 9], bar_width = 0.8, lw=0, ylims=(-3,50), xticks=(1:11,xaxist2))
 p2=plot!( xlabel= "matrix size nxn",  dpi=1000, size = (350, 350), bottom_margin = 5Plots.px,   yticks=(yaxis,yaxist))
title = plot(title = "\n Performance Improvement due to Parameter Variation \n \n ", grid = false, dpi=1000,
showaxis = false, bottom_margin = -20Plots.px, titlefontsize=12)
p2=myplotchars()

yaxis=[-50,-25, 0,25,50,75]
yaxist=["-50%"," ","0%", " ","50%",""]

p3=groupedbar(1:11,xticks=(1:11,xaxist2), yticks=(yaxis,yaxist), tilevarcu,legend=false, color=[4 9], bar_width = 0.8, lw=0, ylims=(-80,80))
p3=plot!( xlabel= "matrix size nxn",  dpi=1000, size = (350, 350),bottom_margin = 5Plots.px)
mylegend2 = groupedbar(zeros(2,2), title="Relative to TILESIZE=16", lw=0,legend = :top , label = ["TILESIZE 64" "TILESIZE 32" ],color=[4 9], legend_columns = 2, legendfontsize = 12, size=(600,100), 
 framestyle = :none, bottom_margin = -30Plots.px, legend_foreground_color = :transparent)
p3=myplotchars()

p4=groupedbar(1:11,xticks=(1:11,xaxist2), yticks=(yaxis,yaxist), tilevaramd,legend=false, color=[4 9], bar_width = 0.8, lw=0, ylims=(-80,80))
p4=plot!( xlabel= "matrix size nxn",  dpi=1000, size = (350, 350),bottom_margin = 5Plots.px)
p4=myplotchars()
 plot(title, mylegend,p1, p2,mylegend2, p3,p4,size=(600,850), dpi=1200, bottom_margin = 20Plots.px, layout = @layout [ a{0.01h}; b{0.01h}; grid(1,2);c{0.01h};grid(1,2)   ])
 
 savefig("params.png")
 savefig("title.png")


 xaxis=[64,128,256,512,1024,2048,4096,1024*8,1024*16, 1024*32]
xaxist=[string.(xaxis[1:5]);"2k";"4k"; "8k";"16k";"32k"]
xaxist2=xaxist
xaxist2[1:2:end].=""
yaxis=[0,10,20,30,40,50]
yaxist=["0%"," ", "20%"," ", "40%"," "]


data_rtx=zeros(length(xaxis)*4+1,4)
data_rtx[2:4:end,4:-1:1].=[11.8	0.7	80.5	7.0
14.5	2.8	59.8	22.8
15.3	3.8	64.9	16.0
17.0	4.7	60.2	18.2
20.7	6.1	51.8	21.5
26.4	8.6	42.1	22.9
30.0	12.4	29.5	28.2
29.0	19.6	25.8	25.7
23.9	34.4	23.6	18.1
18.2	51.2	17.7	12.9]
mylabels=["Bidiagonal to diagonal" "Banddiagonal to bidiagonal" "Trailing submatrix update" " Panel factorization"]
p1=groupedbar(1:length(xaxis)*4+1, data_rtx,label=mylabels,  bar_width = 1, lw=0.4, fillstyle = [nothing ://  nothing nothing], fillalpha=[0 1 0.5 1], fillcolor=green1, linecolor=green1, bar_position = :stack)


data_h100=zeros(length(xaxis)*4+1,4)
data_h100[3:4:end,4:-1:1].=[9.7	0.3	85.2	4.8
13.1	3.0	78.7	5.3
14.8	4.1	74.1	6.9
17.5	5.3	67.9	9.3
22.1	7.2	56.9	13.9
28.2	9.6	45.3	16.9
34.4	12.3	34.4	18.9
38.8	15.5	25.0	20.6
41.8	18.3	20.1	19.8
42.1	21.9	17.6	18.4]

p1=groupedbar!(1:length(xaxis)*4+1, label=mylabels,data_h100,  bar_width = 1, lw=0.4, fillstyle = [nothing ://  nothing nothing], fillalpha=[0 1 0.5 1], fillcolor=green4, linecolor=green4, bar_position = :stack)


p1=plot!(ylims=(0,100), xticks=(3:4:(length(xaxis)*4+1),xaxist))
p1=myplotchars()

data_amd=zeros(length(xaxis)*4+1,4)
data_amd[4:4:end,4:-1:1].=[25.5	0.7	70.9	2.9
30.7	10.3	56.6	2.4
31.4	13.7	51.7	3.1
31.7	16.1	47.8	4.4
33.4	18.7	41.9	6.0
36.0	22.0	34.6	7.4
39.0	26.0	25.0	10.0
41.3	30.1	18.9	9.8
41.8	34.5	14.2	9.4
39.3	40.6	11.8	8.4]

p1=groupedbar!(1:length(xaxis)*4+1, data_amd,label=mylabels,  bar_width = 1, lw=0.4, fillstyle = [nothing ://  nothing nothing], fillalpha=[0 1 0.5 1], fillcolor=red,  bar_position = :stack,linecolor=red)
yaxis=[0,20,40,60,80,100]
yaxist=["0%","20%", "40%","60%", "80%","100%"]
p1=plot!(yticks=(yaxis,yaxist), xlabel= "matrix size nxn",  dpi=1000, size = (650, 250),bottom_margin = 20Plots.px)

p1= plot!(legend = :outerright ,  legend_foreground_color = :transparent, legendfontsize = 8)
savefig("subkernelratio.png")
#p1=groupedbar!(zeros(length(xaxis)*4+1,4), legend = :outerright , label = [ "bidiagonal -> diagonal" "band -> bidiagonal" "trailing matrix update" "panel factorization"],fillstyle = [nothing ://  nothing nothing], fillalpha=[0 1 0.5 1],color=:black,  legend_foreground_color = :transparent, legendfontsize = 8) 






data_rtx=[0.32	0.02	1.56	0.19
0.88	0.17	4.38	1.38
2.36	0.59	9.53	2.47
6.64	1.84	22.39	7.11
21.3	6.28	55.77	22.14
73.99	24.09	102.82	64.24
269.52	111.25	225.93	253.51
1023.73	692.01	568.45	908.31
3971	5703.42	1735.48	2997.28
15595.34	43909.86	5907	11079.6]
data_h100=[0.32	0.01	1.86	0.16
1.01	0.23	4.03	0.41
2.86	0.8	9.86	1.34
8.45	2.58	24.58	4.49
27.16	8.86	55.52	17.09
94.79	32.31	117.23	56.92
351.65	126.27	244.81	193.44
1349.78	539.58	510.37	717.05
5283.83	2311.57	1050.92	2509.5
20901.12	10875.61	2312.9	9140.7]
data_amd=[2.22	0.06	3.43	0.25
6.93	2.32	7.3	0.55
16.88	7.37	16.78	1.69
39.87	20.2	39.24	5.51
101	56.53	86.27	18.29
283.83	173.13	182.79	58.21
891.14	595.32	378.45	227.51
3076.79	2243.92	783.11	726.93
11357.04	9374.06	1703.9	2564.04
43677.65	45050.99	4437.9	9289.31]



groupedbar(xaxist,data_rtx)


data_h100=zeros(length(xaxis)*4+1,4)


p1=groupedbar!(1:length(xaxis)*4+1, label="",data_h100,  bar_width = 1, lw=0.4, fillstyle = [nothing ://  nothing nothing], fillalpha=[0 1 0.5 1], fillcolor=green4, linecolor=green4, bar_position = :stack)


p1=plot!(xticks=(3:4:(length(xaxis)*4+1),xaxist))
p1=myplotchars()

data_amd=zeros(length(xaxis)*4+1,4)



p1=groupedbar!(1:length(xaxis)*4+1, data_amd,label="",  bar_width = 1, lw=0.4, fillstyle = [nothing ://  nothing nothing], fillalpha=[0 1 0.5 1], fillcolor=red,  bar_position = :stack,linecolor=red)

p1=plot!(yticks=(yaxis,yaxist), xlabel= "matrix size nxn",  dpi=1000, size = (850, 350),bottom_margin = 20Plots.px)

p1=groupedbar!(zeros(length(xaxis)*4+1,4), legend = :outerright , label = [ "bidiagonal -> diagonal" "band -> bidiagonal" "trailing matrix update" "panel factorization"],fillstyle = [nothing ://  nothing nothing], fillalpha=[0 1 0.5 1],color=:black,  legend_foreground_color = :transparent, legendfontsize = 8) 

xaxis=[64,128,256,512,1024,2048,4096,1024*8,1024*16,1024*32]
xaxist=[string.(xaxis[1:5]);"2k";"4k"; "8k";"16k";"32k"]
yaxis=[0.5,1,5,10,50,100,500,1000,5000,10000]
yaxist=["","1ms","","","","100ms","","","","10s"]


#make gridlines darker

p1=groupedbar(
    xaxist,
    [data_rtx[:,1] data_h100[:,1] data_amd[:,1]],
    bar_width = 0.8,
    label = ["NVIDIA RTX4060"  "NVIDIA H100" "AMD MI250"],
    color = [green1  green4 red], ylims=(0.1,50000), yaxis=:log10,
    xlabel = "\n Matrix Size (nxn)\n",
    size = (600, 300), 
    lw=0, title="Panel factorization",
    legend = false,
    yticks=(yaxis,yaxist)
)
p1=myplotchars()
p2=groupedbar(
    xaxist,
    [data_rtx[:,2] data_h100[:,2] data_amd[:,2]],
    bar_width = 0.8,
    label = ["NVIDIA RTX4060"  "NVIDIA H100" "AMD MI250"],
    color = [green1  green4 red], ylims=(0.1,50000), 
    xlabel = "\n Matrix Size (nxn)\n",
    size = (600, 300), 
    lw=0, title="Trailing matrix update",
    legend = false, alpha=0.5,
    yticks=(yaxis,yaxist)
)
p2=myplotchars()
p3=groupedbar(
    xaxist,
    [data_rtx[:,3] data_h100[:,3] data_amd[:,3]],
    bar_width = 0.8,
    label = ["NVIDIA RTX4060"  "NVIDIA H100" "AMD MI250"],
    color = [green1  green4 red], ylims=(0.1,50000), 
    xlabel = "\n Matrix Size (nxn)\n",
    size = (600, 300), title="Banddiagonal to bidiagonal",
    lw=0.5, 
    legend = false, fillstyle=:/,
    yticks=(yaxis,yaxist)
)
p3=myplotchars()
p4=groupedbar(
    xaxist,
    [data_rtx[:,4] data_h100[:,4] data_amd[:,4]],
    bar_width = 0.8,
    label = ["NVIDIA RTX4060"  "NVIDIA H100" "AMD MI250"],
    color = [green1  green4 red], ylims=(0.1,50000), 
    xlabel = "\n Matrix Size (nxn)\n",
    size = (600, 300), 
    lw=0.5, title="Bidiagonal to diagonal",
    legend = false, fillstyle=:x,
    yticks=(yaxis,yaxist)
)
p4=myplotchars()

xaxist2=xaxist
xaxist2[1:2:end].=""

 plot( p1,p2, p3,p4,size=(500,900),left_margin=30Plots.px, dpi=1200,layout = @layout [ grid(4,1) ])
yaxis=[-0.5,-0.25,0,0.25,0.5]
yaxist=["-50%","","0%","","+50%"]


 tsvar=[-0.394	-0.391	-0.218	-0.198
-0.377	-0.394	-0.302	-0.299
-0.409	-0.423	-0.329	-0.363
-0.400	-0.412	-0.317	-0.383
-0.334	-0.342	-0.251	-0.372
-0.228	-0.234	-0.146	-0.350
-0.120	-0.116	-0.025	-0.335
-0.017	-0.007	0.098	-0.371
0.068	0.071	0.199	-0.536
0.121	0.074	0.209	-0.496]

p1=plot(1:10, tsvar, label = ["H100 FP32"  "H100 FP64" "MI250 FP32" "MI250 FP64"], 
    color=[green4 green4 red red], linestyle=[:solid :dash :solid :dash], lw=2, 
    size=(300,200), xticks=(1:11,xaxist2), legend=false,
    yticks=(yaxis,yaxist),legend_foreground_color = :transparent,ylims=(-0.6,0.6) )
p1=myplotchars()
p1=hline!([0 0], linestyle = :dash, color = :black, label="", lw=1)

cpbvar=[0.00	0.00	0.00	0.00
-0.02	0.00	0.00	0.01
-0.01	0.00	-0.01	0.00
-0.01	0.00	0.00	0.00
0.00	0.00	0.00	0.00
-0.01	0.00	0.00	0.00
-0.01	-0.01	0.01	0.02
0.00	0.00	0.04	0.07
0.00	0.00	0.08	0.32
0.04	0.10	0.21	0.38]

p2=plot(1:10, cpbvar, label = ["H100 FP32"  "H100 FP64" "MI250 FP32" "MI250 FP64"], 
    color=[green4 green4 red red], linestyle=[:solid :dash :solid :dash], lw=2, 
    size=(400,200),legend=:bottomright, xticks=(1:11,xaxist2), 
    yticks=(yaxis,yaxist),legend_foreground_color = :transparent,ylims=(-0.6,0.6) )
p2=myplotchars()
p2=hline!([1 1], linestyle = :dash, color = :black, label="", lw=1)
plot( p1,p2,size=(600,200), dpi=1200,layout = @layout [ a{0.5w} b ])
 savefig("params3.png")





#type: cu 32, cu 64, amd32, amd64
datamul=zeros(9,4,4,4) #size,mulsize, type,  tilesize
datamul[:,:,3,1].=[0.00	-5.56	-5.56	-5.56
0.00	-1.85	-3.70	-7.41
0.00	-2.29	-4.00	-9.71
0.16	-2.40	-4.33	-10.90
2.61	0.49	-1.39	-8.56
7.14	7.53	6.71	0.99
17.08	20.77	22.06	18.82
26.56	33.75	35.07	30.56
49.51	55.98	57.04	56.45]
datamul[:,:,3,2].=[4.76	0.00	0.00	0.00
1.79	0.00	0.00	-3.57
1.27	1.27	-1.90	-5.06
1.63	1.83	-2.64	-6.50
5.59	6.32	1.24	-3.16
14.51	17.69	13.38	10.43
26.62	35.26	34.19	32.42
19.29	35.99	38.56	34.16
37.10	56.72	61.81	60.88]
datamul[:,:,3,3].=[0.00	0.00	0.00	0.00
2.63	2.63	2.63	2.63
6.51	6.05	5.58	3.72
10.78	10.78	10.14	6.97
18.75	19.08	18.52	14.47
29.65	32.68	32.32	28.92
37.29	46.63	48.45	46.02
43.94	56.66	61.62	58.15
67.93	79.38	84.62	84.19]
datamul[:,:,3,4]=[0.00	0.00	0.00	0.00
0.93	1.85	1.85	0.93
2.67	4.53	5.14	4.53
4.50	7.31	8.10	7.36
6.84	10.69	11.35	10.44
20.59	25.54	26.18	25.57
35.40	49.67	48.31	49.92
40.47	55.49	60.27	61.29
46.90	52.60	68.24	68.82]
datamul[:,:,4,1].=[4.35	4.35	4.35	0.00
11.27	9.86	7.04	0.00
13.62	11.49	8.94	0.85
15.81	13.60	10.81	0.47
18.26	16.91	14.51	3.78
20.30	20.86	20.73	12.03
27.21	26.07	27.41	21.18
35.03	34.98	32.31	29.95
46.73	64.35	63.75	59.03]
datamul[:,:,4,2].=[4.35	6.52	4.35	4.35
9.29	8.74	7.65	6.01
11.98	11.16	8.54	7.02
13.55	12.69	9.59	7.74
14.61	14.35	11.29	8.76
19.83	21.68	19.14	16.39
45.46	50.04	49.08	47.54
55.15	68.29	69.88	69.02
55.57	68.89	75.89	76.08]
datamul[:,:,4,3].=[2.22	2.22	2.22	2.22
1.69	2.12	2.54	1.27
2.07	2.26	2.85	1.57
2.62	3.00	3.38	1.43
8.17	9.24	9.64	7.11
29.97	34.65	35.23	33.13
29.95	47.46	50.57	49.07
38.81	45.70	56.62	56.25
47.45	52.55	68.85	68.98]
datamul[:,:,4,4]=[0.00	0.00	0.00	0.00
0.99	0.80	0.87	-0.74
0.71	0.60	0.33	-1.51
1.22	0.91	0.17	-1.49
17.54	17.67	17.07	15.22
24.22	36.76	36.54	34.78
58.01	37.02	94.16	37.97
36.72	38.00	95.08	49.47
45.87	47.87	96.65	65.78]

datamul[:,:,1,1].=[0.00	0.00	0.00	0.00
-5.88	-15.79	6.25	6.25
-3.23	-14.29	15.00	13.33
-5.83	-15.36	16.37	14.60
-6.07	-17.11	16.61	14.99
-5.31	-16.90	15.49	13.11
-3.15	-10.73	12.50	9.30
1.86	-3.62	7.85	10.03
-0.91	-3.55	-12.61	4.08]
datamul[:,:,1,2].=[0.00	0.00	0.00	0.00
-9.09	0.00	0.00	0.00
-10.26	5.71	5.71	2.86
-9.72	10.00	10.00	5.38
-10.71	10.98	10.98	6.71
-11.12	9.22	9.32	5.32
-6.84	5.64	4.70	4.31
-4.23	-11.08	0.81	-1.38
-3.29	-45.53	-9.33	-0.74]
datamul[:,:,1,3].=[0.00	0.00	0.00	0.00
0.00	0.00	0.00	0.00
10.71	10.71	10.71	7.14
10.42	11.46	10.42	7.29
11.36	13.35	11.65	8.24
9.97	12.23	10.92	8.15
-1.76	6.14	7.58	5.33
-41.69	-3.41	1.13	4.50
-60.16	-34.01	1.36	7.30]
datamul[:,:,1,4].=[0.00	0.00	0.00	0.00
0.00	0.00	-12.50	-22.22
2.94	0.00	-15.00	-19.05
1.48	0.00	-20.59	-23.73
3.31	1.65	-21.50	-25.17
8.49	8.49	-16.45	-20.54
29.49	34.09	15.82	12.14
48.84	61.57	53.63	52.31
74.45	83.23	79.89	76.88]
datamul[:,:,2,1].=[0.00	0.00	16.67	16.67
-5.00	-13.64	10.53	10.53
-4.17	-13.75	11.59	10.14
-4.69	-14.29	13.26	11.74
-5.43	-15.68	13.53	12.27
-3.70	-12.87	11.02	10.40
-2.30	-7.79	8.72	7.74
0.67	-2.68	6.52	8.84
-3.18	-7.56	-32.00	4.36]
datamul[:,:,2,2].=[0.00	0.00	0.00	0.00
-7.69	0.00	0.00	0.00
-10.42	4.65	4.65	2.33
-8.43	7.36	7.36	3.68
-8.97	8.11	7.95	4.77
-7.49	4.89	5.69	3.05
-4.12	2.36	2.45	2.38
-2.88	-34.02	-0.32	-2.07
-6.38	-57.31	-27.37	-2.56]
datamul[:,:,2,3].=[0.00	0.00	0.00	0.00
0.00	9.09	9.09	0.00
5.13	7.69	7.69	5.13
6.43	9.29	7.86	6.43
6.42	10.19	9.25	7.55
2.68	7.71	7.14	6.06
-12.89	3.05	4.52	3.81
-50.14	-9.02	0.87	3.74
-63.58	-34.33	0.48	3.38]
datamul[:,:,2,4].=[0.00	0.00	0.00	0.00
9.09	9.09	0.00	-8.33
7.41	5.56	-3.57	-23.94
8.73	5.68	-6.91	-31.02
10.60	7.48	-7.23	-32.68
28.35	27.84	14.96	-17.37
52.11	62.75	57.26	38.41
67.69	82.10	73.79	77.80
56.60	83.83	75.85	76.53]

xaxis=[128,256,512,1024,2048,4096,1024*8,1024*16, 1024*32]
xaxist=[string.(xaxis[1:4]);"2k";"4k"; "8k";"16k";"32k"]
xaxist2=xaxist
xaxist2[2:2:end].=""
yaxis=[-50,0,50,100]
yaxist=["-50%", "0%","+50%", "+100%"]
xlabels=["","","", "matrix size nxn"]
bottommargins=[0,0,0,5]
outputplots=repeat([plot(1)], 16)


for i in 1:4
    for j in 1:4
        p1=groupedbar(1:9, datamul[:,:,i,j],label = ["16" "32" "64" "128" ],legend=false, bar_width = 0.8, lw=0)
        p1=plot!(ylims=(-75,100), xticks=(1:9,xaxist2))
        p1=plot!( xlabel= xlabels[i],  dpi=1000, size = (300, 200))
        p1=plot!( bottom_margin = bottommargins[i]*Plots.px,  yticks=(yaxis,yaxist))
        p1=myplotchars()
        outputplots[(i-1)*4+j]= p1

    end
end

#p1=plot!(legend=:outerright,     legend_columns = 1, legendfontsize = 12, size=(600,600), 
#        framestyle = :none, legend_foreground_color = :transparent)


 plot(outputplots...,size=(1600,800), dpi=1200, bottom_margin = 20Plots.px, 
    layout = @layout [ grid(4,4)   ])
 savefig("mulvar.png")


 #type: cu 32, cu 64, amd32, amd64
dataqr=zeros(9,3,4,4) #size,qrsplit, type,  tilesize
dataqr[:,:,3,1].=[33.33	35.29	35.29
43.64	47.27	47.27
51.69	55.42	55.25
56.25	60.27	60.09
58.57	62.75	62.57
60.00	64.22	64.02
60.59	64.87	64.60
60.64	64.99	64.71
61.02	65.19	64.95]
dataqr[:,:,3,2].=[23.08	27.69	29.23
32.60	38.67	39.78
42.28	49.75	51.61
49.74	58.26	60.44
53.94	63.17	65.66
56.35	66.02	68.52
57.53	67.39	69.97
58.13	68.22	70.74
58.55	68.52	71.12]
dataqr[:,:,3,3].=[19.10	22.47	23.60
27.13	31.58	32.39
36.55	42.59	43.45
47.66	55.51	56.44
55.72	64.87	66.02
60.65	70.85	71.94
63.45	74.06	75.32
64.88	75.71	77.06
65.67	76.61	77.95]
dataqr[:,1:2,3,4]=[0.00	0.00
31.28	34.24
59.57	62.05
74.92	77.25
82.83	85.28
86.92	89.57
88.85	91.67
89.95	92.75
90.49	93.30]
dataqr[:,:,4,1].=[32.26	37.10	33.87
43.72	48.74	45.23
51.75	57.08	53.16
56.40	61.98	57.84
58.98	64.76	60.37
60.33	66.23	61.73
60.89	66.91	62.28
61.01	67.06	62.36
61.45	67.47	62.78]
dataqr[:,:,4,2].=[25.00	27.50	28.75
33.18	38.12	38.57
43.13	49.38	49.65
50.45	57.80	57.96
55.21	63.20	63.20
57.53	65.89	65.91
58.74	67.29	67.50
59.52	68.17	68.25
59.78	68.51	68.67]
dataqr[:,:,4,3].=[26.83	30.08	30.89
44.74	48.17	48.66
59.83	63.58	63.87
70.71	74.98	75.19
77.00	81.60	81.83
80.40	85.17	85.39
82.00	86.96	87.22
82.81	87.81	88.07
83.26	88.29	88.558]
dataqr[:,1:2,4,4]=[-90.01	-90.10
-84.87	-83.65
-73.50	-68.52
-59.97	-45.45
-47.59	-15.95
-38.87	12.08
-32.46	27.90
-28.55	36.02
-26.23	40.398]

dataqr[:,:,1,1].=[54.55	59.09	59.09
60.00	65.88	67.06
64.52	70.38	71.55
66.72	72.95	74.12
67.80	74.15	75.34
67.99	74.35	75.55
68.25	74.63	75.84
68.32	74.71	75.92
68.40	74.82	76.01]
dataqr[:,:,2,1].=[51.61	58.06	58.06
57.94	65.08	66.67
61.93	69.43	71.01
63.85	71.51	73.08
64.53	72.23	73.88
64.96	72.67	74.35
65.20	72.90	74.59
65.30	73.03	74.73
65.37	73.10	74.79]
dataqr[:,:,1,2].=[37.50	43.75	50.00
50.00	62.50	62.50
56.42	70.43	71.60
59.69	75.00	76.36
61.48	77.29	78.74
62.22	78.28	79.71
62.65	78.87	80.28
62.85	79.14	80.56
62.96	79.29	80.71]
dataqr[:,:,2,2].=[44.00	48.00	52.00
54.29	62.86	66.67
60.51	69.28	73.67
63.39	72.77	77.43
64.85	74.37	79.23
65.56	75.18	80.10
65.92	75.59	80.55
66.10	75.80	80.78
66.19	75.91	80.89]
dataqr[:,:,1,3].=[23.08	23.08	23.08
40.74	46.30	48.15
53.33	62.22	63.56
60.26	69.98	71.72
63.72	74.08	75.91
65.39	76.01	77.98
65.59	77.01	79.01
66.02	77.53	79.53
66.24	77.79	79.81]
dataqr[:,:,2,3].=[21.05	26.32	26.32
46.07	51.69	53.93
57.73	64.95	67.78
63.56	71.53	74.55
66.46	74.77	77.95
67.84	76.35	79.61
68.53	77.12	80.44
68.90	77.53	80.86
69.09	77.74	81.09]
dataqr[:,1:2,1,4].=[0.00	-9.09
22.45	22.45
42.54	49.56
53.01	63.86
58.10	70.87
60.60	74.35
61.84	76.07
62.48	76.94
62.78	77.33]
dataqr[:,1:2,2,4].=[0.00	-6.67
22.54	21.13
43.70	44.28
54.01	55.94
58.93	61.46
61.34	64.29
62.48	65.61
63.07	66.26
63.38	66.60]

yaxis=[-100,-50,0,50,100]
yaxist=["-100%","-50%", "0%","+50%","+100%"]
outputplots=repeat([plot(1)], 16)


for i in 1:4
    for j in 1:4
        p1=groupedbar(1:9, dataqr[:,:,i,j],label = ["4" "8" "16" ],legend=false, bar_width = 0.8, lw=0)
        p1=plot!(ylims=(-100,100), xticks=(1:9,xaxist2))
        p1=plot!( xlabel= xlabels[i],  dpi=1000, size = (300, 200))
        p1=plot!( bottom_margin = bottommargins[i]*Plots.px,  yticks=(yaxis,yaxist))
        p1=myplotchars()
        outputplots[(i-1)*4+j]= p1

    end
end

#p1=plot!(legend=:outerright,     legend_columns = 1, legendfontsize = 12, size=(600,600), 
 #       framestyle = :none, legend_foreground_color = :transparent)


 plot(outputplots...,size=(1600,800), dpi=1200, bottom_margin = 20Plots.px, 
    layout = @layout [ grid(4,4)   ])
 savefig("splitvar.png")



outputplots=repeat([plot(1)], 4)
datats=zeros(9,3,4) #size,tilesize, type
datats[:,:,3].=[-23.88059701	-43.95604396	-40.69767442
-14.54545455	-41.49377593	-62.29946524
-1.569506726	-26.34228188	-60.9430605
11.60184575	-2.443729904	-54.3622142
20.69454288	20.57051736	-46.94990131
26.89341595	35.33187715	-39.31450522
32.43467706	44.90406525	-30.03933137
37.04386357	52.37866613	-32.87444034
43.79803479	59.51110698	-38.33789293]
datats[:,:,4].=[-39.6039604	-52.71317829	-44.57568599
-46.07843137	-62.5	-54.7039284
-50.43816943	-65.77000672	-47.52469123
-52.70962524	-67.15355805	-31.88375967
-53.56762513	-67.40268155	-12.31010579
-52.28885059	-66.30057285	7.879031031
-49.4540077	-65.6518369	-71.48917016
-42.37812917	-68.91616928	-71.52649792
-39.49712973	-63.90859378	-62.95678359]
datats[:,:,1].=[78.43137255	76.47058824	74.50980392
75.88652482	74.46808511	68.08510638
75.62642369	75.39863326	66.28701595
75.80751483	77.05998682	67.50164799
76.25797307	78.45499646	69.13536499
76.6281817	79.75750011	70.83760531
77.14282672	80.91420175	72.09693562
78.69189171	83.10096655	71.18035144
82.51857895	86.45954633	66.68578298]
    
datats[:,:,2].=[73.7704918	72.13114754	73.7704918
71.51515152	68.48484848	60
69.35166994	68.17288802	52.84872299
68.41505131	69.04218928	50.22805017
68.25688073	70.12232416	49.40366972
68.6035899	71.6078491	50.87085488
69.97056462	73.65712247	49.8492552
74.9984806	78.83774556	44.88372093
83.83185845	85.66390715	26.05091233]

for i in 1:4
        p1=groupedbar(1:9, datats[:,:,i],label = ["32" "64" "128" ],legend=false, bar_width = 0.8, lw=0)
        p1=plot!(ylims=(-100,100), xticks=(1:9,xaxist2))
        p1=plot!( xlabel= xlabels[4],  dpi=1000, size = (300, 200))
        p1=plot!( bottom_margin = bottommargins[4]*Plots.px,  yticks=(yaxis,yaxist))
        p1=myplotchars()
        outputplots[i]= p1
end

#p1=plot!(legend=:outerright,     legend_columns = 1, legendfontsize = 12, size=(600,600), 
        #framestyle = :none, legend_foreground_color = :transparent)

         plot(outputplots...,size=(1600,350), dpi=1200, bottom_margin = 50Plots.px, 
    layout = @layout [ grid(1,4)   ])
 savefig("tilevar.png")