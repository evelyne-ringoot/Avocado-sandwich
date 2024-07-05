import Pkg; 
Pkg.add("https://github.com/JuliaLang/julia/tree/master/stdlib/LinearAlgebra"); 
Pkg.add("https://github.com/JuliaLang/julia/tree/master/stdlib/Random"); 

using Random, LinearAlgebra

a=rand(5,5)
triu!(a)