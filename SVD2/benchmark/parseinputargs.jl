function parse_commandline(args)
    # initialize the settings (the description is for the help screen)
    s = ArgParseSettings(description = "GPU-accelerated Bandreduction")


    add_arg_group!(s, "Required mutually exclusive options", exclusive=true, required=true)
    @add_arg_table! s begin
        "--half"
            action = :store_const
            dest_name = "data type"
            constant = Float16
            help = "half precision"
        "--single", "-s"
            action = :store_const
            dest_name = "data type"
            constant = Float32
            help = "double precision"
        "--double", "-d"
            action = :store_const
            dest_name = "data type"
            constant = Float64
            help = "double precision"
    end
    add_arg_group!(s, "Required options", required=true)
    @add_arg_table! s begin
        "--hardware"
            arg_type = String
            help = "GPU backend: specify amd, cuda, oneapi, metal"
    end

   add_arg_group!(s, "Optional arguments")
    @add_arg_table! s begin
        "--brdwidth"
            arg_type = Int
            default = 64
        "--bandwidth"
            arg_type = Int
            default = 64
        "--brdmulsize"
            arg_type = Int
            default = 64
        "--maxblocks"
            arg_type = Int
            default = 100
        "--mintime"
            arg_type = Int
            default = 2000
        "--numruns"
            arg_type = Int
            default = 20
        "--tilesizemul"
            arg_type = Int
            default = 64
        "--qrsplit"
            arg_type = Int
            default = 8
        "--tilesize"
            arg_type = Int
            default = 32

    end

    output=parse_args(args,s)
    if !(output["hardware"] in ["amd", "cuda", "oneapi", "metal"])
        error("Argument hardware requires one of the following options: amd, cuda, oneapi, metal")
    end
    println("benchmarking for parameters ")
    for (key, value) in output
        println(key, " : ", value)
    end
    return output
end
parsed_args = parse_commandline(ARGS)
elty=parsed_args["data type"]
include("vendorspecific/benchmark_"*parsed_args["hardware"]*".jl")
#include("vendorspecific/benchmark_cuda.jl")
arty=typeof(KernelAbstractions.zeros(backend,elty,2,2))
const BRDWIDTH = parsed_args["brdwidth"]
const BW= parsed_args["bandwidth"]
const BRDMULSIZE = parsed_args["brdmulsize"]
const MAXBLOCKS = parsed_args["maxblocks"]
const MINTIME = parsed_args["mintime"]
const NUMRUMS= parsed_args["numruns"]
const TILESIZE = parsed_args["tilesize"]
const BANDOFFSET = Int(BRDWIDTH/TILESIZE)
const TILESIZEMUL =  parsed_args["tilesizemul"]
const QRSPLIT = parsed_args["qrsplit"]