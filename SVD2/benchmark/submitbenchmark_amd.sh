#!/bin/sh

for N in 64 92 128 160 192 224; do flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh single 64 128 $N 832 ; done
for N in 64 92 128 160 192 224; do flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh single 64 256 $N 832 ; done
for N in 64 92 128 160 192 224; do flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh single 64 512 $N 832 ; done

for N in 264 1056; do flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh single 64  128  128  $N ; done
for N in 264 1056; do flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh single 64  256  160  $N ; done
for N in 264 1056; do flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh single 64 512  192  $N ; done

for N in 32 128; do flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh single $N 128  128  528 ; done
for N in 32 128; do flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh single $N 256  160  528 ; done
for N in 32 128; do flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh single $N 512  192  528 ; done
for N in 32 128; do flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh single $N  128  64  528 ; done
for N in 32 128; do flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh single $N  256  96  528 ; done
for N in 32 128; do flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh single $N  512  128  528 ; done

flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia_svd.sh single 32  32  64  528  32 
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia_svd.sh single 64  64  96  528  64 
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia_svd.sh single 64  128  128  528  128 
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia_svd.sh single 64  256  160  528  256
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia_svd.sh single 64  512  192  528  64 
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia_svd.sh single 64  1024  224  528  64 

flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh double 32 32  64  528 
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh double 32 64  96  528 
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh double 32 128  128  528 
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh double 32 256  160  528
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh double 32 512  192  528
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh double 32 1024  224  528 

flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh half 32  32  64  528 
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh half 64  64  96  528 
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh half 128  128  128  528 
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh half 128  256  160  528
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh half 128  512  192  528
flux batch -t 120 -N 1 -n 1 --exclusive -g 1 ./submit_julia.sh half 128  1024  224  528 