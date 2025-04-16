compilation:

 ```
 nvcc  --std=c++17 --expt-relaxed-constexpr -O3  -arch=sm_89 -o cusvd cusvd.cu -lcublas -lcurand -lcusolver
 ```
or for AMD:
 ```
hipcc -O2 rocsvd.cpp -o svd_rocsolver  -lrocsolver -lrocblas -lrocrand
 ```
