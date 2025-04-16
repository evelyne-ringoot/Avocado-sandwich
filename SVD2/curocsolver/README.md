compilation:

 ```
nvcc -o svd_cusolver cusvd.cu -lcusolver -lcurand
 ```
or for AMD:
 ```
hipcc -O2 rocsvd.cpp -o svd_rocsolver  -I/opt/rocm-6.0.0/include/rocsolver  -L/opt/rocm-6.0.0/lib  -lrocsolver -lrocblas -lrocrand
 ```
