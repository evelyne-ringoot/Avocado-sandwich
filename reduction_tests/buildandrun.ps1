cp src_cpp\reduction.cu build\
cp src_cpp\gen_test_data.py build\
cd build
py gen_test_data.py
nvcc  --std=c++17 --expt-relaxed-constexpr -O3  -arch=sm_89 -o red reduction.cu -lcublas -lcurand -lcusolver
.\red.exe