#!/usr/bin/env python3

import numpy as np
import os
import json

np.random.seed(0xCA7CAFE)

script_dir = os.path.dirname(os.path.realpath(__file__))

tilesize = 128
sizes = [32,128,512, 2048,1024*8,1024*32,1024*128]



for size_i in sizes:
    a = np.random.randn(size_i, tilesize).astype(np.float32)
    a_fname = os.path.join(script_dir, f"test_a_{tilesize}_{size_i}.bin")
    with open(a_fname, "wb") as f:
        f.write(a.tobytes())
    b=np.sum(a,axis=1)
    b_fname = os.path.join(script_dir, f"ref_a_{tilesize}_{size_i}.bin")
    with open(b_fname, "wb") as f:
        f.write(b.tobytes())

sizes= [32,64,128,256,512,1024,2048]

for size_i in sizes:
    a = np.random.randn(size_i, size_i).astype(np.float32)
    a_fname = os.path.join(script_dir, f"test_a_{size_i}_{size_i}.bin")
    with open(a_fname, "wb") as f:
        f.write(np.transpose(a).tobytes())
    b = np.random.randn(size_i, size_i).astype(np.float32)
    b_fname = os.path.join(script_dir, f"test_b_{size_i}_{size_i}.bin")
    with open(b_fname, "wb") as f:
        f.write(np.transpose(b).tobytes() )  
    c_fname = os.path.join(script_dir, f"ref_c_{size_i}_{size_i}.bin")
    with open(c_fname, "wb") as f:
        f.write(np.transpose(np.matmul(a , b)).tobytes())