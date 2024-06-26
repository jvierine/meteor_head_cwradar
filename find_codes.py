#!/usr/bin/env python

import numpy as n
import digital_rf as drf
import matplotlib.pyplot as plt
import glob
import h5py
from mpi4py import MPI
import prc_lib
import stuffr

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
dfile="/data0/interstellar_2023/And-Sau/ch000/rf@1690408920.000.h5"

h=h5py.File(dfile,"r")
print(h.keys())
z=h["rf_data"][:,0]
#plt.plot(h["rf_data"][:,0].real)
#plt.plot(h["rf_data"][:,0].imag)
#plt.show()

code_len=1000
seeds=[2,9,814,4081,4399]

def bpsk(seed,length):
    n.random.seed(seed)
    code=n.random.random(length)
    code=n.exp(2.0*n.pi*1.0j*code)
    code=n.angle(code)
    code=-1.0*n.sign(code)
    code=code.real
    code=n.array(code,dtype=n.complex64)
    return(code)

codes=[]
for s in seeds:
    codes.append(bpsk(s,code_len))

Z=n.fft.fft(z[(len(z)-code_len):len(z)])
best_seed=0
best=0
mfs=[]
for seed_idx in range(8192):
    mf=n.max(n.abs(n.fft.ifft(n.conj(n.fft.fft(bpsk(seed_idx,code_len)))*Z))**2.0)
    mfs.append(10.0*n.log10(mf))

print(n.sort(mfs)[::-1][0:10])
seedis=n.arange(8192)
print(seedis[n.argsort(mfs)[::-1][0:10]])
