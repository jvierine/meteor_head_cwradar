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

code_len=1000
seeds=[2,9,814,4081,4399]

def bpsk(seed,length):
#    print(seed)
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
#print(n.real(codes[0]))

d=drf.DigitalRFReader("/mnt/data/juha/peru_bolide/rawdata")

c=d.get_channels()
#print(d.get_digital_rf_metadata(c[0]))
#print(c)
b=d.get_bounds(c[0])
N_per_seg=1000
N_segments=(b[1]-b[0])/(N_per_seg*code_len)
print(N_segments)
print(b[0])
r_max=600

i0=12*code_len*N_per_seg+367*code_len+b[0]
z=d.read_vector_c81d(i0,code_len,c[0])
Z=n.fft.fft(z)
best_seed=0
best=0
for seed_idx in range(8192):
    mf=n.max(n.abs(n.fft.ifft(n.conj(n.fft.fft(bpsk(seed_idx,code_len)))*Z))**2.0)
    if 10.0*n.log10(mf)>80.0:
        print("sed %d mf=%1.2f"%(seed_idx,10.0*n.log10(mf)))    
#    if mf > best:
 #       best_seed=seed_idx
  #      best=mf
   #     print("sed %d mf=%1.2f"%(seed_idx,10.0*n.log10(best)))
    
