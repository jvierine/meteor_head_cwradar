#!/usr/bin/env python

import numpy as n
import digital_rf as drf
import matplotlib.pyplot as plt
import glob
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

code_len=1000
seeds=[2,9,814,4081,4399]

def bpsk(seed,length):
    n.random.seed(seed)
    code=n.random.random(length)
    code=n.exp(2.0*n.pi*1.0j*code)
    code=n.angle(code)
    code=-1.0*n.sign(code)
    code=n.array(code,dtype=n.complex64)
    return(code)

codes=[]
for s in seeds:
    codes.append(bpsk(s,code_len))

d=drf.DigitalRFReader("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/peru_bolide/rawdata")

c=d.get_channels()
print(c)
b=d.get_bounds(c[0])
N_segments=(b[1]-b[0])/code_len
print(N_segments)
print(b[0])
r_max=330
ridx=n.arange(code_len)
for si in range(rank,N_segments,size):
    
    S=n.zeros([r_max,code_len],dtype=n.float32)
    for ci in range(len(codes)):
        for chi in range(len(c)):
            for ri in range(r_max):
                z0=d.read_vector_c81d(si*code_len+b[0],2*code_len,c[chi])
                S[ri,:]+=n.fft.fftshift(n.abs(n.fft.fft(n.conj(codes[ci])*z0[ri+ridx]))**2.0)
    noise_floor=n.median(S)
    snr=(S-noise_floor)/noise_floor
    peak_snr=n.max(snr)
    print("segment %d (%1.3f s)  max_snr=%1.2f"%( si, si*code_len/100e3, peak_snr ))

    
#    plt.pcolormesh(S)
 #   plt.colorbar()
  #  plt.show()
