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
#seeds=[2,9,814,4081,4399]
seeds=[8, 4195, 2660,   19,  344,   95]
#seeds=[1,238,681,3099,3263]

def bpsk(seed,length):
    print(seed)
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
dname="/mnt/data/juha/peru_bolide/santarosa/PeruMeteorSantaRosa"

d=drf.DigitalRFReader(dname)

c=d.get_channels()
#print(d.get_digital_rf_metadata(c[0]))
#print(c)
b=d.get_bounds(c[0])
N_per_seg=1000
N_segments=(b[1]-b[0])/(N_per_seg*code_len)
print(N_segments)
print(b[0])
r_max=600
Nr=600
ch_idx=n.arange(len(c))
ch_idx=[1]
for si in range(rank,N_segments,size):
    S=n.zeros([N_per_seg,Nr])
    C=n.zeros([N_per_seg,code_len])
    RTI=n.zeros([N_per_seg,Nr])

    # all channels
    for ci in ch_idx:
        z=d.read_vector_c81d(si*code_len*N_per_seg+b[0],N_per_seg*code_len,c[ci])

        std_est=n.median(n.abs(z))
        bidx=n.where(n.abs(z)>6*std_est)
        z[bidx]=0.0
#        plt.plot(z[0:10000].real)
 #       plt.plot(z[0:10000].imag)
  #      plt.show()
        Z=n.fft.fft(z)
        ZS=n.abs(Z)
        std_est=n.median(ZS)
        bad_idx=n.where( ZS>15.0*std_est)[0]
        Z[bad_idx]=0        
        z=n.fft.ifft(Z)
#        cf=n.conj(n.fft.fft(bpsk(seeds[0],code_len)))
 #       for ti in range(N_per_seq):
  #          C[ti,:]=n.fft.ifft(cf*n.fft.fft(z[(ti*code_len):((ti+1)*code_len)]))
   #     plt.pcolormesh(n.abs(C))
    #    plt.show()
        
        
        
        for seed_idx in seeds:
            r=prc_lib.analyze_prc(n.copy(z),Nranges=600,code=bpsk(seed_idx,code_len),gc_rem=False,rfi_rem=False,dec=1,station=seed_idx)
            S+=n.abs(r["spec"])**2.0
            RTI+=n.abs(r["res"])**2.0
    plt.figure(figsize=(16,10))
    plt.subplot(211)
    peak_snr=n.max(n.abs(r["spec"])**2.0)
    plt.title("%s-%s"%(stuffr.unix2datestr( (si*code_len*N_per_seg+b[0])/100e3),
                       stuffr.unix2datestr( ((si+1)*code_len*N_per_seg+b[0])/100e3)))
    n_floor_r=n.median(RTI)
    n_floor_s=n.median(S)
    S=(S-n_floor_s)/n_floor_s
    RTI=(RTI-n_floor_r)/n_floor_r
    plt.pcolormesh(10.0*n.log10(n.transpose(n.abs(RTI))),vmin=-3,vmax=20)
    plt.colorbar()
    plt.subplot(212)
    plt.pcolormesh(n.transpose(S),vmin=-3,vmax=20)
    plt.colorbar()
    plt.show()
    
