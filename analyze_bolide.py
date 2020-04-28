#!/usr/bin/env python

import numpy as n
import digital_rf as drf
import matplotlib.pyplot as plt
import glob
import h5py
from mpi4py import MPI
import stuffr
import scipy.interpolate as sio
import scipy.signal as ss
import scipy.constants as sc
import itertools

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("comm rank %d size %d"%(rank,size))


def bpsk(seed,length):
    n.random.seed(seed)
    code=n.random.random(length)
    code=n.exp(2.0*n.pi*1.0j*code)
    code=n.angle(code)
    code=-1.0*n.sign(code)
    code=n.array(code,dtype=n.complex64)
    return(code)

def get_codes(int_f=8,
              code_len=1000,
              n_codes=4,
              seeds=[1,238,681,3099,3263]):
    codes=[]
    plot_code=False
    wf=n.repeat(1.0/float(int_f),int_f)#ss.hann(int(int_f))
    W=n.fft.fft(wf,code_len*int_f)
    for s in seeds:
        c0=bpsk(s,code_len)
        # interpolate using a rectangular filter
        c0=n.repeat(c0,int_f)
        c0f=n.fft.ifft(n.fft.fft(c0)*W)
        c0f=n.roll(c0f,-int_f/2)
        c0f=c0f/n.max(n.abs(c0f))    
        if plot_code:
            plt.plot(c0f.real)
            plt.plot(c0.real)
            plt.show()
        codes.append(n.tile(c0f,n_codes))
    return(codes)
    
#d=drf.DigitalRFReader("/mnt/data/juha/peru_bolide/rawdata")
d=drf.DigitalRFReader("/mnt/data/juha/peru_bolide/huancayo/MeteorHuancayo/")
out="/mnt/data/juha/peru_bolide/huancayo"
print(d.get_bounds("ch000"))


def range_doppler_matched_filter(d,
                                 out,
                                 code_seeds=[1,238,681,3099,3263],
                                 code_len=1000,
                                 i0=158699892900000, # first index to analyze
                                 i1=158699892900000, # last sample index to analyze
                                 int_f=4,   # range interpolation factor
                                            # this is  multiplied by two in the code
                                 n_codes=8, # number of codes to integrate coherently together
                                 sr=100e3,
                                 ignore_freq=250.0, # don't include Doppler shifts
                                                    # smaller than this
                                 r_min_an=400,  # minimum range to analyze (km)
                                 r_max_an=600,   # maximum range to analyze (km)
                                 f_min_an=-4e3, # minimum doppler to analyze (Hz)
                                 f_max_an=1e3, # maximum doppler to analyze
                                 noise_bw=10e3, # receiver noise bandwidth
                                 codes_per_step=1 #
                                 ):
    int_f=int_f*2

    codes=get_codes(int_f,code_len=code_len,seeds=code_seeds,n_codes=n_codes)

    dr = sc.c/sr/1e3

    c=d.get_channels()
    b=d.get_bounds(c[0])
    
    # index of ranges
    ridx=n.arange(n_codes*code_len*int_f)
    
    # frequency shifts 
    fvec=n.fft.fftshift(n.fft.fftfreq(int_f*n_codes*code_len,d=1/(int_f*sr)))
    
    fidx=n.where(n.abs(fvec) < 1e3)[0]

    # one-way range (distance from transmitter to receiver)
    rvec=n.arange(r_max_an*int_f)*(dr/int_f)

    # range of Doppler shifts to consider
    gfidx=n.where( (fvec > f_min_an ) &(fvec < f_max_an))[0]
    fvec0=fvec[gfidx]
    
    # do not detect these doppler frequencies 
    bfidx=n.where( n.abs(fvec0) < ignore_freq)[0]

    # noise band
    sfidx=n.where( n.abs(fvec0) < noise_bw)[0]

    # range gates to analyze
    gridx=n.where( (rvec > r_min_an ) &(rvec < r_max_an))[0]
    rvec0=rvec[gridx]

    n_steps=int(n.floor((i1-i0)/(code_len*codes_per_step)))

    # figure out all correlations and cross-correlations
    ch_pairs=list(itertools.combinations(n.arange(len(code_seeds),dtype=n.int),2))
    for i in range(len(code_seeds)):
        ch_pairs.append((i,i))
    ch_pairs = n.array(ch_pairs)
    n_pairs=ch_pairs.shape[0]
    
    # go through all time steps
    for si in range(rank,n_steps,size):
        S=n.zeros([n_pairs,len(gridx),len(gfidx)],dtype=n.complex64)
        
        read_idx=si*code_len*codes_per_step+i0
        read_len=n_codes*code_len+code_len
        
        # go through all pairs
        for chi in range(len(c)):
            z0=d.read_vector_c81d(read_idx,read_len,c[chi])

            # interpolate using a rectangular filter
            # make sure we shift the signal back so that no range offset is created
            z0=n.repeat(z0,int_f)
            wf=n.repeat(1.0/float(int_f),int_f)
            z0i=n.fft.ifft(n.fft.fft(z0)*n.fft.fft(wf,len(z0)))
            z0i=n.roll(z0i,-int_f/2)

            # go through all transmit code pairs
            for ci in range(n_pairs):
                c0i=ch_pairs[ci,0]
                c1i=ch_pairs[ci,1]
                
                # go through all range gates
                for rii,ri in enumerate(gridx):
                    Z0=n.fft.fft(n.conj(codes[c0i])*z0i[ri+ridx])
                    Z1=n.fft.fft(n.conj(codes[c1i])*z0i[ri+ridx])
                    S[ci,rii,:]+=n.fft.fftshift(Z0*n.conj(Z1))[gfidx]
                    
        # average over all XCs to obtain power estimate
        S0=n.mean(n.abs(S),axis=0)
        
        noise_floor=n.median(S0)
        # create signal_to_noise ratio estimate
        snr=(S0-noise_floor)/noise_floor

        # create a copy 
        snr0=n.copy(snr)

        # because we are repeating the code, we are introducing some periodic
        # features in the range-doppler matched filter output.
        # gently try to filter these out using a 2d FFT
        F=n.fft.fft2(snr0)
        for fri in range(F.shape[0]):
            med_amp=n.median(n.abs(F[fri,:]))
            bad_f_idx=n.where(n.abs(F[fri,:]) > 5.0*med_amp)[0]
            F[fri,bad_f_idx]=0.0
        snr0=n.abs(n.fft.ifft2(F))
#        snr0[n.min(gridx):n.max(gridx),n.min(sfidx):n.max(sfidx)]=snr0F
    
        plt.figure(figsize=(16*1.5,1.5*12))
        n_floor=n.median(snr0)#[n.min(gridx):n.max(gridx),n.min(gfidx):n.max(gfidx)])
        snr0=snr0-n_floor
        snr0[snr0<0]=1e-6
        plt.pcolormesh(fvec0,rvec0,10.0*n.log10(snr0),vmin=-10,vmax=10.0,cmap="gnuplot2")
        plt.title("%s\n"%(stuffr.unix2datestr(read_idx/sr)))
        plt.xlim([f_min_an,f_max_an])
        plt.ylim([r_min_an,r_max_an])
        plt.xlabel("Doppler shift (Hz)")
        plt.ylabel("Transmitter to receiver range (km)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("%s/rd-%06d.png"%(out,si))
        plt.clf()
        plt.close()

        snr0[:,n.min(bfidx):n.max(bfidx)]=0.0
        rp,fp=n.unravel_index(n.argmax(snr0),snr.shape)
        f0=fvec0[fp]
        r0=rvec0[rp]
        peak_snr=n.max(snr0)
    
        high_snr=n.max(snr0,axis=1)
        low_snr=n.max(snr,axis=1)
        h_r=n.argmax(high_snr)
        l_r=n.argmax(low_snr)

        h=h5py.File("%s/snr_%06d.h5"%(out,read_idx),"w")
        h["XC"]=S
        h["channel_pairs"]=ch_pairs
        h["channel_code_seeds"]=code_seeds
        h["high_snr"]=high_snr
        h["low_snr"]=low_snr
        h["high_r"]=h_r
        h["low_r"]=l_r
        h["dop_freq"]=fvec0
        h["rvec"]=rvec0
        h["f0"]=f0
        h["r0"]=r0
        h["t0"]=read_idx 
        h["sr"]=sr
        h["snr0"]=peak_snr
        h.close
        print("segment %d/%d (%s)  max_snr=%1.2f range=%1.2f (km) doppler=%1.2f (Hz)"%( si, n_steps, stuffr.unix2datestr(read_idx/sr), peak_snr, r0, f0 ))


range_doppler_matched_filter(d,
                             out=out,
                             r_min_an=550,
                             r_max_an=600,
                             i0=158699892000000,
                             i1=158699894000000)
               
