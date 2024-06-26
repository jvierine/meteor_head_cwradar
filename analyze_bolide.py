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
from astropy.nddata import block_reduce

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

def rfi_rem(zin):
    z=n.copy(zin)
    # remove spikes
    std_est=n.median(n.abs(z))
    bidx=n.where(n.abs(z)>6*std_est)
    z[bidx]=0.0

    # remove spectral spikes
    Z=n.fft.fft(z)
    ZS=n.abs(Z)
#    plt.plot(ZS)
 #   plt.show()
    std_est=n.median(ZS)
    bad_idx=n.where( ZS>15.0*std_est)[0]
    Z[bad_idx]=0        
    z=n.fft.ifft(Z)
    return(z)


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
        print(int_f/2)
        c0f=n.roll(c0f,-int(int_f/2))
        c0f=c0f/n.max(n.abs(c0f))    
        if plot_code:
            plt.plot(c0f.real,label="filtered")
            plt.plot(c0.real,label="repeated")
            plt.legend()
            plt.show()
        codes.append(n.tile(c0f,n_codes))
    return(codes)
    
def range_doppler_matched_filter(d,
                                 out,
                                 code_seeds=[8, 4195, 2660,   19,  344,   95], # simone norway 2023
                                 code_len=1000,
                                 i0=169040892000000+50*100000, # first index to analyze
                                 i1=169040892000000+60*100000, # last sample index to analyze
                                 int_f=4,   # range interpolation factor
                                            # this is  multiplied by two in the code
                                 n_codes=4, # number of codes to integrate coherently together
                                 sr=100e3,
                                 ignore_freq=250.0, # don't include Doppler shifts
                                                    # smaller than this
                                 r_min_an=300,  # minimum range to analyze (km)
                                 r_max_an=600,   # maximum range to analyze (km)
                                 f_min_an=-30e3, # minimum doppler to analyze (Hz)
                                 f_max_an=30e3, # maximum doppler to analyze
                                 noise_bw=10e3, # receiver noise bandwidth
                                 codes_per_step=1, #
                                 rfi_remove=False,
                                 rx_channels=[0,1]
                                 ):
    # force this to be one!
    int_f=1

    codes=get_codes(int_f,code_len=code_len,seeds=code_seeds,n_codes=n_codes)

    # range gate (one-way)
    dr = sc.c/sr/1e3

    c=d.get_channels()
    b=d.get_bounds(c[0])
    
    # index of ranges
    ridx=n.arange(n_codes*code_len*int_f)
    
    # frequency shifts 
    fvec=n.fft.fftshift(n.fft.fftfreq(int_f*n_codes*code_len,d=1/(int_f*sr)))
    
#    fidx=n.where(n.abs(fvec) < 1e3)[0]

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
    print(rvec0)

    n_steps=int(n.floor((i1-i0)/(code_len*codes_per_step)))
    print(n_steps)

    # figure out all correlations and cross-correlations
#    ch_pairs=list(itertools.combinations(n.arange(len(code_seeds),dtype=n.int),2))
    ch_pairs=[]
    for i in range(len(code_seeds)):
        ch_pairs.append((i,i))
    ch_pairs = n.array(ch_pairs)
    n_pairs=ch_pairs.shape[0]
    
    # go through all time steps
    for si in range(rank,n_steps,size):
#        S=n.zeros([n_pairs,len(gridx),len(gfidx)],dtype=n.complex64)
        S=n.zeros([len(gridx),len(gfidx)],dtype=n.float)
        
        read_idx=si*code_len*codes_per_step+i0
        read_len=n_codes*code_len+code_len
        
        # go through all rx channels 
        for chi in rx_channels:
            z0=d.read_vector_c81d(read_idx,read_len,c[chi])
            z0=z0-n.median(z0)
#            plt.plot(z0.real)
 #           plt.plot(z0.imag)
  #          plt.show()
            
            if rfi_remove:
                z0=rfi_rem(z0)

            # interpolate using a rectangular filter
            # make sure we shift the signal back so that no range offset is created
            z0=n.repeat(z0,int_f)
            wf=n.repeat(1.0/float(int_f),int_f)
            z0i=n.fft.ifft(n.fft.fft(z0)*n.fft.fft(wf,len(z0)))
            z0i=n.roll(z0i,-int(int_f/2))

            # go through all transmit code pairs
            for ci in range(n_pairs):
                c0i=ch_pairs[ci,0]
                c1i=ch_pairs[ci,1]
  #          for ci in range(len(code_seeds)):
                # go through all range gates
                for rii,ri in enumerate(gridx):
                    Z0=n.fft.fft(n.conj(codes[c0i])*z0i[ri+ridx])
                    Z1=n.fft.fft(n.conj(codes[c1i])*z0i[ri+ridx])
                    S[rii,:]+=n.fft.fftshift(n.abs(Z0*n.conj(Z1)))[gfidx]

        for fi in range(S.shape[1]):
            S[:,fi]=(S[:,fi]-n.median(S[:,fi]))/n.std(S[:,fi])
        
        # average over all XCs to obtain power estimate
#        S0=n.sqrt(n.mean(n.abs(S)**2.0,axis=0))
#        S0=S
#        noise_floor=n.median(S0)
        # create signal_to_noise ratio estimate
        snr=S#S0/noise_floor

        # create a copy 
        snr0=n.copy(snr)

        # because we are repeating the code, we are introducing some periodic
        # features in the range-doppler matched filter output.
        # gently try to filter these out using a 2d FFT
#        F=n.fft.fft2(snr0)
#        FA=n.abs(F)
 #       MFA=ss.medfilt2d(FA,21)
  #      plt.pcolormesh(10.0*n.log10(FA-MFA))
   #     plt.colorbar()
    #    plt.show()
        
 #       for fri in range(F.shape[0]):
  #          med_amp=n.median(n.abs(F[fri,:]))
   #         bad_f_idx=n.where(n.abs(F[fri,:]) > 10.0*med_amp)[0]
    #        F[fri,bad_f_idx]=0.0
     #   snr0=n.abs(n.fft.ifft2(F))
        

    
        plt.figure(figsize=(16,9))
        #        n_floor=n.median(snr0)
        #       snr0=snr0-n_floor
        #      snr0=snr0/n.median(n.abs(snr0))
#        snrm=block_reduce(snr0,(1,2),func=n.max)
 #       print(snrm.shape)
        snrm=snr0        
  
        plt.pcolormesh(snrm,vmin=-2,vmax=20.0,cmap="inferno")
        plt.title("%s\n"%(stuffr.unix2datestr(read_idx/sr)))
#        plt.xlim([f_min_an,f_max_an])
 #       plt.ylim([r_min_an,r_max_an])
        plt.xlabel("Doppler shift (Hz)")
        plt.ylabel("Transmitter to receiver range (km)")
        plt.colorbar()
        plt.tight_layout()
#        plt.show()
        plt.savefig("%s/rd-%06d.png"%(out,read_idx))
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


# huancayo
def huancayo():
    range_doppler_matched_filter(d=drf.DigitalRFReader("/mnt/data/juha/peru_bolide/huancayo/MeteorHuancayo/"),
                                 out="/mnt/data/juha/peru_bolide/huancayo/",
                                 int_f=1,
                                 n_codes=16,
                                 rfi_remove=True,
                                 r_min_an=500,
                                 r_max_an=650,
                                 f_min_an=-1e3,
                                 f_max_an=1e3,                                 
                                 i0=158699893000000-300000, #I'm guessing timeing 3 s ahead at huancayo.
                                 i1=158699893000000+100000*2-300000)
                                 
#                                 i0=158699892000000,
 #                                i1=158699894000000)
def an_azpitia():
    # azpitia
    range_doppler_matched_filter(d=drf.DigitalRFReader("/mnt/data/juha/peru_bolide/rawdata"),
                                 out="/mnt/data/juha/peru_bolide/azpitia"   ,
                                 r_min_an=400,
                                 r_max_an=600,
                                 n_codes=8,
                                 int_f=1,
                                 rfi_remove=True,
                                 i0=158699893000000,
                                 i1=158699893000000+100000*2)
#                                 i0=158699893056000,
 #                                i1=158699893056000+100000*3)

def santa_rosa():
    # santa rosa
    range_doppler_matched_filter(d=drf.DigitalRFReader("/mnt/data/juha/peru_bolide/santarosa/PeruMeteorSantaRosa"),
                                 out="/mnt/data/juha/peru_bolide/santarosa",
                                 r_min_an=500,
                                 r_max_an=600,
                                 n_codes=8,
                                 int_f=1,                        
                                 rx_channels=[1], # channel 0 is busted
                                 rfi_remove=True, # santa rosa has tons of rfi
                                 f_min_an=-5e3,
                                 f_max_an=5e3,
                                 i0=158699893000000,
                                 i1=158699893000000+100000*2)
                                 
 #                                i0=158699893056000,
#                                 i1=158699893056000+100000*3)
                                 
#                                 i0=158699892000000,
 #                                i1=158699894000000)

def saura():
    # santa rosa
    range_doppler_matched_filter(d=drf.DigitalRFReader("/data0/interstellar_2023/And-Sau/"),
                                 out="/data0/interstellar_2023/saura_head",
                                 r_min_an=0,
                                 r_max_an=700,
                                 n_codes=50,
                                 int_f=1,                        
                                 rx_channels=[0], # channel 0 is busted
                                 rfi_remove=False, # santa rosa has tons of rfi
                                 f_min_an=-400,
                                 f_max_an=400,
                                 codes_per_step=50,
                                 i0=169040892000000+49*100000,
                                 i1=169040892000000+60*100000)


saura()

 
#an_azpitia()
#santa_rosa()
#huancayo()


