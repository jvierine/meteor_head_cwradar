#!/usr/bin/env python

import glob
import h5py
import numpy as n
import matplotlib.pyplot as plt
import stuffr

fl=glob.glob("snr8/snr*.h5")
fl.sort()
h=h5py.File(fl[0],"r")
rvec=n.copy(h["rvec"].value)
n_r=len(h["rvec"].value)
h.close()
S=n.zeros([len(fl),n_r])
T=n.zeros([len(fl),n_r])
rp=[]
tv=[]
dp=[]

si=0
for fi,f in enumerate(fl):
    try:
        h=h5py.File(f,"r")
        print(h.keys())
        S[si,:]=h["high_snr"].value
        T[si,:]=h["low_snr"].value

        rp.append(h["r0"].value)
        dp.append(h["f0"].value)
        tv.append(stuffr.unix2date(h["t0"].value/100e3))
        h.close()
        si=si+1
    except:
        print("error %d"%(fi))
        pass
dB=10.0*n.log10(S[0:si,:]+T[0:si,:])
#dB=10.0*n.log10(S[0:si,:])
for ti in range(dB.shape[0]):
    dB[ti,:]=dB[ti,:]-n.median(dB[ti,:])
    
plt.pcolormesh(tv,rvec,n.transpose(dB),vmin=0,vmax=20)
plt.title("Signal-to-noise ratio (dB)")
plt.xlabel("Time (UTC)")
plt.ylabel("Full propagation distance (km)")

plt.colorbar()
plt.show()
plt.figure(figsize=(8,6))
plt.subplot(121)
plt.plot(tv,rp,".")
plt.ylabel("Delay (km)")
plt.xlabel("Time (UTC)")
plt.subplot(122)


#36e6*v=3e8*df/36e6
dp=n.array(dp)
plt.plot(tv,dp,".")
plt.ylabel("Doppler (Hz)")
plt.xlabel("Time (UTC)")
plt.show()
