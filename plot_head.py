#!/usr/bin/env python

import glob
import h5py
import numpy as n
import matplotlib.pyplot as plt
import stuffr

fl=glob.glob("snr*.h5")
fl.sort()
S=n.zeros([len(fl),600])
rp=[]
tv=[]
dp=[]

for fi,f in enumerate(fl):
    try:
        h=h5py.File(f,"r")
        #    S[fi,:]=h["high_snr"].value

        if h["snr0"].value > 1.0:
            rp.append(h["r0"].value)
            dp.append(h["f0"].value)
            tv.append(stuffr.unix2date(h["t0"].value/100e3))
        h.close()
    except:
        print("error")
        pass

plt.figure(figsize=(16,9))
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
