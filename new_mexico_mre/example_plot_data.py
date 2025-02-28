import h5py
import matplotlib.pyplot as plt
import numpy as n

files=["deco_rd_na.png.h5","deco_rd_sv.png.h5","deco_rd_ws.png.h5"]
def plot_file(fname):
    hna=h5py.File(fname,"r")
    print(hna.keys())
    doppler_na=hna["doppler_Hz"][()]
    time_na=hna["transmit_time_sec"][()]
    snr_na=hna["range_delay_power"][()]
    echo_delay_us=hna["echo_delay_us"][()]
    epoch_time=hna["time_epoch_unix_sec"][()]
    hna.close()

    plt.subplot(221)
    plt.pcolormesh(time_na,echo_delay_us,10.0*n.log10(snr_na.T),vmin=2)
    plt.xlabel("Time (since %d)"%(epoch_time))
    plt.ylabel(r"Delay ($\mu s$")
    plt.xlim([35,39])
    plt.subplot(222)
    plt.pcolormesh(time_na,echo_delay_us,10.0*n.log10(snr_na.T),vmin=2)
    plt.xlabel("Time (since %d)"%(epoch_time))
    plt.ylabel(r"Delay ($\mu s$")
    cb=plt.colorbar()
    cb.set_label("SNR (dB)")


    plt.subplot(223)
    plt.pcolormesh(time_na,echo_delay_us,doppler_na.T,cmap="turbo")
    plt.xlabel("Time (since %d)"%(epoch_time))
    plt.ylabel(r"Delay ($\mu s$")
    plt.xlim([35,39])
    plt.subplot(224)
    plt.pcolormesh(time_na,echo_delay_us,doppler_na.T,cmap="turbo")
    plt.xlabel("Time (since %d)"%(epoch_time))
    plt.ylabel(r"Delay ($\mu s$")
    cb=plt.colorbar()
    cb.set_label("Doppler-shift (Hz)")

    plt.show()


for f in files:
    plot_file(f)
