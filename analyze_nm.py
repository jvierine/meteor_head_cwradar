import h5py
import numpy as n
import matplotlib.pyplot as plt
import analyze_bolide as ab
from mpi4py import MPI
import scipy.constants as c

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("comm rank %d size %d"%(rank,size))

receivers = [ 'IS', 'NA', 'SV','WS','LA','KH']
tx_name = 'NM'
seed = [5,20,608,1686,2755,4972]

nm_rx_gps = [
      [34.9482, -106.512, 1707.2],    #   IS: ISOON:     
      [34.2475, -107.641, 2147.4],    #   NA: North Arm VLA or SKYWAVE VLA:
      [34.3491, -106.886, 1478.9],    #   SV: SEVILLETA:
      [33.7204, -106.739, 1473.6],    #   SW:  WSMR:   
      [35.8701, -106.326, 2272],      #   LA: LOS ALAMOS:
      [35.0698, -106.29, 2274.8]      #   KH KEN'S HOUSE:
    ]

tx_ant = [
    [ 0, 0, 0],
    [0.0208, 24.9132, -0.353],
    [23.8559, 7.6436, 0.26],
    [14.668, -20.2828, 0.494],
    [-14.5978, -20.2824, 0.077],
    [-23.5428, 7.5366, -0.495]
]

tx_gps = [35.002, -106.526, 1674.8]

tx_freq=32.8e6
tx_wavelength=c.c/tx_freq
      

def overview(fname="/data0/simone-nm/raw_data/NA/CH000/rf@1705582560.000.h5",
             title="overview",
             plot_file="tmp.png"):
    h=h5py.File(fname,"r")
    print(h.keys())

    interp_len=1
    
    zo=h["rf_data"][()][:,0]
    # TODO: WE NEED TO DO A SPIKE REMOVAL
    # some stations are just too noisy
    plt.plot(zo.real)
    plt.plot(zo.imag)
    plt.show()
    plt.plot(10.0*n.log10(n.abs(n.fft.fftshift(n.fft.fft(zo)))**2.0))
    plt.title(title)
    plt.show()

    z=n.repeat(zo,interp_len)

    code_len=1000
    codes=ab.get_codes(interp_len,code_len=code_len,seeds=seed,n_codes=1)
    codes_f=[]

    for c in codes:
        print("code length",len(c))
        codes_f.append(n.conj(n.fft.fft(c)))


    n_window=int(len(z)/(code_len*interp_len))

    P=n.zeros([n_window,code_len*interp_len],dtype=n.float32)
    for i in range(n_window):
        print(i,n_window)
        Z=n.fft.fft(z[(i*code_len*interp_len):(i*code_len*interp_len+interp_len*code_len)])
        print(Z.shape)
        print(len(Z))
        for ci in range(len(codes_f)):
            print(len(codes_f[ci]))
            cc=n.abs(n.fft.ifft(Z*codes_f[ci]))**2.0
            P[i,:]+=cc

    dB=10*n.log10(P.T)
    vlow,vhigh=n.percentile(dB.flatten(), [10 ,90])
    plt.pcolormesh(dB,vmin=vlow,vmax=vhigh,cmap="turbo")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

def decimate(a,declen=10):
    a.shape=(int(len(a)/declen),declen)
    return(n.mean(a,axis=1))


#b=decimate(n.arange(1000))
#plt.plot(b)
#plt.show()

def range_doppler_matched_filter(fname,         # data file name
                                 rg=[50,120],   # range ranges to analyze (in data samples)
                                 i0=3500*1000,  # first sample to process
                                 i1=3900*1000,  # last sample to process
                                 step=250,      # how many samples to step in time
                                 interp_len=10, # interpolation factor for sub sample delay search
                                 fftlen=4096*2, # how many fft points for doppler search
                                 fftdec=2,      # reduce fft bandwidth by interp_len*fftdec
                                 n_codes = 1,   # how many codes are analyzed together
                                 plot=False,
                                 plot_file="tmp.png",
                                 title="Range-doppler MF",
                                 use_spline=False,
                                 remove_spikes=True
                                 ):
    """
    range-doppler matched filter
    sub sample range resolution
    """
    
    h=h5py.File(fname,"r")
    
    
    # interpolate received signal to a higher sample-rate
    zo=h["rf_data"][()][:,0]

    tidx0 = h["rf_data_index"][()][0,0]
    
#    use_spline=
    if use_spline:
        import scipy.interpolate as sint
        x=n.arange(len(zo))+0.5
        x[0]-=1
        x[-1]+=1
        xi=n.arange(len(zo)*interp_len)/interp_len+0.5/interp_len
        zf=sint.interp1d(x,zo)
        z=zf(xi)
    else:
        z=n.repeat(zo,interp_len)

    code_len=1000
    codes=ab.get_codes(interp_len,code_len=code_len,seeds=seed,n_codes=1)

    # repeat the code so that we can read the transmitted
    # code at an arbitrary delay and get sub code length time steps
    repcodes=[]
    nrep=int(n.round(len(z)/(interp_len*code_len)))
    for ci in range(len(codes)):
        repcodes.append(n.tile(codes[ci],nrep))
        
    # number of time steps
    n_window=int(n.floor((i1-i0)*interp_len/(interp_len*step)))

    # time indices
    tx_idx=[]

    rgs_interp=n.arange(rg[0]*interp_len,rg[1]*interp_len)
    n_rg=len(rgs_interp)

    P=n.zeros([n_window,n_rg],dtype=n.float32)
    # doppler shifts
    if fftlen < (n_codes*code_len/fftdec):
        fftlen=(n_codes*code_len/fftdec)
        print("adjusting fftlen to %d"%(fftlen))
        
    freqs=n.fft.fftfreq(fftlen,d=1/(100e3/fftdec))
    dopvel=3e8*freqs/2/32.8e6
    D=n.zeros([n_window,n_rg],dtype=n.float32)
    D_Hz=n.zeros([n_window,n_rg],dtype=n.float32)    

    
    for i in range(n_window):
        print(i,n_window)
        # index of transmit pulse start
        idx0 = i0*interp_len + i*step*interp_len
        # record transmit time for this window
        tx_idx.append(idx0/interp_len)
        for rgi in range(n_rg):
            # range shifted echo
            echo = z[ (idx0+rgs_interp[rgi]):(idx0+n_codes*code_len*interp_len + rgs_interp[rgi]) ]
            
            # notch spikes to zero
            if remove_spikes:
                abs_echo=n.abs(echo)
                noise_std_est=n.nanmedian(abs_echo)
                echo[abs_echo>(3*noise_std_est)]=0.0
            
            # allow sub code length steps!
            DP=n.zeros(fftlen,dtype=n.float32)
            for ci in range(len(codes)):
                # what was transmitted for this echo?
                code = repcodes[ci][(idx0):(idx0+n_codes*code_len*interp_len)]
                # average the doppler spectrum for all codes
                DP+=n.abs(n.fft.fft(decimate(echo*n.conj(code),fftdec*interp_len),fftlen))**2.0
                
            # record the doppler shift
            dopidx=n.argmax(DP)                
            D[i,rgi]=dopvel[dopidx]
            D_Hz[i,rgi]=freqs[dopidx]
            P[i,rgi]=DP[dopidx]

    dB=10*n.log10(P.T)
    tx_idx=n.array(tx_idx)
    tx_ts=tx_idx/100e3
    vlow,vhigh=n.percentile(dB.flatten(), [25 ,90])
    ho=h5py.File("%s.h5"%(plot_file),"w")
    ho["tx_idx"]=tx_idx
    ho["tidx0"]=tidx0
    ho["tx_ts"]=tx_ts
    ho["rgs"]=rgs_interp
    ho["interp"]=interp_len
    ho["P"]=P
    ho["D_Hz"]=D_Hz
    ho["D"]=D    
    ho.close()
    if plot:
        fig=plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.pcolormesh(tx_ts,10*rgs_interp/interp_len,dB,vmin=vlow,vmax=vlow+3)
        plt.xlabel("Time (s)")        
        plt.title(title)
        cb=plt.colorbar()
        cb.set_label("Power (dB)")
        plt.ylabel("Delay ($\\mu$s)")
        plt.subplot(122)
        plt.pcolormesh(tx_ts,10*rgs_interp/interp_len,D.T,cmap="turbo",vmin=-70e3,vmax=0)
        plt.title(title)
        cb=plt.colorbar()
        cb.set_label("Doppler shift (m/s)")
        plt.ylabel("Delay ($\\mu$s)")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(plot_file,dpi=150)
        plt.close()
    return(P)


if __name__ == "__main__":
    plot_overviews  = False
    if plot_overviews:
        overview("/data0/simone-nm/raw_data/NA/CH000/rf@1705582560.000.h5",title="NA0",plot_file="o_na0.png")
        overview("/data0/simone-nm/raw_data/NA/CH001/rf@1705582560.000.h5",title="NA1",plot_file="o_na1.png")
        overview("/data0/simone-nm/raw_data/IS/CH000/rf@1705582560.000.h5",title="IS0",plot_file="o_is0.png")
        overview("/data0/simone-nm/raw_data/IS/CH001/rf@1705582560.000.h5",title="IS1",plot_file="o_is1.png")
        overview("/data0/simone-nm/raw_data/SV/CH000/rf@1705582560.000.h5",title="SV0",plot_file="o_sv0.png")
        overview("/data0/simone-nm/raw_data/SV/CH001/rf@1705582560.000.h5",title="SV1",plot_file="o_sv1.png")
        overview("/data0/simone-nm/raw_data/WS/CH000/rf@1705582560.000.h5",title="WS0",plot_file="o_ws0.png")
        overview("/data0/simone-nm/raw_data/WS/CH001/rf@1705582560.000.h5",title="WS1",plot_file="o_ws1.png")

    
    pars=[
        {"fname":"/data0/simone-nm/raw_data/NA/CH000/rf@1705582560.000.h5",
         "title":"NA0",
         "plot_file":"rd_na0.png"},
        {"fname":"/data0/simone-nm/raw_data/NA/CH001/rf@1705582560.000.h5",
         "title":"NA1",
         "plot_file":"rd_na1.png"},
        {"fname":"/data0/simone-nm/raw_data/SV/CH000/rf@1705582560.000.h5",
         "title":"SV0",
         "plot_file":"rd_sv0.png"},
        {"fname":"/data0/simone-nm/raw_data/SV/CH001/rf@1705582560.000.h5",
         "title":"SV1",
         "plot_file":"rd_sv1.png"},
        {"fname":"/data0/simone-nm/raw_data/WS/CH000/rf@1705582560.000.h5",
         "title":"WS0",
         "plot_file":"rd_ws0.png"},
        {"fname":"/data0/simone-nm/raw_data/WS/CH001/rf@1705582560.000.h5",
         "title":"WS1",
         "plot_file":"rd_ws1.png"},
    ]

    for pi in range(rank,len(pars),size):
        p=pars[pi]
        ps=range_doppler_matched_filter(p["fname"],plot=True,title=p["title"],plot_file=p["plot_file"])


    
