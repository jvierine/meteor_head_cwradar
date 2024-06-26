import h5py
import numpy as n
import matplotlib.pyplot as plt
import analyze_bolide as ab

receivers = [ 'IS', 'NA', 'SV','WS','LA','KH']
tx_name = 'NM'
seed = [5,20,608,1686,2755,4972]

nm_rx_gps = [
      [34.9482, -106.512, 1707.2],    #   IS: ISOON:     
      [34.2475, -107.641, 2147.4],    #   NA: North Arm VLA or SKYWAVE VLA:
      [34.3491, -106.886, 1478.9],    #   SV: SEVILLETA:
      [33.7204, -106.739, 1473.6],    #   SW:  WSMR:   
      [35.8701, -106.326, 2272],      #   LA: LOS ALAMOS:
      [35.0698, -106.29, 2274.8]      #    KH KEN'S HOUSE:
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

    print(z.shape)

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

def range_doppler_matched_filter(fname,
                                 rg=[50,120],
                                 i0=3400*1000,
                                 i1=4000*1000,
                                 step=100,
                                 interp_len=10,
                                 fftlen=4000,
                                 plot=False,
                                 plot_file="tmp.png",
                                 title="Range-doppler MF"
                                 ):
    """
    range-doppler matched filter
    sub sample range resolution
    """
    
    h=h5py.File(fname,"r")
    z=n.repeat(h["rf_data"][()][:,0],interp_len)

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
    freqs=n.fft.fftfreq(fftlen,d=1/(100e3))
    dopvel=3e8*freqs/2/32.8e6
    D=n.zeros([n_window,n_rg],dtype=n.float32)    
    
    for i in range(n_window):
        print(i,n_window)
        # index of transmit pulse start
        idx0 = i0*interp_len + i*step*interp_len
        # record transmit time for this window
        tx_idx.append(idx0/interp_len)
        for rgi in range(n_rg):
            echo = z[ (idx0+rgs_interp[rgi]):(idx0+code_len*interp_len + rgs_interp[rgi]) ]
            # notch spikes to zero
            abs_echo=n.abs(echo)
            noise_std_est=n.nanmedian(abs_echo)
            echo[abs_echo>(3*noise_std_est)]=0.0
            
            # allow sub code length steps!
            DP=n.zeros(fftlen,dtype=n.float32)
            for ci in range(len(codes)):
                code = repcodes[ci][(idx0):(idx0+code_len*interp_len)]
                DP+=n.abs(n.fft.fft(decimate(echo*n.conj(code),interp_len),fftlen))**2.0
                
            # record the doppler shift
            dopidx=n.argmax(DP)                
            D[i,rgi]=dopvel[dopidx]
            P[i,rgi]=DP[dopidx]

    dB=10*n.log10(P.T)
    tx_idx=n.array(tx_idx)
    tx_ts=tx_idx/100e3
    vlow,vhigh=n.percentile(dB.flatten(), [25 ,90])
    ho=h5py.File("%s.h5"%(plot_file),"w")
    ho["tx_idx"]=tx_idx
    ho["tx_ts"]=tx_ts
    ho["rgs"]=rgs_interp
    ho["interp"]=interp_len
    ho["P"]=P
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

    
    ps={}
    # IS is pure noise. not usable
    #ps["IS0"]=range_doppler_matched_filter("/data0/simone-nm/raw_data/IS/CH000/rf@1705582560.000.h5",plot=True)
    #ps["IS1"]=range_doppler_matched_filter("/data0/simone-nm/raw_data/IS/CH001/rf@1705582560.000.h5",plot=True)

    ps["NA0"]=range_doppler_matched_filter("/data0/simone-nm/raw_data/NA/CH000/rf@1705582560.000.h5",plot=True,title="NA0",plot_file="rd_na0.png")
    ps["NA1"]=range_doppler_matched_filter("/data0/simone-nm/raw_data/NA/CH001/rf@1705582560.000.h5",plot=True,title="NA1",plot_file="rd_na1.png")
        

    ps["SV0"]=range_doppler_matched_filter("/data0/simone-nm/raw_data/SV/CH000/rf@1705582560.000.h5",plot=True,title="SV0",plot_file="rd_sv0.png")
    ps["SV1"]=range_doppler_matched_filter("/data0/simone-nm/raw_data/SV/CH001/rf@1705582560.000.h5",plot=True,title="SV1",plot_file="rd_sv1.png")

    ps["WS0"]=range_doppler_matched_filter("/data0/simone-nm/raw_data/WS/CH000/rf@1705582560.000.h5",plot=True,title="WS0",plot_file="rd_ws0.png")
    ps["WS1"]=range_doppler_matched_filter("/data0/simone-nm/raw_data/WS/CH001/rf@1705582560.000.h5",plot=True,title="WS1",plot_file="rd_ws1.png")


    
