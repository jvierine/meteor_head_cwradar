import h5py
import numpy as n
import matplotlib.pyplot as plt
import analyze_bolide as ab
from mpi4py import MPI
import scipy.constants as c
import scipy.interpolate as sint
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
      

def decimate(a,declen=10):
    a.shape=(int(len(a)/declen),declen)
    return(n.mean(a,axis=1))


#b=decimate(n.arange(1000))
#plt.plot(b)
#plt.show()

def range_doppler_matched_filter(fname,         # data file name
                                 rg=[50,120],   # range ranges to analyze (in data samples)
                                 i0=35*100000,  # first sample to process
                                 i1=120*100000, # last sample to process
                                 step=500,      # how many samples to step in time
                                 interp_len=2,  # interpolation factor for sub sample delay search
                                 fftlen=4096*2, # how many fft points for doppler search
                                 fftdec=1,      # reduce fft bandwidth by interp_len*fftdec
                                 n_codes = 1,   # how many codes are analyzed together
                                 plot=False,
                                 plot_file="tmp.png",
                                 title="Range-doppler MF",
                                 use_spline=False,
                                 remove_spikes=False,
                                 rmodel=None,
                                 dmodel=None,
                                 t_model=None
                                 ):
    """
    range-doppler matched filter
    sub sample range resolution
    """
#    ,rmodel=p["rmodel"],dmodel=p["dmodel"],t_model=t_model
    z=[]
    tidx0=0
    for fn in fname:
        zo=n.array([],dtype=n.complex64)
        for f in fn:
            print(f)
            h=h5py.File(f,"r")
            # interpolate received signal to a higher sample-rate
            zo=n.concatenate((zo,h["rf_data"][()][:,0]))
            if tidx0 == 0:
                tidx0 = h["rf_data_index"][()][0,0]

        if use_spline:
            import scipy.interpolate as sint
            x=n.arange(len(zo))+0.5
            x[0]-=1
            x[-1]+=1
            xi=n.arange(len(zo)*interp_len)/interp_len+0.5/interp_len
            zf=sint.interp1d(x,zo)
            z.append(zf(xi))
        else:
            z.append(n.repeat(zo,interp_len))

    code_len=1000
    codes=ab.get_codes(interp_len,code_len=code_len,seeds=seed,n_codes=1)

    # repeat the code throughout the echo length so that we can
    # read the transmitted code at an arbitrary delay and get sub code length time steps
    repcodes=[]
    nrep=int(n.round(len(z[0])/(interp_len*code_len)))
    for ci in range(len(codes)):
        repcodes.append(n.tile(codes[ci],nrep))
        
    # number of time steps
    n_window=int(n.floor((i1-2*code_len-i0)*interp_len/(interp_len*step)))

    # time indices
    tx_idx=[]

    rgs_interp=n.arange(rg[0]*interp_len,rg[1]*interp_len)
    n_rg=len(rgs_interp)
    drg=c.c/100e3/interp_len

    P=n.zeros([n_window,n_rg],dtype=n.float32)
    # doppler shifts
    if fftlen < (n_codes*code_len/fftdec):
        fftlen=(n_codes*code_len/fftdec)
        print("adjusting fftlen to %d"%(fftlen))
        
    freqs=n.fft.fftshift(n.fft.fftfreq(fftlen,d=1/(100e3/fftdec)))
    dopvel=c.c*freqs/2/32.8e6
    D=n.zeros([n_window,n_rg],dtype=n.float32)
    D_Hz=n.zeros([n_window,n_rg],dtype=n.float32)    

    specs=[]
    tnows=[]
    for i in range(n_window):
      #  print("%d/%d"%(i,n_window))
        # index of transmit pulse start
        idx0 = i0*interp_len + i*step*interp_len
        # record transmit time for this window
        tx_idx.append(idx0/interp_len)
        noise_floors=[]
        
        tnow=(idx0/interp_len+tidx0)/100e3 
        
        if (tnow > n.min(t_model)) and (tnow < n.max(t_model)):
            #print("has model %f %d %f %d"%(rmodel(tnow),dmodel(tnow)))
            rg_now = int(rmodel(tnow)/drg)
            fi=n.argmin(n.abs(freqs-dmodel(tnow)))
            print("has model %f %d %f %d"%(rmodel(tnow),rg_now,dmodel(tnow),fi))

            tnows.append(tnow)
            if True:
                # allow sub code length steps!
                DP=n.zeros(fftlen,dtype=n.float32)
                # both pols
                for zi in range(len(z)):
                    # range shifted echo
                    echo = z[zi][ (idx0+rg_now):(idx0+n_codes*code_len*interp_len + +rg_now) ]
                    # notch spikes to zero
                    if remove_spikes:
                        abs_echo=n.abs(echo)
                        noise_std_est=n.nanmedian(abs_echo)
                        echo[abs_echo>(3*noise_std_est)]=0.0

                    for ci in range(len(codes)):
                        # what was transmitted for this echo?
                        code = repcodes[ci][(idx0):(idx0+n_codes*code_len*interp_len)]
                        # average the doppler spectrum for all codes
                        DP+=n.fft.fftshift(n.abs(n.fft.fft(decimate(echo*n.conj(code),fftdec*interp_len),fftlen))**2.0)
                    noise_floors.append(n.median(DP))
                specs.append(DP[(fi-200):(fi+200)])

    specs=n.array(specs)
    mfi=n.argmin(n.abs(freqs))
    print(mfi)
    freqw=freqs[(mfi-200):(mfi+200)]
    tnows=n.array(tnows)
    dB=10.0*n.log10(specs.T)#-0.6*nfloor)/(0.6*nfloor))
    nfloor=n.median(dB)
    h=h5py.File("meteor_fit.h5","r")
    m_t=h["model_time_unix"][()]+h["epoch_unix"][()]
    llh=h["model_lat_lon_h"][()]
    h.close()
#    nfloor=n.nanmedian(dB)
    fig,(ax0,ax1)=plt.subplots(2,1,sharex=True,figsize=(8,8))
    #plt.subplot(122)
    s=ax0.pcolormesh(tnows-tidx0/100e3,freqw,dB-nfloor,vmin=0)
    ax0.set_title("Meteor head echo Doppler spectrum (SV)")
    ax0.set_ylabel("Doppler shift relative to model (Hz)")
    cb=fig.colorbar(s,ax=ax0,location="bottom")
    cb.set_label("SNR (dB)")
    ax1.plot(m_t-tidx0/100e3,llh[:,2]/1e3,color="C0")
    ax1.set_ylabel("Height (km)",color="C0")
    ax1.grid()
    ax11=ax1.twinx()
    ax11.plot(tnows-tidx0/100e3,rmodel(tnows)/1e3,color="C1")
    ax11.set_ylabel("TX-SV range (km)",color="C1")
    ax1.set_xlabel("Time (seconds since 12:56 UT)")
#    cb=plt.colorbar()
 #   cb.set_label("SNR (dB)")
  #  plt.xlabel("Time (s after 12:56 UT)")
   # plt.ylabel("Doppler shift (Hz)")
    #plt.colorbar()
    plt.show()
#                plt.plot(10.0*n.log10(DP[(fi-100):(fi+100)]))
 #               plt.show()


#                # record the doppler shift
 #               dopidx=n.argmax(DP)                
  #              D[i,rgi]=dopvel[dopidx]
   #             D_Hz[i,rgi]=freqs[dopidx]
    #            P[i,rgi]=DP[dopidx]
     #           noise_floor=n.median(noise_floors)
      #          P[i,:]=(P[i,:]-noise_floor)/noise_floor



if __name__ == "__main__":

        # Obenberger, K., Pfeffer, N., Chau, J., & Vierinen, J. (2025). New Mexico 2024-01-18 Bolide head and trail echo [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14945152
    
    ho=h5py.File("tristatic_model.h5","r")
    r_na=ho["r_na"][()]
    r_sv=ho["r_sv"][()]
    r_ws=ho["r_ws"][()]
    t_model=ho["t"][()]
    d_na=ho["d_na"][()]
    d_ws=ho["d_ws"][()]
    d_sv=ho["d_sv"][()]
    ho.close()

    def get_model(t,f):
        t0=n.mean(t)
        nm=len(t)
        A=n.zeros([nm,4])
        tn=t-t0
        A[:,0]=1
        A[:,1]=tn
        A[:,2]=tn**2
        A[:,3]=tn**3
        xhat=n.linalg.lstsq(A,f)[0]
        def pfun(ti):
            return(xhat[0]+xhat[1]*(ti-t0)+xhat[2]*(ti-t0)**2+xhat[3]*(ti-t0)**3)
        return(pfun)
    rnamodel=get_model(t_model,r_na)
    #rnamodel=sint.interp1d(t_model,r_na)

#    plt.plot(t_model,r_na,".")
 #   plt.plot(t_model,rnamodel0(t_model))
  #  plt.show()
    dnamodel=get_model(t_model,d_na)
    rsvmodel=get_model(t_model,r_sv)
    dsvmodel=get_model(t_model,d_sv)
    rwsmodel=get_model(t_model,r_ws)
    dwsmodel=get_model(t_model,d_ws)

    pars=[
        {"fname":[["data/NA/CH000/rf@1705582560.000.h5","data/NA/CH000/rf@1705582620.000.h5"],
                  ["data/NA/CH001/rf@1705582560.000.h5","data/NA/CH001/rf@1705582620.000.h5"]],
         "title":"NA",
         "rmodel":rnamodel,
         "dmodel":dnamodel,
         "plot_file":"rd_na.png"},
        {"fname":[["data/SV/CH000/rf@1705582560.000.h5","data/SV/CH000/rf@1705582620.000.h5"],
                  ["data/SV/CH001/rf@1705582560.000.h5","data/SV/CH001/rf@1705582620.000.h5"]],
         "title":"SV",
         "rmodel":rsvmodel,
         "dmodel":dsvmodel,
         "plot_file":"rd_sv.png"},
        {"fname":[["data/WS/CH000/rf@1705582560.000.h5","data/WS/CH000/rf@1705582620.000.h5"],
                  ["data/WS/CH001/rf@1705582560.000.h5","data/WS/CH001/rf@1705582620.000.h5"]],
         "title":"WS",
         "rmodel":rwsmodel,
         "dmodel":dwsmodel,
         "plot_file":"rd_ws.png"},
    ]
    t_model[0]=t_model[0]-0.2
    t_model[-1]=t_model[-1]+0.1

    for pi in range(rank,len(pars),size):
        p=pars[pi]

        ps=range_doppler_matched_filter(p["fname"],plot=True,title=p["title"],plot_file=p["plot_file"],rmodel=p["rmodel"],dmodel=p["dmodel"],t_model=t_model)


    
