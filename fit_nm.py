import numpy as n
import matplotlib.pyplot as plt
import h5py
import scipy.constants as c
import analyze_nm as nm
import jcoord
import fastecef2h 



import cartopy.crs as ccrs
from datetime import datetime
from cartopy.feature.nightshade import Nightshade
import cartopy.feature as cfeature

p_tx=jcoord.geodetic2ecef(nm.tx_gps[0],nm.tx_gps[1],nm.tx_gps[2])

p_0=jcoord.geodetic2ecef(nm.tx_gps[0],nm.tx_gps[1],100e3)
ecef2h=fastecef2h.get_fastecef2h(p_0)

p_na=jcoord.geodetic2ecef(nm.nm_rx_gps[1][0],
                          nm.nm_rx_gps[1][1],
                          nm.nm_rx_gps[1][2])
p_sv=jcoord.geodetic2ecef(nm.nm_rx_gps[2][0],
                          nm.nm_rx_gps[2][1],
                          nm.nm_rx_gps[2][2])
p_ws=jcoord.geodetic2ecef(nm.nm_rx_gps[3][0],
                          nm.nm_rx_gps[3][1],
                          nm.nm_rx_gps[3][2])


def get_head_echo(fname="rd_na0.png.h5",
                  plot=False):

    h=h5py.File(fname,"r")
    print(h.keys())

    P=h["P"][()]
    P_masked=n.copy(P)
    D=h["D"][()]
    tx_idx=h["tx_idx"][()]
    D_Hz=h["D_Hz"][()]

    interp=h["interp"][()]
    rgs_m=c.c*h["rgs"][()]/(interp*100e3)

    t0=h["tidx0"][()]/100e3
    D_masked=n.copy(D)
    h.close()
    print(P.shape)
    # snr
    for i in range(P.shape[0]):
        noise_floor=n.median(P[i,:])
        P[i,:]=(P[i,:]-noise_floor)/noise_floor
        P_masked[i,:]=(P_masked[i,:]-noise_floor)/noise_floor    


    D_masked[P<0.5]=n.nan
    D_Hz[P<0.5]=n.nan
    D_Hz[n.abs(D)<10e3]=n.nan
    P_masked[n.abs(D) < 10e3]=0.0
    D_masked[n.abs(D) < 10e3]=n.nan

    dops=[]
    rgs=[]
    snrs=[]
    tms=[]
    for i in range(P.shape[0]):
        rgmax=n.argmax(P_masked[i,:])

        if P_masked[i,rgmax] > 1.05:
            rgs.append(rgs_m[rgmax])
            dops.append(D_Hz[i,rgmax])
            snrs.append(P_masked[i,rgmax])
            tms.append(tx_idx[i])

    if plot:
        plt.subplot(121)
        plt.pcolormesh(tx_idx,rgs_m,P_masked.T,vmin=0,vmax=2)
        plt.plot(tms,rgs,"x")
        plt.colorbar()
        plt.subplot(122)
        plt.pcolormesh(tx_idx,rgs_m,D_masked.T,cmap="turbo",vmin=-76e3,vmax=0)
        plt.plot(tms,rgs,"x")
        plt.colorbar()
        plt.show()
        
        

    return(n.array(tms)/100e3+t0,n.array(rgs),n.array(dops),n.array(snrs))

def polyfit(t,r):
    t0=n.mean(t)
    A=n.zeros([len(t),4])
    A[:,0]=1.0
    A[:,1]=t-t0
    A[:,2]=(t-t0)**2.0
    A[:,3]=(t-t0)**3.0
    xhat=n.linalg.lstsq(A,r)[0]
    def rfun(tt):
        return(xhat[0]+xhat[1]*(tt-t0)+xhat[2]*(tt-t0)**2.0+xhat[3]*(tt-t0)**3.0)
    return(rfun)


def fit_pos(rna,rws,rsv):
    """
    Given three ranges between transmit and receive, find position in ECEF
    """
    rs=n.array([rna,rws,rsv])
    def ss(x):
        txdist=n.linalg.norm(x-p_tx)
        mrna=txdist + n.linalg.norm(p_na-x)
        mrws=txdist + n.linalg.norm(p_ws-x)
        mrsv=txdist + n.linalg.norm(p_sv-x)
        model_range=n.array([mrna,mrws,mrsv])
        s=n.sum(n.abs(rs-model_range)**2.0 )
        return(s)
    
    import scipy.optimize as so
    xhat=so.fmin(ss,p_tx)
    print(xhat)
    return(xhat)


def simple_trajectory_fit(tna,
                          tws,
                          tsv,
                          rna,
                          rws,
                          rsv,
                          plot=True):
    # initial fit just based on time delays
    # interpolate also to get the same time for all three links
    # fit polynomial for range
    rnafun=polyfit(tna,rna)
    rwsfun=polyfit(tws,rws)
    rsvfun=polyfit(tsv,rsv)

    tmin=n.min(tna)
    tmax=n.max(tna)
    ts=n.linspace(tmin,tmax,num=50)
    poss=[]
    llhs=[]
    
    for i in range(len(ts)):
        pos=fit_pos(rnafun(ts[i]),
                    rwsfun(ts[i]),
                    rsvfun(ts[i]))
        llh=jcoord.ecef2geodetic(pos[0],pos[1],pos[2])
        llhs.append(llh)
        poss.append(pos)
        
    poss=n.array(poss)
    llhs=n.array(llhs)
    
    dt=n.diff(ts)[0]
    print(dt)
    vx=n.gradient(poss[:,0],dt)
    vy=n.gradient(poss[:,1],dt)
    vz=n.gradient(poss[:,2],dt)

    p0_est = poss[0,:]
    v0_est = (poss[-1,:]-poss[0,:])/(ts[-1]-ts[0])

    if plot:
 #       plt.subplot(221)
        plt.plot(tna,rna/1e3,".",label="NA")
        plt.plot(tna,rnafun(tna)/1e3)
        plt.plot(tws,rws/1e3,".",label="WS")
        plt.plot(tws,rwsfun(tws)/1e3)
        plt.plot(tsv,rsv/1e3,".",label="SV")
        plt.plot(tsv,rsvfun(tsv)/1e3)
        plt.legend()
        plt.xlabel("Time (unix)")
        plt.ylabel("Range (km)")
#        plt.subplot(222)
#        plt.plot(tna,dna,".")
#        plt.plot(tws,dws,".")
#        plt.plot(tsv,dsv,".")
#        plt.subplot(223)
#        plt.plot(tna,sna,".")
#        plt.plot(tws,sws,".")
#        plt.plot(tsv,ssv,".")
        plt.show()
        
    
    return(poss,llhs,vx,vy,vz,ts,p0_est,v0_est)






def fit_trajectory():
    # extract the range, doppler and snr of the head echo
    tna,rna,dna,sna=get_head_echo("rd_na0.png.h5")
    tws,rws,dws,sws=get_head_echo("rd_ws1.png.h5")
    tsv,rsv,dsv,ssv=get_head_echo("rd_sv0.png.h5")

    poss, llhs, vx, vy, vz, ts, p0_est, v0_est = simple_trajectory_fit(tna,tws,tsv,rna,rws,rsv)
    print(v0_est)
    print(n.linalg.norm(v0_est))

    plt.subplot(131)
    plt.plot(llhs[:,1],llhs[:,0])
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.subplot(132)
    plt.plot(ts,llhs[:,2]/1e3)
    plt.xlabel("Time (unix)")
    plt.ylabel("Height (km)")
    
    plt.subplot(133)
    plt.plot(ts,n.sqrt(vx**2.0+vy**2.0+vz**2.0)/1e3)
    plt.xlabel("Time (unix)")
    plt.ylabel("Velocity (km/s)")
    plt.show()


    fig = plt.figure(figsize=[8, 6.4])
    ax = fig.add_subplot( 1,1, 1, projection=ccrs.Orthographic(nm.tx_gps[1], nm.tx_gps[0]))
    data_projection = ccrs.PlateCarree()
    this_frame_date=datetime.utcfromtimestamp(n.min(tna))
    ax.coastlines(zorder=3)
    ax.stock_img()
    ax.gridlines()

    ax.add_feature(Nightshade(this_frame_date))
#    ax.set_extent((-60,-100,20,60),crs=ccrs.PlateCarree())
    lons=n.copy(llhs[:,0])
 #   negidx=n.where(lons<0)[0]
#    lons[negidx]=lons[negidx]+360
    # make a scatterplot
    mp=ax.scatter(llhs[:,1],
                  lons,

                  c=llhs[:,2]/1e3,
                  vmin=70,
                  vmax=120,
                  transform=data_projection,
                  zorder=2)
    cb=plt.colorbar(mp,ax=ax)
    cb.set_label("Altitude (km)")
    plt.show()
    

fit_trajectory()
