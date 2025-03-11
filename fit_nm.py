import numpy as n
import matplotlib.pyplot as plt
import h5py
import scipy.constants as c
import analyze_nm as nm
import jcoord
#import fastecef2h

import meteor_trajectory_fit as tf


import cartopy.crs as ccrs
from datetime import datetime
from cartopy.feature.nightshade import Nightshade
import cartopy.feature as cfeature

p_tx=jcoord.geodetic2ecef(nm.tx_gps[0],nm.tx_gps[1],nm.tx_gps[2])

#p_0=jcoord.geodetic2ecef(nm.tx_gps[0],nm.tx_gps[1],100e3)
#ecef2h=fastecef2h.get_fastecef2h(p_0)

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
        plt.pcolormesh(tx_idx/100e3+t0,rgs_m,P_masked.T,vmin=0,vmax=2)
        plt.plot(n.array(tms)/100e3+t0,rgs,"x")
        plt.colorbar()
        plt.subplot(122)
        plt.pcolormesh(tx_idx/100e3+t0,rgs_m,D_masked.T,cmap="turbo",vmin=-76e3,vmax=0)
        plt.plot(n.array(tms)/100e3+t0,rgs,"x")
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


def forward_model_meas3(p0,  # meteor position initially
                        v0,  # meteor velocity initially
                        tm0,tm1,tm2, # measurement times
                        p_tx,        # transmit positoin
                        p_rx0,p_rx1,p_rx2,  # receiver positions
                        rho_m_r=0.1,plot=False,
                        rho_m=1000):

    all_time=n.concatenate([tm0,tm1,tm2])
    t0=n.min(all_time)
    max_time=n.max(all_time)-n.min(all_time)
    
    m_t, m_p, m_u=tf.forward_model(p0=p0,
                                   v0=v0,
                                   plot=plot,
                                   max_t=max_time+10e-3,
                                   rho_m_r=rho_m_r)
    
    import scipy.interpolate as interp
    
    posfun_x=interp.interp1d(m_t, m_p[:,0])
    posfun_y=interp.interp1d(m_t, m_p[:,1])
    posfun_z=interp.interp1d(m_t, m_p[:,2])
    dt=1e-3
    model_pos0=n.vstack([posfun_x(tm0-t0),posfun_y(tm0-t0),posfun_z(tm0-t0)])
    model_pos0_dt=n.vstack([posfun_x(tm0-t0+dt),posfun_y(tm0-t0+dt),posfun_z(tm0-t0+dt)])
    
    model_pos1=n.vstack([posfun_x(tm1-t0),posfun_y(tm1-t0),posfun_z(tm1-t0)])
    model_pos1_dt=n.vstack([posfun_x(tm1-t0+dt),posfun_y(tm1-t0+dt),posfun_z(tm1-t0+dt)])
    
    model_pos2=n.vstack([posfun_x(tm2-t0),posfun_y(tm2-t0),posfun_z(tm2-t0)])
    model_pos2_dt=n.vstack([posfun_x(tm2-t0+dt),posfun_y(tm2-t0+dt),posfun_z(tm2-t0+dt)])

    # distance from meteor to transmitter
    dist_tx_meteor0 = n.linalg.norm(p_tx[:,None]-model_pos0,axis=0)
    dist_meteor_rx0 = n.linalg.norm(p_rx0[:,None]-model_pos0,axis=0)

    dist_tx_meteor0_dt = n.linalg.norm(p_tx[:,None]-model_pos0_dt,axis=0)
    dist_meteor_rx0_dt = n.linalg.norm(p_rx0[:,None]-model_pos0_dt,axis=0)

    dist_tx_meteor1 = n.linalg.norm(p_tx[:,None]-model_pos1,axis=0)
    dist_meteor_rx1 = n.linalg.norm(p_rx1[:,None]-model_pos1,axis=0)

    dist_tx_meteor1_dt = n.linalg.norm(p_tx[:,None]-model_pos1_dt,axis=0)
    dist_meteor_rx1_dt = n.linalg.norm(p_rx1[:,None]-model_pos1_dt,axis=0)

    
    dist_tx_meteor2 = n.linalg.norm(p_tx[:,None]-model_pos2,axis=0)
    dist_meteor_rx2 = n.linalg.norm(p_rx2[:,None]-model_pos2,axis=0)

    dist_tx_meteor2_dt = n.linalg.norm(p_tx[:,None]-model_pos2_dt,axis=0)
    dist_meteor_rx2_dt = n.linalg.norm(p_rx2[:,None]-model_pos2_dt,axis=0)

    dop0=((dist_tx_meteor0_dt+dist_meteor_rx0_dt) - (dist_tx_meteor0+dist_meteor_rx0))/nm.tx_wavelength/dt
    dop1=((dist_tx_meteor1_dt+dist_meteor_rx1_dt) - (dist_tx_meteor1+dist_meteor_rx1))/nm.tx_wavelength/dt
    dop2=((dist_tx_meteor2_dt+dist_meteor_rx2_dt) - (dist_tx_meteor2+dist_meteor_rx2))/nm.tx_wavelength/dt
    
    return(dist_tx_meteor0+dist_meteor_rx0,
           dist_tx_meteor1+dist_meteor_rx1,
           dist_tx_meteor2+dist_meteor_rx2,
           dop0,dop1,dop2)

def correlation_from_covariance(covariance):
    """
    from github.com/wiso
    """
    v = n.sqrt(n.diag(covariance))
    outer_v = n.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return(correlation)

def msis_meteor_fit(p0_est,
                    v0_est,
                    tna,
                    tws,
                    tsv,
                    rna,
                    rws,
                    rsv,
                    dna,
                    dws,
                    dsv,
                    p_tx,
                    p_na,
                    p_ws,
                    p_sv,
                    plot=False,
                    ofname="meteor_fit.h5",
                    overlap_factor=4): # how much do the measurements overlap (4 means that 1/4 th of a code length time step)
    """
    fit with msis atmospheric density for drag
    """
    calc_cov=False
    fix_rho_m_r=False 
    fixed_rho_m_r=0.1
    C=None
    def ss(x):
        global C
        p0 = x[0:3]
        v0 = x[3:6]
        rho_m_r=x[6]
        if fix_rho_m_r:
            rho_m_r=fixed_rho_m_r
            
        model_rna,model_rws,model_rsv,model_dna,model_dws,model_dsv, = forward_model_meas3(p0,v0,
                                                                                           tna,tws,tsv,
                                                                                           p_tx,
                                                                                           p_na,p_ws,p_sv,
                                                                                           rho_m_r=rho_m_r,plot=plot,
                                                                                           rho_m=1000)
        
                
            
            
        s=0.0
        s+=n.sum(n.abs(model_rna-rna)**2.0)
        s+=n.sum(n.abs(model_rws-rws)**2.0)
        s+=n.sum(n.abs(model_rsv-rsv)**2.0)
        # weight the doppler down by this factor
        dopw=1e-2
        dsumna=n.abs(model_dna-dna)**2.0
        s+=dopw*n.sum(dsumna)
        dsumws=n.abs(model_dws-dws)**2.0
        s+=dopw*n.sum(dsumws)
        dsumsv=n.abs(model_dsv-dsv)**2.0
        s+=dopw*n.sum(dsumsv)

        # estimate error standard deviation
        na_std=n.std(model_rna-rna)*n.sqrt(overlap_factor)
        ws_std=n.std(model_rws-rws)*n.sqrt(overlap_factor)
        sv_std=n.std(model_rsv-rsv)*n.sqrt(overlap_factor)
        dna_std=n.std(model_dna-dna)*n.sqrt(overlap_factor)
        dws_std=n.std(model_dws-dws)*n.sqrt(overlap_factor)
        dsv_std=n.std(model_dsv-dsv)*n.sqrt(overlap_factor)

        if calc_cov:
            n_meas=2*len(tna)+2*len(tws)+2*len(tsv)
            n_par=len(x)
            J=n.zeros([n_meas,n_par])
            dpar=[100,100,100,100,100,100,1]
            for i in range(len(x)):
                x2=n.copy(x)
                x2[i]+=dpar[i]
                p0 = x2[0:3]
                v0 = x2[3:6]
                rho_m_r=x2[6]
                model_rna2,model_rws2,model_rsv2,model_dna2,model_dws2,model_dsv2 = forward_model_meas3(p0,v0,
                                                                                                        tna,tws,tsv,
                                                                                                        p_tx,
                                                                                                        p_na,p_ws,p_sv,
                                                                                                        rho_m_r=rho_m_r,plot=False,
                                                                                                        rho_m=1000)
                J[0:len(tna),i]=(model_rna2-model_rna)/dpar[i]/na_std
                J[len(tna):(len(tna)+len(tws)),i]=(model_rws2-model_rws)/dpar[i]/ws_std
                J[(len(tna)+len(tws)):(len(tna)+len(tws)+len(tsv)),i]=(model_rsv2-model_rsv)/dpar[i]/sv_std
                J[(len(tna)+len(tws)+len(tsv)):(2*len(tna)+len(tws)+len(tsv)),i]=(model_dna2-model_dna)/dpar[i]/dna_std
                J[(2*len(tna)+len(tws)+len(tsv)):(2*len(tna)+2*len(tws)+len(tsv)),i]=(model_dws2-model_dws)/dpar[i]/dws_std
                J[(2*len(tna)+2*len(tws)+len(tsv)):(2*len(tna)+2*len(tws)+2*len(tsv)),i]=(model_dsv2-model_dsv)/dpar[i]/dsv_std
            C=n.linalg.inv(n.dot(n.transpose(J),J))
            print(C)
            plt.figure(figsize=(3*8,6.4))
            plt.subplot(131)
            plt.pcolormesh(C)
            plt.title("Covariance matrix")
            plt.xlabel("$x_1$, $x_2$, $x_3$, $v_1$, $v_2$, $v_3$, $\\rho_m r$")
            plt.ylabel("$x_1$, $x_2$, $x_3$, $v_1$, $v_2$, $v_3$, $\\rho_m r$")

            plt.colorbar()
            plt.subplot(132)
            plt.pcolormesh(C[0:6,0:6])
            plt.title("Covariance matrix")
            plt.xlabel("$x_1$, $x_2$, $x_3$, $v_1$, $v_2$, $v_3$, $\\rho_m r$")
            plt.ylabel("$x_1$, $x_2$, $x_3$, $v_1$, $v_2$, $v_3$, $\\rho_m r$")
            
#            plt.xlabel("pos1,pos2,pos3,vel1,vel2,vel3")
 #           plt.ylabel("pos1,pos2,pos3,vel1,vel2,vel3")
            plt.colorbar()
            
            NC=correlation_from_covariance(C)
            plt.subplot(133)
            plt.pcolormesh(NC)
            plt.title("Correlation matrix")
            plt.xlabel("$x_1$, $x_2$, $x_3$, $v_1$, $v_2$, $v_3$, $\\rho_m r$")
            plt.ylabel("$x_1$, $x_2$, $x_3$, $v_1$, $v_2$, $v_3$, $\\rho_m r$")
            
#            plt.xlabel("pos1,pos2,pos3,vel1,vel2,vel3,$\\rho_m r$")
 #           plt.ylabel("pos1,pos2,pos3,vel1,vel2,vel3,$\\rho_m r$")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("fit_cov.png",dpi=150)
            plt.show()
            print(n.sqrt(n.diag(C)))
            
        
        if plot:

            model_rna2,model_rws2,model_rsv2,model_dna2,model_dws2,model_dsv2 = forward_model_meas3(p0,v0,
                                                                                                    tsv,tsv,tsv,
                                                                                                    p_tx,
                                                                                                    p_na,p_ws,p_sv,
                                                                                                    rho_m_r=rho_m_r,plot=False,
                                                                                                    rho_m=1000)
            
            ho=h5py.File("tristatic_model.h5","w")
            ho["t"]=tsv
            ho["r_na"]=model_rna2
            ho["r_ws"]=model_rws2
            ho["r_sv"]=model_rsv2
            ho["d_na"]=model_dna2
            ho["d_ws"]=model_dws2
            ho["d_sv"]=model_dsv2
            ho.close()


            plt.figure(figsize=(2*8,6.4))
            plt.subplot(121)
            plt.plot(tna,rna,".",color="C0",label="NA")
            plt.title("$\sigma$=%1.2f,%1.2f,%1.2f (m)"%(na_std,ws_std,sv_std))
            plt.plot(tsv,model_rna2,color="C0")
            plt.plot(tws,rws,".",color="C1",label="WS")
            plt.plot(tsv,model_rws2,color="C1")
            plt.plot(tsv,rsv,".",color="C2",label="SV")
            plt.plot(tsv,model_rsv2,color="C2")
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("Propagation distance (km)")
            plt.subplot(122)
            plt.title("$\sigma$=%1.2f,%1.2f,%1.2f (Hz)"%(dna_std,dws_std,dsv_std))
            plt.plot(tna,dna,".",label="NA",color="C0")
            plt.plot(tsv,model_dna2,color="C0")
            plt.plot(tws,dws,".",label="WS",color="C1")
            plt.plot(tsv,model_dws2,color="C1")
            plt.plot(tsv,dsv,".",label="SV",color="C2")
            plt.plot(tsv,model_dsv2,color="C2")
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("Doppler-shift (Hz)")
            plt.tight_layout()
            plt.savefig("fitres.png",dpi=150)
            plt.show()
        return(s)
    
    import scipy.optimize as sio
    x0=n.zeros(7)
    x0[0:3]=p0_est
    x0[3:6]=v0_est
    x0[6]=0.1
    xhat=sio.fmin(ss,x0)
    
    # one more time
    plot=True
    calc_cov=True
    ss(xhat)
    print(xhat)

    test_smaller_masses=False
    if test_smaller_masses:
        # try out smaller masses to determine what is no longer supported by the data
        plot=False
        calc_cov=False
        fix_rho_m_r=True
        fixed_rho_m_r=xhat[6]*0.46
        xhat2=sio.fmin(ss,xhat)
        plot=True
        calc_cov=False
        ss(xhat2)

        plot=False
        calc_cov=False
        fix_rho_m_r=True
        # 
        fixed_rho_m_r=xhat[6]*0.46**2.0
        xhat2=sio.fmin(ss,xhat)
        plot=True
        calc_cov=False
        ss(xhat2)


        plot=False
        calc_cov=False
        fix_rho_m_r=True
        fixed_rho_m_r=xhat[6]*0.46**4.0
        xhat2=sio.fmin(ss,xhat)
        plot=True
        calc_cov=False
        ss(xhat2)

        plot=False
        calc_cov=False
        fix_rho_m_r=True
        fixed_rho_m_r=xhat[6]*0.46**6.0
        xhat2=sio.fmin(ss,xhat)
        plot=True
        calc_cov=False
        ss(xhat2)

        plot=False
        calc_cov=False
        fix_rho_m_r=True
        fixed_rho_m_r=xhat[6]*0.46**7.0
        xhat2=sio.fmin(ss,xhat)
        plot=True
        calc_cov=False
        ss(xhat2)
        
        plot=False
        calc_cov=False
        fix_rho_m_r=True
        fixed_rho_m_r=xhat[6]*0.46**8.0
        xhat2=sio.fmin(ss,xhat)
        plot=True
        calc_cov=False
        ss(xhat2)
        
    all_time=n.concatenate([tna,tws,tsv])
    t0=n.min(all_time)
    max_time=n.max(all_time)-n.min(all_time)

    m_t, m_p, m_u=tf.forward_model(p0=xhat[0:3],
                                   v0=xhat[3:6],
                                   plot=True,
                                   max_t=max_time,
                                   rho_m_r=xhat[6])
    
    llhs=[]
    
    for i in range(len(m_t)):
        llh=jcoord.ecef2geodetic(m_p[i,0],m_p[i,1],m_p[i,2])
        llhs.append(llh)
        
    llhs=n.array(llhs)
    
    print("position %1.1f,%1.1f,%1.1f (ECEF meters) velocity %1.1f,%1.1f,%1.1f (ECEF m/s) epoch %1.2f (unix)"%(xhat[0],xhat[1],xhat[2],xhat[3],xhat[4],xhat[5],t0))
    print(llhs.shape)
    
    plt.subplot(121)
    plt.plot(llhs[:,1],llhs[:,0])
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.subplot(122)
    plt.plot(m_t+t0,llhs[:,2]/1e3)
    plt.xlabel("Time (unix)")
    plt.ylabel("Altitude (km)")
    plt.show()
    
    ho=h5py.File(ofname,"w")
    ho["model_time_unix"]=m_t
    ho["model_lat_lon_h"]=llhs
    ho["model_ecef"]=m_p
    ho["epoch_unix"]=t0
    ho["x0_ecef"]=xhat[0:3]
    ho["v0_ecef"]=xhat[3:6]
    ho["ml_pars"]=xhat
    print(C)
    ho["covariance"]=C
    ho["rho_m_r"]=xhat[6]
    ho.close()
    

def fit_trajectory(t0=1705582596,t1=1705582598):
    # extract the range, doppler and snr of the head echo
    tna,rna,dna,sna=get_head_echo("rd_na.png.h5")
    tws,rws,dws,sws=get_head_echo("rd_ws.png.h5")
    tsv,rsv,dsv,ssv=get_head_echo("rd_sv.png.h5")

    gidx=n.where( (tna>t0) & (tna<t1) )[0]
    tna=tna[gidx];rna=rna[gidx];dna=dna[gidx];sna=sna[gidx]
    gidx=n.where( (tws>t0) & (tws<t1) )[0]
    tws=tws[gidx];rws=rws[gidx];dws=dws[gidx];sws=sws[gidx]
    gidx=n.where( (tsv>t0) & (tsv<t1) )[0]
    tsv=tsv[gidx];rsv=rsv[gidx];dsv=dsv[gidx];ssv=ssv[gidx]

    poss, llhs, vx, vy, vz, ts, p0_est, v0_est = simple_trajectory_fit(tna,tws,tsv,rna,rws,rsv)

    msis_meteor_fit(p0_est,
                    v0_est,
                    tna,tws,tsv,
                    rna,rws,rsv,
                    dna,dws,dsv,
                    p_tx,
                    p_na,
                    p_ws,
                    p_sv)

    
    if False:
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
