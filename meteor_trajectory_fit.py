from pymsis import msis
import matplotlib.pyplot as plt
import numpy as n
import h5py
import scipy.interpolate as interp
import os
import scipy.optimize as so
import scipy.constants as c

import fastecef2h
import analyze_nm as nm
import jcoord

# todo make this an option
obs_lat=35.0
obs_lon=-106

p_0=jcoord.geodetic2ecef(obs_lat,obs_lon,100e3)
ecef2h=fastecef2h.get_fastecef2h(p_0)

def create_msis_file(plot=False):
    msis_dates = n.array([n.datetime64("2023-01-01T12:00"),n.datetime64("2023-04-01T12:00"),n.datetime64("2023-06-20T12:00"),n.datetime64("2023-09-01T12:00")])
    
    hgt=n.linspace(0,200,num=200)
    # lon, lat, h
    data=msis.run(msis_dates, obs_lon, obs_lat, hgt, geomagnetic_activity=-1)
    print(data.shape)    
    rho=data[0,0,0,:,0]

    ho=h5py.File("msis_density.h5","w")
    ho["hgt"]=hgt
    ho["rho"]=rho
    ho.close()

    if plot:
        plt.semilogx(data[0,0,0,:,0],hgt,label="Winter")
        plt.semilogx(data[1,0,0,:,0],hgt,label="Spring")
        plt.semilogx(data[2,0,0,:,0],hgt,label="Summer")
        plt.semilogx(data[3,0,0,:,0],hgt,label="Fall")    
        plt.xlabel("Atmospheric density (kg/m$^3$)")
        plt.ylabel("Height (km)")
        plt.axhline(80,color="gray")
        plt.axhline(120,color="gray")
        plt.ylim([50,150])
        plt.legend()
        plt.title("MSIS-2.0 Atmospheric Density")
        plt.grid()
        plt.tight_layout()    
        plt.savefig("figs/atmospheric_density.png",dpi=300)
        plt.show()


if os.path.exists("msis_density.h5") == False:
    create_msis_file()

ho=h5py.File("msis_density.h5","r")
hgt=n.copy(ho["hgt"][()])
rho=n.copy(ho["rho"][()])
rhof=interp.interp1d(hgt,rho)
ho.close()
    

def enu2h(enu):
    """
    TO BE DONE: fix this so that geographic altitude is given back.
    """
    return(enu[2])

def rho_a(h):
    """ hgt in meters """
    if h>140e3:
        h=140e3
    if h<0:
        h=0
    return(rhof(h/1e3))

def forward_model(p0=[0,0,120e3],
                  v0=[0,0,-70e3],
                  rho_m_r=0.1,
                  alpha=0,       # don't touch this!
                  max_t=0.5,
                  rho_m=1000.0,  # 1000 kg/m^3
                  dt=1e-3,
                  plot=False
                  ):
    """
    Atmospheric drag only. Ignore Magnus effect.
    """
    # positions
    ps = []
    h=ecef2h(p0)
    # unit vector
    u=n.linalg.norm(v0)    
    u0=v0/u
    
    us = []
    p_prev = p0
    ts=[]
    t=0
    while(t <= max_t):
        rho_m_r_t = rho_m_r*n.exp(-alpha*t)
        # propagate object forward
        # rho_m_r is meteoroid density multiplied with meteoroid radius
        k = (3/8)*(rho_a(h)/rho_m_r)
        # distance that we move forward in 1 ms
        us.append(u)
        u = 1/((1/u)+k*dt)
        x_1ms = n.log(k*dt*u + 1)/k
        p_1ms = u0*x_1ms + p_prev
        ps.append(p_prev)
        p_prev=p_1ms
        ts.append(t)
        t+=dt
        # update height

        h = ecef2h(p_1ms)#[2]

        #print(h)

    ts=n.array(ts)
    ps=n.array(ps)
    us=n.array(us)
    r = rho_m_r/rho_m
    mass = rho_m*(4/3)*n.pi*(r**3.0)
    if plot:

        #plt.subplot(121)
        
    #    plt.plot(ts,ps[:,0])
   #     plt.plot(ts,ps[:,1])
  #      plt.plot(ts,ps[:,2])
 #       plt.xlabel("Time (s)")
 #      plt.ylabel("ECEF position (meters)")
#        plt.subplot(122)
        plt.plot(ts,us/1e3)
        plt.title("Dynamic mass: %1.2g kg  (assuming 1000 kg/m$^3$)\n $\\rho_m r$ = %1.2g (kg/m$^2$)"%(mass,rho_m_r))
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (km/s)")
        plt.show()
    return(ts,ps,us)

def forward_model_meas(pos0,v0,rho_m_r,time_vec,rho_m=1e3):
    """
    Evaluate e,n,u positions and pulse-to-pulse doppler
    """
    dt=mc.ipp
    # numerically propagate meteor with atmospheric drag
    m_t, m_p, m_u=forward_model(p0=pos0,
                                v0=v0,
                                rho_m_r=rho_m_r,
                                max_t=n.max(time_vec)+10*mc.ipp,
                                rho_m=rho_m,     # assumption of meteor density, 
                                dt=dt)           # not really used here, as we estimate pressure on meteor surface: (rho_m * r) (kg/m^3)*m = kg/m^2
 
    n_t=len(time_vec)
    n_pairs=len(mc.doppler_pairs)
    zpp_model=n.zeros([1,n_t],dtype=n.complex64)
    tau=mc.ipp
    posfun_e=interp.interp1d(m_t, m_p[:,0])
    posfun_n=interp.interp1d(m_t, m_p[:,1])
    posfun_u=interp.interp1d(m_t, m_p[:,2])

    # calculate pulse-to-pulse Doppler phase progression
    # for i in range(n_t):
    #     for j in range(n_pairs):
    #         model_pos = n.array([posfun_e(time_vec[i]),posfun_n(time_vec[i]),posfun_u(time_vec[i])])
    #         model_pos_tau = n.array([posfun_e(time_vec[i]+tau),posfun_n(time_vec[i]+tau),posfun_u(time_vec[i]+tau)])

    #         # distance from transmitter to meteor
    #         L = n.linalg.norm(model_pos)
    #         # distance from transmitter to meteor at time lag tau
    #         L_tau = n.linalg.norm(model_pos_tau)

    #         # iterate to include approximate speed of light propagation time
    #         model_pos = n.array([posfun_e(time_vec[i]+L/c.c),posfun_n(time_vec[i]+L/c.c),posfun_u(time_vec[i]+L/c.c)])
    #         model_pos_tau = n.array([posfun_e(time_vec[i]+tau +L_tau/c.c),posfun_n(time_vec[i]+tau +L_tau/c.c),posfun_u(time_vec[i]+tau +L_tau/c.c)])

    #         # distance from transmitter to meteor
    #         L = n.linalg.norm(model_pos)
    #         # distance from transmitter to meteor at time lag tau
    #         L_tau = n.linalg.norm(model_pos_tau)

    #         # distance from transmitter to meteor
    #         L0 = L
    #         # distance from transmitter to meteor at time lag tau
    #         L1_tau = L_tau

    #         zpp_model[j,i]=n.exp(-1j*2.0*n.pi*(L - L_tau)/mc.lam)*n.exp(-1j*2.0*n.pi*(L0 - L1_tau)/mc.lam)

    model_pos = n.vstack([posfun_e(time_vec),posfun_n(time_vec),posfun_u(time_vec)])
    # distance from transmitter to meteor
    L = n.linalg.norm(model_pos,axis=0)
    
    # iterate to include approximate speed of light propagation time
    model_pos = n.vstack([posfun_e(time_vec+L/c.c),posfun_n(time_vec+L/c.c),posfun_u(time_vec+L/c.c)])

    model_pos_tau = n.vstack([posfun_e(time_vec+tau),posfun_n(time_vec+tau),posfun_u(time_vec+tau)])
    # distance from transmitter to meteor
    L_tau = n.linalg.norm(model_pos_tau,axis=0)
    
    # iterate to include approximate speed of light propagation time
    model_pos_tau = n.vstack([posfun_e(time_vec+tau+L_tau/c.c),posfun_n(time_vec+tau+L_tau/c.c),posfun_u(time_vec+tau+L_tau/c.c)])

    zpp_model[0,:] = n.exp(-1j*2.0*n.pi*(L - L_tau)/mc.lam)*n.exp(-1j*2.0*n.pi*(L - L_tau)/mc.lam)
    
    return({"posfun_e":posfun_e,
            "posfun_n":posfun_n,
            "posfun_u":posfun_u,
            "model_t":m_t,
            "pos_e":model_pos[0,:],
            "pos_n":model_pos[1,:],
            "pos_u":model_pos[2,:],
            "zpp":zpp_model
            })

    
def fit_drag_model(fname="test_data/20231212_003545766_event.ud3.h5",
                   flip_en=True,
                   plot_fit=True,
                   rho_m=1000,
                   ):
    h=h5py.File(fname,"r")
    print(h.keys())
    pos_e=h["pos_e"][()]
    pos_n=h["pos_n"][()]
    pos_u=h["pos_u"][()]
    if flip_en:
        pe = n.copy(-pos_n)
        pn = n.copy(-pos_e)
        pos_e=pe
        pos_n=pn

    gidx=h["gidx"][()]
    t_idx=h["t_idx"][gidx]
    snr=h["snr"][gidx]

    snr_weight=n.copy(snr)
    snr_weight[snr_weight>100]=100

    zpp=n.copy(h["zpp"][()])
    zpp=zpp[:,gidx]

    time_vec = t_idx*mc.ipp

    # use a linear least-squares fit to guess what the initial position and initial
    # velocity is
    x0,v0=initial_guess(pos_e,
                        pos_n,
                        pos_u,
                        time_vec,                        
                        snr,
                        plot=True)

    u0=v0/n.linalg.norm(v0)
    vabs=n.linalg.norm(v0)

    min_alt=n.min(pos_u)-2e3
    global sprev
    sprev=1e99
    def ss(x,plot_fit=False):
        global sprev
        pos0=x[0:3]
        v0=x[3:6]
        rho_m_r=x[6]

        model=forward_model_meas(pos0,v0,rho_m_r,time_vec)
        #    return({"posfun_e":posfun_e,
        #           "posfun_n":posfun_n,
        #          "posfun_u":posfun_u,
        #         "pos_e":model_pos[:,0],
        #        "pos_n":model_pos[:,1],
        #       "pos_u":model_pos[:,2],
        #      "zpp":zpp_model
        #     })
        
        # up directions is about 4 times better than e-w and n-s
        # in terms of measurement error std
        up_w = 4**2.0 
        s=n.sum( snr_weight*(model["pos_e"]-pos_e)**2.0 +
                 snr_weight*(model["pos_n"]-pos_n)**2.0 +
                 snr_weight*up_w*(model["pos_u"]-pos_u)**2.0)

        # pulse to pulse doppler phase is about 200 times better in terms of
        # measurement uncertainty std compared to e-w and n-s
        s+=(200**2)*n.sum( snr_weight*n.angle( n.exp(1j*n.angle(zpp[0,:]))*n.exp(-1j*n.angle(model["zpp"][0,:])))**2.0 )

        print(s)
        if plot_fit:
            plt.subplot(231)
            plt.plot(time_vec, pos_e/1e3,".")
            plt.plot(time_vec, model["pos_e"]/1e3)
            plt.xlabel("Time (s)")
            plt.ylabel("East (km)")            
            plt.subplot(232)    
            plt.plot(time_vec, pos_n/1e3,".")
            plt.plot(time_vec, model["pos_n"]/1e3)
            plt.xlabel("Time (s)")
            plt.ylabel("North (km)")            
            
            plt.subplot(233)    
            plt.plot(time_vec, pos_u/1e3,".")
            plt.plot(time_vec, model["pos_u"]/1e3)
            plt.xlabel("Time (s)")
            plt.ylabel("Up (km)")            
            
            plt.subplot(234)
            plt.plot(time_vec, n.angle(zpp[0,:]),".")
            plt.plot(time_vec, n.angle(model["zpp"][0,:]))
            
            plt.xlabel("Time (s)")
            plt.ylabel("Pulse-to-pulse Doppler phase (rad/ipp)")
            plt.title("$\\rho_m r$=%1.2f $v_0$=%1.2f km/s"%(rho_m_r,n.linalg.norm(v0)/1e3))
            plt.subplot(235)
            vel_e=n.gradient(model["posfun_e"](model["model_t"]),1e-3,edge_order=2)
            vel_n=n.gradient(model["posfun_n"](model["model_t"]),1e-3,edge_order=2)
            vel_u=n.gradient(model["posfun_u"](model["model_t"]),1e-3,edge_order=2)            
            plt.plot(model["model_t"],n.sqrt(vel_e**2.0+vel_n**2.0+vel_u**2.0)/1e3)
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity (km/s)")            
            
            plt.subplot(236)
            acc_e=n.gradient(vel_e,1e-3)
            acc_n=n.gradient(vel_n,1e-3)
            acc_u=n.gradient(vel_u,1e-3)            
            plt.plot(model["model_t"],n.sqrt(acc_e**2.0+acc_n**2.0+acc_u**2.0)/1e3)
            plt.xlabel("Time (s)")
            plt.ylabel("Deceleration (km/s$^2$)")
            plt.tight_layout()
            plt.show()
            sprev=s

        return(s)

    xhat=so.fmin(ss,[x0[0],x0[1],x0[2],v0[0],v0[1],v0[2],1])
    print(xhat)
    
    ss(xhat,plot_fit=True)
    h.close()

if __name__ == "__main__":
#    forward_model(plot=True)    
    create_msis_file(plot=True)

#    fit_drag_model(fname="test_data/20231212_003520280_event.ud3.h5")
 #   fit_drag_model(fname="test_data/20231212_003545766_event.ud3.h5")
  #  fit_drag_model(fname="test_data/20231212_003553036_event.ud3.h5")    
