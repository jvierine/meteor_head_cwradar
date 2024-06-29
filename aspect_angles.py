import numpy as n
import jcoord
import h5py
import matplotlib.pyplot as plt





def aspect_angle(xyz,p_tx,p_rx):
    """
    given cartesian positions of scatterers (N,3) and tx & rx positions (3,)
    calculate the aspect angles
    """
    a=p_tx[:].T-xyz
    
    b=xyz-p_rx[:].T
    print(b.shape)

#    plt.plot(n.sqrt(n.sum(a*a,axis=1)))
#    plt.plot(n.sqrt(n.sum(b*b,axis=1)))    
#    plt.show()
    
    angle=n.arccos(n.sum(a*b,axis=1)/(n.sqrt(n.sum(a*a,axis=1))*n.sqrt(n.sum(a*a,axis=1))))
    return(angle)




if __name__ == "__main__":
    import analyze_nm as nm

    h=h5py.File("meteor_fit.h5","r")
    xyz=h["model_ecef"][()]
    llh=h["model_lat_lon_h"][()]
    t=h["model_time_unix"][()]    
    p_tx=jcoord.geodetic2ecef(nm.tx_gps[0],nm.tx_gps[1],nm.tx_gps[2])
    p_na=jcoord.geodetic2ecef(nm.nm_rx_gps[1][0],nm.nm_rx_gps[1][1],nm.nm_rx_gps[1][2])
    p_sv=jcoord.geodetic2ecef(nm.nm_rx_gps[2][0],nm.nm_rx_gps[2][1],nm.nm_rx_gps[2][2])
    p_ws=jcoord.geodetic2ecef(nm.nm_rx_gps[3][0],nm.nm_rx_gps[3][1],nm.nm_rx_gps[3][2])

    fig=plt.figure(figsize=(2*8,6.4))
    plt.subplot(121)
    plt.scatter(nm.tx_gps[1],nm.tx_gps[0],label="TX",color="C0")
    plt.scatter(nm.nm_rx_gps[1][1],nm.nm_rx_gps[1][0],label="NA",color="C1")
    plt.scatter(nm.nm_rx_gps[2][1],nm.nm_rx_gps[2][0],label="SV",color="C2")
    plt.scatter(nm.nm_rx_gps[3][1],nm.nm_rx_gps[3][0],label="WS",color="C3")    
    plt.scatter(llh[:,1],llh[:,0],c=t,cmap="turbo")
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")    
    cb=plt.colorbar()
    cb.set_label("Time (unix)")
    plt.legend()
    #    plt.show()
    plt.subplot(122)   
    a_na=aspect_angle(xyz,p_tx,p_na)
    a_sv=aspect_angle(xyz,p_tx,p_sv)
    a_ws=aspect_angle(xyz,p_tx,p_ws)
    plt.plot(t,180*a_na/n.pi,label="TX-NA",color="C1")
    plt.plot(t,180*a_sv/n.pi,label="TX-SV",color="C2")
    plt.plot(t,180*a_ws/n.pi,label="TX-WS",color="C3")
    plt.xlabel("Time (unix)")
    plt.ylabel("Aspect angle (deg)")
    plt.tight_layout()
    plt.savefig("aspect_angles.png",dpi=150)
    plt.show()
    
    
