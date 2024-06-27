import numpy as n
import jcoord
import h5py
import os

"""
   Fast conversion of ECEF to height using \second order polynomial
"""

def get_fastecef2h(p0,dx=700e3):
    N=100
    dxs=n.linspace(-dx,dx,num=N)
    step=n.diff(dxs)[0]

    fname="fastecef2h-%1.1f%1.1f%1.1f.h5"%(p0[0]/1e3,p0[1]/1e3,p0[2]/1e3)
    xhat=None
    if os.path.exists(fname):
        print("using stored coefficients")
        hi=h5py.File(fname,"r")
        xhat=n.copy(hi["xhat"][()])
        hi.close()
    else:
        print("creating coefficients")
        h=n.zeros([N,N,N])
        x=n.zeros([N,N,N])
        y=n.zeros([N,N,N])
        z=n.zeros([N,N,N])
        for i in range(N):
            print(i)
            for j in range(N):
                for k in range(N):
                    pos_this=p0 + n.array([dxs[i],dxs[j],dxs[k]])
                    x[i,j,k]=pos_this[0]
                    y[i,j,k]=pos_this[1]
                    z[i,j,k]=pos_this[2]
                    llh=jcoord.ecef2geodetic(pos_this[0],pos_this[1],pos_this[2])
                    h[i,j,k]=llh[2]

        minx=n.min(x)
        miny=n.min(y)
        minz=n.min(z)
        N_meas=N**3
        # h(x,y,z) = adc + a0 * (x-x0) + a1*(y-y0) + a2*(z-z0) + a3*(x-x0)**2.0 +
        #            a4*(y-y0)**2.0 + a5*(z-z0)**2.0 +
        #             a6*(x-x0)*(y-y0) + a7*(x-x0)*(z-z0) + a8*(y-y0)(z-z0)
        A = n.zeros([N_meas,10])
    #    xx = (x.flatten())/1e3
     #   yy = (y.flatten())/1e3
      #  zz = (z.flatten())/1e3
        xx = (x.flatten()-p0[0])/1e3
        yy = (y.flatten()-p0[1])/1e3
        zz = (z.flatten()-p0[2])/1e3
        hh = h.flatten()/1e3
        A[:,0]=1.0
        A[:,1]=xx
        A[:,2]=yy
        A[:,3]=zz
        A[:,4]=xx**2.0
        A[:,5]=yy**2.0
        A[:,6]=zz**2.0
        A[:,7]=xx*yy
        A[:,8]=xx*zz
        A[:,9]=yy*zz    
    #    A[:,10]=(xx**2)*yy
        # A[:,11]=(xx**2)*zz
        # A[:,12]=(yy**2)*zz
        # A[:,13]=(yy**2)*xx
        # A[:,14]=(zz**2)*xx
        # A[:,15]=(zz**2)*yy

        xhat=n.linalg.lstsq(A,hh)[0]
        ho=h5py.File(fname,"w")
        ho["xhat"]=xhat
        ho.close()
    

    def get_h(p):
        x=(p[0]-p0[0])/1e3
        y=(p[1]-p0[1])/1e3
        z=(p[2]-p0[2])/1e3
        return( (xhat[0]+xhat[1]*x+xhat[2]*y+xhat[3]*z
               +xhat[4]*x**2.0+xhat[5]*y**2.0+xhat[6]*z**2.0+
               +xhat[7]*x*y+xhat[8]*x*z+xhat[9]*y*z)*1e3)
 #              +xhat[10]*(x**2)*y)#+xhat[11]*(x**2)*z+xhat[12]*(y**2)*z
 #              +xhat[13]*(y**2)*x+xhat[14]*(z**2)*x+xhat[15]*(z**2)*y)
               
    return(get_h)


if __name__ == "__main__":
    p_tx=jcoord.geodetic2ecef(35.002, -106.526, 100e3)
    h_fun=get_fastecef2h(p_tx)
    for i in range(120):
        xyz=jcoord.geodetic2ecef(35.002, -106.526, i*1e3)
        print("h %1.2f km err %1.2f m"%(i,h_fun(xyz)-i*1e3))


        

