# Daniel Kastinen's orbit determination tool, which is based on Rebound.
import dasst

# other standard stuff
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from astropy.time import Time, TimeDelta
import numpy as np
import numpy as n
import numpy.random as nr
import scipy.stats as st
import scipy.constants as sc
import spiceypy as spice
import numpy as np
from astropy import units as u

spice.furnsh("naif0012.tls")             # Leap seconds
spice.furnsh("de430.bsp")   


def mean_to_true_anomaly(M, e, tol=1e-10, max_iter=100):
    """
    Convert mean anomaly to true anomaly for elliptical or hyperbolic orbits.

    Parameters:
        M : float
            Mean anomaly [radians]
        e : float
            Eccentricity (e >= 0, e ≠ 1)
        tol : float
            Tolerance for Newton-Raphson convergence
        max_iter : int
            Maximum number of iterations

    Returns:
        ν : float
            True anomaly [radians]
    """
    if e < 1.0:
        # Elliptical orbit
        M = M % (2 * np.pi)  # Normalize

        # Initial guess
        E = M if e < 0.8 else np.pi

        for _ in range(max_iter):
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            delta = f / f_prime
            E -= delta
            if abs(delta) < tol:
                break
        else:
            raise RuntimeError("Kepler's Equation (elliptic) did not converge")

        # Compute true anomaly
        sin_nu = np.sqrt(1 - e**2) * np.sin(E) / (1 - e * np.cos(E))
        cos_nu = (np.cos(E) - e) / (1 - e * np.cos(E))
        ν = np.arctan2(sin_nu, cos_nu)

    elif e > 1.0:
        # Hyperbolic orbit
        # Initial guess for hyperbolic eccentric anomaly H
        H = np.log(2 * abs(M) / e + 1.8) if M >= 0 else -np.log(2 * abs(M) / e + 1.8)

        for _ in range(max_iter):
            f = e * np.sinh(H) - H - M
            f_prime = e * np.cosh(H) - 1
            delta = f / f_prime
            H -= delta
            if abs(delta) < tol:
                break
        else:
            raise RuntimeError("Kepler's Equation (hyperbolic) did not converge")

        # Compute true anomaly
        sin_nu = np.sqrt(e**2 - 1) * np.sinh(H) / (e * np.cosh(H) - 1)
        cos_nu = (e - np.cosh(H)) / (e * np.cosh(H) - 1)
        ν = np.arctan2(sin_nu, cos_nu)

    else:
        raise ValueError("Parabolic case (e = 1) is not supported.")

    return ν

# km & km/s
def cart2kep(state,epoch_j2000_s):
    xform = spice.sxform("J2000", "ECLIPJ2000", epoch_j2000_s)
    state_ecl = spice.mxvg(xform, state)
    mu_sun = 1.32712440018e11  # km^3/s^2
    elements = spice.oscelt(state_ecl, epoch_j2000_s, mu_sun)
    return(elements)


# NASA's navigation files (precise solar system ephemeris data)
# https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
kernel = "./de430.bsp"

# file containing the meteor information
h=h5py.File("new_mexico_mre/meteor_fit.h5","r")
# maximum likelihood parameters
states_ecef=h["ml_pars"][()][0:6]
states_cov=h["covariance"][()]


epoch = Time(h["epoch_unix"][()], scale="utc", format="unix")
print(epoch.utc.iso)
import stuffr
print(stuffr.unix2datestr(h["epoch_unix"][()]))

kepler_out_frame = ["ICRS", "HeliocentricMeanEcliptic"]
radiant_out_frame = ["GCRS", "GeocentricMeanEcliptic"]

N_samples = 100
all_results = []
vgs=[]
keps_long=[]
j2000 = Time("2000-01-01T12:00:00", scale='tdb')
for si in range(N_samples):
    print("sample %d"%(si))
    # sample errors from correct measurement error distribution
    err=nr.multivariate_normal(n.repeat(0,6),states_cov[0:6,0:6]),
    # randomly perturbed state
    pstate=n.copy(states_ecef+err)
    print(pstate[0])
    vgs.append(n.linalg.norm(pstate[0][3:6]))
    pstate.shape=(6,1)

#    settings = dict(
 #       in_frame="HCRS",
  #      out_frame="HCRS",
   #     time_step=dt,  # s
 #       termination_check=False,
  #      tqdm=False,
   # )
    results_hat = dasst.orbit_determination.rebound_od(
        pstate,
        epoch,
        kernel,  #
        kepler_out_frame=kepler_out_frame,     # frame that you want orbit elements in
        radiant_out_frame=radiant_out_frame,   # frame that you want orbit elements in
        termination_check=False,                # do we stop propagation once at sun-earth hill sphere
        dt=10.0,                               #
        max_t=7 * 24 * 3600.0+10,                 #
        settings=None,
        progress_bar=False,
    )
    print(results_hat.keys())
   
#    print(results_hat['hcrs_states'].shape)
 #   print(results_hat['hcrs_states'])

#    print(results_hat["t"].utc.iso)
    print((epoch+results_hat["t"][-1]).utc.iso)
    short_epoch=results_hat["t"][-1]
    results_hat["t"] = results_hat["t"].sec    # astropy time delta, seconds since epoch
    print(results_hat["t"])
    t_short=short_epoch + epoch
    seconds_since_j2000 = (t_short.tdb - j2000).to(u.s).value
    state_short=results_hat["states"][:, -1, 0]
    kep_short=cart2kep(state_short/1e3,seconds_since_j2000)
    kep_short[0]=1e3*kep_short[0]/sc.au
    kep_short[2]=180*kep_short[2]/n.pi
    kep_short[3]=180*kep_short[3]/n.pi
    kep_short[4]=180*kep_short[4]/n.pi

    kep_short[5]=180*mean_to_true_anomaly(kep_short[5], kep_short[1], tol=1e-10, max_iter=100)/n.pi

#    kep_short[5]=180*kep_short[5]/n.pi
    print(kep_short)
    # setup long term propation
    dt = 3600.0 * 24
    settings = dict(
        in_frame="HCRS",
        out_frame="HCRS",
        time_step=dt,  # s
        termination_check=False,
        tqdm=False,
    )    
    prop = dasst.propagators.Rebound(
        kernel=kernel,
        settings=settings,
    )
    results={}
    for key in results_hat:
        results[key + "_hat"] = results_hat[key]
    #del results_hat
    # 200 days
    t_long = np.arange(0, 3600.0 * 24 * 194, dt)
#    t_long[-1]=
    t_long = TimeDelta(-t_long, format="sec")
    p_lt_states, m_lt_states = prop.propagate(
        t_long,
        results["states_hat"][:, -1, :],
        epoch + TimeDelta(results["t_hat"][-1], format="sec"),
        massive_states=results["massive_states_hat"][:, -1, :],
    )
    epoch_long = short_epoch + t_long[-1] + epoch
    print(epoch_long.utc.iso)
    # Define J2000 epoch
    
    #j2000 = Time("2000-01-01T12:00:00", scale='tdb')
    # Compute difference in seconds
    seconds_since_j2000 = (epoch_long.tdb - j2000).to(u.s).value
 #   print("seconds since:")
    print(seconds_since_j2000)
#    print(p_lt_states.shape)
    kep_long=cart2kep(p_lt_states[:,-1]/1e3,seconds_since_j2000)
#        kep_short=cart2kep(state_short/1e3,seconds_since_j2000)
    kep_long[0]=1e3*kep_long[0]/sc.au
    kep_long[2]=180*kep_long[2]/n.pi
    kep_long[3]=180*kep_long[3]/n.pi
    kep_long[4]=180*kep_long[4]/n.pi    
    kep_long[5]=180*mean_to_true_anomaly(kep_long[5], kep_long[1], tol=1e-10, max_iter=100)/n.pi

#    kep_long[5]=180*kep_long[5]/n.pi
#    print(kep_short)


    keps_long.append(kep_long)
    print(kep_long)
    # save results
    results["earth_ind"] = prop._earth_ind
    results["sun_ind"] = prop._sun_ind
    results["long_term_states"] = n.copy(p_lt_states)          # test particle states
    results["long_term_massive_states"] = n.copy(m_lt_states)  # states for planets and moon
    results["long_term_t"] = t_long.sec
    results["epoch"] = epoch.unix
    results["kepler_out_frame"] = kepler_out_frame
#    print(kepler_out_frame.shape)
    print(kepler_out_frame)

    all_results.append(results)

keps_long=n.array(keps_long)
print(keps_long.shape)
#keps_long[:,2]=180*keps_long[:,2]/n.pi
#keps_long[:,3]=180*keps_long[:,3]/n.pi
#keps_long[:,4]=180*keps_long[:,4]/n.pi
#keps_long[:,5]=180*keps_long[:,5]/n.pi


for i in range(6):
    print("keps long %d"%(i))
    print(n.mean(keps_long[:,i]))
    print(n.std(keps_long[:,i]))
    print(n.percentile(keps_long[:,i],[5,95]))

#for i in range(6):
 #   keps_long[:,0]

vgs=n.array(vgs)
print("Vg")
print(vgs)
print(n.mean(vgs))
print(n.std(vgs))
print(n.percentile(vgs,[5,95]))

plt.subplot(121)
keps=[]
eccs=[]

for results in all_results:
    plt.plot(results["t_hat"]/3600/24,results["kepler_HeliocentricMeanEcliptic_hat"][1,:],alpha=0.3,color="black")
    eccs.append(results["kepler_HeliocentricMeanEcliptic_hat"][1,-1][0])
    print(results["kepler_HeliocentricMeanEcliptic_hat"][:,-1])
#    print(results["kepler_HeliocentricMeanEcliptic_hat"][:,-1][0])
    keps.append(results["kepler_HeliocentricMeanEcliptic_hat"][:,-1])
plt.xlabel("Time (days relative to impact)")
plt.ylabel("Eccentricity")

plt.subplot(122)
print(eccs)
plt.hist(eccs)
plt.ylabel("Eccentricity")
plt.show()
keps=n.array(keps)

print(keps.shape)
q=(1-keps[:,1,0])*keps[:,0,0]/sc.au
print(q)
plt.hist(n.abs(q),bins=10)
plt.show()
print("Q")
print(n.mean(n.abs(q)))
print(n.std(n.abs(q)))
print(n.percentile(n.abs(q),[5,95]))

keps[:,0,0]=keps[:,0,0]/sc.au
for i in range(6):
    print("keps %d"%(i))
    print(n.mean(keps[:,i,0]))
    print(n.std(keps[:,i,0]))
    print(n.percentile(keps[:,i,0],[5,95]))

fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection="3d")
au=1.496e+11


for resi,results in enumerate(all_results):
    print(results["long_term_states"].shape)
    sp=ax_3d.plot(
        results["long_term_states"][0,:]/au,
        results["long_term_states"][1,:]/au,
        results["long_term_states"][2,:]/au,
        alpha=0.1,
        color="gray"
    )

#    sp=ax_3d.scatter(
 #       results["long_term_states"][0,:]/au,
  #      results["long_term_states"][1,:]/au,
   #     results["long_term_states"][2,:]/au,
    #    c=results["long_term_t"]/3600/24/365.25,cmap="turbo",s=0.01)
#    if resi==0:
 #       cb=fig_3d.colorbar(sp,ax=ax_3d)
  #      cb.set_label("Time before impact (years)")



r=all_results[0]
m_lt_states=r["long_term_massive_states"]
#symbols=["$☉$","$☿$","$♀︎$","$\u1F728$","$☾$","♂︎","♃","♄","U","N"]
symbols=["$☉$","$♁$","$☿$","$♀︎$","$☾$","$♂︎$","$♃$","$♄$","$♆$","$⛢$"]
for ind in range(m_lt_states.shape[2]):
    if ind != 4: # don't show moon, as it overlaps with earth
        ax_3d.plot(
            m_lt_states[0, :, ind]/au,
            m_lt_states[1, :, ind]/au,
            m_lt_states[2, :, ind]/au,
            "--",
            color="black",
        )
        ax_3d.plot(
            m_lt_states[0, 0, ind]/au,
            m_lt_states[1, 0, ind]/au,
            m_lt_states[2, 0, ind]/au,
            marker="o",
            markersize=20,
            color="black",
        )
        ax_3d.plot(
            m_lt_states[0, 0, ind]/au,
            m_lt_states[1, 0, ind]/au,
            m_lt_states[2, 0, ind]/au,
            marker=symbols[ind],
            markersize=15,
            color="white",
        )
ax_3d.set_box_aspect([1,1,1])  
ax_3d.set_xlabel("x (AU)")
ax_3d.set_ylabel("y (AU)")
ax_3d.set_zlabel("z (AU)")

plt.show()

""" 

    63285.86732206    28674.38849133]
2024-01-11 12:56:46.485
2023-07-02 12:56:46.485
['ICRS', 'HeliocentricMeanEcliptic']
Vg
[72353.54068754 72448.31663055 72461.77313346 72273.6004403 ]
72384.30772296066
76.32290445335842
[72273.72035067 72275.87873734]
#q
[-0.97622573 -0.97565372 -0.97633217 -0.97683083]
-0.976260613731416
0.00041826994469800985
[-0.97683008 -0.97681662]
keps 0
99.73784616866257
62.13516435173095
[48.58262437 49.07139386]
keps 1
1.0130047589108546
0.005512385184488211
[1.00475775 1.00496788]
keps 2
158.21555804420893
0.27950724149249884
[157.96318591 157.96404184]
keps 3
349.96220012316246
0.2700874935261947
[349.56772185 349.5776255 ]
keps 4
117.58464345543061
3.7428459297799834e-05
[117.58458371 117.58458523]
keps 5
269.9390067460753
155.66078498228407
[ 0.8656287  10.56602749]
(6, 194)
(6, 194)





2024-01-11 12:56:36.485
2023-07-02 12:56:36.485
741574665.6691122
[ 1.46066912e+08  1.02423411e+00  2.46959638e+00  2.58172819e+00
  4.29268703e-01 -1.29924473e-02  7.41574666e+08  1.32712440e+11]
['ICRS', 'HeliocentricMeanEcliptic']
(4, 8)
keps long 0
0.0009758247335785578
6.200758552788386e-07
[0.00097513 0.00097513]
keps long 1
1.0169870816862274
0.015668818049236994
[0.99221875 0.99288398]
keps long 2
141.7927812105835
0.3401455964832472
[141.41074567 141.4130902 ]
keps long 3
148.2543142976311
0.3815147683684141
[147.82663627 147.82921846]
keps long 4
24.46860076715015
0.1322785960722875
[24.31528756 24.31647907]
keps long 5
89.35323163947741
156.17963656381784
[-1.27568627 -1.26132032]
Vg
[72490.0560426  72235.87907816 72584.55167533 72488.04601417]
72449.63320256823
129.42504595977687
[72236.25732856 72243.06583584]
 """