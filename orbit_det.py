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

# NASA's navigation files (precise solar system ephemeris data)
# https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
kernel = "/home/j/src/maarsy_meteors/daniel/de430.bsp"

# file containing the meteor information
h=h5py.File("new_mexico_mre/meteor_fit.h5","r")
# maximum likelihood parameters
states_ecef=h["ml_pars"][()][0:6]
states_cov=h["covariance"][()]


epoch = Time(h["epoch_unix"][()], scale="utc", format="unix")
import stuffr
print(stuffr.unix2datestr(h["epoch_unix"][()]))

kepler_out_frame = ["ICRS", "HeliocentricMeanEcliptic"]
radiant_out_frame = ["GCRS", "GeocentricMeanEcliptic"]

N_samples = 100
all_results = []
for si in range(N_samples):
    print("sample %d"%(si))
    # sample errors from correct measurement error distribution
    err=nr.multivariate_normal(n.repeat(0,6),states_cov[0:6,0:6]),
    # randomly perturbed state
    pstate=n.copy(states_ecef+err)
    print(pstate)
    pstate.shape=(6,1)
    results_hat = dasst.orbit_determination.rebound_od(
        pstate,
        epoch,
        kernel,  #
        kepler_out_frame=kepler_out_frame,     # frame that you want orbit elements in
        radiant_out_frame=radiant_out_frame,   # frame that you want orbit elements in
        termination_check=True,                # do we stop propagation once at sun-earth hill sphere
        dt=10.0,                               #
        max_t=7 * 24 * 3600.0,                 #
        settings=None,
        progress_bar=False,
    )
    results_hat["t"] = results_hat["t"].sec    # astropy time delta, seconds since epoch

    # setup long term propation
    dt = 3600.0 * 48
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
    del results_hat

    t_long = np.arange(0, 3600.0 * 24 * 365.24 * 100, dt)
    t_long = TimeDelta(-t_long, format="sec")
    p_lt_states, m_lt_states = prop.propagate(
        t_long,
        results["states_hat"][:, -1, :],
        epoch + TimeDelta(results["t_hat"][-1], format="sec"),
        massive_states=results["massive_states_hat"][:, -1, :],
    )

    # save results
    results["earth_ind"] = prop._earth_ind
    results["sun_ind"] = prop._sun_ind
    results["long_term_states"] = n.copy(p_lt_states)          # test particle states
    results["long_term_massive_states"] = n.copy(m_lt_states)  # states for planets and moon
    results["long_term_t"] = t_long.sec
    results["epoch"] = epoch.unix
    results["kepler_out_frame"] = kepler_out_frame

    all_results.append(results)


plt.subplot(121)
eccs=[]
for results in all_results:
    plt.plot(results["t_hat"]/3600/24,results["kepler_HeliocentricMeanEcliptic_hat"][1,:],alpha=0.3,color="black")
    eccs.append(results["kepler_HeliocentricMeanEcliptic_hat"][1,-1][0])
plt.xlabel("Time (days relative to impact)")
plt.ylabel("Eccentricity")

plt.subplot(122)
print(eccs)
plt.hist(eccs)
plt.ylabel("Eccentricity")
plt.show()


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



