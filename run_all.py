import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
import matplotlib.pyplot as plt
import pickle
import os,sys
import rv_analyse,llh_max,hgca_inference
import gaiascanlaw
from astropy.timeseries import LombScargle
def single_eval(inc, Omega, pmra, pmdec, const):
    (rv_params, mass, ra, dec, dist ,hgca_data, t_gaia, t_hip, scan_gaia, t_dr4, scan_dr4) = const
    mu_map = jnp.array([pmra,pmdec])
    #inc = jnp.array([inc1,inc2])
    #Omega = jnp.array([Omega1,Omega2])
    params = [
        rv_params['P'],
        rv_params['e'],
        rv_params['q'] / jnp.sin(inc),
        inc,
        Omega,
        rv_params['omega'],
        rv_params['T0'],
        mass,
        dist,
    ]
    llh_val = llh_max.loglike_total(
        params, mu_map, hgca_data, t_gaia, t_hip, scan_gaia, ra, dec
    )
    x_dr3 = calc_offset(
        t_gaia, scan_gaia,
        ra,
        dec,
        params, mu_map, dist
    )
    x_dr4 = calc_offset(
        t_dr4, scan_dr4,
        ra,
        dec,
        params, mu_map, dist
    )
    return llh_val, x_dr3, x_dr4

def rv_llh(P,e,q,omega,M0,offset,const):
    (mass, inc, rv_data) = const
    params = (
        P,
        e,
        q,
        omega,
        M0,
        mass,
        inc,
        offset)
    llh_val = llh_max.loglike_rv(params, rv_data)
    return llh_val
def calc_offset(t,scan_angle,ra,dec,params,mu_map,dist):
	cond = jnp.max(t) > 2019
	epoch = jnp.where(cond, 2017.5, 2016.0)
	x_orbit_dr4, y_orbit_dr4 = llh_max.sky_position(t, params)
	x_star_dr4, y_star_dr4 = llh_max.star_model(t, ra, dec, 0, 0, dist, mu_map[0], mu_map[1], epoch)
	x_dr4 = x_orbit_dr4 + x_star_dr4
	y_dr4 = y_orbit_dr4 + y_star_dr4
	x_AL_dr4 = x_dr4 * jnp.sin(scan_angle) + y_dr4 * jnp.cos(scan_angle)
	A = llh_max.gaia_matrix_AL(t,scan_angle,ra,dec,epoch)
	params_est = llh_max.gaia_params_iter(x_AL_dr4,A,0.216,n_iter=10)
	(x_model_dr4,y_model_dr4) = llh_max.star_model(t, ra, dec, params_est[0],params_est[1],1000./params_est[2],params_est[3],params_est[4],epoch)
	x_AL_model_dr4 = x_model_dr4*jnp.sin(scan_angle)+y_model_dr4*jnp.cos(scan_angle)
	x_diff = x_AL_dr4-x_AL_model_dr4
	return x_diff

def planet_positions(t_year, params):
    P, e, q, inc, Omega, omega, T0, M_star, distance = params
    inc *= jnp.ones(len(P))
    Omega *= jnp.ones(len(P))
    JD0 = 2457388.5
    t_days = (t_year - 2016.0) * 365.25

    x_planets = []
    y_planets = []

    for ii in range(len(P)):
        n = 2 * jnp.pi / P[ii]
        M0 = n * (JD0 - T0[ii])

        f = llh_max.true_anomaly(t_days, n, e[ii], M0)

        # star reflex motion from this planet
        x_star, y_star = llh_max.astrometric_model(
            f, P[ii], e[ii], q[ii], inc[ii], Omega[ii], omega[ii], M_star, distance
        )

        # convert to planet position
        x_p = -x_star / q[ii]
        y_p = -y_star / q[ii]

        x_planets.append(x_p)
        y_planets.append(y_p)

    return jnp.array(x_planets), jnp.array(y_planets)
n_planets = 2
all_data = pickle.load(open('all_data.pkl','rb'))
name = 'HD65216'
data = all_data[name]
hgca_data = {}
hgca_data['y_obs'] = llh_max.build_y_obs(data['hg_obs'])
hgca_data['C'] = llh_max.build_covariance(data['hg_obs'])
hgca_data['Cinv'] = np.linalg.inv(hgca_data['C'])
hgca_data['C_gaia'] = llh_max.build_gaia_covariance(data)
hgca_data['gaia_obs'] = np.array([data['ra'],data['dec'],0,0,data['parallax'],data['pmra'],data['pmdec']])
hgca_data['Cinv'] = np.linalg.inv(hgca_data['C'])
hgca_data['Cinv_gaia'] = np.linalg.inv(hgca_data['C_gaia'])
t_hip = llh_max.find_hip('hip_epochs.csv',data['hg_obs']['hip_id'])
t_gaia,scan_gaia = gaiascanlaw.scanlaw(data['hg_obs']['gaia_ra'],data['hg_obs']['gaia_dec'],tstart=gaiascanlaw.tstart,tend=gaiascanlaw.tdr3)
t_dr4,scan_dr4 = gaiascanlaw.scanlaw(data['hg_obs']['gaia_ra'],data['hg_obs']['gaia_dec'],tstart=gaiascanlaw.tstart,tend=gaiascanlaw.tdr4)
M_star = data['mass']    
samples_rv,log_prob_rv = pickle.load(open('samples_rv_'+name+'.pkl','rb'))
el = jnp.argmax(log_prob_rv)
els = np.where(log_prob_rv<np.max(log_prob_rv))[0]
el = els[np.argmax(log_prob_rv[els])]
rv_params = llh_max.get_rv_params_one(samples_rv,n_planets,idx=el)
(samples,log_prob) = hgca_inference.run_astro(name,1,el)
pickle.dump((samples,log_prob),open('samples_hgca_'+name+'.pkl','wb'))