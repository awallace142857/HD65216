import copy
import logging
import numpy as np,matplotlib.pyplot as plt
import pandas as pd
import healpy as hp
from astropy import units as u
from astropy import constants, coordinates, time
from typing import Iterable, Optional, Tuple, Union
from gaiaunlimited.utils import coord2healpix
import scanninglaw,scanninglaw.times
from astromet import sigma_ast
import sys,pickle
import astromet
import corner
import os
import matplotlib.lines as mlines
from matplotlib import rcParams
import pygtc
from matplotlib import animation
from IPython.display import HTML
from astroquery.gaia import Gaia
import gaiascanlaw
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
import llh_max
from astropy.timeseries import LombScargle
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
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
def calc_offset(t,scan_angle,ra,dec,params,mu_map):
	cond = jnp.max(t) > 2019
	epoch = jnp.where(cond, 2017.5, 2016.0)
	x_orbit, y_orbit = sky_position(t, params)
	x_star, y_star = star_model(t, ra, dec, 0, 0, params[8], mu_map[0], mu_map[1], epoch)
	x = x_orbit + x_star
	y = y_orbit + y_star
	x_AL = x * jnp.sin(scan_angle) + y * jnp.cos(scan_angle)
	A = gaia_matrix_AL(t,scan_angle,ra,dec,epoch)
	params_est = gaia_params_iter(x_AL,A,0.216,n_iter=10)
	(x_model,y_model) = star_model(t, ra, dec, params_est[0],params_est[1],1000./params_est[2],params_est[3],params_est[4],epoch)
	x_AL_model = x_model*jnp.sin(scan_angle)+y_model*jnp.cos(scan_angle)
	x_diff = x_AL-x_AL_model
	return x_diff

def true_anomaly(t_rel,n,e,M0):
    M = n * t_rel + M0
    x = jnp.cos(M)
    y = jnp.sin(M)
    M = jnp.arctan2(y,x)
    E = kepler_eq(M, e)
    return 2 * jnp.arctan2(jnp.sqrt(1 + e) * jnp.sin(E / 2), jnp.sqrt(1 - e) * jnp.cos(E / 2))

def kepler_eq(M, e, max_iter=20):
    def body_fn(E, _):
        f = E - e * jnp.sin(E) - M
        f_prime = 1 - e * jnp.cos(E)
        E_new = E - f / f_prime
        return E_new, None
    E0 = M + e * jnp.sin(M)
    E_final, _ = jax.lax.scan(body_fn, E0, xs=None, length=max_iter)
    return E_final

# Sky-projected orbital position (scaled to 1 AU)
def projected_orbit(f, P, e, inc, Omega, omega):
    r = (1 - e**2) / (1 + e * jnp.cos(f))  # semi-major axis scaled to 1 AU
    # Sky projection
    x = r * (jnp.sin(Omega) * jnp.cos(omega + f) + jnp.cos(Omega) * jnp.sin(omega + f) * jnp.cos(inc))
    y = r * (jnp.cos(Omega) * jnp.cos(omega + f) - jnp.sin(Omega) * jnp.sin(omega + f) * jnp.cos(inc))
    return x, y

# Astrometric wobble model
def astrometric_model(f, P, e, q, inc, Omega, omega, M_star, distance):
    x, y = projected_orbit(f, P, e, inc, Omega, omega)
    P_year = P/365.25
    #Semi-major axis of the star's orbit in AU
    a_star_m = ((M_star * P_year**2)**(1 / 3)) * q / (1 + q)  # in AU
    au_to_mas = 1000.0 / distance # AU to mas conversion
    scale = a_star_m * au_to_mas

    return x * scale, y * scale

import csv
def find_hip(csv_file,hip_id):
    ts = []
    sinscan = []
    cosscan = []
    plx_factor = []
    used = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in enumerate(reader):
            if row[1]['Epoch'] not in used and str(row[1]['HIP'])==str(hip_id):
                used.append(row[1]['Epoch'])
                ts.append(float(row[1]['Epoch']))
                sinscan.append(float(row[1]['SinPsi']))
                cosscan.append(float(row[1]['CosPsi']))
                plx_factor.append(float(row[1]['PlxFactor']))
    return np.array(ts),np.array(plx_factor),np.array(sinscan),np.array(cosscan)
def fit_slope(t, x):
    """
    Returns slope dx/dt via linear least squares.
    t: (N,)
    x: (N,)
    """
    t_mean = jnp.mean(t)
    x_mean = jnp.mean(x)

    dt = t - t_mean
    dx = x - x_mean

    return jnp.sum(dt * dx) / jnp.sum(dt**2)
def gaia_position(t):
    """3D position of Gaia, with Sun at [0,0,0], in celestial coordinates"""
    T0 = 3./365.25
    Omega = 0
    inc = 23.4*jnp.pi/180
    e = 0.0167
    omega = 103.33284959908069*jnp.pi/180
    P = 1
    a = 1.01
    n = 2 * jnp.pi / P
    M0 = -n*T0
    f = true_anomaly(t,n,e,M0)
    r = a * (1 - e**2) / (1 + e * jnp.cos(f))
    x = r * (jnp.cos(Omega) * jnp.cos(omega + f) - jnp.sin(Omega) * jnp.sin(omega + f) * jnp.cos(inc))
    y = r * (jnp.sin(Omega) * jnp.cos(omega + f) + jnp.cos(Omega) * jnp.sin(omega + f) * jnp.cos(inc))
    z = r * (jnp.sin(omega + f) * jnp.sin(inc))
    return [x,y,z]
    
def star_model(t_year, ra, dec, dra, ddec, distance, pmra, pmdec,epoch):
    parallax = 1000./distance
    [xG,yG,zG] = gaia_position(t_year)
    ra_off = dra+(xG*jnp.sin(jnp.deg2rad(ra))-yG*jnp.cos(jnp.deg2rad(ra)))*parallax+pmra*(t_year-epoch)
    dec_off = ddec+(xG*jnp.cos(jnp.deg2rad(ra))*jnp.sin(jnp.deg2rad(dec))+yG*jnp.sin(jnp.deg2rad(ra))*jnp.sin(jnp.deg2rad(dec))-zG*jnp.cos(jnp.deg2rad(dec)))*parallax+pmdec*(t_year-epoch)
    return ra_off, dec_off
def gaia_matrix_AL(t, scan_angle, ra, dec, epoch):
    [xG,yG,zG] = gaia_position(t)
    nVar = 5
    A = jnp.column_stack((jnp.sin(scan_angle),
                    jnp.cos(scan_angle),
                    (xG*jnp.sin(jnp.deg2rad(ra))-yG*jnp.cos(jnp.deg2rad(ra)))*jnp.sin(scan_angle)+(xG*jnp.cos(jnp.deg2rad(ra))*jnp.sin(jnp.deg2rad(dec))+yG*jnp.sin(jnp.deg2rad(ra))*jnp.sin(jnp.deg2rad(dec))-zG*jnp.cos(jnp.deg2rad(dec)))*jnp.cos(scan_angle),
                    (t-epoch)*jnp.sin(scan_angle),
                    (t-epoch)*jnp.cos(scan_angle)))
    return A
    
def gaia_mat(t, sinscan, cosscan, plx_factor):
    A = jnp.column_stack((sinscan,
                    cosscan,
                    plx_factor,
                    t*sinscan,
                    t*cosscan))
    return A  

def gaia_matrix_AL(t, scan_angle, ra, dec, epoch):
    [xG,yG,zG] = gaia_position(t)
    nVar = 5
    A = jnp.column_stack((jnp.sin(scan_angle),
                    jnp.cos(scan_angle),
                    (xG*jnp.sin(jnp.deg2rad(ra))-yG*jnp.cos(jnp.deg2rad(ra)))*jnp.sin(scan_angle)+(xG*jnp.cos(jnp.deg2rad(ra))*jnp.sin(jnp.deg2rad(dec))+yG*jnp.sin(jnp.deg2rad(ra))*jnp.sin(jnp.deg2rad(dec))-zG*jnp.cos(jnp.deg2rad(dec)))*jnp.cos(scan_angle),
                    (t-epoch)*jnp.sin(scan_angle),
                    (t-epoch)*jnp.cos(scan_angle)))
    return A  

def gaia_params_iter(x_obs, A, al_err, n_iter=5):
    """AGIS-like robust iterative WLS with constant design matrix A."""
    
    # initial weights = 1 / sigma^2
    w0 = jnp.ones(len(x_obs)) / (al_err**2)

    def step(carry, _):
        w = carry
        # Weighted normal matrix
        Aw = A * w[:, None]         # weight each row
        AtAw = A.T @ Aw
        AtWx = A.T @ (w * x_obs)

        # Solve for params
        params = jnp.linalg.solve(AtAw, AtWx)

        # Compute residuals
        R = x_obs - A @ params

        # Update weights (Huber-like, Gaia uses similar scheme)
        sigma = jnp.median(jnp.abs(R)) * 1.4826 + 1e-12
        w_new = 1.0 / (sigma**2 + (R**2))   # simple robustifier

        return w_new, params

    w_final, params_last = jax.lax.scan(step, w0, None, length=n_iter)
    return params_last[-1]
def sky_position(t_year, params):
    P, e, q, inc, Omega, omega, T0, M_star, distance = params
    #inc *= jnp.ones(len(P))
    #Omega *= jnp.ones(len(P))
    JD0 = 2457388.5

    # convert to days relative to reference
    t_days = (t_year - 2016.0) * 365.25
    x_tot = 0
    y_tot = 0
    for ii in range(len(P)):
        n = 2 * jnp.pi / P[ii]
        M0 = n * (JD0 - T0[ii])
        f = true_anomaly(t_days, n, e[ii], M0)
        x,y = astrometric_model(
            f, P[ii], e[ii], q[ii], inc, Omega, omega[ii], M_star, distance
        )
        x_tot+=x
        y_tot+=y
    return x_tot,y_tot

def model_pm(ts, params):
    #ts = jnp.linspace(t1, t2, N)

    x, y = jax.vmap(lambda t: sky_position(t, params))(ts)
    x = x[:, 0] if x.ndim > 1 else x
    y = y[:, 1] if y.ndim > 1 else y

    pmra  = fit_slope(ts, x)
    pmdec = fit_slope(ts, y)

    return pmra, pmdec
def hgca_model_vector(params,mu_com,
                      tG_all,hip_epochs,scan_gaia,ra,dec,
                      tG1=2014.56, tG2=2017.83,
                      tH1=1989.85, tH2=1993.21,
                      tHmid=1991.25, tGmid=2016.0):

    # Gaia PM
    #pmra_G, pmdec_G = model_pm(tG_all, params)
    params_gaia = gaia_model(params,tG_all,scan_gaia,mu_com,ra,dec)
    pmra_G = params_gaia[3]
    pmdec_G = params_gaia[4]
    plx_G = params_gaia[2]

    # Hipparcos PM
    #pmra_H, pmdec_H = model_pm(tH_all, params)
    params_hip = hip_model(params,hip_epochs[0],hip_epochs[1],hip_epochs[2],hip_epochs[3],mu_com)
    pmra_H = params_hip[3]
    pmdec_H = params_hip[4]

    # Hip-Gaia baseline
    pmra_HG, pmdec_HG = model_pm(jnp.array([tHmid,tGmid]), params)

    # Stack into 6D vector
    return jnp.array([
        pmra_G+0*mu_com[0], pmdec_G+0*mu_com[1],
        pmra_H+0*mu_com[0], pmdec_H+0*mu_com[1],
        pmra_HG+mu_com[0], pmdec_HG+mu_com[1]
    ])

def residuals(params, mu_com, y_obs, tG_all, tH_all):
    y_model = hgca_model_vector(params, mu_com, tG_all, tH_all)
    return y_obs - y_model

def hgca_loglike(params, mu_com, y_obs, Cinv, tG_all, tH_all):
    r = residuals(params, mu_com, y_obs, tG_all, tH_all)
    return -0.5 * (r @ Cinv @ r)

def gaia_model(params,tG_all,scan_angle,mu_com,ra,dec,al_err=0.18):
	x_orbit, y_orbit = sky_position(tG_all, params)
	cond = jnp.max(tG_all) > 2019
	epoch = jnp.where(cond, 2017.5, 2016.0)
	"""if max(tG_all)>2019:
		epoch = 2017.5
	else:
		epoch = 2016"""
	x_star, y_star = star_model(tG_all, ra, dec, 0, 0, params[8], mu_com[0], mu_com[1], epoch)
	x = x_orbit + x_star
	y = y_orbit + y_star
	x_AL = x * jnp.sin(scan_angle) + y * jnp.cos(scan_angle)
	A = gaia_matrix_AL(tG_all,scan_angle,ra,dec,epoch)
	params_est = gaia_params_iter(x_AL,A,al_err,n_iter=1)
	return params_est

def gaia_offset(params,tG_all,scan_angle,mu_com,ra,dec):
	x_orbit, y_orbit = sky_position(tG_all, params)
	#cond = jnp.max(tG_all) > 2019
	#epoch = jnp.where(cond, 2017.5, 2016.0)
	tmax = jnp.max(tG_all)
	epoch = jnp.where(tmax > 2021,2020.0,jnp.where(tmax > 2019,2017.5,2016.0))
	x_star, y_star = star_model(tG_all, ra, dec, 0, 0, params[8], mu_com[0], mu_com[1], epoch)
	x = x_orbit + x_star
	y = y_orbit + y_star
	x_AL = x * jnp.sin(scan_angle) + y * jnp.cos(scan_angle)
	A = gaia_matrix_AL(tG_all,scan_angle,ra,dec,epoch)
	params_est = gaia_params_iter(x_AL,A,0.2,n_iter=1)
	x_AL0 = A @ params_est
	return x_AL-x_AL0
def hip_matrix(tH_all,sinscan,cosscan,plx_factor):
    A = jnp.column_stack((sinscan,
                    cosscan,
                    plx_factor,
                    tH_all*sinscan,
                    tH_all*cosscan))
    return A

def hip_params_iter(x_obs, A, al_err, n_iter=5):
    """AGIS-like robust iterative WLS with constant design matrix A."""
    
    # initial weights = 1 / sigma^2
    w0 = jnp.ones(len(x_obs)) / (al_err**2)

    def step(carry, _):
        w = carry
        # Weighted normal matrix
        Aw = A * w[:, None]         # weight each row
        AtAw = A.T @ Aw
        AtWx = A.T @ (w * x_obs)

        # Solve for params
        params = jnp.linalg.solve(AtAw, AtWx)

        # Compute residuals
        R = x_obs - A @ params

        # Update weights (Huber-like, Gaia uses similar scheme)
        sigma = jnp.median(jnp.abs(R)) * 1.4826 + 1e-12
        w_new = 1.0 / (sigma**2 + (R**2))   # simple robustifier

        return w_new, params

    w_final, params_last = jax.lax.scan(step, w0, None, length=n_iter)
    return params_last[-1]
    
def hip_model(params,tH_all,sinscan,cosscan,plx_factor,mu_com):
    x_orbit, y_orbit = sky_position(tH_all+1991.25, params)
    xAL_orbit = x_orbit*sinscan+y_orbit*cosscan
    x_AL_star = plx_factor*1000/params[8]+mu_com[0]*tH_all*sinscan++mu_com[1]*tH_all*cosscan
    x_AL = xAL_orbit+x_AL_star
    A = hip_matrix(tH_all,sinscan,cosscan,plx_factor)
    params_est = hip_params_iter(x_AL, A, 1.0, n_iter=1)
    return params_est
	
def gaia_loglike(params,tG_all,scan_angle,mu_com,hgca_data,Cinv):
	r = hgca_data['gaia_obs'][2:7]-gaia_model(params,tG_all,scan_angle,mu_com,hgca_data['gaia_obs'][0],hgca_data['gaia_obs'][1])
	return -0.5 * (r @ Cinv @ r)
	
def loglike_total(params, mu_com, hgca_data, tG_all, tH_all, scan_angle, rv_data=None):
    # --- HGCA ---
    ll = hgca_loglike(params, mu_com, hgca_data["y_obs"], hgca_data["Cinv"], tG_all, tH_all)
    ll += gaia_loglike(params,tG_all,scan_angle,mu_com,hgca_data,hgca_data["Cinv_gaia"])

    # --- RV (optional) ---
    if rv_data is not None:
        t, v, sigma = rv_data

        v_model = rv_model(t, params)  # you already have this
        ll += -0.5 * jnp.sum(((v - v_model) / sigma)**2)

    return ll

def cov_2x2(sigma_ra, sigma_dec, rho):
    return jnp.array([
        [sigma_ra**2, rho * sigma_ra * sigma_dec],
        [rho * sigma_ra * sigma_dec, sigma_dec**2]
    ])
    
def build_covariance(row):

    C = jnp.zeros((6, 6))

    # --- Gaia block ---
    CG = cov_2x2(
        row["pmra_gaia_error"],
        row["pmdec_gaia_error"],
        row["pmra_pmdec_gaia"]
    )

    # --- Hipparcos block ---
    CH = cov_2x2(
        row["pmra_hip_error"],
        row["pmdec_hip_error"],
        row["pmra_pmdec_hip"]
    )

    # --- HG block ---
    CHG = cov_2x2(
        row["pmra_hg_error"],
        row["pmdec_hg_error"],
        row["pmra_pmdec_hg"]
    )

    # place diagonal blocks
    C = C.at[0:2, 0:2].set(CG)
    C = C.at[2:4, 2:4].set(CH)
    C = C.at[4:6, 4:6].set(CHG)

    # --- Cross terms ---
    # Example: Hip-Gaia RA correlation
    C = C.at[2, 0].set(
        0
        * row["pmra_hip_error"]
        * row["pmra_gaia_error"]
    )
    C = C.at[0, 2].set(C[2, 0])

    # Dec-Dec
    C = C.at[3, 1].set(
        0
        * row["pmdec_hip_error"]
        * row["pmdec_gaia_error"]
    )
    C = C.at[1, 3].set(C[3, 1])
    # Repeat for:
    # - Hip - HG
    # - Gaia - HG
    # (HGCA provides all of these)

    return C
import jax.numpy as jnp

def build_gaia_covariance(row):
    # standard deviations
    sig = jnp.array([
        row["ra_error"],
        row["dec_error"],
        row["parallax_error"],
        row["pmra_error"],
        row["pmdec_error"]
    ])

    # start with identity (diagonal = 1)
    corr = jnp.eye(5)

    # fill correlation matrix
    corr = corr.at[0, 1].set(row["ra_dec_corr"])
    corr = corr.at[1, 0].set(row["ra_dec_corr"])

    corr = corr.at[0, 2].set(row["ra_parallax_corr"])
    corr = corr.at[2, 0].set(row["ra_parallax_corr"])

    corr = corr.at[0, 3].set(row["ra_pmra_corr"])
    corr = corr.at[3, 0].set(row["ra_pmra_corr"])

    corr = corr.at[0, 4].set(row["ra_pmdec_corr"])
    corr = corr.at[4, 0].set(row["ra_pmdec_corr"])

    corr = corr.at[1, 2].set(row["dec_parallax_corr"])
    corr = corr.at[2, 1].set(row["dec_parallax_corr"])

    corr = corr.at[1, 3].set(row["dec_pmra_corr"])
    corr = corr.at[3, 1].set(row["dec_pmra_corr"])

    corr = corr.at[1, 4].set(row["dec_pmdec_corr"])
    corr = corr.at[4, 1].set(row["dec_pmdec_corr"])

    corr = corr.at[2, 3].set(row["parallax_pmra_corr"])
    corr = corr.at[3, 2].set(row["parallax_pmra_corr"])

    corr = corr.at[2, 4].set(row["parallax_pmdec_corr"])
    corr = corr.at[4, 2].set(row["parallax_pmdec_corr"])

    corr = corr.at[3, 4].set(row["pmra_pmdec_corr"])
    corr = corr.at[4, 3].set(row["pmra_pmdec_corr"])

    # convert to covariance: C = D * corr * D
    D = jnp.diag(sig)
    C = D @ corr @ D

    return C
def build_y_obs(row):
    return jnp.array([
        row["pmra_gaia"],
        row["pmdec_gaia"],
        row["pmra_hip"]+row["crosscal_pmra_hip"],
        row["pmdec_hip"]+row["crosscal_pmdec_hip"],
        row["pmra_hg"]+row["crosscal_pmra_hg"]+row["nonlinear_dpmra"],
        row["pmdec_hg"]+row["crosscal_pmdec_hg"]+row["nonlinear_dpmdec"]
    ])

def pole_vector(i, Omega):
    return jnp.array([
        jnp.sin(i) * jnp.sin(Omega),
        -jnp.sin(i) * jnp.cos(Omega),
        jnp.cos(i)
    ])

# Radial velocity model
def rv_model(f, P, e, q, omega, M_star, inc):
    P_year = P/365.25
    a = ((M_star*P_year**2))**(1./3)
    # RV semi-amplitude
    K = (1.496e11/24/3600) * ((2 * jnp.pi * a) / P) * (q/(1+q)) * jnp.sin(inc) / jnp.sqrt(1 - e ** 2)
    return K * (jnp.cos(omega + f) + e * jnp.cos(omega))
def model_astro(rv_params, hgca_data, gaia_data, hip_epochs, t_gaia, scan_gaia, M_star=1.0, return_epochs=False,return_dr4_sim=False):
    JD0 = 2457388.5
    parallax = gaia_data['parallax']#numpyro.sample("parallax", dist.Uniform(gaia_data['parallax']-10, gaia_data['parallax']+10))
    pmra = numpyro.sample("pmra", dist.Normal(gaia_data['pmra'],0.2))
    pmdec = numpyro.sample("pmdec", dist.Normal(gaia_data['pmdec'],0.2))
    cosi = numpyro.sample("cos_i", dist.Uniform(-1, 1))
    Omega = numpyro.sample("Omega", dist.Uniform(-2*jnp.pi, 2*jnp.pi))
    inc = jnp.arccos(cosi)
    params = (rv_params['P'],rv_params['e'],rv_params['q'],inc,Omega,rv_params['omega'],rv_params['T0'],M_star,1000./parallax)
    mu_com = jnp.array([pmra,pmdec])
    y_sim = hgca_model_vector(params,mu_com,t_gaia,hip_epochs,scan_gaia,gaia_data['ra'],gaia_data['dec'])
    gaia_sim = gaia_model(params,t_gaia,scan_gaia,mu_com,gaia_data['ra'],gaia_data['dec'])
    #gaia_obs = jnp.array([0,0,gaia_data['parallax'],gaia_data['pmra'],gaia_data['pmdec']])
    #numpyro.sample("gaia_sim",dist.MultivariateNormal(loc=gaia_sim, precision_matrix=hgca_data['Cinv_gaia']),obs=gaia_obs)
    numpyro.sample("hgca_pm",dist.MultivariateNormal(loc=y_sim, precision_matrix=hgca_data['Cinv']),obs=hgca_data['y_obs'])
    numpyro.deterministic('hgca_sim',y_sim)
    numpyro.deterministic('gaia_params_dr3',gaia_sim)
    if return_epochs:
        x_AL = gaia_offset(params,t_gaia,scan_gaia,mu_com,gaia_data['ra'],gaia_data['dec'])
        numpyro.deterministic('x_dr3',x_AL)
        t_dr4,scan_dr4 = gaiascanlaw.scanlaw(gaia_data['ra'],gaia_data['dec'],tstart=gaiascanlaw.tstart,tend=gaiascanlaw.tdr4)
        x_AL = gaia_offset(params,t_dr4,scan_dr4,mu_com,gaia_data['ra'],gaia_data['dec'])
        numpyro.deterministic('x_dr4',x_AL)
        t_dr5,scan_dr5 = gaiascanlaw.scanlaw(gaia_data['ra'],gaia_data['dec'],tstart=gaiascanlaw.tstart,tend=gaiascanlaw.tdr5)
        x_AL = gaia_offset(params,t_dr5,scan_dr5,mu_com,gaia_data['ra'],gaia_data['dec'])
        numpyro.deterministic('x_dr5',x_AL)
    if return_dr4_sim:
        t_dr4,scan_dr4 = gaiascanlaw.scanlaw(gaia_data['ra'],gaia_data['dec'],tstart=gaiascanlaw.tstart,tend=gaiascanlaw.tdr4)
        dr4_gaia = gaia_model(params,t_dr4,scan_dr4,mu_com,gaia_data['ra'],gaia_data['dec'],al_err=0.14)
        numpyro.deterministic('gaia_params_dr4',dr4_gaia)
    
def model_all(rv_obs, hgca_data, gaia_data, hip_epochs, t_gaia, scan_gaia, M_star=1.0, n_planets=1, return_epochs=False):
    JD0 = 2457388.5
    t_rv = rv_obs[:,0]
    rv_data = rv_obs[:,1]
    t_ref = 0.0
    parallax = gaia_data['parallax']#numpyro.sample("parallax", dist.Uniform(gaia_data['parallax']-10, gaia_data['parallax']+10))
    pmra = numpyro.sample("pmra", dist.Uniform(gaia_data['pmra']-40, gaia_data['pmra']+40))
    pmdec = numpyro.sample("pmdec", dist.Uniform(gaia_data['pmdec']-40, gaia_data['pmdec']+40))
    cosi = numpyro.sample("cos_i", dist.Uniform(-1, 1))
    Omega = numpyro.sample("Omega", dist.Uniform(-2*jnp.pi, 2*jnp.pi))
    inc = jnp.arccos(cosi)
    P = jnp.zeros(n_planets)
    e = jnp.zeros(n_planets)
    q = jnp.zeros(n_planets)
    omega = jnp.zeros(n_planets)
    M0 = jnp.zeros(n_planets)
    T0 = jnp.zeros(n_planets)
    if np.min(t_rv)<2455000 and np.max(t_rv)>2455000:
        diff = True
        offset1 = numpyro.sample("offset1", dist.Uniform(-2000, 2000))
        offset2 = numpyro.sample("offset2", dist.Uniform(-2000, 2000))
        els1 = np.where(t_rv<=2455000)[0]
        els2 = np.where(t_rv>2455000)[0]
        rv1 = jnp.ones(len(els1))*offset1
        rv2 = jnp.ones(len(els2))*offset2
        f1 = jnp.zeros((n_planets,len(els1)))
        f2 = jnp.zeros((n_planets,len(els2)))
    else:
        diff = False    
        offset = numpyro.sample("offset", dist.Uniform(-2000, 2000))
        rv = jnp.ones(len(t_rv))*offset
        f = jnp.zeros((n_planets,len(t_rv)))
    for ii in range(n_planets):
        logP = numpyro.sample("logP"+str(ii+1), dist.Uniform(jnp.log(100), jnp.log(10000)))
        P = P.at[ii].set(jnp.exp(logP))
        e = e.at[ii].set(numpyro.sample("e"+str(ii+1), dist.Beta(1.0, 3.0)))
        logq = numpyro.sample("logq"+str(ii+1),dist.Uniform(jnp.log(2e-4), jnp.log(1e-1)))
        q = q.at[ii].set(jnp.exp(logq))
        omega = omega.at[ii].set(numpyro.sample("omega"+str(ii+1), dist.Uniform(-2*jnp.pi, 2*jnp.pi)))
        M0 = M0.at[ii].set(numpyro.sample("M0"+str(ii+1), dist.Uniform(-jnp.pi, jnp.pi)))
        n = 2 * jnp.pi / P[ii]
        T0 = T0.at[ii].set(t_ref - M0[ii] / n + JD0)
        if diff:
            f1 = f1.at[ii].set(true_anomaly(t_rv[els1] - JD0 - t_ref, n, e[ii], M0[ii]))
            f2 = f2.at[ii].set(true_anomaly(t_rv[els2] - JD0 - t_ref, n, e[ii], M0[ii]))
            rv1 = rv1+rv_model(f1[ii], P[ii], e[ii], q[ii], omega[ii], M_star, inc)
            rv2 = rv2+rv_model(f2[ii], P[ii], e[ii], q[ii], omega[ii], M_star, inc)
        else:
            f = f.at[ii].set(true_anomaly(t_rv - JD0 - t_ref, n, e[ii], M0[ii]))
            rv = rv+rv_model(f[ii], P[ii], e[ii], q[ii], omega[ii], M_star, inc)
    if diff:
        jitter1 = numpyro.sample("jitter1", dist.HalfNormal(20.0))
        jitter2 = numpyro.sample("jitter2", dist.HalfNormal(20.0))
        rv_err1 = jnp.sqrt(rv_obs[els1,2]**2 + jitter1**2)
        rv_err2 = jnp.sqrt(rv_obs[els2,2]**2 + jitter2**2)
        numpyro.sample("RV1", dist.Normal(rv1, rv_err1), obs=rv_data[els1])
        numpyro.deterministic("RV_sim1", rv1)
        numpyro.sample("RV2", dist.Normal(rv2, rv_err2), obs=rv_data[els2])
        numpyro.deterministic("RV_sim2", rv2)
    else:
        jitter = numpyro.sample("jitter", dist.HalfNormal(20.0))
        rv_err = jnp.sqrt(rv_obs[:,2]**2 + jitter**2)
        numpyro.sample("RV", dist.Normal(rv, rv_err), obs=rv_data)
        numpyro.deterministic("RV_sim", rv)
    params = (P,e,q,inc,Omega,omega,T0,M_star,1000./parallax)
    mu_com = jnp.array([pmra,pmdec])
    y_sim = hgca_model_vector(params,mu_com,t_gaia,hip_epochs,scan_gaia,gaia_data['ra'],gaia_data['dec'])
    gaia_sim = gaia_model(params,t_gaia,scan_gaia,mu_com,gaia_data['ra'],gaia_data['dec'])
    #gaia_obs = jnp.array([0,0,gaia_data['parallax'],gaia_data['pmra'],gaia_data['pmdec']])
    #numpyro.sample("gaia_sim",dist.MultivariateNormal(loc=gaia_sim, precision_matrix=hgca_data['Cinv_gaia']),obs=gaia_obs)
    numpyro.sample("hgca_pm",dist.MultivariateNormal(loc=y_sim, precision_matrix=hgca_data['Cinv']),obs=hgca_data['y_obs'])
    numpyro.deterministic('hgca_sim',y_sim)
    numpyro.deterministic('gaia_params',gaia_sim)
    numpyro.deterministic("T0", T0)
    if return_epochs:
        x_AL = gaia_offset(params,t_gaia,scan_gaia,mu_com,gaia_data['ra'],gaia_data['dec'])
        numpyro.deterministic('x_dr3',x_AL)
        t_dr4,scan_dr4 = gaiascanlaw.scanlaw(gaia_data['ra'],gaia_data['dec'],tstart=gaiascanlaw.tstart,tend=gaiascanlaw.tdr4)
        x_AL = gaia_offset(params,t_dr4,scan_dr4,mu_com,gaia_data['ra'],gaia_data['dec'])
        numpyro.deterministic('x_dr4',x_AL)
        t_dr5,scan_dr5 = gaiascanlaw.scanlaw(gaia_data['ra'],gaia_data['dec'],tstart=gaiascanlaw.tstart,tend=gaiascanlaw.tdr5)
        x_AL = gaia_offset(params,t_dr5,scan_dr5,mu_com,gaia_data['ra'],gaia_data['dec'])
        numpyro.deterministic('x_dr5',x_AL)

all_data = pickle.load(open('all_data.pkl','rb'))
def run_astro(name,n_planet):
    #samples,log_prob = pickle.load(open('results/samples_'+name+'_rv_test_'+str(n_planet)+'.pkl','rb'))
    samples,log_prob = pickle.load(open('results/rv_good.pkl','rb'))
    data = all_data[name]
    rv_obs = data['rv_obs']
    rv_params = {'P':np.exp(np.median(samples['logP2'])),'e':np.median(samples['e2']),'q':np.exp(np.median(samples['logq2'])),'omega':np.median(samples['omega2']),'T0':np.median(samples['T02'])}
    P = [jnp.exp(samples['logP1']),jnp.exp(samples['logP2'])]
    e = [samples['e1'],samples['e2']]
    q = [jnp.exp(samples['logq1']),jnp.exp(samples['logq2'])]
    omega = [samples['omega1'],samples['omega2']]
    M0 = [samples['M01'],samples['M02']]
    offset = samples['offset']
    el = jnp.argmax(log_prob)
    rv_params = llh_max.get_rv_params_one(samples,n_planet,idx=el)
    frequency, power = LombScargle(rv_obs[:,0], rv_obs[:,1]).autopower(samples_per_peak=10)
    period = 1./frequency[np.argmax(power)]
    hgca_data = {}
    hgca_data['y_obs'] = build_y_obs(data['hg_obs'])
    hgca_data['C'] = build_covariance(data['hg_obs'])
    hgca_data['C_gaia'] = build_gaia_covariance(data)
    hgca_data['gaia_obs'] = np.array([data['ra'],data['dec'],0,0,data['parallax'],data['pmra'],data['pmdec']])
    t_hip,pf_hip,sin_hip,cos_hip = find_hip('hip_epochs.csv',data['hg_obs']['hip_id'])
    #t_hip = find_hip('hip_'+name+'.csv')
    t_gaia,scan_gaia = gaiascanlaw.scanlaw(data['hg_obs']['gaia_ra'],data['hg_obs']['gaia_dec'],tstart=gaiascanlaw.tstart,tend=gaiascanlaw.tdr3)
    t_dr4,scan_dr4 = gaiascanlaw.scanlaw(data['hg_obs']['gaia_ra'],data['hg_obs']['gaia_dec'],tstart=gaiascanlaw.tstart,tend=gaiascanlaw.tdr4)
    hgca_data['Cinv'] = np.linalg.inv(hgca_data['C'])
    hgca_data['Cinv_gaia'] = np.linalg.inv(hgca_data['C_gaia'])
    M_star = data['mass']
    init_values = {"cos_i": 0, "Omega": 0, "pmra": hgca_data['y_obs'][0], "pmdec": hgca_data['y_obs'][1]}
    #init_values = {"logP1":jnp.log(period),"logq1":jnp.log(1e-3),"M01":0,"e1":0.01,"offset1":0,"offset2":0,"offset":0,"cos_i": 0, "Omega": 0, "pmra": hgca_data['y_obs'][0], "pmdec": hgca_data['y_obs'][1]}
    kernel = NUTS(model_astro,init_strategy=init_to_value(values=init_values))
    #kernel = NUTS(model_all,init_strategy=init_to_value(values=init_values))
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=10000)
    mcmc.run(jax.random.PRNGKey(1), rv_params=rv_params, hgca_data=hgca_data, gaia_data=data, hip_epochs=(t_hip,sin_hip,cos_hip,pf_hip), t_gaia=t_gaia, scan_gaia=scan_gaia, M_star=M_star, return_epochs=True, return_dr4_sim=True, extra_fields=("potential_energy",))
    #mcmc.run(jax.random.PRNGKey(1), rv_obs=rv_obs, hgca_data=hgca_data, gaia_data=data, hip_epochs=(t_hip,sin_hip,cos_hip,pf_hip), t_gaia=t_gaia, scan_gaia=scan_gaia, M_star=M_star, n_planets=1, return_epochs=True, extra_fields=("potential_energy",))
    mcmc.print_summary()
    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    log_prob = -extra["potential_energy"]
    pickle.dump((samples,log_prob),open('results/samples_hgca_'+name+'.pkl','wb'))