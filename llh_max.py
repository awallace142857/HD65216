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
#import hgca_inference
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
def ruwe_calc(params,tG_all,scan_gaia,mu_com,ra,dec,epoch=2016,al_err=0,noise=0):
	x_orbit, y_orbit = sky_position(tG_all, params)
	x_star, y_star = star_model(tG_all, ra, dec, 0, 0, params[8], mu_com[0], mu_com[1], epoch)
	x = x_orbit + x_star
	y = y_orbit + y_star
	x_AL = x * jnp.sin(scan_gaia) + y * jnp.cos(scan_gaia)+noise
	A = gaia_matrix_AL(tG_all,scan_gaia,ra,dec,epoch)
	params_est = gaia_params_iter(x_AL,A,al_err,n_iter=1)
	x_AL0 = A @ params_est
	N = len(tG_all)
	ruwe = jnp.sqrt(jnp.sum((x_AL-x_AL0)**2)/((N-5)*al_err**2))
	return ruwe
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
    JD0 = 2457388.5
    inc *= jnp.ones(len(P))
    Omega *= jnp.ones(len(P))
    # convert to days relative to reference
    t_days = (t_year - 2016.0) * 365.25
    x_tot = 0
    y_tot = 0
    for ii in range(len(P)):
        n = 2 * jnp.pi / P[ii]
        M0 = n * (JD0 - T0[ii])
        f = true_anomaly(t_days, n, e[ii], M0)
        x,y = astrometric_model(
            f, P[ii], e[ii], q[ii], inc[ii], Omega[ii], omega[ii], M_star, distance
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
    params_est = hip_params_iter(x_AL, A, 1.0, n_iter=10)
    return params_est

def residuals(params, mu_com, y_obs, tG_all, scan_gaia, hip_epochs, ra, dec):
    y_model = hgca_model_vector(params, mu_com, tG_all, hip_epochs, scan_gaia, ra, dec)
    return y_obs - y_model

def hgca_loglike(params, mu_com, y_obs, Cinv, tG_all, scan_gaia, hip_epochs, ra, dec):
    r = residuals(params, mu_com, y_obs, tG_all, scan_gaia, hip_epochs, ra, dec)
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
	params_est = gaia_params_iter(x_AL,A,al_err,n_iter=10)
	return params_est
def gaia_loglike(params,tG_all,scan_angle,mu_com,hgca_data,Cinv):
	r = hgca_data['gaia_obs'][2:7]-gaia_model(params,tG_all,scan_angle,mu_com,hgca_data['gaia_obs'][0],hgca_data['gaia_obs'][1])
	return -0.5 * (r @ Cinv @ r)
	
def loglike_total(params, mu_com, hgca_data, tG_all, hip_epochs, scan_angle, ra, dec, rv_data=None):
    # --- HGCA ---
    ll = hgca_loglike(params, mu_com, hgca_data["y_obs"], hgca_data["Cinv"], tG_all, scan_angle, hip_epochs, ra, dec)
    #ll += gaia_loglike(params,tG_all,scan_angle,mu_com,hgca_data,hgca_data["Cinv_gaia"])

    # --- RV (optional) ---
    if rv_data is not None:
        t, v, sigma = rv_data

        v_model = rv_model(t, params)  # you already have this
        ll += -0.5 * jnp.sum(((v - v_model) / sigma)**2)

    return ll
    
def true_anomaly(t_rel,n,e,M0):
    M = n * t_rel + M0
    #x = jnp.cos(M)
    #y = jnp.sin(M)
    #M = jnp.arctan2(y,x)
    E = kepler_eq(M, e)
    return 2 * jnp.arctan2(jnp.sqrt(1 + e) * jnp.sin(E / 2), jnp.sqrt(1 - e) * jnp.cos(E / 2))

def kepler_eq(M, e, max_iter=20):
    def body_fn(E, _):
        f = E - e * jnp.sin(E) - M
        f_prime = 1 - e * jnp.cos(E)
        E_new = E - f / f_prime
        return E_new, None

    E0 = M# + e * jnp.sin(M)
    E_final, _ = jax.lax.scan(body_fn, E0, xs=None, length=max_iter)
    return E_final

# Radial velocity model
def rv_model(f, P, e, q, omega, M_star, inc, offset):
    P_year = P/365.25
    a = ((M_star*P_year**2))**(1./3)
    # RV semi-amplitude
    K = (1.496e11/24/3600) * ((2 * jnp.pi * a) / P) * (q/(1+q)) * jnp.sin(inc) / jnp.sqrt(1 - e ** 2)
    return K * (jnp.cos(omega + f) + e * jnp.cos(omega))+offset
    
def loglike_rv(params, rv_data):
    t = rv_data[:,0]
    v = rv_data[:,1]
    sigma = rv_data[:,2]
    P,e,q,omega,M0,M_star,inc,offset = params
    JD0 = 2457388.5
    v_model = 0
    n_planets = len(P)
    for ii in range(n_planets):
        n = 2*jnp.pi/P[ii]
        f = true_anomaly(t - JD0, n, e[ii], M0[ii])
        v_model += rv_model(f, P[ii], e[ii], q[ii], omega[ii], M_star, inc, offset)
    ll = -0.5 * jnp.sum(((v - v_model) / sigma)**2)
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
        row["pmdec_hg"]+row["crosscal_pmdec_hg"]+row["nonlinear_dpmdec"],
    ])

def pole_vector(i, Omega):
    return jnp.array([
        jnp.sin(i) * jnp.sin(Omega),
        -jnp.sin(i) * jnp.cos(Omega),
        jnp.cos(i)
    ])

def angle2(i1,Omega1,phi,psi):
    h1 = pole_vector(i1, Omega1)
    tmp = jnp.array([1., 0., 0.])
    tmp = jnp.where(jnp.abs(h1[0]) > 0.9,jnp.array([0.,1.,0.]),tmp)
    u = jnp.cross(h1, tmp)
    u = u / jnp.linalg.norm(u)
    v = jnp.cross(h1, u)
    h2 = (jnp.cos(phi) * h1 + jnp.sin(phi) * (jnp.cos(psi) * u + jnp.sin(psi) * v))
    cosi2 = h2[2]
    Omega2 = jnp.arctan2(h2[0], -h2[1])
    Omega2 = jnp.mod(Omega2, 2*jnp.pi)
    return jnp.arccos(cosi2),Omega2
def unpack_theta(theta, fixed):
    inc, Omega, mu_ra, mu_dec = theta
    P, e, qsini, omega, T0, M_star, parallax = fixed
    q = qsini/jnp.sin(inc)
    params = (P, e, q, inc, Omega, omega, T0, M_star, 1000./parallax)
    mu_com = jnp.array([mu_ra, mu_dec])

    return params, mu_com
def logprior_rv(theta):
    P,e,q,omega,M0,offset = theta
    valid = jnp.all((jnp.array(e) >= 0.0) & (jnp.array(e) < 1.0))
    return jnp.where(valid, 0.0, -1e30)
def unpack_theta_rv(theta, fixed):
    P,e,q,omega,M0,offset = theta
    M_star,inc = fixed
    params = (P,e,q,omega,M0,M_star,inc,offset)
    return params

def log_posterior(theta, fixed, hgca_data, tG_all, hip_epochs, scan, ra, dec, rv_data=None):
    params, mu_com = unpack_theta(theta, fixed)
    ll = loglike_total(params, mu_com, hgca_data, tG_all, hip_epochs, scan, ra, dec, rv_data)
    return ll
    
def log_posterior_rv(theta, fixed, rv_data):
    lp = logprior_rv(theta)
    params = unpack_theta_rv(theta, fixed)
    ll = loglike_rv(params, rv_data)
    return ll+lp
import optax

def find_map(theta0, fixed, hgca_data, tG_all, hip_epochs, scan, ra, dec):
    loss = lambda th: -log_posterior(
        th, fixed, hgca_data,
        tG_all, hip_epochs,
        scan, ra, dec
    )
    opt = optax.adam(learning_rate=1e-3)
    @jax.jit
    def optimize(theta0):
        opt_state = opt.init(theta0)
        def body_fun(i, carry):
            theta, opt_state = carry
            grads = jax.grad(loss)(theta)
            updates, opt_state = opt.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)
            return (theta, opt_state)
        theta, opt_state = jax.lax.fori_loop(
            0,
            2000,
            body_fun,
            (theta0, opt_state)
        )
        return theta
    return optimize(theta0)
    
def find_map_rv(theta0, fixed, rv_data):

    opt = optax.adam(learning_rate=1e-2)
    opt_state = opt.init(theta0)

    @jax.jit
    def step(theta, opt_state):
        loss = lambda th: -log_posterior_rv(th, fixed, rv_data)
        grads = jax.grad(loss)(theta)

        updates, opt_state = opt.update(grads, opt_state)
        theta = optax.apply_updates(theta, updates)

        return theta, opt_state

    theta = theta0
    for _ in range(400):
        print(theta)
        theta, opt_state = step(theta, opt_state)

    return theta
all_data = pickle.load(open('all_data.pkl','rb'))
def get_rv_params(name,n_planet):
	samples = pickle.load(open('results/samples_'+name+'_rv_test.pkl','rb'))
	if n_planet==1:
		rv_params = {'P':jnp.array([np.exp(np.median(samples['logP']))]),'e':jnp.array([np.median(samples['e'])]),'q':jnp.array([np.exp(np.median(samples['logq']))]),'omega':jnp.array([np.median(samples['omega'])]),'T0':jnp.array([np.median(samples['T0'])])}
	else:
		P = jnp.zeros(n_planet)
		e = jnp.zeros(n_planet)
		q = jnp.zeros(n_planet)
		omega = jnp.zeros(n_planet)
		T0 = jnp.zeros(n_planet)
		for ii in range(n_planet):
			P = P.at[ii].set(jnp.exp(np.median(samples['logP'+str(ii+1)])))
			e = e.at[ii].set(jnp.median(samples['e'+str(ii+1)]))
			q = q.at[ii].set(jnp.exp(np.median(samples['logq'+str(ii+1)])))
			omega = omega.at[ii].set(jnp.median(samples['omega'+str(ii+1)]))
			T0 = T0.at[ii].set(jnp.median(samples['T0'][ii]))
		rv_params = {'P':P,'e':e,'q':q,'omega':omega,'T0':T0}
	return rv_params
def get_rv_params_one(samples,n_planet,idx=0):
	P = jnp.zeros(n_planet)
	e = jnp.zeros(n_planet)
	q = jnp.zeros(n_planet)
	omega = jnp.zeros(n_planet)
	T0 = jnp.zeros(n_planet)
	for ii in range(n_planet):
		P = P.at[ii].set(jnp.exp(samples['logP'+str(ii+1)][idx]))
		e = e.at[ii].set(samples['e'+str(ii+1)][idx])
		q = q.at[ii].set(jnp.exp(samples['logq'+str(ii+1)][idx]))
		omega = omega.at[ii].set(samples['omega'+str(ii+1)][idx])
		T0 = T0.at[ii].set(samples['T0'][ii][idx])
	rv_params = {'P':P,'e':e,'q':q,'omega':omega,'T0':T0}
	return rv_params
def do_astrometry(name,samples,n_planets):
    N = len(samples['logP1'])
    data = all_data[name]
    t_gaia,scan_gaia = gaiascanlaw.scanlaw(data['hg_obs']['gaia_ra'],data['hg_obs']['gaia_dec'],tstart=gaiascanlaw.tstart,tend=gaiascanlaw.tdr3)
    hip_epochs = find_hip('hip_epochs.csv',data['hg_obs']['hip_id'])
    new_keys = ['inc','Omega']
    for key in new_keys:
        samples[key] = jnp.zeros(N)
    samples['hgca_sim'] = jnp.zeros((N,6))
    for ii in range(N):
        rv_params = get_rv_params_one(samples,n_planets,idx=ii)
        params_map,mu_map = run_astro(name,rv_params,n_planets,inc0=jnp.pi/2,Omega0=1.0)
        samples['inc'] = samples['inc'].at[ii].set(params_map[3][0])
        samples['Omega'] = samples['Omega'].at[ii].set(params_map[4][0])
        y = hgca_model_vector(params_map,mu_map,t_gaia,hip_epochs,scan_gaia,data['ra'],data['dec'])
        samples['hgca_sim'] = samples['hgca_sim'].at[ii].set(y)
        if ii%100==0:
            print(ii)
        if ii%1000==0:
            pickle.dump(samples,open('results/samples_'+name+'_test_'+str(n_planets)+'.pkl','wb'))
    return samples

def prepare_star(name):
    data = all_data[name]
    hgca_data = {}
    hgca_data['y_obs'] = build_y_obs(data['hg_obs'])
    hgca_data['C'] = build_covariance(data['hg_obs'])
    hgca_data['C_gaia'] = build_gaia_covariance(data)

    hgca_data['gaia_obs'] = np.array([
        data['ra'],
        data['dec'],
        0,
        0,
        data['parallax'],
        data['pmra'],
        data['pmdec']
    ])

    t_hip,pf_hip,sin_hip,cos_hip = find_hip(
        'hip_epochs.csv',
        data['hg_obs']['hip_id']
    )

    t_gaia,scan_gaia = gaiascanlaw.scanlaw(
        data['hg_obs']['gaia_ra'],
        data['hg_obs']['gaia_dec'],
        tstart=gaiascanlaw.tstart,
        tend=gaiascanlaw.tdr3
    )
    t_dr4,scan_dr4 = gaiascanlaw.scanlaw(
        data['hg_obs']['gaia_ra'],
        data['hg_obs']['gaia_dec'],
        tstart=gaiascanlaw.tstart,
        tend=gaiascanlaw.tdr4
    )
    hgca_data['Cinv'] = np.linalg.inv(hgca_data['C'])
    hgca_data['Cinv_gaia'] = np.linalg.inv(hgca_data['C_gaia'])

    const = {
        'hgca_data': hgca_data,
        't_gaia': t_gaia,
        'hip': (t_hip,sin_hip,cos_hip,pf_hip),
        'scan_gaia': scan_gaia,
        't_dr4': t_dr4,
        'scan_dr4': scan_dr4,
        'ra': data['ra'],
        'dec': data['dec'],
        'M_star': data['mass'],
        'parallax': data['hg_obs']['parallax_gaia'],
        'pmra0': data['hg_obs']['pmra_gaia'],
        'pmdec0': data['hg_obs']['pmdec_gaia']
    }
    return const	

def eval_one(rv_params, const):
    theta0 = jnp.array([
        const['i0'],
        jnp.pi,
        const['pmra0'],
        const['pmdec0']
    ])
    fixed = [
        rv_params.P,
        rv_params.e,
        rv_params.q,
        rv_params.omega,
        rv_params.T0,
        const['M_star'],
        const['parallax']
    ]
    theta_map = find_map(
        theta0,
        fixed,
        const['hgca_data'],
        const['t_gaia'],
        const['hip'],
        const['scan_gaia'],
        const['ra'],
        const['dec']
    )
    params_map, mu_map = unpack_theta(theta_map, fixed)
    x_diff = hgca_inference.gaia_offset(params_map,const['t_dr4'],const['scan_dr4'],mu_map,const['ra'],const['dec'])
    dr3_gaia = hgca_inference.gaia_model(params_map,const['t_gaia'],const['scan_gaia'],mu_map,const['ra'],const['dec'])
    dr4_gaia = hgca_inference.gaia_model(params_map,const['t_dr4'],const['scan_dr4'],mu_map,const['ra'],const['dec'],al_err=0.14)
    return params_map, mu_map, x_diff, dr3_gaia, dr4_gaia
from typing import NamedTuple

class RVParams(NamedTuple):
    P: jnp.ndarray
    e: jnp.ndarray
    q: jnp.ndarray
    omega: jnp.ndarray
    T0: jnp.ndarray
    
def solve_astro(rv_samples, const):
    batched_eval = jax.vmap(
    eval_one,
    in_axes=(0, None))
    results = batched_eval(rv_samples, const)
    return results
def run_astro(const,rv_params,n_planet,inc0=jnp.pi/2,Omega0=jnp.pi):
	hgca_data = const['hgca_data']
	t_gaia = const['t_gaia']
	(t_hip,sin_hip,cos_hip,pf_hip) = const['hip']
	scan_gaia = const['scan_gaia']
	ra = const['ra']
	dec = const['dec']
	M_star = const['M_star']
	parallax = const['parallax']
	pmra0 = const['pmra0']
	pmdec0 = const['pmdec0']
	theta0 = jnp.array([
		inc0,        # inc
		Omega0,        # Omega
		pmra0,        # mu_ra offset
		pmdec0,         # mu_dec offset
	])
	fixed = [rv_params['P'], rv_params['e'], rv_params['q'], rv_params['omega'], rv_params['T0'], M_star, parallax]
	theta_map = find_map(theta0, fixed, hgca_data, t_gaia, (t_hip,sin_hip,cos_hip,pf_hip), scan_gaia, ra,dec)
	params_map, mu_map = unpack_theta(theta_map, fixed)
	return params_map,mu_map
def run_rv(name,rv_data,P0=[1000.],e0=[0.],q0=[0.001],omega0=[jnp.pi],M00=[0.],offset0=0.):
	theta0 = [P0,e0,q0,omega0,M00,offset0]
	fixed = [all_data[name]['mass'],jnp.pi/2]
	theta_map = find_map_rv(theta0, fixed, rv_data)
	params_map = unpack_theta_rv(theta_map, fixed)
	return params_map