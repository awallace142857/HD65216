import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
import matplotlib.pyplot as plt
import pickle
import sys
import rv_inference as inference
import astromet,os
from astropy.timeseries import LombScargle
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
def rv_model(f, P, e, q, omega, M_star, inc):
    P_year = P/365.25
    a = ((M_star*P_year**2))**(1./3)
    # RV semi-amplitude
    K = (1.496e11/24/3600) * ((2 * jnp.pi * a) / P) * (q/(1+q)) * jnp.sin(inc) / jnp.sqrt(1 - e ** 2)
    return K * (jnp.cos(omega + f) + e * jnp.cos(omega))

def plot_rv(t,samples,M_star,n_planet,el):
    JD0 = 2457388.5
    P,e,q,omega,T0,M0,n,f = [],[],[],[],[],[],[],[]
    for ii in range(n_planet):
        P.append(np.exp(samples['logP'+str(ii+1)][el]))
        e.append(samples['e'+str(ii+1)][el])
        q.append(np.exp(samples['logq'+str(ii+1)][el]))
        omega.append(samples['omega'+str(ii+1)][el])
        #T0.append(np.median(samples['T0'+str(ii+1)]))
        M0.append(samples['M0'+str(ii+1)][el])
        n.append(2*np.pi/P[ii])
        #M0.append(n[ii]*(JD0-T0[ii]))
        f.append(true_anomaly(t-JD0,n[ii],e[ii], M0[ii]))
    rv = 0
    for ii in range(n_planet):
        rv+=rv_model(f[ii], P[ii], e[ii], q[ii], omega[ii], M_star, np.pi/2)
    plt.plot(t-2450000,rv)
#data = pickle.load(open('datafile.pkl','rb'))
all_data = pickle.load(open('all_data.pkl','rb'))
def run_rv(name,n_planets):
    data = all_data[name]
    M_star = data['mass']
    rv_obs = data['rv_obs']
    t_rv = rv_obs[:,0]
    rv_data = rv_obs[:,1]
    els = np.where(rv_obs[:,0]<2454000)[0]
    #rv_obs = rv_obs[els]
    frequency, power = LombScargle(t_rv, rv_data).autopower(samples_per_peak=10)
    period = 1./frequency[np.argmax(power)]
    p_init = [period,1]
    mcmc = inference.run_rv_inference_2planet(name, n_planets=n_planets, rv_obs=rv_obs, p_init=p_init, M_star=M_star, fit_type='')
    mcmc.print_summary()
    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    log_prob = -extra["potential_energy"]
    el = jnp.argmax(log_prob)
    T0 = jnp.zeros((n_planets,len(samples['logP1'])))
    for ii in range(n_planets):
        M0_best = samples['M0'+str(ii+1)][el]
        M0 = samples['M0'+str(ii+1)]%(2*jnp.pi)
        els = jnp.where(M0>M0_best+jnp.pi)[0]
        M0 = M0.at[els].set(M0[els]-2*jnp.pi)
        els = jnp.where(M0<M0_best-jnp.pi)[0]
        M0 = M0.at[els].set(M0[els]+2*jnp.pi)
        n = 2*jnp.pi/jnp.exp(samples['logP'+str(ii+1)])
        T0 = T0.at[ii].set(M0 / n + 2457388.5)
    samples['T0'] = T0.T
    #pickle.dump((samples,log_prob),open('results/samples_'+name+'_rv_test_'+str(n_planets)+'.pkl','wb'))
    pickle.dump((samples,log_prob),open('results/samples_rv_'+name+'.pkl','wb'))
    t = np.linspace(np.min(rv_obs[:,0]),np.max(rv_obs[:,0]),1000)
    plot_rv(t,samples,M_star,n_planets,el)
    if rv_obs.shape[1]==3:
        errs = rv_obs[:,2]
    else:
        errs = np.zeros(rv_obs.shape[0])
    if np.min(rv_obs[:,0])<2455000 and np.max(rv_obs[:,0])>=2455000:
        els1 = np.where(rv_obs[:,0]<2455500)[0]
        els2 = np.where(rv_obs[:,0]>=2455500)[0]
        plt.errorbar(rv_obs[els1,0]-2450000,rv_obs[els1,1]-samples['offset1'][el],yerr=np.sqrt(errs[els1]**2+samples['jitter1'][el]**2),marker='o',color='k',linestyle='none')
        plt.errorbar(rv_obs[els2,0]-2450000,rv_obs[els2,1]-samples['offset2'][el],yerr=np.sqrt(errs[els2]**2+samples['jitter2'][el]**2),marker='o',color='k',linestyle='none')
    else:
        plt.errorbar(rv_obs[:,0]-2450000,rv_obs[:,1]-samples['offset'][el],yerr=np.sqrt(errs**2+samples['jitter'][el]**2),marker='o',color='k',linestyle='none')
    plt.show()
    
