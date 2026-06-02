import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
import matplotlib.pyplot as plt
import pickle
import sys
# Orbital mechanics
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
def rv_model(f, P, e, q, omega, M_star, inc):
    P_year = P/365.25
    a = ((M_star*P_year**2))**(1./3)
    # RV semi-amplitude
    K = (1.496e11/24/3600) * ((2 * jnp.pi * a) / P) * (q/(1+q)) * jnp.sin(inc) / jnp.sqrt(1 - e ** 2)
    return K * (jnp.cos(omega + f) + e * jnp.cos(omega))

def model_rv_vector(rv_obs, n_planets=1, M_star=1.0):
    JD0 = 2457388.5
    t_rv = rv_obs[:,0]
    rv_data = rv_obs[:,1]
    t_ref = 0.0
    inc = jnp.pi/2
    P = jnp.zeros(n_planets)
    e = jnp.zeros(n_planets)
    q = jnp.zeros(n_planets)
    omega = jnp.zeros(n_planets)
    M0 = jnp.zeros(n_planets)
    T0 = jnp.zeros(n_planets)
    offset = numpyro.sample("offset", dist.Uniform(-2000, 2000))
    rv = jnp.ones(len(t_rv))*offset
    f = jnp.zeros((n_planets,len(t_rv)))
    p_init = [500,7000]
    for ii in range(n_planets):
        logP = numpyro.sample("logP"+str(ii+1), dist.Uniform(jnp.log(100), jnp.log(10000)))
        #logP = numpyro.sample("logP"+str(ii+1), dist.Normal(jnp.log(p_init[ii]), jnp.log(2)))
        P = P.at[ii].set(jnp.exp(logP))
        #e = e.at[ii].set(numpyro.sample("e"+str(ii+1), dist.Uniform(0.0, 0.5)))
        e = e.at[ii].set(numpyro.sample("e"+str(ii+1), dist.Beta(1.0, 3.0)))
        logq = numpyro.sample("logq"+str(ii+1),dist.Uniform(jnp.log(2e-4), jnp.log(1e-1)))
        q = q.at[ii].set(jnp.exp(logq))
        omega = omega.at[ii].set(numpyro.sample("omega"+str(ii+1), dist.Uniform(-2*jnp.pi, 2*jnp.pi)))
        M0 = M0.at[ii].set(numpyro.sample("M0"+str(ii+1), dist.Uniform(-7*jnp.pi, 7*jnp.pi)))
        n = 2 * jnp.pi / P[ii]
        #T0 = T0.at[ii].set(t_ref - (M0[ii]%(2*jnp.pi)) / n + JD0)
        f = f.at[ii].set(true_anomaly(t_rv - JD0 - t_ref, n, e[ii], M0[ii]))
        rv = rv+rv_model(f[ii], P[ii], e[ii], q[ii], omega[ii], M_star, inc)
    jitter = numpyro.sample("jitter", dist.HalfNormal(20.0))
    rv_err = jnp.sqrt(rv_obs[:,2]**2 + jitter**2)
    numpyro.sample("RV", dist.Normal(rv, rv_err), obs=rv_data)
    numpyro.deterministic("RV_sim", rv)
    #numpyro.deterministic("T0", T0)
def model_rv_2planet(rv_obs, M_star=1.0):

    JD0 = 2457388.5

    t_rv = rv_obs[:,0]
    rv_data = rv_obs[:,1]
    #rv_err = rv_obs[:,2]

    t_ref = 0.0

    # -------- Planet 1 --------
    logP1 = numpyro.sample("logP1", dist.Uniform(jnp.log(100), jnp.log(20000)))
    P1 = jnp.exp(logP1)
    n1 = 2 * jnp.pi / P1

    e1 = numpyro.sample("e1", dist.Uniform(0.0, 0.3))
    logq1 = numpyro.sample("logq1", dist.Uniform(jnp.log(2e-4), jnp.log(1e-1)))
    q1 = jnp.exp(logq1)
    omega1 = numpyro.sample("omega1", dist.Uniform(-jnp.pi, jnp.pi))
    M01 = numpyro.sample("M01", dist.Uniform(-jnp.pi, jnp.pi))

    # -------- Planet 2 --------
    logP2 = numpyro.sample("logP2", dist.Uniform(jnp.log(100), jnp.log(20000)))
    P2 = jnp.exp(logP2)
    n2 = 2 * jnp.pi / P2

    e2 = numpyro.sample("e2", dist.Uniform(0.0, 0.3))
    logq2 = numpyro.sample("logq2",dist.Uniform(jnp.log(2e-4), jnp.log(1e-1)))
    q2 = jnp.exp(logq2)
    omega2 = numpyro.sample("omega2", dist.Uniform(-jnp.pi, jnp.pi))
    M02 = numpyro.sample("M02", dist.Uniform(-jnp.pi, jnp.pi))

    # inclination (shared)
    inc = jnp.pi / 2

    # offsets (same as your logic)
    offset1 = numpyro.sample("offset1", dist.Uniform(-2000, 2000))
    offset2 = numpyro.sample("offset2", dist.Uniform(-2000, 2000))

    # jitter (better than fixed 50)
    #log_jitter = numpyro.sample("log_jitter", dist.Uniform(jnp.log(1.0), jnp.log(200)))
    sigma_jit = 0#jnp.exp(log_jitter)
    rv_err = 5*np.ones(len(t_rv))

    # -------- True anomalies --------
    if np.min(t_rv) < 2455500 and np.max(t_rv) >= 2455500:

        els1 = np.where(t_rv < 2455500)[0]
        els2 = np.where(t_rv >= 2455500)[0]

        t1 = t_rv[els1]
        t2 = t_rv[els2]

        f1_p1 = true_anomaly(t1 - JD0 - t_ref, n1, e1, M01)
        f2_p1 = true_anomaly(t2 - JD0 - t_ref, n1, e1, M01)

        f1_p2 = true_anomaly(t1 - JD0 - t_ref, n2, e2, M02)
        f2_p2 = true_anomaly(t2 - JD0 - t_ref, n2, e2, M02)

    else:
        t = t_rv
        f_p1 = true_anomaly(t - JD0 - t_ref, n1, e1, M01)
        f_p2 = true_anomaly(t - JD0 - t_ref, n2, e2, M02)

    # -------- RV signals --------
    if np.min(t_rv) < 2455500 and np.max(t_rv) >= 2455500:

        rv1 = (
            rv_model(f1_p1, P1, e1, q1, omega1, M_star, inc) +
            rv_model(f1_p2, P2, e2, q2, omega2, M_star, inc) +
            offset1
        )

        rv2 = (
            rv_model(f2_p1, P1, e1, q1, omega1, M_star, inc) +
            rv_model(f2_p2, P2, e2, q2, omega2, M_star, inc) +
            offset1
        )

        sigma1 = jnp.sqrt(rv_err[els1]**2 + sigma_jit**2)
        sigma2 = jnp.sqrt(rv_err[els2]**2 + sigma_jit**2)

        numpyro.sample("RV1", dist.Normal(rv1, sigma1), obs=rv_data[els1])
        numpyro.sample("RV2", dist.Normal(rv2, sigma2), obs=rv_data[els2])

        numpyro.deterministic("RV1_sim", rv1)
        numpyro.deterministic("RV2_sim", rv2)

    else:
        rv = (
            rv_model(f_p1, P1, e1, q1, omega1, M_star, inc) +
            rv_model(f_p2, P2, e2, q2, omega2, M_star, inc) +
            offset1
        )

        sigma = jnp.sqrt(rv_err**2 + sigma_jit**2)

        numpyro.sample("RV", dist.Normal(rv, sigma), obs=rv_data)
        numpyro.deterministic("RV_sim", rv)

    # -------- Derived quantities --------
    T01 = t_ref - M01 / n1 + JD0
    T02 = t_ref - M02 / n2 + JD0

    numpyro.deterministic("T01", T01)
    numpyro.deterministic("T02", T02)
            
def model_rv(rv_obs, M_star=1.0):
     # --- Time conversions ---
    JD0 = 2457388.5
    t_rv = rv_obs[:,0]
    rv_data = rv_obs[:,1]
    rv_err = rv_obs[:,2]
    t_ref = 0#jnp.mean(t_rv-JD0)
    sec_per_day = 24.0 * 3600.0
    sec_per_year = 365.25 * sec_per_day
    # Sample logP and derive P
    logP = numpyro.sample("logP", dist.Uniform(jnp.log(100), jnp.log(20000)))  # days
    P = jnp.exp(logP)
    n = 2 * jnp.pi / P  # mean motion

    # Orbital parameters
    e = numpyro.sample("e", dist.Uniform(0.0, 0.9))
    logq = numpyro.sample("logq", dist.Uniform(jnp.log(1e-4), jnp.log(1e-1)))
    q = jnp.exp(logq)
    omega = numpyro.sample("omega", dist.Uniform(-9 * jnp.pi, 9 * jnp.pi))
    inc = jnp.pi/2
    offset1 = numpyro.sample("offset1", dist.Uniform(-2000,2000))
    offset2 = numpyro.sample("offset2", dist.Uniform(-2000,2000))
    # Mean anomaly at global t_ref
    M0 = numpyro.sample("M0", dist.Uniform(-jnp.pi, jnp.pi))
    if np.min(t_rv)<2455500 and np.max(t_rv)>=2455500:
        els1 = np.where(t_rv<2455500)[0]
        els2 = np.where(t_rv>=2455500)[0]
        t_rv1 = t_rv[els1]
        t_rv2 = t_rv[els2]
        f1 = true_anomaly(t_rv1-JD0-t_ref,n,e,M0)
        f2 = true_anomaly(t_rv2-JD0-t_ref,n,e,M0)
    else:
        f = true_anomaly(t_rv-JD0-t_ref,n,e,M0)
    T0 = t_ref - M0 / n
    # Noise scales
    sigma_rv = rv_err#2*np.ones(len(t_rv))
    # Models
    if np.min(t_rv)<2455500 and np.max(t_rv)>=2455500:
        rv1 = rv_model(f1, P, e, q, omega, M_star, inc)+offset1
        rv2 = rv_model(f2, P, e, q, omega, M_star, inc)+offset2
    else:
        rv = rv_model(f, P, e, q, omega, M_star, inc)+offset1

    numpyro.deterministic("T0", T0+JD0)
    if np.min(t_rv)<2455500 and np.max(t_rv)>=2455500:
        numpyro.sample("RV1", dist.Normal(rv1, sigma_rv[els1]), obs=rv_data[els1])
        numpyro.sample("RV2", dist.Normal(rv2, sigma_rv[els2]), obs=rv_data[els2])
        numpyro.deterministic("RV1_sim", rv1)
        numpyro.deterministic("RV2_sim", rv2)
    else:
        numpyro.sample("RV", dist.Normal(rv, sigma_rv), obs=rv_data)
        numpyro.deterministic("RV_sim", rv)

    
# Inference

def run_rv_inference(name, rv_obs, p_init=1000, M_star=1.0, fit_type=''):
    init_values = {"logP": jnp.log(p_init), "logq": jnp.log(2e-3), "M0": 0.01, "e": 0.01, "omega": 0, "offset1":1000,"offset2":1000}
    kernel = NUTS(model_rv,init_strategy=init_to_value(values=init_values))
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=10000)
    mcmc.run(jax.random.PRNGKey(1), rv_obs=rv_obs, M_star=M_star)

    samples = mcmc.get_samples()
    P_median = jnp.median(jnp.exp(samples["logP"]))
    q_median = jnp.median(jnp.exp(samples["logq"]))
    e_median = jnp.median(samples["e"])
    o_median = jnp.median(samples["omega"]%(2*jnp.pi))
    T0_median = jnp.median(samples["T0"])
    print('Results for '+name+':'+fit_type)
    print(f"  Msini = {q_median*1000*M_star:.2f} M_jup")
    print(f"  P     = {P_median:.2f} days")
    print(f"  e     = {e_median:.3f}")
    print(f"  omega     = {o_median*180/jnp.pi:.2f} deg")
    print(f"  T0     = {T0_median:.2f}")
    #pickle.dump(samples,open('samples_'+name+'_rv.pkl','wb'))
    return mcmc

def run_rv_inference_2planet(name, n_planets, rv_obs, p_init=[1000,2000], M_star=1.0, fit_type=''):
    init_values = {}
    for ii in range(n_planets):
        init_values['logP'+str(ii+1)] = jnp.log(p_init[ii])
        init_values['logq'+str(ii+1)] = jnp.log(1e-3)
        init_values['M0'+str(ii+1)] = 0.0
        init_values['e'+str(ii+1)] = 0.01
        init_values['omega'+str(ii+1)] = 0.0
    init_values['offset'] = 0.0
    init_values['offset1'] = 0.0
    init_values['offset2'] = 0.0
    kernel = NUTS(model_rv_vector,init_strategy=init_to_value(values=init_values))
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=10000)
    mcmc.run(jax.random.PRNGKey(1), rv_obs=rv_obs, M_star=M_star, n_planets=n_planets, extra_fields=("potential_energy",))

    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    log_prob = -extra["potential_energy"]
    el = np.argmax(log_prob)
    print('Results for '+name+':'+fit_type)
    for ii in range(n_planets):
        P_best = jnp.exp(samples["logP"+str(ii+1)][el])
        q_best = jnp.exp(samples["logq"+str(ii+1)][el])
        e_best = samples["e"+str(ii+1)][el]
        o_best = samples["omega"+str(ii+1)][el]%(2*jnp.pi)
        #T0_median = jnp.median(samples["T0"][ii])#jnp.median(samples["T01"])
        print(f"  M"+str(ii+1)+f"sini = {q_best*1000*M_star:.2f} M_jup")
        print(f"  P"+str(ii+1)+f"     = {P_best:.2f} days")
        print(f"  e"+str(ii+1)+f"     = {e_best:.3f}")
        print(f"  omega"+str(ii+1)+f"     = {o_best*180/jnp.pi:.2f} deg")
        #print(f"  T0"+str(ii+1)+f"     = {T0_median:.2f}")
    #pickle.dump(samples,open('samples_'+name+'_rv.pkl','wb'))
    return mcmc
#model_astro(rv_params, t_gaia, scan_gaia=None,xAL_obs=None,ruwe_obs=None,param_obs=None,hg_params=None,hg_dr4=None,M_star=1.0,epoch=2016.0,errs=None,al_err=0.2,t_dr4=None,scan_dr4=None):
