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
    offset1 = numpyro.sample("offset1", dist.Normal(-100,100))
    offset2 = numpyro.sample("offset2", dist.Normal(-100,100))
    offset3 = numpyro.sample("offset3", dist.Normal(-100,100))
    times = rv_obs[:, 0]
    offset = jnp.where(
        times < 2452849,
        offset1,
        jnp.where(
            times < 2458000,
            offset2,
            offset3,
        ),
    )
    rv = offset
    f = jnp.zeros((n_planets,len(t_rv)))
    for ii in range(n_planets):
        logP = numpyro.sample("logP"+str(ii+1), dist.Uniform(jnp.log(100), jnp.log(10000)))
        P = P.at[ii].set(jnp.exp(logP))
        e = e.at[ii].set(numpyro.sample("e"+str(ii+1), dist.Beta(0.867, 3.03)))
        logq = numpyro.sample("logq"+str(ii+1),dist.Uniform(jnp.log(2e-4), jnp.log(1e-1)))
        q = q.at[ii].set(jnp.exp(logq))
        omega = omega.at[ii].set(numpyro.sample("omega"+str(ii+1), dist.Uniform(0, 2*jnp.pi)))
        M0 = M0.at[ii].set(numpyro.sample("M0"+str(ii+1), dist.Uniform(0, 2*jnp.pi)))
        n = 2 * jnp.pi / P[ii]
        T0 = T0.at[ii].set(t_ref - (M0[ii]%(2*jnp.pi)) / n + JD0)
        f = f.at[ii].set(true_anomaly(t_rv - JD0 - t_ref, n, e[ii], M0[ii]))
        rv = rv+rv_model(f[ii], P[ii], e[ii], q[ii], omega[ii], M_star, inc)
    jitter1 = numpyro.sample("jitter1", dist.HalfNormal(10.0))
    jitter2 = numpyro.sample("jitter2", dist.HalfNormal(10.0))
    jitter3 = numpyro.sample("jitter3", dist.HalfNormal(10.0))
    times = rv_obs[:, 0]
    jitter = jnp.where(
        times < 2452849,
        jitter1,
        jnp.where(
            times < 2458000,
            jitter2,
            jitter3,
        ),
    )
    rv_err = jnp.sqrt(rv_obs[:, 2]**2 + jitter**2)
    numpyro.sample("RV", dist.Normal(rv, rv_err), obs=rv_data)
    numpyro.deterministic("RV_sim", rv)
    numpyro.deterministic("T0", T0)

def run_rv_inference_2planet(name, n_planets, rv_obs, p_init=[1000,6000], M_star=1.0, fit_type=''):
    init_values = {}
    for ii in range(n_planets):
        init_values['logP'+str(ii+1)] = jnp.log(p_init[ii])
        init_values['logq'+str(ii+1)] = jnp.log(1e-3)
        init_values['M0'+str(ii+1)] = jnp.pi
        init_values['e'+str(ii+1)] = 0.01
        init_values['omega'+str(ii+1)] = 0.0
    init_values['omega1'] = 233*jnp.pi/180
    init_values['omega2'] = 110*jnp.pi/180
    init_values['e1'] = 0.3
    init_values['e2'] = 0.5
    init_values['offset'] = 0.0
    init_values['offset1'] = -20.0
    init_values['offset2'] = -20.0
    init_values['offset3'] = -20.0
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
        M0 = samples["M0"+str(ii+1)][el]
        n = 2*jnp.pi/P_best
        T0 = M0 / n + 2457388.5
        #T0_median = jnp.median(samples["T0"][ii])#jnp.median(samples["T01"])
        print(f"  M"+str(ii+1)+f"sini = {q_best*1000*M_star:.2f} M_jup")
        print(f"  P"+str(ii+1)+f"     = {P_best:.2f} days")
        print(f"  e"+str(ii+1)+f"     = {e_best:.3f}")
        print(f"  omega"+str(ii+1)+f"     = {o_best*180/jnp.pi:.2f} deg")
        print(f"  T0"+str(ii+1)+f"     = {T0:.2f}")
    pickle.dump(samples,open('samples_rv_'+name+'.pkl','wb'))
    return mcmc
