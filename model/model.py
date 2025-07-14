#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:38:19 2022

@author: Giulio Colombini
"""

import numpy as np
from   scipy import stats 
from   tqdm  import tqdm

# Global parameters to keep track of run parameters

_NORM_ = False
_DT_   = 1./24

# Social activity rate
# The example is one year of simulation with a quite strong lockdown 30 days after the
# arrival of patient zero. The lockdown is later lifted partially, and then reinstated
# with modulations over time, with stronger measures after the arrival of a variant 
# at day 290. The variant has relative transmissivity $\tau = 1.56$.
# At day 310, a vaccination campaign begins, with 100 doses administered each day.
s_test = (np.array([    0,   30,   40,   80,  150,  160,  170,  190,  300, np.inf]), 
          np.array([ 1.00, 0.25, 0.15, 0.25, 0.50, 0.20, 0.15, 0.30, 0.25,   0.75]))

# Variant arrival schedule 
var_test = (np.array([  0,  290, np.inf]), 
            np.array([1.0, 1.56,  1.56]))

# Vaccination schedule in individuals/day. 
vacc_test = (np.array([  0,  310, np.inf]), 
             np.array([ 0., 150.,   100.]))

# Parameters of \rho_E over time
pars_e_test = (np.array([0., np.inf]), np.array([2.,2.]), np.array([.1,.1]))

# Parameters of \rho_I over time
pars_i_test = (np.array([0., np.inf]), np.array([21, 21]), np.array([2.3,2.3]))

# Parameters of \rho_U over time
pars_u_test = (np.array([0., np.inf]), np.array([5.5,5.5]), np.array([2.3,2.3]))

# Parameters of \rho_R over time
pars_r_test = (np.array([0., np.inf]), np.array([180, 180]), np.array([10,10]))

def discrete_gamma(mean, std_dev, min_t = None, max_t = None):
    if (min_t == None) and (max_t == None):
        min_t = np.rint(mean) - np.rint(3 * std_dev)
        max_t = np.rint(mean) + np.rint(5 * std_dev)
    
    RV = stats.gamma(a = (mean/std_dev)**2, scale = (std_dev**2/mean))
    low_i, high_i = np.int32(np.round([min_t, max_t]))
    low_i = max([1, low_i])
    
    c = np.zeros(high_i, dtype = np.double)
    
    for j in range(low_i, high_i):
        c[j] = RV.cdf((j + 1/2)) - RV.cdf((j-1/2))
    c /= c.sum()
    return (c, low_i, high_i)

def dirac_delta(tau0, min_t = None, max_t = None):
    if (min_t == None) and (max_t == None):
        min_t = 0
        max_t = tau0
        c = np.zeros(shape = int(max_t+1))
        c[-1] += 1.
    return (c, int(min_t), int(max_t+1))

# Forward buffered convolution
def propagate_forward(t, max_t, donor, acceptors, kernel_tuple, branching_ratios = np.array([1.])):
    kernel, i0, lker = kernel_tuple
    if t + i0 > max_t:
        return
    if t + lker - 1 > max_t:
        k = kernel[i0 : max_t - t + 1]
        lk = len(k)
    else:
        k = kernel[i0:]
        lk = len(k)
    buffer = np.empty(shape = (lk,) + donor.shape)
    for i in range(lk):
        buffer[i] = donor * k[i]
    for a, r in zip(acceptors, branching_ratios):
        a[t + i0 : t + i0 + lk] += r * buffer

def run_simulation(days = 60, dt = _DT_, beta = 1/1.2, alpha = .14, 
                   N = 886891, norm = _NORM_, s = s_test,
                   pars_e = pars_e_test, pars_i = pars_i_test, 
                   pars_u = pars_u_test, pars_r = pars_r_test, 
                   variants = var_test, vaccines = vacc_test,
                   return_new_positives = True, return_vaccinated = True):
    '''
    Launch a simulation of the epidemic using the specified parameters.

    Parameters
    ----------
    days : float, optional
        Number of days to simulate. The default is 60.
    dt : float, optional
        Timestep, expressed as a fraction of day. The default is 1./24..
    beta : float, optional
        Infection probability. The default is 1/1.2.
    alpha : float, optional
        Probability of manifesting symptoms. The default is .14.
    N : float or int, optional
        Total population in the model. The default is 886891.
    norm : bool, optional
        Normalise populations if True, otherwise keep numbers unnormalised. The default is False.
    s : tuple of np.arrays, optional
        Days in which sociability is changed and the sociability value to consider 
        up to the next change. The last value in the days array must be np.inf, 
        with a repeated final value in the second one. The default is s_test.
    pars_e : tuple of np.arrays, optional
        Same rules as s but with format (days, means, standard_deviations) 
        for the Exposed exit distribution. If dirac_delta is used in the simulation,
        provide a placeholder value for the standard deviation, 
        which will be ignored. The default is pars_e_test.
    pars_i : tuple of np.arrays, optional
        Same as pars_e but for the Isolated Infected category. The default is pars_i_test.
    pars_u : tuple of np.arrays, optional
        Same as pars_e but for the Unreported category. The default is pars_u_test.
    pars_r : tuple of np.arrays, optional
        Same as pars_e but for the Recovered category. The default is pars_r_test.
    variants : tuple of np.arrays, optional
        Days in which variants are considered to arrive in the region of interest,
        with the associated transmissivity multiplier \tau with respect to the
        wild type. The last value in the days array must be np.inf, 
        with a repeated final value in the second one. The default is var_test.
    vaccines : tuple of np.arrays, optional
        Days in which the number of administed vaccines changes,
        with the associated number of daily doses. The last value 
        in the days array must be np.inf, with a repeated final value 
        in the second one. The default is vacc_test.
    return_new_positives : bool, optional
        If True, return series of new positives, i.e. Phi_EU.
        This quantity corresponds to the daily number of reported cases.
        The default is True.
    return_vaccinated : bool, optional
        If True, return series of daily vaccinations. The default is True.

    Returns
    -------
    t : np.array
        Simulation timestamps.
    S : np.array
        Time series for the Susceptibles compartment.
    E : np.array
        Time series for the Exposed compartment.
    I : np.array
        Time series for the Infected compartment.
    U : np.array
        Time series for the Unreported compartment.
    R : np.array
        Time series for the Removed compartment.
    new_positives : np.array (if return_new_positives is True)
        Time series of the new reported positive cases.
    R : np.array (if return_vaccinated is True)
        Time series of daily vaccinations.
    TOT : np.array
        Time series for the sum of all compartments, to check consistency.

    '''
    global _DT_, _NORM_
    _DT_ = dt
    _NORM_ = norm
    # Calculate number of iterations
    max_step = int(np.rint(days / dt))
    
    # Initialise compartments and flows memory locations
    S = np.zeros(max_step+1)
    E = np.zeros(max_step+1)
    I = np.zeros(max_step+1)
    U = np.zeros(max_step+1)
    R = np.zeros(max_step+1)
    TOT = np.zeros(max_step+1)
    
    Phi_SE = np.zeros(max_step+1)
    Phi_EU = np.zeros(max_step+1)
    Phi_UI = np.zeros(max_step+1)
    Phi_IR = np.zeros(max_step+1)
    Phi_UR = np.zeros(max_step+1)
    Phi_RS = np.zeros(max_step+1)
    Phi_V  = np.zeros(max_step+1) # Vaccination flow
    
    # Unpack parameter tuples and rescale them with dt.
    
    s_t    = s[0] / dt
    s_vals = s[1]
    
    s_array= np.array([s_vals[np.searchsorted(s_t, t, side = 'right') - 1] 
                       for t in range(max_step+1)])

    tau_t    = variants[0] / dt
    tau_vals = variants[1]
    
    tau_array= np.array([tau_vals[np.searchsorted(tau_t, t, side = 'right') - 1] 
                         for t in range(max_step+1)])
    v_t    = vaccines[0] / dt
    v_vals = vaccines[1]
    
    v_array= np.array([v_vals[np.searchsorted(v_t, t, side = 'right') - 1] 
                       for t in range(max_step+1)])
    
    # Unpack distribution tuples and generate distributions
    
    rho_e_t      = pars_e[0] / dt
    rho_e_mus    = pars_e[1] / dt
    rho_e_sigmas = pars_e[2] / dt
    
    rho_es = [discrete_gamma(mu, sigma) for mu, sigma in zip(rho_e_mus, rho_e_sigmas)]

    rho_i_t      = pars_i[0] / dt
    rho_i_mus    = pars_i[1] / dt
    rho_i_sigmas = pars_i[2] / dt

    rho_is = [discrete_gamma(mu, sigma) for mu, sigma in zip(rho_i_mus, rho_i_sigmas)]

    rho_u_t      = pars_u[0] / dt
    rho_u_mus    = pars_u[1] / dt
    rho_u_sigmas = pars_u[2] / dt

    rho_us = [discrete_gamma(mu, sigma) for mu, sigma in zip(rho_u_mus, rho_u_sigmas)]
    
    rho_r_t      = pars_r[0] / dt
    rho_r_mus    = pars_r[1] / dt
    rho_r_sigmas = pars_r[2] / dt

    rho_rs = [dirac_delta(mu) for mu, sigma in zip(rho_r_mus, rho_r_sigmas)]

    # Add initial population to Susceptibles
    if norm:
        S[0]   = 1
        TOT[0] = 1
    else:
        S[0]   = N
        TOT[0] = N
    
    # Add patient zero to flow
    if norm:
        Phi_SE[0] += 1./N
    else:
        Phi_SE[0] += 1.
    
    # Intialize indices for distribution selection
    cur_rho_e_idx = 0
    cur_rho_i_idx = 0
    cur_rho_u_idx = 0
    cur_rho_r_idx = 0
    
    # Main simulation loop
    
    for t in tqdm(range(max_step)):
        
        # Update distribution indices
        cur_rho_e_idx = np.searchsorted(rho_e_t, t, side = 'right') - 1
        cur_rho_i_idx = np.searchsorted(rho_i_t, t, side = 'right') - 1
        cur_rho_u_idx = np.searchsorted(rho_u_t, t, side = 'right') - 1
        cur_rho_r_idx = np.searchsorted(rho_r_t, t, side = 'right') - 1

        # Get current parameters
        cur_s     = s_array[t]
        cur_tau   = tau_array[t]
        cur_v     = v_array[t]

        # Evaluate active population
        P = S[t] + E[t] + U[t] + R[t]
        
        # Evolve contagion flow
        Phi_SE[t] += beta * cur_tau * cur_s * S[t] * (U[t]) * dt / P
        Phi_V[t]  += min(S[t]-Phi_SE[t], cur_v) 
        
        # Propagate flows
        propagate_forward(t, max_step, Phi_SE[t],
                          [Phi_EU], rho_es[cur_rho_e_idx],
                          branching_ratios = np.array([1.]))
        propagate_forward(t, max_step, Phi_EU[t],
                          [Phi_UI, Phi_UR], rho_us[cur_rho_u_idx],
                          branching_ratios = np.array([alpha, 1. - alpha]))
        propagate_forward(t, max_step, Phi_UI[t], 
                          [Phi_IR], rho_is[cur_rho_i_idx],
                          branching_ratios = np.array([1.]))
        propagate_forward(t, max_step, Phi_IR[t]+Phi_UR[t]+Phi_V[t],
                          [Phi_RS], rho_rs[cur_rho_r_idx],
                          branching_ratios = np.array([1.]))
        
        # Evolve compartments
       
        S[t+1] = S[t] - Phi_SE[t] + Phi_RS[t] - Phi_V[t]
        E[t+1] = E[t] + Phi_SE[t] - Phi_EU[t]
        U[t+1] = U[t] + Phi_EU[t] - Phi_UI[t] - Phi_UR[t]
        I[t+1] = I[t] + Phi_UI[t] - Phi_IR[t]
        R[t+1] = R[t] + Phi_IR[t] + Phi_UR[t] + Phi_V[t] - Phi_RS[t]
        TOT[t+1] = S[t+1] + E[t+1] + I[t+1] + U[t+1] + R[t+1]

    t = np.array([t for t in range(max_step+1)])
    if return_new_positives:
        if return_vaccinated:
            return (t, S, E, I, U, R, Phi_EU, Phi_V, TOT)
        else:
            return (t, S, E, I, U, R, Phi_EU, TOT)
    else:
        if return_vaccinated:
            return(t, S, E, I, U, R, Phi_V, TOT)
        else:
            return (t, S, E, I, U, R, TOT)

def test_model(days = 365, dt = 1/24., norm = False):
    print("Simulate", days, "days with a {:.2f}".format(dt), "day resolution.")
    print("The example is one year of simulation with a quite strong lockdown 30 days\nafter the arrival of patient zero. The lockdown is later lifted partially,\nand then reinstated with modulations over time, with stronger measures after the arrival of a variant\nat day 290. The variant has relative transmissivity \\tau = 1.56. At day 310,\na vaccination campaign begins, with 100 doses administered to susceptible\nindividuals each day.")

    t,s,e,i,u,r,newp,vaccinated,tot = run_simulation(days = days, dt = dt, 
                                     s = s_test,
                                     norm = norm,
                                     return_vaccinated    = True,
                                     return_new_positives = True)
#%% Graphics
    from matplotlib import pyplot as plt
    
    plt.rcParams["figure.autolayout"] = True
     
    fig, ax = plt.subplots(2,1, figsize = (12,8), sharex = True)
    ax[0].plot(t * dt,i, label = 'I')
    ax[0].plot(t * dt,u, label = 'U')
    ax[0].plot(t * dt,newp, label = 'New positive cases')    
    ax[0].plot(t * dt,vaccinated, label = 'Daily vaccinations') 

    if norm:
        ax[0].set_ylim(bottom = 0, top = 0.0025)
        ax[0].set_ylabel('Population Fraction', fontsize = 14)
    else:
        ax[0].set_ylim(bottom = 0, top = 2000)
        ax[0].set_ylabel('Individuals', fontsize = 14)
    ax[0].set_xlim([0, max(t * dt)])
    ax[0].vlines(var_test[0], *ax[0].get_ylim(), linestyle = 'dashed',
                 color = 'red', label = 'Variant arrival')
    ax[0].vlines(vacc_test[0], *ax[0].get_ylim(), linestyle = 'dashed',
                 color = 'blue', label = 'Beginning of vaccination')

    ax[1].set_xlabel('Days since the beginning of the epidemic', fontsize = 14)
    ax[1].set_ylabel('Adimensional units', fontsize = 14)
    ax[1].scatter(s_test[0], s_test[1], s = 100, marker = '+', color = 'red',
                  label = 'Sociability parameter')
    ax[1].scatter(var_test[0], var_test[1], s = 100, marker = 'x', color = 'green',
                  label = 'Transmissivity parameter')
    ax[1].set_ylim([0,2])

    ax[0].legend(fontsize = 14)
    ax[1].legend(fontsize = 14, loc = "center right")
    plt.show()

if __name__ == "__main__":
    test_model()
