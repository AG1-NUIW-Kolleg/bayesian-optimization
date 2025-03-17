# Simulation of Mass-Spring-Damper System for a Hyperelastic-Solid with uniaxial simple tension with Heun solver
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import timeit

import jax as jx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from diffrax import diffeqsolve
from diffrax import ODETerm
from diffrax import PIDController
from diffrax import SaveAt
from diffrax import Tsit5
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# Enable double precision
jx.config.update('jax_enable_x64', True)
# Uncomment this line to force using the CPU
jx.config.update('jax_platform_name', 'cpu')
start = timeit.default_timer()

# input stretch total stretch
# tendon berechnen
# lobs_m1 = 15.0  # Observed prestretched length of muscle 1 [cm]
# lobs_m2 = 15.0  # Observed prestretched length of muscle 2 [cm]
# lobs_td = 5.0  # Observed prestretched length of tendon [cm]


extmaxobs_muscle_1 = 18.3  # Maximal extension of muscle 1
extminobs_muscle_1 = 8.7  # Minimal extension of muscle 1
extmaxobs_muscle_2 = 21.00  # Maximal extension of muscle 2
extminobs_muscle_2 = 11.5  # Minimal extension of muscle 2

observed_data = [extmaxobs_muscle_1, extminobs_muscle_1,
                 extmaxobs_muscle_2, extminobs_muscle_2]

# Parameters

ode_solver = 1  # 1 = Thelen, 2 = Van Soest, 3 = Silva, 4 = Hyperelastic

# Define range of possible model input parameters

# lslack_muscle_1_input = [9.0, 17.0]  # Stress-free length of muscle 1
# lslack_muscle_2_input = [9.0, 17.0]  # Stress-free length of muscle 2
# lslack_tendon_input = [4.95,7.0] # Stress-free length of tendon

# Simulation parameters
Tstart_input = 0.0
Tend_input = 6.0

# MCMC parameters

NUM_BINS = 2  # Numbers of blocks in Histogramm
NUM_DRAWS = 1000  # Number of HMC draws
burn_in = 500  # Number of iterations for the burn-in of the HMC

# min_sample_boundary = [lslack_muscle_1_input[0], lslack_muscle_2_input[0]]
# max_sample_boundary = [lslack_muscle_1_input[1], lslack_muscle_2_input[1]]

# Allow number of different tries to choose a valid start sample for the HMC algorithm
maximum_start_counter = 5

# Define a priori parameters of distributions
# Standard deviation of proposal distribution
STD_DEVIATION_PROP_DIST = [0.1, 0.1]
# Standard deviation of prior distribution
STD_DEVIATION_PRIOR_DIST = [6.0, 6.0]
EXP_VALUE_PRIOR_DIST = [13.5, 13.5]  # Expected value of prior distribution
STD_DEVIATION_DATA = [1.0, 1.0, 1.0, 1.0]  # Standard deviation of data

length_slack_m1 = 12
length_slack_m2 = 13
# Model parameters
model_parameters = {'Length_Slack_M1': length_slack_m1,
                    'Length_Slack_M2': length_slack_m2,
                    'Total_Length': 35,
                    'Obs_Ext_Max_M1': extmaxobs_muscle_1,
                    'Obs_Ext_Min_M1': extminobs_muscle_1,
                    'Obs_Ext_Max_M2': extmaxobs_muscle_2,
                    'Obs_Ext_Min_M2': extminobs_muscle_2,
                    'Std_Data': STD_DEVIATION_DATA,
                    'Exp_Prior': EXP_VALUE_PRIOR_DIST,
                    'Std_Prior': STD_DEVIATION_PRIOR_DIST,
                    'Mass_M1': 1.0,
                    'Mass_M2': 1.0,
                    'Force_M1': 0.0,
                    'Force_M2': 0.0,
                    'Force_Ref_M0': 6820.0,
                    'Force_Ref_Max': 1.6}

# Model Parameters

solver = 2  # 1 = explicit Euler, 2 = Heun

ode_solver = 1  # 1 = Thelen, 2 = Van Soest, 3 = Silva, 4 = Hyperelastic

model_num = 1  # 1 = Mooney, 2 = Yeoh, 3 = Gent

# Model Parameters Mooney

# Opendihu

# c1 = 3.176e-10  # [N/cm^2] Material constant Monney-Rivilin
# c2 = 1.813e+0  # [N/cm^2] Material constant Monney-Rivilin

# Wikipedia (Factor "..e-.." is added extra to scale the models properly)

c1 = 103.0*0.2e-2  # [N/cm^2] Material constant Monney-Rivilin
c2 = 11.4*0.2e-2  # [N/cm^2] Material constant Monney-Rivilin

# Model Parameters Yeoh

# Wikipedia

c3 = 120.2*0.2e-2  # [N/cm^2] Material constant Yeoh
c4 = -5.7*0.2e-2  # [N/cm^2] Material constant Yeoh
c5 = 0.4*0.2e-2  # [N/cm^2] Material constant Yeoh

# Model Parameters Gent

# Wikipedia

c6 = 229.0*0.2e-2  # [N/cm^2] Material constant Gent
Jm = 30.0  # Material constant Gent

d = 0.0  # Damping constant

fr = 0.0  # Friction constant

# k = 2.0 # Stiffness of the spring

# l0_muscle1 = 14.5 # Stress-free length of muscle 1
# l0_muscle2 = 14.5 # Stress-free length of muscle 2
# l0_tendon = 5.0 # Stress-free length of tendon
mass_1 = 1.0  # Mass of mass point 1
mass_2 = 1.0  # Mass of mass point 2
F_m1 = 0.0  # Force pulling on mass point 1 (external)
F_m2 = 0.0  # Force pulling on mass point 2 (external)
A = 1.0  # Surface of rubber on which force F is pulling

# Observed Parameters

lobs_muscle1 = 15.0  # Observed prestretched length of muscle 1 [cm]
lobs_muscle2 = 15.0  # Observed prestretched length of muscle 2 [cm]
lobs_tendon = 5.0  # Observed prestretched length of tendon [cm]
extobs = 1.5  # Maximal extension of rubber band with pulling force F

# Simulation parameters

deltaT = 0.001
Tstart = 0.0
Tend = 6.0

# Hill-Type Parameters
Fm0 = 6820.0  # Maximum isometric force [N] #[gf] #[gf/cm^2]
tauc = 0.1  # Time constant [s]
lslack_muscle1 = 12.5  # Stress-free length of muscle 1
lslack_muscle2 = 14.5  # Stress-free length of muscle 2
lslack_tendon = 5.0  # Stress-free length of tendon
kpe = 3.0  # Shape factor passiv model Thelen
strain_com = 1  # Strain of parallel element Thelen constant == 1, calculated == 0
gamma = 0.45  # Half-width of curve active force-length model Thelen
Ar = 5.2  # 0.41 # Hill's constant corrections van Soest
Br = 0.41  # 5.2 # Hill's constant corrections van Soest
fac = 1.0  # Muscle activation factor van Soest normally min(1,3.33a)
fmax = 1.6  # Maximum normailzed achievable muscle force
Umax = 0.04  # The relative elongation of the SEE at maximal isometric force
Sf = 2.0  # Slope factor van Soest
width = 0.56  # Max range of force production relative to stress-free length Van Soest
kce1 = 0.25  # Shape factor active force-velocity Thelen
kce2 = 0.06  # Shape factor active force-velocity Thelen
t_start_m1 = 0.0  # Start activation of muscle 1
t_act_m1 = 1.0  # Time of activation of muscle 1
t_rest_m1 = 10.0  # Rest time of muscle 1
t_start_m2 = 2.0  # Start activation of muscle 2
t_act_m2 = 1.0  # Time of activation of muscle 2
t_rest_m2 = 10.0  # Rest time of muscle 2
tau_rise = 20.0*1e-3  # Delay of activation start
tau_fall = 200.0*1e-3  # Delay of deactivation start
alpha_min = 0.0  # Minimal activation (for numeric stability)


# Calculating Displacement and Prestretch at time 0

# Displacement of muscle 1 at time 0
dis0_muscle1 = lobs_muscle1 - lslack_muscle1
# Displacement of muscle 2 at time 0
dis0_muscle2 = lobs_muscle2 - lslack_muscle2
dis0_tendon = lobs_tendon - lslack_tendon  # Displacement of tendon at time 0
# incorporating prestretch as optimisation parameter
prestretch_muscle1 = lobs_muscle1/lslack_muscle1  # Prestretch of muscle 1
prestretch_muscle2 = lobs_muscle2/lslack_muscle2  # Prestretch of muscle 2
prestretch_tendon = lobs_tendon/lslack_tendon  # Prestretch of tendon

# Tendon model Thelen
# Passive tendon force (eq. 16)


def Tendon_Thelen_NF(displacement, lslack, Forcem0):
    strain = displacement/lslack
    stress = jnp.where(jnp.logical_and(strain >= 0.0, strain <= 0.01516), Forcem0 * 0.10377 * (
        jnp.exp(91*strain) - 1), jnp.where(strain >= 0.01516, Forcem0 * (37.526*strain - 0.26029), 0.0))

    return stress

# Passive muscle model Thelen
# Passive force (eq. 15)


def Passive_Muscle_Thelen_NF(displacement, lslack, strain_comp, kpe, Forcem0):
    strain = jnp.where(strain_comp == 0, displacement/lslack, 0.5)

    stress = Forcem0 * \
        (jnp.exp(kpe*(((displacement+lslack)/lslack)-1)/strain)-1.0)/(jnp.exp(kpe)-1.0)
    return stress

# Active length muscle model Thelen
# force-length relationship f_l (siehe eq. 13)


def Active_Muscle_Length_Thelen_NF(displacement, lslack, gamma):
    stretch = (displacement+lslack)/lslack
    stress = jnp.exp(-jnp.power(stretch-1.0, 2.0)/gamma)

    return stress

# Active velocity muscle model Thelen
# force-velocity relationship f_v (siehe eq. 13)


def Active_Muscle_Velocity_Thelen_NF(velocity, lslack, fmax, kce1, kce2, tau_ar):
    velstretch = (velocity/(lslack/tau_ar))
    stress = jnp.where(velstretch > 0.0, (1.0+velstretch*(fmax/kce2))/(1.0+(velstretch/kce2)), jnp.where(
        jnp.logical_and(velstretch > (-1.0), velstretch <= 0.0), (1.0+velstretch)/(1.0-(velstretch/kce1)), 0.0))

    return stress

# warum zwei separate funktionen? -> im prinzip die gleiche funktionalität
# Neural input of muscle 1


def activation_u_NF(t, t_start_m1_arg, t_act_m1_arg, t_rest_m1_arg, n1_arg):
    activation = 0.0
    activation = jnp.where(t < t_start_m1_arg, 0.0, activation)

    def body_fun(i, val):
        t, activation, t_start_m1_arg, t_act_m1_arg, t_rest_m1_arg, n1_arg = val
        activation = jnp.where(jnp.logical_and(t >= (i*(t_act_m1_arg+t_rest_m1_arg)+t_start_m1_arg), t <= (i*(t_act_m1_arg+t_rest_m1_arg)+t_act_m1_arg+t_start_m1_arg)), 1.0, jnp.where(
            jnp.logical_and(t <= ((i+1)*(t_act_m1_arg+t_rest_m1_arg)+t_start_m1_arg), t >= (i*(t_act_m1_arg+t_rest_m1_arg)+t_act_m1_arg+t_start_m1_arg)), 0.0, activation))
        return [t, activation, t_start_m1_arg, t_act_m1_arg, t_rest_m1_arg, n1_arg]

    val = [t, activation, t_start_m1_arg, t_act_m1_arg, t_rest_m1_arg, n1_arg]
    for i in range(0, n1_arg+1):
        val = body_fun(i, val)
    t, activation, t_start_m1_arg, t_act_m1_arg, t_rest_m1_arg, n1_arg = val
    return activation

# Neural input muscle 2


def activation_v_NF(t, t_start_m2_arg, t_act_m2_arg, t_rest_m2_arg, n2_arg):
    activation = 0.0
    activation = jnp.where(t < t_start_m2_arg, 0.0, activation)

    def body_fun(i, val):
        t, activation, t_start_m2_arg, t_act_m2_arg, t_rest_m2_arg, n2_arg = val
        activation = jnp.where(jnp.logical_and(t >= (i*(t_act_m2_arg+t_rest_m2_arg)+t_start_m2_arg), t <= (i*(t_act_m2_arg+t_rest_m2_arg)+t_act_m2_arg+t_start_m2_arg)), 1.0, jnp.where(
            jnp.logical_and(t <= ((i+1)*(t_act_m2_arg+t_rest_m2_arg)+t_start_m2_arg), t >= (i*(t_act_m2_arg+t_rest_m2_arg)+t_act_m2_arg+t_start_m2_arg)), 0.0, activation))
        return [t, activation, t_start_m2_arg, t_act_m2_arg, t_rest_m2_arg, n2_arg]

    val = [t, activation, t_start_m2_arg, t_act_m2_arg, t_rest_m2_arg, n2_arg]
    for i in range(0, n2_arg + 1):
        val = body_fun(i, val)
    t, activation, t_start_m2_arg, t_act_m2_arg, t_rest_m2_arg, n2_arg = val
    return activation

# Two-Muscle-One-Tendon Hill-type model (by Thelen) ODE


def RHS_Thelen_ODE(t, y, args):
    Forcem0, mass_m1, mass_m2, Force_at_m1, Force_at_m2, dis0_m1, dis0_m2, dis0_td, length_slack_m1, length_slack_m2, length_slack_td, tau_rise_arg, tau_fall_arg, alpha_min_arg, t_start_m1_arg, t_act_m1_arg, t_rest_m1_arg, n1_arg, t_start_m2_arg, t_act_m2_arg, t_rest_m2_arg, n2_arg, Force_max, kce1_arg, kce2_arg, tau_ar, gamma_arg, strain_arg, kpe_arg = args
    x1, x2, x3, x4, x5, x6 = y
    derx1 = x2
    derx2 = (1/mass_m1) * (Force_at_m1 + Tendon_Thelen_NF(dis0_td+(x3-x1), length_slack_td, Forcem0) - Passive_Muscle_Thelen_NF(dis0_m1+x1, length_slack_m1, strain_arg, kpe_arg, Forcem0) -
                           Forcem0 * x5 * Active_Muscle_Length_Thelen_NF(dis0_m1+x1, length_slack_m1, gamma_arg) * Active_Muscle_Velocity_Thelen_NF(x2, length_slack_m1, Force_max, kce1_arg, kce2_arg, tau_ar))
    derx3 = x4
    derx4 = (1/mass_m2) * (Force_at_m2 - Tendon_Thelen_NF(dis0_td+(x3-x1), length_slack_td, Forcem0) + Passive_Muscle_Thelen_NF(dis0_m2-x3, length_slack_m2, strain_arg, kpe_arg, Forcem0) +
                           Forcem0 * x6 * Active_Muscle_Length_Thelen_NF(dis0_m2-x3, length_slack_m2, gamma_arg) * Active_Muscle_Velocity_Thelen_NF(-x4, length_slack_m2, Force_max, kce1_arg, kce2_arg, tau_ar))
    derx5 = ((1/tau_rise_arg) * (1 - x5) * activation_u_NF(t, t_start_m1_arg, t_act_m1_arg, t_rest_m1_arg, n1_arg)) + \
        ((1/tau_fall_arg) * (alpha_min_arg - x5) * (1 - activation_u_NF(t,
         t_start_m1_arg, t_act_m1_arg, t_rest_m1_arg, n1_arg)))
    derx6 = ((1/tau_rise_arg) * (1 - x6) * activation_v_NF(t, t_start_m2_arg, t_act_m2_arg, t_rest_m2_arg, n2_arg)) + \
        ((1/tau_fall_arg) * (alpha_min_arg - x6) * (1 - activation_v_NF(t,
         t_start_m2_arg, t_act_m2_arg, t_rest_m2_arg, n2_arg)))
    d_y = derx1, derx2, derx3, derx4, derx5, derx6
    return d_y


def Hill_System_ODE_Solve(input_parameters, params):

    # Read input parameters
    total_length = params['Total_Length']
    obs_length_m1, obs_length_m2 = input_parameters
    obs_length_td = total_length - obs_length_m1-obs_length_m2
    # Model Parameters

    length_slack_m1 = params['Length_Slack_M1']
    length_slack_m2 = params['Length_Slack_M2']
    mass_m1 = params['Mass_M1']
    mass_m2 = params['Mass_M2']
    Force_at_m1 = params['Force_M1']
    Force_at_m2 = params['Force_M2']
    Force_m0 = params['Force_Ref_M0']
    Force_max = params['Force_Ref_Max']
    length_slack_td = obs_length_td

    # Additional Parameters

    strain_arg = 1.0
    kpe_arg = 3.0
    gamma_arg = 0.45
    kce1_arg = 0.25
    kce2_arg = 0.06
    tauc_arg = 0.1
    tau_rise_arg = 20.0*1e-3
    tau_fall_arg = 200.0*1e-3
    alpha_min_arg = 0.0
    Ts_arg = 0.0
    Te_arg = 6.0
    t_act_m1_arg = 1.0
    t_rest_m1_arg = 10.0
    t_start_m1_arg = 0.0
    t_act_m2_arg = 1.0
    t_rest_m2_arg = 10.0
    t_start_m2_arg = 2.0

    # Prestretch and Displacement at time zero

    dis0_m1 = obs_length_m1 - length_slack_m1  # Displacement of muscle 1 at time 0
    dis0_m2 = obs_length_m2 - length_slack_m2  # Displacement of muscle 2 at time 0
    dis0_td = obs_length_td - length_slack_td  # Displacement of tendon at time 0

    # Calculating number of activation intervalls for neural input function u for muscle 1
    m1_arg = (Te_arg-Ts_arg)/(t_act_m1_arg+t_rest_m1_arg)
    n1_arg = int(math.ceil(m1_arg))

    # Calculating number of activation intervalls for neural input function v for muscle 2
    m2_arg = (Te_arg-Ts_arg)/(t_act_m2_arg+t_rest_m2_arg)
    n2_arg = int(math.ceil(m2_arg))

    # Solve Hill-type model ODE with jax module diffrax

    timesteps = 2**10
    term = ODETerm(RHS_Thelen_ODE)
    solver = Tsit5()
    t0 = Ts_arg
    t1 = Te_arg
    dt0 = 0.0015
    y0 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    args = (Force_m0, mass_m1, mass_m2, Force_at_m1, Force_at_m2, dis0_m1, dis0_m2, dis0_td, length_slack_m1, length_slack_m2, length_slack_td, tau_rise_arg, tau_fall_arg, alpha_min_arg,
            t_start_m1_arg, t_act_m1_arg, t_rest_m1_arg, n1_arg, t_start_m2_arg, t_act_m2_arg, t_rest_m2_arg, n2_arg, Force_max, kce1_arg, kce2_arg, tauc_arg, gamma_arg, strain_arg, kpe_arg)
    saveat = SaveAt(ts=jnp.linspace(t0, t1, timesteps))
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    sol = diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat,
                      stepsize_controller=stepsize_controller, throw=False)  # ,max_steps = 7000)

    # Read results of ODE

    t_ODE = sol.ts
    dis1_ODE = sol.ys[0]
    vel1_ODE = sol.ys[1]
    dis2_ODE = sol.ys[2]
    vel2_ODE = sol.ys[3]
    act_m1_ODE = sol.ys[4]
    act_m2_ODE = sol.ys[5]

    # Calculate length of Muscles and Tendon at each time step

    total_len_muscle1 = length_slack_m1 + dis0_m1 + dis1_ODE
    total_len_muscle2 = length_slack_m2 + dis0_m2 - dis2_ODE
    total_len_tendon = length_slack_td + dis0_td + (dis2_ODE - dis1_ODE)

    # Calculate neural input to muscels (critical because if the for loop)

    # total_neural_input_muscle_1 = [activation_u_NF(i) for i in t_ODE]
    # total_neural_input_muscle_2 = [activation_v_NF(i) for i in t_ODE]

    output_parameters = jnp.asarray([total_len_muscle1, total_len_muscle2])

    return output_parameters


# Compute Hill-type simulation output from input values
def black_box_simulation(input_value, params):
    input_value = jnp.array(input_value)
    calculated_data_m1_raw_jax, calculated_data_m2_raw_jax = Hill_System_ODE_Solve(
        input_value, params)

    output_data_1 = jnp.asarray(calculated_data_m1_raw_jax, dtype='float')
    output_data_2 = jnp.asarray(calculated_data_m2_raw_jax, dtype='float')

    output_data = [output_data_1, output_data_2]
    return output_data


# Compute forward simulation output from input values
def observe_blackbox_simulation(sample_input, params=None):
    if params is None:
        params = model_parameters
    calculated_data_m1_raw, calculated_data_m2_raw = black_box_simulation(
        sample_input, params)

    value_maximal_length_m1 = jnp.amax(calculated_data_m1_raw)
    value_minimal_length_m1 = jnp.amin(calculated_data_m1_raw)
    value_maximal_length_m2 = jnp.amax(calculated_data_m2_raw)
    value_minimal_length_m2 = jnp.amin(calculated_data_m2_raw)

    calculated_data_blackbox = [value_maximal_length_m1, value_minimal_length_m1, value_maximal_length_m2,
                                value_minimal_length_m2]

    return calculated_data_blackbox


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 2)
    outputs_m1, outputs_m2 = Hill_System_ODE_Solve(
        [15.5, 15], params=model_parameters)

    outputs_tendon = np.full(
        len(outputs_m2), (model_parameters['Total_Length']))-outputs_m1-outputs_m2
    ax[0].plot(range(len(outputs_m1)), outputs_m1, color='red')
    ax[0].plot(range(len(outputs_m2)), outputs_m2, color='blue')
    ax[1].plot(range(len(outputs_tendon)), outputs_tendon)
    plt.tight_layout()
    plt.show()
