############################################################################################################################################################################
import numpy as np
############################################################################################################################################################################

# Define the constants and parameters for our calculation.

# Universal constants.
H_BAR = 1.0545718e-34 # J*s
m_e = 9.10938356e-31 # kg
m_eff = 0.6 * m_e
i = complex(0, 1)
PI =np.pi
Q = 1.6021773e-19 # C
# experimental parameters.

v = 20.8 # meV
psi = 107.7 # degrees
psi_ = psi/180.0 * PI
w = -23.8 # meV
twist_angle = 3.89 # degrees
twist_angle_ = twist_angle/180.0 * PI

a_prim = 3.52e-10 # m
a_moire = a_prim / twist_angle_

# Useful points or vectors in the k-space.
coef_G = 4 * PI / (np.sqrt(3) * a_moire)
G_vec = {i: coef_G * np.array([np.cos(PI/3 * (i - 1)), np.sin(PI/3 * (i - 1))]) for i in range(1, 7)}
G_vec_unit = {i: np.array([np.cos(PI/3 * (i - 1)), np.sin(PI/3 * (i - 1))]) for i in range(1, 7)}

coef_K = 4 * PI / (3 * a_moire)
K_b = coef_K * np.array([np.sqrt(3)/2, 1/2])
K_t = coef_K * np.array([np.sqrt(3)/2, -1/2])
K = K_b
Gamma = np.array([0,0]) 
M = coef_K * np.array([np.sqrt(3)/2, 0])

K_b_unit = np.array([1/2, np.sqrt(3)/6])
K_t_unit = np.array([1/2, -np.sqrt(3)/6])
K_unit = K_b_unit
Gamma_unit = np.array([0,0]) 
M_unit = np.array([1/2, 0])

# Some useful variables for our calculations.
Kinetic_Coef = - H_BAR**2 * (1e3) / (2 * m_eff * Q) # meV! Not J!
Exp_psi_plus = v * np.exp(i * psi_)
Exp_psi_minus = v * np.exp(- i * psi_)