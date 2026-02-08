import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import os 
import sys
import time
sys.path.append(os.path.expanduser(f'~/harmonic3dhfb/src'))
import harmonic3dhfb

# Independent Parameters.
Z = 8 # Proton number.
N = 12 # Neutron number.
## Note 
nshellsx = 12 # Number of shells to consider in x.
nshellsy = 12 # Number of shells to consider in y.
nshellsz = 12 # Number of shells to consider in z.

ecut = 100.0 # Cutoff energy.

# Iterations.
niter = 5

# Tolerance.
tol = 1e-6
kappa_n = 3.0 # kappa_q \sim \sqrt{S_{q}} where S_{q} is the proton or neutron sep. energy
kappa_p = 3.0

R_eff = -np.log(tol)/min(kappa_n,kappa_p)

gammax = np.sqrt(2*nshellsx)/R_eff
gammay = np.sqrt(2*nshellsy)/R_eff
gammaz = np.sqrt(2*nshellsz)/R_eff

# Mixing parameter.
alpha_mix = 0.2

# Writing parameters.
iio = 0 # 0 writes densities, 1 writes wavefunctions also. 
outputDir = 'output' # Output parameters. 

# Debug parameters. 
iharmonic = 0 # Uses harmonic oscillator potential for testing purposes. (Should be set to 0 for typical run).  
iconstpair = 0 # Uses a constant pairing field. (Should be set to 0 for typical run). 
isotest = 0 # Uses a vec(W) = vec(r) potential. (Should be set to 0 for typical run).  

# Constants.
massp = 938.272013 
massn = 939.565346 
hbarc = 197.3269631 
hbar2m = pow( hbarc , 2.0 ) / ( massp + massn ) 
mass = (massp + massn)/2.0
alphainv = 137.035999679
e2 = hbarc/alphainv
nstates,qnums = harmonic3dhfb.obtainNumStates(nshellsx,nshellsy,nshellsz)
print(nstates)
print(qnums)



cPnts_x = harmonic3dhfb.get_collocation_points(nshellsx,gammax)
cPnts_y = harmonic3dhfb.get_collocation_points(nshellsy,gammay)
cPnts_z = harmonic3dhfb.get_collocation_points(nshellsz,gammaz)

print(cPnts_x)
Phi_x = harmonic3dhfb.basisHermite(cPnts_x,gammax,nshellsx)
weightArr_x = harmonic3dhfb.get_quadrature_weights(nshellsx,Phi_x) 


test_func = harmonic3dhfb.test_func(cPnts_x)
D1_test_func = harmonic3dhfb.D1_test_func(cPnts_x)
#plt.plot(cPnts_x,test_func)
#plt.show()


