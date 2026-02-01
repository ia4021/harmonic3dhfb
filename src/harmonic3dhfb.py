import numpy as np
import scipy as sci
import jax
import jax.numpy as jnp
import math

#import matplotlib as mpl
#import matplotlib.pyplot as plt

import h5py
import os
import time



jax.config.update('jax_enable_x64', True)


###################################################################################################
# Basis Definitions
###################################################################################################

# We use Scaled-weighted Hermite Functions defined as 
# \tilde{\phi}_{n}(x_{i}) = \sqrt{w_{i}}\phi_{n}(x_{i})
 
# \phi_{n} = \frac{\sqrt{\gamma}}{\pi^{1/4}\sqrt{2^{n}n!}}e^{-\gamma^{2}x^{2}/2}H_{n}(\gamma x) (scaled Hermite functions)
# \gamma = scaling parameter
# H_{n}(\gamma x) = physicist Hermite polynomials 
# w_{i} are the integration weights for the Hermite function interpolation (see notes for definition)
# x_{i} are roots of \phi_{N}(x).

###################################################################################################



#jax.jit()
def obtainNumStates(nshellsx,nshellsy,nshellsz):
    nx = jnp.arange(nshellsx)
    ny = jnp.arange(nshellsy)
    nz = jnp.arange(nshellsz)
    nshells = jnp.max(jnp.array([nshellsx, nshellsy, nshellsz]))

    
    # broadcasting trick to create the sum grid
    # x[:, None, None] is shape (nshellsx, 1, 1)
    # y[None, :, None] is shape (1, nshellsy, 1)
    # z[None, None, :] is shape (1, 1, nshellsz)
    # bascially creates a (nshellsx,nshelly,nshellz) array for x,y,z directions.
    # Then, to get all the states with nshell >= nx + ny + nz, we just add the arrays together
    # to get all possible combinations at onces. Then we use numpy logic to grab the ones that
    # satisfy the condition.
    grid_sum = nx[:, None, None] + ny[None, :, None] + nz[None, None, :]
     
    # This creates a boolean mask and sums the True values
    return jnp.sum(grid_sum <= nshells_limit)

# Counts total number of allowed states given nshells. 
def obtainNumStates_old(nshellsx,nshellsy,nshellsz):
    nshells = max(nshellsx,nshellsy,nshellsz)
    nstates = 0 
    for i in range(nshells):
        for ix in range(min(i+1,nshellsx)): # ix can go from 0 to N-1, which is represented by i. 
            for iy in range(min(i+1,nshellsy)): # iy can go from 0 to N-1, which is represented by i. 
                for iz in range(min(i+1,nshellsz)): # iz can go from 0 to N-1, which is represented by i. 
                    if iz+iy+ix != i:
                        continue
                    nstates +=1     
    return nstates

# Records the energies and quantum numbers of the allowed states.
def obtainQuanNums(nstates,nshellsx,nshellsy,nshellsz,hbaromegax,hbaromegay,hbaromegaz):
    nshells = max(nshellsx,nshellsy,nshellsz)
    eho = np.zeros(nstates)
    qnums = np.zeros([4,nstates])
    ifill = 0
    for i in range(nshells):
        for ix in range(min(i+1,nshellsx)): # ix can go from 0 to N, which is represented by i. 
            for iy in range(min(i+1,nshellsy)): # ix can go from 0 to N, which is represented by i. 
                for iz in range(min(i+1,nshellsz)): # ix can go from 0 to N, which is represented by i. 
                    if iz+iy+ix != i:
                        continue
                    eho[ifill] = hbaromegax*(ix+0.5)+hbaromegay*(iy+0.5)+hbaromegaz*(iz+0.5)
                    qnums[0,ifill] = i
                    qnums[1,ifill] = ix
                    qnums[2,ifill] = iy
                    qnums[3,ifill] = iz
                    ifill +=1 # For spin!          
    return eho, qnums


#@jax.jit
def get_collocation_points(Nmax,gamma):
    roots,weights = sci.special.roots_hermite(Nmax)
    cPnts = roots/gamma
    return cPnts
#@jax.jit
def get_quadrature_weights(Nmax,Phi):
    weights = 1.0/(Nmax*Phi[:,Nmax-1]**2)
    return weights
#@jax.jit
def basisHermite(cPnts,weights,gamma,Nmax):
    '''
    Calculates the basis matrix \Phi_{ni} defined as 
    
    \tilde{\Phi}_{ni} = \tilde{\phi}_{n}(x_{i}))

    This is calculated using recursion relation

    \tilde{\phi}_{n+1}(x) = \gamma x \sqrt{\frac{2}{n+1}}\tilde_{\phi}_{n}(x) - \sqrt{\frac{n}{n+1}}\tilde{\phi}_{n-1}(x)

    \tilde{\phi}_{0}(x) = \frac{\sqrt{\gamma}}{\pi^{1/4}} e^{-\gamma^{2}x^{2}/2}

    Here we assume n goes from 0 to Nmax - 1
    '''    
        y = gamma * cPnts
        Phi = jnp.zeros((len(y),Nmax))
        
        # recurrence relation
        Phi = Phi.at[:,0].set(jnp.sqrt(gamma)/jnp.pi**(0.25) * jnp.exp(-0.5 * y**2))
        if Nmax > 1:
            Phi = Phi.at[:, 1].set(y*jnp.sqrt(2)*Phi[:,0])        
        
        for n in range(1, Nmax-1):
            Phi = Phi.at[:,n+1].set(jnp.sqrt(2.0/(n+1)) * y * Phi[:,n] - jnp.sqrt(n/(n+1)) * Phi[:,n-1])
        return Phi


def interpolate_1d(fArr,h,N,grid):
    cardinal_func = np.zeros((2*N+1,len(grid)))
    for j in range(2*N +1 ):
        cardinal_func[j] = 0  
    result = np.einsum('i,ij->j',fArr,card)
    return result

