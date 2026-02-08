import numpy as np
import scipy as sci
import jax
import jax.numpy as jnp
import jax.scipy as jsci
import math
import h5py



jax.config.update('jax_enable_x64', True)

def test_func(x):
    return np.exp(-x**2)*np.cos(x)
def D1_test_func(x):
    return -np.exp(-x**2)*(2*x*np.cos(x)+np.sin(x))
def D2_test_func(x):
    return np.exp(-x**2)*((-3 + 4*x**2)*np.cos(x) + 4*x*np.sin(x))
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

def obtainNumStates(nshellsx,nshellsy,nshellsz):
    nx = np.arange(nshellsx)
    ny = np.arange(nshellsy)
    nz = np.arange(nshellsz)
    nshells = max([nshellsx, nshellsy, nshellsz])

    
    # broadcasting trick to create the sum grid
    # x[:, None, None] is shape (nshellsx, 1, 1)
    # y[None, :, None] is shape (1, nshellsy, 1)
    # z[None, None, :] is shape (1, 1, nshellsz)
    # bascially creates a (nshellsx,nshelly,nshellz) array for x,y,z directions.
    # Then, to get all the states with nshell >= nx + ny + nz, we just add the arrays together
    # to get all possible combinations at onces. Then we use numpy logic to grab the ones that
    # satisfy the condition.
    grid_sum = nx[:, None, None] + ny[None, :, None] + nz[None, None, :]
    
    mask = grid_sum < nshells
    ix, iy, iz = np.where(mask)
    itot = grid_sum[mask] 
    qnums = np.vstack([itot, ix, iy, iz])
    
    # sort by itot, then ix, then iy, then iz
    # np.lexsort sorts by the last key first (iz -> iy -> ix -> itot)
    sort_indices = np.lexsort((iz, iy, ix, itot))
    qnums = qnums[:, sort_indices] 
    num_states = qnums.shape[1]

    return num_states, qnums

# Note: Add function that automates the choice of nshellx,nshelly, nshellz that corresponds to
# minimum needed shells to meet error tolerance. 

def get_collocation_points(Nmax,gamma):
    """
    
    Gets collocation points for coordiante axis. Collocation points are chosen to be 
    scaled roots of the physicist Hermite polynomials H_{Nmax}(y) where y = \gamma x.

    Note: sci.special.roots_hermite is stable up to at least Nmax \sim 1000 according to their documentation
    
    """
    roots,weights = sci.special.roots_hermite(Nmax)
    cPnts = roots/gamma
    return cPnts
def basisHermite(cPnts,gamma,Nmax):
    '''
    Calculates the basis matrix \Phi_{ni} for the scaled Harmonic Oscillator Basis defined as 
    
    \Phi_{ni} =\phi_{n}(x_{i})

    This is calculated using recursion relation

    \phi_{n+1}(x) = \gamma x \sqrt{\frac{2}{n+1}}\phi_{n}(x) - \sqrt{\frac{n}{n+1}}\phi_{n-1}(x)

    \phi_{0}(x) = \frac{\sqrt{\gamma}}{\pi^{1/4}} e^{-\gamma^{2}x^{2}/2}

    Here we assume n goes from 0 to Nmax - 1

    See notes for proof of recursion relation
    '''    
    y = gamma * cPnts
    Phi = np.zeros((len(y),Nmax))    
    # recurrence relation
    Phi[:,0] = np.sqrt(gamma)/np.pi**(0.25) * np.exp(-0.5 * y**2)
    if Nmax > 1:
        Phi[:, 1] = y*np.sqrt(2)*Phi[:,0]          
    for n in range(1, Nmax-1):
        Phi[:,n+1] = y*np.sqrt(2.0/(n+1))*Phi[:,n] - np.sqrt(n/(n+1)) * Phi[:,n-1]
    return Phi

def get_quadrature_weights(Nmax,Phi):
    """
    Integration weights for the scaled hermite basis. See notes for proof.
    """
    weights = 1.0/(Nmax*Phi[:,Nmax-1]**2)
    return weights

def weighted_basisHermite(wArr):
    
    return tilde_Phi
def interpolate_1d(fArr,h,N,grid):
    cardinal_func = np.zeros((2*N+1,len(grid)))
    for j in range(2*N +1 ):
        cardinal_func[j] = 0  
    result = np.einsum('i,ij->j',fArr,card)
    return result

