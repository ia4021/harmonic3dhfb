# harmonic3dhfb

# 1) Need to compute potentials after Hamiltonian, density step, but before compute observables.
# 2) Need to compute divJ density.
# 3) Need to compute Coulomb potential more accurately.
# 4) Need to replace all integrals involving trapz with quadrature integrals.
# 5) Need to remove analytic derivatives in Hamiltonian for kinetic term, for ones using derivatives functions so we can handle a effective mass. Also important since quadrature points are only accurate for derivative functions.
# 6) Need exit condition after computing observables.
# 7) Need to include linear mixing of potentials after computing observables.
# 8) Need to place all functions in a for loop.
# 9) Need to speed up functions.
# ...) Eventually odd terms, other functionals, constraints, restart feature, writing of wfs/densities, and potential energy surface feature. 
