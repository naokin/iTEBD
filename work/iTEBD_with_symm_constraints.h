#ifndef __iTEBD_with_symm_constraints_H
#define __iTEBD_with_symm_constraints_H

void iTEBD_with_symm_constraints
(
  double J, ///< J  coupling constant in 1D-Heisenberg spin hamiltonian
  double Jz,///< Jz coupling constant ...
  double Hz,///< external magnetic field
  double dt,///< step size of time-evolution
  size_t M_sector, ///< Number of symmetry sectors
  size_t M_block, ///< Number of blocks for each symmetry sector
  size_t M_state ///< Number of dense states for each block
);

void imagEvolve
(
  bool isForward,
  MPS<double>& aMps,
  MPS<double>& bMps,
  double J, ///< J  coupling constant in 1D-Heisenberg spin hamiltonian
  double Jz,///< Jz coupling constant ...
  double Hz,///< external magnetic field
  double dt,///< step size of time-evolution
  size_t M_sector, ///< Number of symmetry sectors
  size_t M_block, ///< Number of blocks for each symmetry sector
  size_t M_state ///< Number of dense states for each block
);

#endif // __iTEBD_with_symm_constraints_H
