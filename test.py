import p2d
import dmc
import subprocess
import sys

from pyscf import scf, gto, mcscf

mol = gto.Mole()
#mol.atom = 'H 0 0 1; N 0 0 0; H 1 0 0; H 0 1 0'
mol.atom = 'N 0 0 -1; N 0 0 1'
mol.basis = 'sto-3g'
#mol.basis = 'ccpvdz'
mol.charge = 0
mol.spin = 0 
mol.verbose = 0
mol.build(0, 0)
mol.symmetry = 'dooh'
mf = scf.RHF(mol).run()

eps_var = [1e-2, 5e-3, 1e-3]
eps_var_sched = eps_var
shci_cmd = '/home/tanderson/Projects/dmc/dmc/bin/run_shci .'
shci_path = 'shci.out'
num_dets = 100
out_name = 'dmc.in'
dmc.make_dmc(mol, mf, eps_var, eps_var_sched, 
        shci_cmd, shci_path, num_dets, out_name)
