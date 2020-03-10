from shci4qmc.src.champ import ChampInputFiles
from pyscf import gto

fs = ChampInputFiles()

fs.config = {
	"system": "chem",
	"n_up": 9,
	"n_dn": 7,
	"var_only": True,
	"eps_vars": [
		0.005
	],
	"eps_vars_schedule": [
        0.02,
		0.01,
        0.005
	],
	"chem": {
		"point_group": "Dooh"
	},
    "optorb": True,
    "optimization": {
        "rotation_matrix": True,
        "method": "appnewton",
        "accelerate": True
    }
}

fs.shci_cmd = 'mpirun -np 1 /home/tanderson/shci/shci'
fs.dir_reuse = './reuse'
fs.dir_qmc_inp = './qmc_inp'
fs.tol_det = 0. #tolerance for det coef in shci wf
fs.tol_csf = [1e-1, 3e-2, 7e-3, 0.] #tolerance for (csf coef in shci_wf)/sqrt(ndet in csf)
fs.opt_orbs = True
fs.target_l2 = None

mol = gto.Mole()
mol.unit = 'bohr'
mol.atom = 'O 0 0 -1.14139000; O 0 0 1.14139000'
mol.spin = 2
mol.symmetry = 'dooh'
mol.verbose = 0
mol.build()
fs.mol = mol

fs.make()
