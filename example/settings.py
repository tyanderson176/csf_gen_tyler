from shci4qmc.src.champ import ChampInputFiles
from pyscf import gto

fs = ChampInputFiles()

fs.config = {
	"system": "chem",
	"n_up": 9,
	"n_dn": 7,
	"var_only": True,
	"eps_vars": [
		0.05
	],
	"eps_vars_schedule": [
        0.2,
		0.1,
        0.05
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
fs.dir_in = './in'
fs.dir_out = './out'
fs.tol_det = 1e-2
fs.tol_csf = [0., 1e-1, 1e-2]
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
