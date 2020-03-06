import os
import subprocess

import shci4qmc.lib.load_wf as lwf

from shci4qmc.src.rotate_orbs import OrbRotator
from shci4qmc.src.l2_proj import L2_Projector
from shci4qmc.src.vec import Vec, Det, Config

def load_shci_wf(filename, tol):
    def make_det(orbs):
        [up, dn] = orbs
        reindex_up = [orb+1 for orb in up]
        reindex_dn = [orb+1 for orb in dn]
        return Det(reindex_up, reindex_dn)

    shci_wf_dict = lwf.load(filename)
    pairs = zip(shci_wf_dict['dets'], shci_wf_dict['coefs'])
    shci_wf = Vec.zero()
    for orbs, coef in pairs:
        if abs(coef) > tol:
            shci_wf += coef*make_det(orbs) 
    return shci_wf

def orthogonalize(csfs):
    csfs = Vec.gram_schmidt(csfs, len(csfs), tol=1e-4)
    return csfs

def generate_spin_csfs(n_open_max, spin):
    filepath = "spin_csfs.dat"
    csf_gen_exe = os.path.join(os.path.dirname(__file__), '../lib/spin_csf_gen')
    process = subprocess.Popen('%s %d'% (csf_gen_exe, n_open_max), shell=True, 
                               stdout=subprocess.PIPE, universal_newlines = True)
    with open(filepath, 'w+') as f:
        for line in iter(process.stdout.readline, ''):
            f.write(line)
    spin_csfs = parse_spin_csfs(spin, filepath)
    os.remove(filepath) #cleanup
    return spin_csfs

def parse_spin_csfs(spin, filepath):
    def parse(f):
        coefs, det_strs = [], f.readline().split()
        for line in f:
            if 'END' in line: 
                break
            else:
                coefs.append([float(coef) for coef in line.split()])
        return (coefs, det_strs)

    spin_csfs = {0 : ([[1]], [''])}
    with open(filepath) as f:
        for line in f:
            if 'START' in line:
                match = f.readline().split()
                n_open, s = int(match[1]), float(match[3])
                if s != spin/2:
                    continue
                assert(n_open not in spin_csfs)
                spin_csfs[n_open] = parse(f)
    return spin_csfs

def make_det(config, det_str):
    '''
    'config' is the orbital configuration of the returned determinant

    'det_str' is a string of 0's and 1's containing information about which open orbitals are up/dn. 
    If det_str[i] == '1', then the ith open orbital is up.  
    If det_str[i] == '0', then the ith open orbital in dn.
    '''

    up_orbs, dn_orbs = [], []
    i = 0
    orb_occupation = sorted(config.occs.items(), key = lambda p: p[0])
    for orb, n_elec in orb_occupation:
        spin_up = i < len(det_str) and det_str[i] == '1' 
        spin_dn = i < len(det_str) and det_str[i] == '0'
        if n_elec == 2 or spin_up:
            up_orbs.append(orb)
        if n_elec == 2 or spin_dn:
            dn_orbs.append(orb)
        if n_elec == 1:
            i += 1
    assert(i == len(det_str))
    assert(Config(Det(up_orbs, dn_orbs)) == config)

    return Det(up_orbs, dn_orbs)

def make_csf(coefs, dets):
    assert(len(coefs) == len(dets))
    state = Vec.zero()
    for det, coef in zip(dets, coefs):
        state += coef*det
    return state

def get_n_open_max(wf):
    configs = set(Config(det) for det in wf.dets.keys())
    return max([config.num_open for config in configs])

class CSF_Generator():
    def __init__(self, wf, mol, mf, target_l2 = None):
        self.wf = wf
        self.n_open_max = get_n_open_max(self.wf)
        self.spin_csf_data = generate_spin_csfs(self.n_open_max, abs(mol.spin))
        self.target_l2 = target_l2
        self.linear_sys = (mol.symmetry.lower() in ('dooh', 'coov'))
        self.atomic_sys = (mol.natm == 1 and self.linear_sys and self.target_l2 != None)
        self.real_wf = wf
        if self.linear_sys:
            self.orb_rotator = OrbRotator(mol, mf)
            self.ro_wf = self.orb_rotator.to_real_orbs(self.wf) #wf w/ real orbs
            self.rmag = self.ro_wf.real_part().norm()
            self.imag = self.ro_wf.imag_part().norm()
            self.real_wf = (self.ro_wf.real_part() if self.rmag >= self.imag 
                            else self.ro_wf.imag_part()) #wf w/ real orbs & real coefs
        if self.atomic_sys:
            self.l2_projector = L2_Projector(mol, mf)

    def linear_sys_csfs(self, csfs):
        def partner(config):
            return self.orb_rotator.partner_config(config)

        def real_orbs(csf):
            return self.orb_rotator.to_real_orbs(csf)

        lin_sys_csfs, skip = [], set()
        for csf in csfs:
            config = csf.config_label
            if config in skip:
                continue
            csf = real_orbs(csf)
            csf = csf.real_part() if self.rmag >= self.imag else csf.imag_part()
            csf.config_label = config
            if csf.norm() > 0:
                lin_sys_csfs.append(csf)
            if config != partner(config):
                skip.add(partner(config))
        return lin_sys_csfs

    def atomic_sys_csfs(self, csfs):
        atm_csfs = []
        for csf in csfs:
            atm_csf = self.l2_projector.project(self.target_l2, csf)
            #TODO: use spherical config label (in l2_projector?)
            atm_csf.config_label = csf.config_label
            if atm_csf.norm() > 1e-2:
                err = self.l2_projector.eigen_error(self.target_l2, atm_csf)
                assert(err < 1e-1)
                atm_csfs.append(atm_csf)
        return atm_csfs

    def generate_init(self):
        def configs_of(wf):
            return sorted(set(Config(det) for det in wf.dets.keys()))

        def csfs_from_config(config):
            data, det_strs = self.spin_csf_data[config.num_open]
            dets = [make_det(config, det_str) for det_str in det_strs]
            csfs = [make_csf(coefs, dets) for coefs in data]
            for csf in csfs:
                csf.config_label = config
            return csfs

        configs = configs_of(self.wf)
        csfs = [csf for config in configs for csf in csfs_from_config(config)]
        return csfs

    def generate(self):
        csfs = self.generate_init()
        if self.linear_sys:
            csfs = self.linear_sys_csfs(csfs)
        if self.atomic_sys: 
            csfs = self.atomic_sys_csfs(csfs) 
        csfs = orthogonalize(csfs)
        return csfs

if __name__ == "__main__":
    import numpy as np

    from pyscf import scf, ao2mo, gto, symm

    import shci4qmc.src.gamess as gamess
    import shci4qmc.src.sym_rhf as sym_rhf

    mol = gto.Mole()
    mol.unit = 'bohr'
    mol.atom = 'N 0 0 0.'
    mol.spin = 3
    mol.symmetry = 'dooh'
    mol.basis = 'ccpvdz'
    
    mol.build()
    mol.is_atomic_system = True
    
    mf = sym_rhf.RHF(mol).run()
    print('1 = 1S, 2 = 2S, 3 = 2Px, 4 = 2Py, 5 = 2Pz, 6 = 3Px, 7 = 3Py, 8 = 3Pz')
    
    #Simple approx. sigma-minus state w/ complex orbs
    asym = 0.01
    wf = Vec.zero()
    wf +=  0.9*Det([1, 2, 3, 4, 5], [1, 2])
    wf +=  0.1*Det([2, 3, 4, 5, 8], [2, 8])
    wf += -0.1*Det([2, 3, 4, 5, 6], [2, 7])
    wf += -0.1*Det([2, 3, 4, 5, 7], [2, 6])
    wf += asym*Det([2, 3, 4, 5, 7], [2, 6]) #slight asymmetry

    targ_l2 = 0 #find l2 = 0 csfs
    g = CSF_Generator(wf, mol, mf, targ_l2)
    csfs = g.generate()

    print('CSFs (w/ sigma minus symmetry) generated from WF: ')
    for n, csf in enumerate(csfs):
        print(n)
        for det, coef in csf.dets.items():
            print(det, coef)

    print('WF w/ real orbs (sigma minus part should be linear comb of above CSFs): ')
    print('Sigma plus part: ', g.ro_wf.real_part())  #nonzero from asymmetry
    print('Sigma minus part: ', g.ro_wf.imag_part()) #linear comb of csfs generated from g
