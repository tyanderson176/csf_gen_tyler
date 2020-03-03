import os
import subprocess

import shci4qmc.lib.load_wf as lwf
from shci4qmc.src.rotate_orbs import OrbRotator
from shci4qmc.src.proj_l2 import L2_Projector
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

def parse_csf_file(n_open_max, spin, csf_file_contents):
    lines = csf_file_contents.split('\n')
    max_nelecs = 0
    blocks, csf_info = [], {}
    if (spin == 0): 
        csf_info[0] = ([[1]], [''])
    while 'END' in lines:
        blocks.append(lines[lines.index('START')+1:lines.index('END')])
        lines = lines[lines.index('END')+1:]
    for block in blocks:
        header = block[0].split(' ')
        n, s = int(header[2]), float(header[5])
        coefs = []
        if (n > n_open_max):
            break
        if (spin/2. != s): 
            continue
        max_nelecs = max(n, max_nelecs)
        det_strs = block[1].split(' ')[:-1]
        for line in block[2:]:
            coefs.append([float(num) for num in line.split('\t')[:-1]])
        csf_info[n] = (coefs, det_strs)
    return csf_info, max_nelecs

def make_csf_file(n_open_max, spin):
    filename = "csf_cache.txt"
    csf_gen_exe = os.path.join(os.path.dirname(__file__), '../lib/spin_csf_gen')
    process = subprocess.Popen(
        '%s %d'% (csf_gen_exe, n_open_max), shell=True, stdout=subprocess.PIPE,
        universal_newlines = True)
    csf_cache_file = open(filename, 'w+')
    file_contents = ''
    for line in iter(process.stdout.readline, ''):
        csf_cache_file.write(line)
        file_contents += line
    csf_cache_file.close()
    return parse_csf_file(n_open_max, spin, file_contents)

def load_spin_csfs(wf, spin):
    n_open_max = get_n_open_max(wf)
    csf_data, n_elecs_max = make_csf_file(n_open_max, spin)
    assert(n_elecs_max >= n_open_max) 
    return csf_data

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
    def __init__(self, filename, mol, mf, det_tol, target_l2 = None):
        self.wf = load_shci_wf(filename, det_tol)
        self.spin_csf_data = load_spin_csfs(self.wf, abs(mol.spin))
        self.target_l2 = target_l2
        self.linear_sys = (mol.symmetry.lower() in ('dooh', 'coov'))
        self.atomic_sys = (mol.natm == 1 and self.linear_sys and self.target_l2 != None)
        if self.linear_sys:
            self.orb_rotator = OrbRotator(mol, mf)
            self.real_wf = self.orb_rotator.to_real_orbs(self.wf)
            self.rmag = self.real_wf.real_part().norm()
            self.imag = self.real_wf.imag_part().norm()
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
            if csf.norm() > 0:
                lin_sys_csfs.append(csf)
            if config != partner(config):
                skip.add(partner(config))
        return lin_sys_csfs

    def atomic_sys_csfs(self, csfs):
        atm_csfs = []
        for csf in csfs:
            atm_csf = self.l2_projector.project(self.target_l2, csf)
            if atm_csf.norm() > 1e-2:
                err = self.l2_projector.eigen_error(self.target_l2, atm_csf)
                assert(err < 1e-1)
                atm_csfs.append(atm_csf)
        return atm_csfs

    def generate_init(self):
        def configs_of(wf):
            return set(Config(det) for det in wf.dets.keys())

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
    
    #Overwrite shci load routine
    def load_shci_wf(filename, tol): 
        #Simple sigma-minus state
        shci_wf = Vec.zero()
        shci_wf +=  0.9*Det([1, 2, 3, 4, 5], [1, 2])
        shci_wf +=  0.1*Det([1, 2, 3, 4, 5], [1, 3])
        shci_wf += -0.1*Det([1, 2, 3, 4, 5], [1, 4])
        return shci_wf

    g = CSF_Generator(None, mol, mf, 0., None)
    csfs = g.generate()

    print('CSFs generated from WF: ')
    for csf in csfs:
        for det, coef in csf.dets.items():
            print(det, coef)

    print('WF w/ real orbs (should be linear comb of above CSFs): ')
    print('Sigma plus part: ', g.real_wf.real_part()) #zero
    print('Sigma minus part: ', g.real_wf.imag_part()) #sum of csfs generated from g
