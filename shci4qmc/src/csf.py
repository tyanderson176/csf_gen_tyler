import math
import numpy as np
from numpy import linalg as la
from scipy.sparse import csr_matrix
import scipy.sparse as sparse

import shci4qmc.lib.load_wf as lwf
from shci4qmc.src.gen import CSF_Generator
from shci4qmc.src.ham import Ham
from shci4qmc.src.vec import Vec, Det, Config

def get_csfs(filename, det_tol, mol, mf, target_l2):
    wf = load_shci_wf(filename, det_tol)
    gen = CSF_Generator(wf, mol, mf, target_l2)
    csfs = gen.generate()
    coefs = [csf.dot(gen.real_wf) for csf in csfs]
    csfs, coefs = rotate_by_configs(csfs, coefs)
    return csfs, coefs, gen.real_wf

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

def rotate_by_configs(csfs, coefs):
    def rotate(config, csfs):
        sum_csf = Vec.zero()
        for csf in csfs:
            sum_csf += csf
        sum_csf.config_label = config
        coefs = [sum_csf.norm() if n == 0 else 0. for n, csf in enumerate(csfs)]
        csfs = Vec.gram_schmidt([sum_csf] + csfs, len(csfs))
        return list(zip(csfs, coefs))

    group_by_configs = {csf.config_label: [] for csf in csfs}
    for csf, coef in zip(csfs, coefs):
        group_by_configs[csf.config_label].append(coef*csf)
    pairs = []
    for config, group in group_by_configs.items():
        pairs += rotate(config, group) 
    return zip(*pairs)

def error(pairs, wf):
    p_wf = Vec.zero()
    p_wf += wf
    for csf, coef in pairs:
        p_wf += (-coef)*csf
    return (p_wf.norm()/wf.norm())**2