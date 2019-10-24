import sys
import os
import subprocess
sys.path.append(os.path.join(os.path.dirname(__file__), '../lib'))

from pyscf import symm
import numpy as np
import csfgen as proj
import vec
import symm as sy
from vec import Det, Vec, Config

def parse_csf_file(num_open_shells_max, twice_s, csf_file_contents):
    lines = csf_file_contents.split('\n')
    max_nelecs = 0
    blocks, csf_info = [], {}
    if (twice_s == 0): 
        csf_info[0] = ([[1]], [''])
    while 'END' in lines:
        blocks.append(lines[lines.index('START')+1:lines.index('END')])
        lines = lines[lines.index('END')+1:]
    for block in blocks:
        header = block[0].split(' ')
        n, s = int(header[2]), float(header[5])
        coefs = []
        if (n > num_open_shells_max):
            break
        if (twice_s/2. != s): 
            continue
        max_nelecs = max(n, max_nelecs)
        det_strs = block[1].split(' ')[:-1]
        for line in block[2:]:
            coefs.append([float(num) for num in line.split('\t')[:-1]])
        csf_info[n] = (coefs, det_strs)
    return csf_info, max_nelecs

def make_csf_file(max_open, twice_s):
    filename = "csf_cache.txt"
    try:
        f = open(filename, 'r')
        raise Exception(cache_name + " already exists." + 
                "make_csf_file will not attempt to overwrite it")
    except:
        gen_script = os.path.join(os.path.dirname(__file__), '../bin/run_csfgen')
        out_dir = '.'
        nelecs = str(max_open)
        subprocess.run([gen_script, out_dir, filename, nelecs])
        f = open(filename, 'r')
        file_contents = f.read()
        return parse_csf_file(sys.maxsize, twice_s, file_contents)

def load_csf_file(max_open, twice_s):
    filename = "csf_cache.txt"
    try:
        f = open(filename, 'r')
    except:
        csf_info, max_nelecs = make_csf_file(max_open, twice_s)
        return csf_info
    file_contents = f.read()
    csf_info, max_loaded = parse_csf_file(max_open, twice_s, file_contents)
    if (max_loaded < max_open or len(csf_info) == 0):
        raise Exception("Could not find CSFs with requested nelecs " + 
                "and/or spin eigenvalue in " + filename + 
                ". To re-calculate CSFS, remove " + filename + 
                " from the current directory.")
    return csf_info

def combs(n, r):
    if r == 0:
        yield '0'*n
        return
    for m in range(n-r+1):
        for comb in combs(n-m-1, r-1):
            yield '0'*m + '1' + comb

def get_occ_combs(nup, nopen):
    return {comb:n for n, comb in enumerate(combs(nopen, nup))}

def det2open_occ_str(det):
    #could be faster
    occs = sorted(det.up_occ + det.dn_occ)
    return ''.join(['1' if occ in det.up_occ else '0' for occ in occs])

def symm_configs(dmc, configs):
    #if mol is a sigma state, this avoids creating duplicate csfs
    skip = set()
    for config in configs:
        if config in skip: continue
        yield config
        skip.update([config, sy.partner_config(dmc, config)])

def configs2csfs(dmc, csf_info, configs, rel_parity = True):
    csfs = []
    if dmc.symmetry in ('DOOH', 'COOV'):
        configs = symm_configs(dmc, configs)
    for config in configs:
        csfs += list(config2csfs(dmc, config, csf_info, rel_parity))
    return csfs

def config2csfs(dmc, config, csf_info, rel_parity = True):
    '''
    When csf coefs are generated, they are sometimes generated as though
    the electrons are bosonic. If rel_parity = True, this program will
    correct this issue and insert the appropriate sign.

    The gene.cc program uses no extra sign (bosonic) by default.
    '''
    nopen = len([orb for orb in config.occs if config.occs[orb] == 1])
    csf_coefs, index2occs = csf_info[nopen]
    dets = [make_det(occs, config) for occs in index2occs]
    csfs = []
    for coefs in csf_coefs:
        csf = Vec.zero()
        for n, coef in enumerate(coefs):
            det = dets[n] 
            p = parity(det) if rel_parity else 1
            if dmc.symmetry in ('DOOH', 'COOV'):
                det = sy.convert_det(dmc, det)
            csf += p*coef*det
        if csf.norm() > 0: csfs.append(csf/csf.norm())
    return np.array(csfs)

def make_det(occ_str, config):
    up_occs, dn_occs, orbs, is_open = [], [], [], {}
    orbs = sorted([orb for orb in config.occs])
    for orb in orbs:
        if config.occs[orb] == 1:
            if occ_str[0] == '1':
                up_occs.append(orb)
                occ_str = occ_str[1:]
            else:
                dn_occs.append(orb)
                occ_str = occ_str[1:]
        else:
            up_occs.append(orb)
            dn_occs.append(orb)
    assert(occ_str == '')
    return Det(up_occs, dn_occs)

def parity(det):
    #computes parity relative to ordering s.t. up/down spins for the same
    #spacial orbital are adjacent
    up_occ, dn_occ = det.up_occ, det.dn_occ
    if len(up_occ) == 0 or len(dn_occ) == 0:
        return 1
    up_ptr, dn_ptr = len(up_occ)-1, len(dn_occ)-1
    alt_sum, count = 0, 0
    while -1 < dn_ptr:
        dn = dn_occ[dn_ptr]
        if up_ptr != -1 and up_occ[up_ptr] > dn:
            count += 1
            up_ptr += -1
        else:
            alt_sum = count - alt_sum 
            dn_ptr += -1
    assert(alt_sum > -1)
    return (1 if alt_sum%2 == 0 else -1)

def compute_csfs(config, twice_s, twice_sz, method):
    '''
    Use projection method to compute configurations
    '''
    s, sz = twice_s/2, twice_sz/2
    if method == 'projection':
        config_str = config.make_config_str()
        proj_csfs = proj.compute_csfs(config_str, s, sz)
        csfs = convert_proj_csfs(proj_csfs)
    else:
        raise Exception('Invalid method \'' + method + '\' in compute_csfs')
    return csfs

def convert_proj_csfs(proj_csfs):
    csfs = []
    for proj_csf in proj_csfs:
        csf_dets = {} 
        for proj_det in proj_csf.dets:
            coeff = proj_csf.get_det_coeff(proj_det)
            up_occs, dn_occs = [], []
            for orb in proj_det.orbitals:
                sz = orb.labels['s_z']
                n = orb.labels['n']
                if sz == 0.5: 
                    up_occs.append(n)
                else: 
                    dn_occs.append(n)
            csf_dets[vec.Det(up_occs, dn_occs)] = coeff
        csfs.append(vec.Vec(csf_dets))
    return csfs

class Dummy():
  def __init__(porbs, xorbs):
    self.symmetry = 'DOOH'
    self.xorbs = porbs
    self.porbs = xorbs
    self.ang_mom = ang_mom
    self.real = real

if __name__ == "__main__":
    csf_info = load_csf_file(4, 0)
    print(csf_info[4]) 

