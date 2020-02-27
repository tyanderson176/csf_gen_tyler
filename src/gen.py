import sys
import os
import subprocess
#sys.path.append(os.path.join(os.path.dirname(__file__), '../lib'))

from pyscf import symm
import numpy as np
import shci4qmc.src.vec as vec
import shci4qmc.src.symm as sy
from shci4qmc.src.vec import Det, Vec, Config
from shci4qmc.src.proj_l2 import L2Projector

class GenMethods():
    def __init__(self):
        if self.config['project_l2']:
            assert(self.mol.is_atomic_system and self.symmetry == 'DOOH')
            #TODO: Add some tolerance options here? prune tol + mol matrix tol 
            self.L2projector = L2Projector(self.mol, self.mf)
            self.target_l2 = self.config['target_l2']

    def parse_csf_file(self, num_open_shells_max, twice_s, csf_file_contents):
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

    def make_csf_file(self, max_open, twice_s):
        filename = "csf_cache.txt"
        try:
            f = open(filename, 'r')
            raise Exception(cache_name + " already exists." + 
                    "make_csf_file will not attempt to overwrite it")
        except:
            csf_gen_exe = os.path.join(os.path.dirname(__file__), '../lib/spin_csf_gen')
            process = subprocess.Popen(
                '%s %d'% (csf_gen_exe, max_open), shell=True, stdout=subprocess.PIPE,
                universal_newlines = True)
            csf_cache_file = open(filename, 'w+')
            file_contents = ''
            for line in iter(process.stdout.readline, ''):
                csf_cache_file.write(line)
                file_contents += line
            csf_cache_file.close()

#            csf_gen_exe = os.path.join(os.path.dirname(__file__), '../lib/spin_csf_gen')
#            out_dir = '.'
#            nelecs = str(max_open)
#            subprocess.run([gen_script, out_dir, filename, nelecs])
#            f = open(filename, 'r')
#            file_contents = f.read()

#            gen_script = os.path.join(os.path.dirname(__file__), '../bin/run_csfgen')
#            out_dir = '.'
#            nelecs = str(max_open)
#            subprocess.run([gen_script, out_dir, filename, nelecs])
#            f = open(filename, 'r')
#            file_contents = f.read()
            return self.parse_csf_file(sys.maxsize, twice_s, file_contents)

    def load_csf_file(self, max_open, twice_s):
        filename = "csf_cache.txt"
        try:
            f = open(filename, 'r')
        except:
            csf_info, max_nelecs = self.make_csf_file(max_open, twice_s)
            return csf_info
        file_contents = f.read()
        csf_info, max_loaded = self.parse_csf_file(max_open, twice_s, file_contents)
        if (max_loaded < max_open or len(csf_info) == 0):
            raise Exception("Could not find CSFs with requested nelecs " + 
                    "and/or spin eigenvalue in " + filename + 
                    ". To re-calculate CSFS, remove " + filename + 
                    " from the current directory.")
        return csf_info

    def symm_configs(self, configs):
        #if mol is a sigma state, this avoids creating duplicate csfs
        skip = set()
        for config in configs:
            if config in skip: continue
            yield config
            skip.update([config, self.partner_config(config)])

    def configs2csfs(self, csf_cache, configs):
        csfs, config_labels = [], []
        if self.symmetry in ('DOOH', 'COOV'):
            configs = self.symm_configs(configs)
        for n, config in enumerate(configs):
            config_csfs = list(self.config2csfs(csf_cache, config))
            for csf in config_csfs:
                csf.config_label = n
            csfs += config_csfs
        csfs = Vec.gram_schmidt(csfs, len(csfs), tol=1e-4)
        return csfs

    def config2csfs(self, csf_cache, config):
        nopen = len([orb for orb in config.occs if config.occs[orb] == 1])
        csfs_coefs, index2occs = csf_cache[nopen]
        dets = [self.make_det(occs, config) for occs in index2occs]
        csfs = []
        for coefs in csfs_coefs:
            csf = Vec.zero()
            for n, coef in enumerate(coefs):
                det = dets[n] 
                if self.symmetry in ('DOOH', 'COOV'):
                    det = self.convert_det(det)
                csf += coef*det
            if self.config['project_l2']:
                start = csf
                csf = self.L2projector.project(self.target_l2, csf)
                err = self.L2projector.eigen_error(self.target_l2, csf)
                #if l2_error is too large, skip this csf
                if err > 1e-1 or csf.norm() < 1e-2:
                    if csf.norm() > 1e-2:
                        print('WARNING! Rejecting bad projection.')
                        print('l2_error: ', err, '; l2_proj norm: ', csf.norm())
                        for det, coef in csf.dets.items():
                            print(det, ": ", coef)
                        print("\nStarting det:")
                        for det, coef in start.dets.items():
                            print(det, ": ", coef)
                        print()
                    continue
            if csf.norm() != 0:
                csfs.append(csf)
        return np.array(csfs)

    def check_orthogonal(self, csfs):
        for n, csf1 in enumerate(csfs):
            for m, csf2 in enumerate(csfs):
                if m <= n:
                    continue
                if (csf1.dot(csf2) > 1e-2):
                    print('Not orthogonal: ', csf1.dot(csf2)) 

    def make_det(self, occ_str, config):
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

#def parity(det):
#    #computes parity relative to ordering s.t. up/down spins for the same
#    #spacial orbital are adjacent
#    up_occ, dn_occ = det.up_occ, det.dn_occ
#    if len(up_occ) == 0 or len(dn_occ) == 0:
#        return 1
#    up_ptr, dn_ptr = len(up_occ)-1, len(dn_occ)-1
#    alt_sum, count = 0, 0
#    while -1 < dn_ptr:
#        dn = dn_occ[dn_ptr]
#        if up_ptr != -1 and up_occ[up_ptr] > dn:
#            count += 1
#            up_ptr += -1
#        else:
#            alt_sum = count - alt_sum 
#            dn_ptr += -1
#    assert(alt_sum > -1)
#    return (1 if alt_sum%2 == 0 else -1)

#if __name__ == "__main__":
