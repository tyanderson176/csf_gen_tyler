import sys
import os
#sys.path.append('/home/tanderson/Projects/dmc/dmc/')
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import csfgen as proj
import vec
from vec import Det, Vec, Config
#from csf import compute_csfs 

def make_csf_info(max_open, twice_s, twice_sz):
    '''
    Generate eigenfunctions of S and Sz w/ desired eigenvalues.
        1. iter through nopen vals (1, 3, 5, ...nopen)
        1.5. Generate indexing of dets (depends on nup)
        2. make a dummy config and use proj method to find csfs
        3. add coeffs to CSF info
    '''
    min_open = twice_s if twice_s != 0 else 2
    csf_info = {}
    if (twice_s == 0):
        csf_info[0] = ([[1]], [''])
    for nopen in range(min_open, max_open+1, 2): 
        nup = round((nopen + twice_sz)/2)
        config = Config.fromorbs([n+1 for n in range(nopen)], [])
        print("Computing csfs for n = %d" % (nopen))
        csfs = compute_csfs(config, twice_s, twice_sz, 'projection')
        print("Done computing csfs.")
        occ_strs = list(combs(nopen, nup))
        index_dict = {comb:n for n, comb in enumerate(occ_strs)}
        csfs_coefs = []
        for csf in csfs:
            coef_list = [0]*len(occ_strs)
            for det in csf.dets:
                coef = csf.dets[det]
                index = index_dict[det2open_occ_str(det)]
                coef_list[index] = coef
            csfs_coefs.append(coef_list)
        csf_info[nopen] = (csfs_coefs, occ_strs)
    return csf_info 

def load_csf_info(max_open, twice_s, twice_sz):
    cache_name = "csfinfo" + "_2s" + str(twice_s) + "_2sz" + str(twice_sz) + ".dat"
    try:
        f = open(cache_name, 'r')
    except:
        csf_info = make_csf_info(max_open, twice_s, twice_sz)
        save_csf_info(cache_name, csf_info, twice_s, twice_sz)
        return csf_info
    cache = f.read()
    csf_info, loaded_max, loaded_twice_s, loaded_twice_sz = parse_cache(cache)
    if (loaded_twice_s != twice_s or loaded_twice_sz != twice_sz):
        raise Exception("s & sz values loaded from csfinfo file " +
                "do not match desired values.")
    if (loaded_max < max_open):
        raise Exception("csfinfo file does not contain CSFs for " +
                "desired number of open shells.")
    return csf_info

def parse_cache(cache):
    cache = cache.split('\n\n\n')
    loaded_max, loaded_twice_s, loaded_twice_sz = \
            (int(dat) for dat in cache[0].split(' '))
    csf_info = {}
    for n_cache in cache[1:]:
        nopen, det_strs = None, None
        coefs = []
        for n, line in enumerate(n_cache.split('\n')):
            if (n == 0):
                nopen = int(line)
            elif (n == 1):
                det_strs = line.split(', ')
            elif (not line):
                continue
            else:
                coef = [float(dat) for dat in line.split(', ')]
                coefs.append(coef)
        assert(nopen != None and det_strs != None)
        csf_info[nopen] = (coefs, det_strs)
    return csf_info, loaded_max, loaded_twice_s, loaded_twice_sz

def save_csf_info(cache_name, csf_info, twice_s, twice_sz):
    try:
        f = open(cache_name, 'r')
        raise Exception('File \'' + cache_name + '\' already exists.' +
                'save_csf_info will not overwrite it.')
    except:
        f = open(cache_name, 'w+')
        max_open = max([key for key in csf_info])
        f.write("%d %d %d\n" % (max_open, twice_s, twice_sz))
        for key in csf_info:
            csfs_coefs, det_list = csf_info[key]
            det_strs = ', '.join(det_list)
            f.write('\n\n')
            f.write(str(key) + '\n')
            f.write(det_strs + '\n')
            for csf_coefs in csfs_coefs:
                f.write(', '.join([str(c) for c in csf_coefs]) + '\n')
        f.close()

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
    

def config2csfs(config, csf_info, rel_parity = False):
    nopen = len([orb for orb in config.occs if config.occs[orb] == 1])
    csf_coefs, index2occs = csf_info[nopen]
    dets = [make_det(occs, config) for occs in index2occs]
    csfs = []
    for m, coefs in enumerate(csf_coefs):
        csf = Vec.zero()
        for n, coef in enumerate(coefs):
            p = parity(det) if rel_parity else 1
            det = dets[n] 
            csf += p*coef*det 
        csfs.append(csf)
    return csfs

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

if __name__ == "__main__":
#    csf_info = make_csf_info(2, 0, 0)
    csf_info = load_csf_info(8, 0, 0)
    config = Config.fromorbs([3, 4, 7, 8], [1, 2, 5, 6])
    config2 = Config.fromorbs([1, 2], [1, 3])
    csfs = config2csfs(config, csf_info)
    csfs2 = config2csfs(config2, csf_info)
    print('ncsf: ', len(csfs))
    print(csfs2)
