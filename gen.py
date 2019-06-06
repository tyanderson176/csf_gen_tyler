import sys
#sys.path.append('/home/tanderson/Projects/dmc/dmc/')
from vec import Det, Vec, Config
from csf import compute_csfs 

def make_csf_info(max_open, s, sz):
    '''
    Generate eigenfunctions of S and Sz w/ desired eigenvalues.
        1. iter through nopen vals (1, 3, 5, ...nopen)
        1.5. Generate indexing of dets (depends on nup)
        2. make a dummy config and use proj method to find csfs
        3. add coeffs to CSF info
    '''
    min_open = round(2*s) if s != 0 else 2
    csf_info = {}
    for nopen in range(min_open, max_open+1, 2): 
        nup = round(nopen/2 + sz)
        config = Config.fromorbs([n+1 for n in range(nopen)], [])
        print("Computing csfs for n = %d" % (nopen))
        csfs = compute_csfs(config, s, sz, 'projection')
        print("Done computing csfs.")
        occ_strs = list(combs(nopen, nup))
        index_dict = {comb:n for n, comb in enumerate(occ_strs)}
        csfs_coefs = []
        for csf in csfs:
            coef_list = [0]*len(occ_strs)
            for det, coef in zip(csf.dets, csf.coeffs):
                index = index_dict[det2open_occ_str(det)]
                coef_list[index] = coef
            csfs_coefs.append(coef_list)
        csf_info[nopen] = (csfs_coefs, occ_strs)
    return csf_info 

def load_csf_info(max_open, s, sz):
    cache_name = "csfinfo_" + "s" + str(s) + "_sz" + str(sz) + ".dat"
    try:
        f = open(cache_name, 'r')
    except:
        csf_info = make_csf_info(max_open, s, sz)
        save_csf_info(cache_name, csf_info, s, sz)
        return csf_info
    cache = f.read()
    csf_info, loaded_max, loaded_s, loaded_sz = parse_cache(cache)
    if (loaded_s != s or loaded_sz != sz):
        raise Exception("s & sz values loaded from csfinfo file " +
                "do not match desired values.")
    if (loaded_max < max_open):
        raise Exception("csfinfo file does not contain CSFs for " +
                "desired number of open shells.")
    return csf_info

def parse_cache(cache):
    cache = cache.split('\n\n\n')
    loaded_max, loaded_s, loaded_sz = \
            (int(dat) for dat in cache[0].split(' '))
    csf_info = {}
    for n_cache in cache[1:]:
        nopen, det_strs = None, None
        coefs = []
        for n, line in enumerate(n_cache.split('\n')):
            if (not line):
                continue
            elif (n == 0):
                nopen = int(line)
            elif (n == 1):
                det_strs = line.split(', ')
            else:
                coef = [float(dat) for dat in line.split(', ')]
                coefs.append(coef)
        assert(nopen and det_strs)
        csf_info[nopen] = (coefs, det_strs)
    return csf_info, loaded_max, loaded_s, loaded_sz

def save_csf_info(cache_name, csf_info, s, sz):
    try:
        f = open(cache_name, 'r')
        raise Exception('File \'' + cache_name + '\' already exists.' +
                'save_csf_info will not overwrite it.')
    except:
        f = open(cache_name, 'w+')
        max_open = max([key for key in csf_info])
        f.write("%d %d %d\n" % (max_open, s, sz))
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
    for coefs in csf_coefs:
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

if __name__ == "__main__":
#    csf_info = make_csf_info(2, 0, 0)
    csf_info = load_csf_info(8, 0, 0)
    config = Config.fromorbs([3, 4, 7, 8], [1, 2, 5, 6])
    config2 = Config.fromorbs([1, 2], [1, 3])
    csfs = config2csfs(config, csf_info)
    csfs2 = config2csfs(config2, csf_info)
    print('ncsf: ', len(csfs))
    print(csfs2)
