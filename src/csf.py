import numpy
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/csfgen'))
import csfgen as proj
import vec
import gen
import symm as sy
import wf
from scipy.sparse import csr_matrix
import scipy.sparse as sparse

tol = 1e-15
#coef_tol = 1e-5

def get_det_info(dmc, orbsym, wf_filename, cache = False, rep = 'sparse'):
    wfn = wf.load(wf_filename)
#    dets = [vec.Det([o+1 for o in up], [o+1 for o in dn]) 
#            for n, [up, dn] in enumerate(wfn['dets']) if abs(wfn['coefs'][n]) > eps]
    wf_tol = dmc.config['wf_tol']
    trunc_wf = [(coef, vec.Det([orb+1 for orb in up], [orb+1 for orb in dn]))
                for coef, [up, dn] in zip(wfn['coefs'], wfn['dets']) if abs(coef) > wf_tol]
    dets, wf_coeffs = [det for coef, det in trunc_wf], [coef for coef, det in trunc_wf]
    wf_coeffs = normalize(wf_coeffs)
    twice_s = wfn['n_up'] - wfn['n_dn']
#    dmc.symmetry = 'D2H'
    if dmc.symmetry in ('DOOH', 'COOV'):
        sy.setup_dmc(dmc, orbsym, dets)
    csfs = get_csfs(dmc, dets, twice_s, 'projection', cache)
    det_indices, ovlp = csf_matrix(csfs, rep)
#    Convert to real wf if molecule has linear symm
    if dmc.symmetry in ('DOOH', 'COOV'):
        dets, wf_coeffs = sy.convert_wf(dmc, dets, wf_coeffs)
    wf_det_coeffs = get_det_coeffs(det_indices, wf_coeffs, dets, rep)
    wf_csf_coeffs = matrix_mul(ovlp, wf_det_coeffs, rep)
    perr = get_proj_error(ovlp, wf_det_coeffs, det_indices, rep)
    err = get_error(dets, wf_coeffs, csfs, wf_csf_coeffs)
    print('perr: %.10f, err: %.10f' % (perr, err))
#    err2 = get_bf_error(wf_det_coeffs, det_indices, wf_csf_coeffs, csfs)
    if rep == 'sparse':
        wf_csf_coeffs = wf_csf_coeffs.toarray()
    csfs_info = [
            [(det_indices.index(d), csf.dets[d]) for d in csf.dets] for csf in csfs]

    #test_csf_wf = sorted([(coef, csf) for coef, csf in zip(wf_csf_coeffs, csfs_info)],
    #                     key = lambda pair: -abs(pair[0]))
    #itod = {index:det for det, index in det_indices.indices.items()}
    #for n, (coef, csf_info) in enumerate(test_csf_wf):
    #    print('-------------------')
    #    print('CSF #%d:' % (n,))
    #    for det_index, det_coef in csf_info:
    #        print('\t' + str(coef*det_coef) + ' ' + str(itod[det_index]))

    return wf_csf_coeffs, csfs_info, det_indices, err

def get_det_info2(dmc, orbsym, wf_filename, cache=False, rep='sparse'):
    wfn = wf.load(wf_filename)
    wf_tol = dmc.config['wf_tol']
    trunc_wf = [(coef, vec.Det([orb+1 for orb in up], [orb+1 for orb in dn]))
                for coef, [up, dn] in zip(wfn['coefs'], wfn['dets']) if abs(coef) > wf_tol]
    dets, wf_coeffs = [det for coef, det in trunc_wf], [coef for coef, det in trunc_wf]
    wf_coeffs = normalize(wf_coeffs)
    twice_s = wfn['n_up'] - wfn['n_dn']
    if dmc.symmetry in ('DOOH', 'COOV'):
        sy.setup_dmc(dmc, orbsym, dets)
#    csfs = get_csfs(dmc, dets, twice_s, 'projection', cache)
#    Convert to real wf if molecule has linear symm
    if dmc.symmetry in ('DOOH', 'COOV'):
        dets, wf_coeffs = sy.convert_wf(dmc, dets, wf_coeffs)

    wavefunc = vec.Vec.zero()
    for det, coef in zip(dets, wf_coeffs):
      wavefunc += coef*det
    csfs = [wavefunc]

    det_indices, ovlp = csf_matrix(csfs, rep)
    wf_det_coeffs = get_det_coeffs(det_indices, wf_coeffs, dets, rep)
    wf_csf_coeffs = matrix_mul(ovlp, wf_det_coeffs, rep) 
    err = get_error(dets, wf_coeffs, csfs, wf_csf_coeffs)
    print('err: %.10f' % err)
    if rep == 'sparse':
        wf_csf_coeffs = wf_csf_coeffs.toarray()
    csf_info = [
            [(det_indices.index(d), csf.dets[d]) for d in csf.dets] for csf in csfs]
    return wf_csf_coeffs, csf_info, det_indices, err
    

def normalize(coeffs):
    norm = math.sqrt(sum([c**2 for c in coeffs]))
    return [c/norm for c in coeffs]

def get_det_coeffs(det_indices, wf_coeffs, dets, rep='dense'):
    det_coeffs = numpy.zeros(len(det_indices))
    for det, coeff in zip(dets, wf_coeffs):
        index = det_indices.index(det)
        det_coeffs[index] = coeff
    if rep == 'sparse':
        det_coeffs = csr_matrix(det_coeffs).T
    return det_coeffs

def get_csfs(dmc, dets, twice_s, method='projection', cache=False):
    twice_sz = get_2sz(dets)
    if (twice_sz != twice_s):
        raise Exception("CSFs only saved for sz = s. Cannot find CSFs with " +
                "s = " + str(twice_s/2.) + " and sz = " + str(twice_sz/2.) + 
                " in get_csfs.")
    configs = set(vec.Config(det) for det in dets)
    max_open = max([config.num_open for config in configs])
    csfs = []
    if cache:
        print("Loading CSF data...\n");
        csf_data = gen.load_csf_file(max_open, twice_s)
        print("Converting configs...\n");
        csfs = gen.configs2csfs(dmc, csf_data, configs, rel_parity=False)
    else:
        for config in configs: 
            csfs += gen.compute_csfs(config, twice_s, twice_sz, method)
    return csfs

def matrix_mul(ovlp, wf_det_coeffs, rep = 'dense'):
    if (rep == 'dense'):
        return numpy.dot(ovlp, wf_det_coeffs)
    elif (rep == 'sparse'):
        return ovlp*wf_det_coeffs
    else:
        raise Exception('Unknown matrix rep \'' + rep + '\' in matrix_mul')

def get_proj_error(ovlp, wf_det_coeffs, dis, rep = 'dense'):
    if rep == 'dense':
        err_op = numpy.identity(len(wf_det_coeffs)) - numpy.dot(ovlp.T, ovlp)
        err_vec = numpy.dot(err_op, wf_det_coeffs)
        return numpy.dot(err_vec, err_vec)/numpy.dot(wf_det_coeffs, wf_det_coeffs)
    elif rep == 'sparse':
        err_op = sparse.identity(wf_det_coeffs.shape[0]) - ovlp.T*ovlp 
        err_vec = err_op*wf_det_coeffs
        err = err_vec.T*err_vec/(wf_det_coeffs.T*wf_det_coeffs)
        return err[0,0]
    else:
        raise Exception('Unknown matrix rep \''+ rep + '\' in get_proj_error')

def get_error(dets, det_coeffs, csfs, csf_coeffs):
    wf_diff = vec.Vec.zero()
    for det, coef in zip(dets, det_coeffs):
        wf_diff += coef*det
    for csf, [coef] in zip(csfs, csf_coeffs.toarray()):
        wf_diff += -1*coef*csf
    dnorm = wf_diff.norm()
    return dnorm*dnorm

def get_2sz(dets):
    _2sz_vals = set(round(2*det.get_Sz()) for det in dets)
    if len(_2sz_vals) > 1:
        raise Exception("Different sz values in dets")
    for _2sz in _2sz_vals:
        return _2sz

def csf_matrix(csfs, rep = 'dense'):
    det_indices = IndexList()
    for csf in csfs:
        for det in csf.dets:
            det_indices.add(det)
    if rep == 'dense':
        matrix = numpy.array([get_coeffs(csf, det_indices) for csf in csfs])
        return det_indices, matrix
    elif rep == 'sparse':
        coefs, rows, cols = [], [], []
        for n, csf in enumerate(csfs):
            norm = csf.norm()
            for det in csf.dets:
                rows.append(n)
                cols.append(det_indices.index(det))
                coefs.append(csf.dets[det]/norm)
        matrix = csr_matrix((coefs, (rows, cols)))
        return det_indices, matrix
    else:
        raise Exception('Unknown rep \'' + rep + '\' in csf_matrix')

def get_coeffs(csf, det_indices):
    coeffs = numpy.zeros(len(det_indices))
    for det in csf.dets:
        coeff = csf.dets[det]
        index = det_indices.index(det)
        coeffs[index] = coeff
    return coeffs/csf.norm()

class IndexList:
    def __init__(self):
        self.indices = {}

    def add(self, obj):
        sz = len(self)
        if obj not in self.indices:
            self.indices[obj] = sz

    def index(self, obj):
        return self.indices[obj]

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return str(self.indices)

#if __name__ == '__main__':
