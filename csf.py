import numpy
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import csfgen as proj
import vec
import gen
from scipy.sparse import csr_matrix
import scipy.sparse as sparse

tol = 1e-15
#coef_tol = 1e-5

def get_det_info(shci_out, cache = False, rep = 'sparse'):
    s2, det_strs, wf_coeffs = shci_out
    wf_coeffs = normalize([float(coeff) for coeff in wf_coeffs.split()])
    #estimate S; use <S^2> = s(s+1); S = 1/2 +- sqrt(<S^2 + 1/4>)
    twice_s = round(2*math.sqrt(s2 + 0.25) - 1)
    dets = det_strs2dets(det_strs)
    csfs = get_csfs(dets, twice_s, 'projection', cache)
    #if point_group in ('Coov', 'Dooh'):
    #    csfs = linsymm.symmetrize(csfs)
    det_indices, ovlp = csf_matrix(csfs, rep)
    wf_det_coeffs = get_det_coeffs(det_indices, wf_coeffs, dets, rep)
    wf_csf_coeffs = matrix_mul(ovlp, wf_det_coeffs, rep)
#    wf_csf_coeffs_tol = [coef for coef in wf_csf_coeffs.toarray() 
#            if coef > coef_tol]
    csf_info = [[(det_indices.index(d), csf.dets[d]) for d in csf.dets] 
        for csf in csfs]
#    csf_info_tol = [[(det_indices.index(d), csf.dets[d]) for d in csf.dets] 
#        for n, csf in enumerate(csfs) if wf_csf_coeffs.toarray()[n] > coef_tol]
    err = get_proj_error(ovlp, wf_det_coeffs, rep)
    if rep == 'sparse':
        wf_csf_coeffs = wf_csf_coeffs.toarray()
    return wf_csf_coeffs, csf_info, det_indices, err

def normalize(coeffs):
    norm = math.sqrt(sum([c**2 for c in coeffs]))
    return [c/norm for c in coeffs]
    
def get_det_coeffs(det_indices, wf_coeffs, dets, rep='dense'):
#    wf_coeffs = [float(coeff) for coeff in wf_coeffs.split()]
    det_coeffs = numpy.zeros(len(det_indices))
    for det, coeff in zip(dets, wf_coeffs):
        index = det_indices.index(det)
        det_coeffs[index] = coeff
    if rep == 'sparse':
        det_coeffs = csr_matrix(det_coeffs).T
    return det_coeffs

def get_csfs(dets, twice_s, method='projection', cache=False):
    twice_sz = get_2sz(dets)
    if (twice_sz != twice_s):
        raise Exception("CSFs only saved for sz = s. Cannot find CSFs with " +
                "s = " + str(twice_s/2.) + " and sz = " + str(twice_sz/2.) + 
                " in get_csfs.")
    configs = set(vec.Config(det) for det in dets)
    max_open = max([config.num_open for config in configs])
    csfs = []
    if cache:
        print("Loading CSF data...\n\n");
        csf_data = gen.load_csf_file(max_open, twice_s)
        print("Converting configs...\n\n");
        for n, config in enumerate(configs):
            csfs += gen.config2csfs(config, csf_data, rel_parity=False)
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

def get_proj_error(ovlp, wf_det_coeffs, rep = 'dense'):
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

def det_strs2dets(det_strs):
    #TODO: give a format string i.e. "up_occs     dn_occs"
    #or "up_occs\t\t\tdn_occs"
    dets = []
    for det_str in det_strs:
        up_str, dn_str = tuple(det_str.split('     '))
        up_occs = [int(orb)+1 for orb in up_str.split()]
        dn_occs = [int(orb)+1 for orb in dn_str.split()]
        dets.append(vec.Det(up_occs, dn_occs))
    return dets

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
