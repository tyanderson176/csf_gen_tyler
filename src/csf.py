import numpy
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../lib'))
import vec
import gen
import symm as sy
import wf
from scipy.sparse import csr_matrix
import scipy.sparse as sparse

class CsfMethods():
    def __init__(self):
        self.proj_matrix_rep = 'sparse'

    def get_csf_info(self, wf_filename):
        wfn = wf.load(wf_filename)
        wf_tol = self.config['wf_tol']
        trunc_wf = [(coef, vec.Det([orb+1 for orb in up], [orb+1 for orb in dn]))
                    for coef, [up, dn] in zip(wfn['coefs'], wfn['dets']) if abs(coef) > wf_tol]
        dets, wf_coeffs = [det for coef, det in trunc_wf], [coef for coef, det in trunc_wf]
        wf_coeffs = self.normalize(wf_coeffs)
        self.real_or_imag_part(dets[0])

        csfs = self.get_csfs(dets, abs(wfn['n_up'] - wfn['n_dn']))
#       Convert to real wf if molecule has linear symm
        if self.symmetry in ('DOOH', 'COOV'):
            dets, wf_coeffs = self.convert_wf(dets, wf_coeffs)
        det_indices, ovlp = self.csf_matrix(csfs, self.get_det_indices(dets, wf_coeffs))
        wf_det_coeffs = self.get_det_coeffs(det_indices, wf_coeffs, dets)
        wf_csf_coeffs = self.matrix_mul(ovlp, wf_det_coeffs)
#       We should have perr = err. If perr != err, there is likely an error
#       during the projection part.
        perr = self.get_proj_error(ovlp, wf_det_coeffs)
        err = self.get_error(dets, wf_coeffs, csfs, wf_csf_coeffs)
        print('perr: %.10f, err: %.10f' % (perr, err))
        if self.proj_matrix_rep == 'sparse':
            wf_csf_coeffs = wf_csf_coeffs.toarray()
        csfs_info = [
                [(det_indices.index(d), csf.dets[d]) for d in csf.dets] for csf in csfs]
        wf_csf_coeffs, csfs_info = self.sorted_csfs(wf_csf_coeffs, csfs_info, wf_tol)
        return wf_csf_coeffs, csfs_info, det_indices, err

    def sorted_csfs(self, wf_csf_coeffs, csfs_info, tol):
        sorted_csf_wf = sorted(
            [(coef, csf) for coef, csf in zip(wf_csf_coeffs, csfs_info) if abs(coef) > tol],
            key = lambda pair: -abs(pair[0]))
        return ([coef for coef, csf_info in sorted_csf_wf], 
                [csf_info for coef, csf_info in sorted_csf_wf])

    def normalize(self, coeffs):
        norm = math.sqrt(sum([c**2 for c in coeffs]))
        return [c/norm for c in coeffs]

    def get_det_coeffs(self, det_indices, wf_coeffs, dets):
        det_coeffs = numpy.zeros(len(det_indices))
        for det, coeff in zip(dets, wf_coeffs):
            index = det_indices.index(det)
            det_coeffs[index] = coeff
        if self.proj_matrix_rep == 'sparse':
            det_coeffs = csr_matrix(det_coeffs).T
        return det_coeffs

    def get_csfs(self, dets, twice_s):
        twice_sz = self.get_2sz(dets)
        if (twice_sz != twice_s):
            raise Exception("CSFs only saved for sz = s. Cannot find CSFs with " +
                    "s = " + str(twice_s/2.) + " and sz = " + str(twice_sz/2.) + 
                    " in get_csfs.")
        configs = set(vec.Config(det) for det in dets)
        max_open = max([config.num_open for config in configs])
        print("Loading CSF data...\n");
        csf_data = self.load_csf_file(max_open, twice_s)
        print("Converting configs...\n");
        csfs = self.configs2csfs(csf_data, configs)
        return csfs

    def get_2sz(self, dets):
        _2sz_vals = set(round(2*det.get_Sz()) for det in dets)
        if len(_2sz_vals) > 1:
            raise Exception("Different sz values in dets")
        for _2sz in _2sz_vals:
            return _2sz

    def matrix_mul(self, ovlp, wf_det_coeffs):
        if (self.proj_matrix_rep == 'dense'):
            return numpy.dot(ovlp, wf_det_coeffs)
        elif (self.proj_matrix_rep == 'sparse'):
            return ovlp*wf_det_coeffs
        else:
            raise Exception('Unknown matrix rep \'' + self.proj_matrix_rep + '\' in matrix_mul')

    def get_proj_error(self, ovlp, wf_det_coeffs):
        if self.proj_matrix_rep == 'dense':
            err_op = numpy.identity(len(wf_det_coeffs)) - numpy.dot(ovlp.T, ovlp)
            err_vec = numpy.dot(err_op, wf_det_coeffs)
            return numpy.dot(err_vec, err_vec)/numpy.dot(wf_det_coeffs, wf_det_coeffs)
        elif self.proj_matrix_rep == 'sparse':
            err_op = sparse.identity(wf_det_coeffs.shape[0]) - ovlp.T*ovlp 
            err_vec = err_op*wf_det_coeffs
            err = err_vec.T*err_vec/(wf_det_coeffs.T*wf_det_coeffs)
            return err[0,0]
        else:
            raise Exception('Unknown matrix rep \''+ self.proj_matrix_rep + '\' in get_proj_error')

    def get_error(self, dets, det_coeffs, csfs, csf_coeffs):
        wf_diff = vec.Vec.zero()
        for det, coef in zip(dets, det_coeffs):
            wf_diff += coef*det
        for csf, [coef] in zip(csfs, csf_coeffs.toarray()):
            wf_diff += -1*coef*csf
        dnorm = wf_diff.norm()
        return dnorm*dnorm

    def get_det_indices(self, dets, wf_coeffs):
        sorted_wf = sorted([(coef, det) for coef, det in zip(wf_coeffs, dets)],
                            key = lambda pair: -abs(pair[0]))
        return IndexList([det for coef, det in sorted_wf])

    def csf_matrix(self, csfs, det_indices):
    #    det_indices = IndexList()
        for csf in csfs:
            for det in csf.dets:
                det_indices.add(det)
        if self.proj_matrix_rep == 'dense':
            matrix = numpy.array([self.get_coeffs(csf, det_indices) for csf in csfs])
            return det_indices, matrix
        elif self.proj_matrix_rep == 'sparse':
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
            raise Exception('Unknown rep \'' + self.proj_matrix_rep + '\' in csf_matrix')

    def get_coeffs(self, csf, det_indices):
        coeffs = numpy.zeros(len(det_indices))
        for det in csf.dets:
            coeff = csf.dets[det]
            index = det_indices.index(det)
            coeffs[index] = coeff
        return coeffs/csf.norm()

class IndexList:
    def __init__(self, objects):
        self.indices = {}
        for obj in objects:
            self.add(obj)

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
