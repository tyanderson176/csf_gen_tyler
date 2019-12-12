import numpy
from numpy import linalg as la
import math
from scipy.sparse import csr_matrix
import scipy.sparse as sparse

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../lib'))

import vec
import gen
import symm as sy
import load_wf

class CsfMethods():
    def __init__(self):
        self.proj_matrix_rep = 'sparse'
        self.reduce_csfs = self.config['reduce_csfs']

        self.det_config_labels = None
        self.wf_csf_coeffs = None
        self.csf_data = None
        self.config_data = None
        self.det_data = None
        self.err = None

    def tot_lz(self, det):
        return sum([self.orb2lz[orb-1] for orb in det.up_occ + det.dn_occ])

    def get_csf_info(self, wf_filename):
        #Load serialized SHCI wavefunction
        wfn = load_wf.load(wf_filename)
        wf_tol = self.config['wf_tol']
        trunc_wf = [(coef, vec.Det([orb+1 for orb in up], [orb+1 for orb in dn]))
                    for coef, [up, dn] in zip(wfn['coefs'], wfn['dets']) if abs(coef) > wf_tol]
        dets, wf_coeffs = [det for coef, det in trunc_wf], [coef for coef, det in trunc_wf]
#        self.print_related_configs(dets, wf_coeffs)
#        wf_coeffs = self.normalize(wf_coeffs)

#        lz_dict = {}
#        for det in dets:
#            lz = self.tot_lz(det)
#            if lz not in lz_dict:
#                lz_dict[lz] = []
#            lz_dict[lz].append(det)
#        for lz, dets in lz_dict.items():
#            print(lz)
#            for det in dets:
#                print(det)
         
        if self.symmetry in ('DOOH', 'COOV'):
            self.real_or_imag_part(dets[0])

        csfs, config_labels = self.get_csfs(dets)
#       Convert to real wf if molecule has linear symm
        if self.symmetry in ('DOOH', 'COOV'):
            dets, wf_coeffs = self.convert_wf(dets, wf_coeffs)
        det_indices, ovlp = self.csf_matrix(csfs, self.get_det_indices(dets, wf_coeffs))
        wf_det_coeffs = self.reindex_det_coeffs(det_indices, wf_coeffs, dets)
        wf_csf_coeffs = self.matrix_mul(ovlp, wf_det_coeffs)

#        coef_array = [coef for [coef] in wf_det_coeffs.toarray()]
#        assert(len(det_indices) == len(coef_array))
#        print("Det coefs, original:\n")
#        sorted_dets = sorted([(det, coef_array[n]) for det, n in det_indices.indices.items()],
#                             key = lambda det_coef : -abs(det_coef[1]))
#        for det, coef in sorted_dets:
#            if (abs(coef) > 1e-10):
#                print(det, ": ", coef) 
#        shci_wf = vec.Vec.zero()
#        for csf, coef in zip(csfs, wf_csf_coeffs):
#            shci_wf += coef*csf
#        sorted_dets_wf = sorted([(det, coef) for det, coef in shci_wf.dets.items()],
#                                key = lambda det_coef : -abs(det_coef[1]))
#        print("Det coefs, after projection:")
#        for det, coef in sorted_dets_wf:
#            if (abs(coef) > 1e-10):
#                print(det, ": ", coef) 
#        print('CSFS, wf_csf_coeffs:')
#        for csf, coef, row in zip(csfs, wf_csf_coeffs, ovlp.toarray()):
#            if (abs(coef) > 1e-10):
#                print('--------\n', csf, '\n', coef)
#                print('row: ', ' '.join([str(elem) for elem in row]))
#                print('\n')

        #Sort/Rotate CSFS
        print("num csfs before rotate: ", len(csfs))
        csfs, config_labels, wf_csf_coeffs = self.rotate_csfs(
            csfs, config_labels, wf_csf_coeffs, reduce_csfs = self.reduce_csfs)
        csfs, config_labels, wf_csf_coeffs = self.sorted_csfs(
            csfs, config_labels, wf_csf_coeffs)
        print("num csfs after rotate: ", len(csfs))

        det_indices = self.det_indices_from_csfs(csfs, IndexList([]))

#        shci_wf = vec.Vec.zero()
#        for coef, det in zip(wf_coeffs, dets):
#            shci_wf += coef*det
#        csfs = [shci_wf]
#        wf_csf_coeffs = [1.]
#        config_labels = [0]

#        shci_wf = vec.Vec.zero()
#        for csf, coef in zip(csfs, wf_csf_coeffs):
#            shci_wf += coef*csf
#        sorted_dets_wf = sorted([(det, coef) for det, coef in shci_wf.dets.items()],
#                                key = lambda det_coef : -abs(det_coef[1]))
#        print("Det coefs, after rotation:")
#        for det, coef in sorted_dets_wf:
#            if (abs(coef) > 1e-10):
#                print(det, ": ", coef) 
#    

        #Find Error
        perr = self.get_proj_error(ovlp, wf_det_coeffs)
        err = self.get_error(dets, wf_coeffs, csfs, wf_csf_coeffs)
        print('perr: %.10f, err: %.10f' % (perr, err))
#        print('ON Error: ' + str(self.check_orthonormal(csfs)))
#        if self.proj_matrix_rep == 'sparse':
#            wf_csf_coeffs = wf_csf_coeffs.toarray()
        csfs_data = [
            [(det_indices.index(d), csf.dets[d]) for d in csf.dets] for csf in csfs]

        self.det_config_labels = self.get_det_energy_labels(det_indices)
        self.wf_csf_coeffs = wf_csf_coeffs
        self.csf_data = csfs_data
        self.config_data = config_labels
        self.det_data = det_indices
        self.err = err
        return
#        return wf_csf_coeffs, csfs_info, config_labels, det_indices, err

    def sorted_csfs(self, csfs, config_labels, wf_csf_coeffs):
        csf_tol = self.config['csf_tol']
        sorted_csf_wf = sorted(
            [(coef, label, csf) for coef, label, csf in zip(wf_csf_coeffs, config_labels, csfs)],
             key = lambda triple: -abs(triple[0]))
        csfs = [csf for coef, label, csf in sorted_csf_wf]
        labels = [label for coef, label, csf in sorted_csf_wf] 
        coefs = [coef for coef, label, csf in sorted_csf_wf]
        return csfs, self.reindex_labels(labels), coefs

    def check_orthonormal(self, csfs, original_csfs):
        overlap_matrix = numpy.array([[csf1.dot(csf2) for csf1 in csfs] for csf2 in csfs])
        not_orthog = [(n1, csf1, n2, csf2) for n1, csf1 in enumerate(csfs) 
                      for n2, csf2 in enumerate(csfs) if csf1 != csf2 and csf1.dot(csf2) > 1e-1]
        orig_csfs = [csf for coef, csf in original_csfs]
        det_indices = self.det_indices_from_csfs(orig_csfs, IndexList([]))
        if len(not_orthog) != 0:
            print("ORIGINAL CSF MATRIX:")
            matrix = numpy.array([self.get_coeffs(csf, det_indices) for coef, csf in original_csfs])
            for row in matrix:
                for elem in row:
                    print("%8.4e\t" % elem, end = '')
                print()
            print("ROTATED CSF MATRIX:")
            matrix = numpy.array([self.get_coeffs(csf, det_indices) for csf in csfs])
            for row in matrix:
                for elem in row:
                    print("%8.4e\t" % elem, end = '')
                print()
            print("NOT ORTHOG:")
        for n1, csf1, n2, csf2 in not_orthog:
            print('----------  ' + str(csf1.dot(csf2)))
            print(n1, ' and ', n2)
#            print(csf1)
            print('........AND........')
#            print(csf2)
            print('DETS: ')
            dets = set([det for det in csf1.dets.keys()] + [det for det in csf2.dets.keys()])
            for det in dets:
                coef1 = csf1.dets[det] if det in csf1.dets else 0.
                coef2 = csf2.dets[det] if det in csf2.dets else 0.
                print(str(det) + ': ' + str(coef1) + ", " + str(coef2))
#            print('DOT MATRIX:')
#            print("[ %10.4f , %10.4f ]" % (csf1.dot(csf1), csf1.dot(csf2)))
#            print("[ %10.4f , %10.4f ]" % (csf2.dot(csf1), csf2.dot(csf2)))
        return la.norm(overlap_matrix - numpy.eye(len(csfs)))

    def reindex_labels(self, labels):
        new_labels_dict = {}
        new_labels = []
        for label in labels:
            if label not in new_labels_dict:
                new_labels_dict[label] = len(new_labels)
            new_labels.append(new_labels_dict[label])
        return new_labels

    def normalize(self, coeffs):
        norm = math.sqrt(sum([c**2 for c in coeffs]))
        return [c/norm for c in coeffs]

    def reindex_det_coeffs(self, det_indices, wf_coeffs, dets):
        det_coeffs = numpy.zeros(len(det_indices))
        for det, coeff in zip(dets, wf_coeffs):
            index = det_indices.index(det)
            det_coeffs[index] = coeff
        if self.proj_matrix_rep == 'sparse':
            det_coeffs = csr_matrix(det_coeffs).T
        return det_coeffs

    def get_csfs(self, dets):
        twice_s = self.get_2sz(dets)
        configs = set(vec.Config(det) for det in dets)
        max_open = max([config.num_open for config in configs])
        print("Loading CSF data...\n");
        csf_data = self.load_csf_file(max_open, twice_s)
        print("Converting configs...\n");
        csfs, config_labels = self.configs2csfs(csf_data, configs)
        return csfs, config_labels

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
            return [coef for [coef] in (ovlp*wf_det_coeffs).toarray()]
        else:
            raise Exception('Unknown matrix rep \'' + self.proj_matrix_rep + '\' in matrix_mul')

    def get_proj_error(self, ovlp, wf_det_coeffs):
        if self.proj_matrix_rep == 'dense':
            err_op = numpy.identity(len(wf_det_coeffs)) - numpy.dot(ovlp.T, ovlp)
            err_vec = numpy.dot(err_op, wf_det_coeffs)
            return numpy.sqrt(numpy.dot(err_vec, err_vec)/numpy.dot(wf_det_coeffs, wf_det_coeffs))
        elif self.proj_matrix_rep == 'sparse':
            err_op = sparse.identity(wf_det_coeffs.shape[0]) - ovlp.T*ovlp 
            err_vec = err_op*wf_det_coeffs
            err = err_vec.T*err_vec/(wf_det_coeffs.T*wf_det_coeffs)
            return numpy.sqrt(err[0,0])
        else:
            raise Exception('Unknown matrix rep \''+ self.proj_matrix_rep + '\' in get_proj_error')

    def get_error(self, dets, det_coeffs, csfs, csf_coeffs):
        wf_diff = vec.Vec.zero()
        for det, coef in zip(dets, det_coeffs):
            wf_diff += coef*det
        wf_norm = wf_diff.norm()
        for csf, coef in zip(csfs, csf_coeffs):
            wf_diff += -1*coef*csf
        return wf_diff.norm()/wf_norm

    def det_indices_from_csfs(self, csfs, det_indices):
        for csf in csfs:
            for det in csf.dets:
                det_indices.add(det)
        return det_indices

    def get_det_indices(self, dets, wf_coeffs):
        sorted_wf = sorted([(coef, det) for coef, det in zip(wf_coeffs, dets)],
                            key = lambda pair: -abs(pair[0]))
        return IndexList([det for coef, det in sorted_wf])

    def csf_matrix(self, csfs, det_indices):
    #    det_indices = IndexList()
#        for csf in csfs:
#            for det in csf.dets:
#                det_indices.add(det)
        det_indices = self.det_indices_from_csfs(csfs, det_indices)
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

    def rotate_csfs(self, csfs, config_labels, wf_csf_coeffs, reduce_csfs = True, tol=1e-16):
        '''
        config2csfs = {}
        for csf, coef in zip(csfs, wf_csf_coeffs):
            configs = set([vec.Config(det) for det in csf.dets])
            configs_str = ' '.join(sorted([str(config) for config in configs]))
            if configs_str not in config2csfs:
                config2csfs[configs_str] = []
            if abs(coef) > tol:
                config2csfs[configs_str].append((coef, csf))
        '''
        config2csfs = {} #dictionary mapping configs to csfs
        assert(len(wf_csf_coeffs) == len(csfs) and len(csfs) == len(config_labels))
        for csf, coef, config_label in zip(csfs, wf_csf_coeffs, config_labels):
            if config_label not in config2csfs:
                config2csfs[config_label] = []
            config2csfs[config_label].append((coef, csf))
        orth_err = 0.
        csf_tol = self.config['csf_tol']
        csfs, wf_csf_coeffs, config_labels = [], [], []
        for label, config_csfs in config2csfs.items():
            rotated_csf = vec.Vec.zero()
            for coef, csf in config_csfs:
                rotated_csf += coef*csf
            rotated_coef = rotated_csf.norm()

            csf_subspace = [rotated_csf] + [csf for coef, csf in config_csfs]
            dim = 1 if reduce_csfs else len(config_csfs)
            rotated_csfs = vec.Vec.gram_schmidt(csf_subspace, dim)
            rotated_coefs = [rotated_coef if n == 0 else 0. for n, csf in enumerate(rotated_csfs)]
            if abs(rotated_coef) > csf_tol:
                if len(config_csfs) != len(rotated_csfs):
                    print('Warning: len(in_csfs) = %d != len(rot_csfs) = %d'% (len(config_csfs), len(rotated_csfs)))
                csfs += rotated_csfs
                wf_csf_coeffs += rotated_coefs
                config_labels += [label for csf in rotated_csfs]
            orth_err += self.check_orthonormal(rotated_csfs, config_csfs)
        print('ORTHONORMAL ERROR: ' + str(orth_err)) 
        return csfs, config_labels, wf_csf_coeffs

#    def gram_schmidt(self, vecs, dim, tol):
#        orthonormal_basis = []
#        for vec in numpy.copy(vecs):
#            if len(orthonormal_basis) == dim:
#                return orthonormal_basis
#            for basis_vec in orthonormal_basis:
#                vec -= basis_vec.dot(vec) * basis_vec
#            if vec.norm() <= tol:
#                continue
#            vec /= vec.norm()
#            orthonormal_basis.append(vec)
#        return orthonormal_basis

    def sum_orb_energies(self, det):
        up_energy = sum([self.mf.mo_energy[orb-1] for orb in det.up_occ])
        dn_energy = sum([self.mf.mo_energy[orb-1] for orb in det.dn_occ])
        return up_energy + dn_energy

    def get_det_energy_labels(self, det_indices, tol = 1e-8):
        det_config_labels = {}
        det_energies = sorted([(self.sum_orb_energies(det), det) for det in det_indices.indices],
                              key = lambda energy_and_det: energy_and_det[0])
        prev_energy, label = math.inf, 0
        for energy, det in det_energies:
            if abs(energy - prev_energy) > tol:
                label += 1
            det_config_labels[det] = label
            prev_energy = energy
        return det_config_labels

    def print_related_configs(self, dets, wf_coeffs):
        def rep(config):
            rep_occs = {}
            for orb, num in config.occs.items():
                porb = self.porbs[orb]
                rep_orb = min(porb, orb)
                if rep_orb not in rep_occs:
                    rep_occs[rep_orb] = 0
                rep_occs[rep_orb] += num
            up, dn = [], []
            for orb, num in rep_occs.items():
                if num >= 1:
                    up.append(orb)
                if num >= 2:
                    dn.append(orb)
                if num >= 3:
                    up.append(self.partner_orbs[orb])
                if num >= 4:
                    up.append(self.partner_orbs[orb])
            return vec.Config(vec.Det(up, dn))
        def av_weight(config_weight, related_configs, rep):
            tot_weight = 0
            for config in related_configs[rep]:
                tot_weight += config_weight[config] 
            return tot_weight/len(related_configs[rep])
            

        related_configs = {}
        config_weight = {}
        for det, coef in zip(dets, wf_coeffs):
            config = vec.Config(det)
            if config not in config_weight:
                config_weight[config] = 0.
            config_weight[config] += coef**2
            if rep(config) not in related_configs:
                related_configs[rep(config)] = set()
            related_configs[rep(config)].update([config])
        sorted_reps = sorted(
            related_configs.keys(), 
            key = lambda rep: -av_weight(config_weight, related_configs, rep))
        for rep in sorted_reps: 
            related = related_configs[rep]
            print("--------")
            print('Size: ', len(related))
            for config in related:
                print('>>> ', config, math.sqrt(config_weight[config]))
        assert(False)
    
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

if __name__ == '__main__':
    d1 = vec.Det([1,2,3],[1,2,3])
    d2 = vec.Det([1,2,4],[1,2,3])
    d3 = vec.Det([1,2,3],[1,2,4])
    csf1 = 1*d1 + 1e-8*d2 + 1e-8*d3
    csf2 = 1*d1 + 1e-8*d2
    csf3 = 1*d1 +    0*d2 + 1e-8*d3
    csf4 = 1*d2 - 1*d3
    cm = CsfMethods()
    orthogonalized = cm.gram_schmidt([csf1, csf2, csf3, csf4], 4)
    print(orthogonalized)
