import numpy
from numpy import linalg as la
import math
from scipy.sparse import csr_matrix
import scipy.sparse as sparse

import shci4qmc.src.csf_rf as rf
import shci4qmc.src.vec as vec
import shci4qmc.src.gen as gen
import shci4qmc.lib.load_wf as load_wf
from shci4qmc.src.ham import Ham

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

    def get_csf_info(self, wf_filename):
        #Load serialized SHCI wavefunction
        wfn = load_wf.load(wf_filename)
        wf_tol = self.config['wf_tol']
        trunc_wf = [(coef, vec.Det([orb+1 for orb in up], [orb+1 for orb in dn]))
                    for coef, [up, dn] in zip(wfn['coefs'], wfn['dets']) if abs(coef) > wf_tol]
        dets, wf_coeffs = [det for coef, det in trunc_wf], [coef for coef, det in trunc_wf]
        if self.symmetry in ('DOOH', 'COOV'):
            self.real_or_imag_part(dets[0])
        csfs, config_labels = self.get_csfs(dets)

        print('ORIG SHCI WF LENGTH: ', len(dets))
        print('Init CSFS: ')
        for n, csf in enumerate(csfs):
            print("CSF ", n)
            for det, coef in csf.dets.items():
                print(det, '%.8f'%coef)
            print()

        g = rf.CSF_Generator(wf_filename, self.mol, self.mf, wf_tol, self.target_l2)
        csfs_rf = g.generate()
        '''
        print('RF CSFS: ')
        for n, csf in enumerate(csfs_rf):
            print("CSF ", n)
            for det, coef in csf.dets.items():
                print(det, '%.8f'%coef)
            print()
        '''
        for n, (csf_rf, csf) in enumerate(zip(csfs_rf, csfs)):
            print("CSF ", n)
            for (det_rf, coef_rf), (det, coef) in zip(csf_rf.dets.items(), csf.dets.items()):
                print(det_rf, '%.8f'%coef_rf, det, '%.8f'%coef)
            diff = vec.Vec.zero()
            diff += csf
            diff += -1*csf_rf
            if (diff.norm() > 1e-8):
                print("Warning! diff= ", diff.norm())
        assert(False)

#       Convert to real wf if molecule has linear symm
        if self.symmetry in ('DOOH', 'COOV'):
            dets, wf_coeffs = self.convert_wf(dets, wf_coeffs)
        det_indices, ovlp = self.csf_matrix(csfs, self.get_det_indices(dets, wf_coeffs))
        wf_det_coeffs = self.reindex_det_coeffs(det_indices, wf_coeffs, dets)
        wf_csf_coeffs = self.matrix_mul(ovlp, wf_det_coeffs)

        #Sort/Rotate CSFS
        csfs, config_labels, wf_csf_coeffs = self.truncate_csfs(
            csfs, config_labels, wf_csf_coeffs)
        print("num csfs before rotate: ", len(csfs))
        csfs, config_labels, wf_csf_coeffs = self.rotate_csfs(
            csfs, config_labels, wf_csf_coeffs, reduce_csfs = self.reduce_csfs)
        csfs, config_labels, wf_csf_coeffs = self.sorted_csfs(
            csfs, config_labels, wf_csf_coeffs)
        print("num csfs after rotate: ", len(csfs))

        det_indices = self.det_indices_from_csfs(csfs, IndexList([]))

        #Find Error
        perr = self.get_proj_error(ovlp, wf_det_coeffs)
        err = self.get_error(dets, wf_coeffs, csfs, wf_csf_coeffs)
        print('projection err: %.10f' % perr)
        print('total err: %.10f'% err, ' (includes proj. err + csf_tol err + rotation err)')
        if self.config['project_l2']:
            #self.print_important_dets(dets, wf_coeffs, csfs, wf_csf_coeffs)
            err_shci, err_proj = self.get_eigen_error(dets, wf_coeffs, csfs, wf_csf_coeffs)
            print('SHCI eigen err: %.10f'% err_shci)
            print('Proj eigen err: %.10f'% err_proj)
            l2_shci, l2_proj = self.l2_expectation(dets, wf_coeffs, csfs, wf_csf_coeffs)
            print('SHCI l2: %.10f'% l2_shci)
            print('Proj l2: %.10f'% l2_proj)
        csfs_data = [
            [(det_indices.index(d), csf.dets[d]) for d in csf.dets] for csf in csfs]

        filename = "FCIDUMP" if self.symmetry not in ('DOOH', 'COOV') else "FCIDUMP_real_orbs"
        self.ham = Ham(filename) 
        init_wf = self.sum_states(dets, wf_coeffs)
        init_e = self.ham.expectation(init_wf)
        print("Init_energy: %12.8f"% init_e)
        csfs_wf = self.sum_states(csfs, wf_csf_coeffs)
        csfs_e = self.ham.expectation(csfs_wf)
        print("Csfs_energy: %12.8f"% csfs_e)

        self.det_config_labels = self.get_det_energy_labels(det_indices)
        self.wf_csf_coeffs = wf_csf_coeffs
        self.csf_data = csfs_data
        self.config_data = config_labels
        self.det_data = det_indices
        self.err = err
        return

    def sum_states(self, states, coefs):
        res = vec.Vec.zero()
        for state, coef in zip(states, coefs):
            res += coef*state
        return res

    def sorted_csfs(self, csfs, config_labels, wf_csf_coeffs):
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
            print('\nDetected non-orthogonal csfs after rotation\n.')
            print("ORIGINAL CSF MATRIX:")
            matrix = numpy.array([self.get_coeffs(csf, det_indices) for coef, csf in original_csfs])
            print('Coefs: ', ''.join(["%12.8e"%c for c, csf in original_csfs]))
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
        csfs = self.configs2csfs(csf_data, configs)
        config_labels = [csf.config_label for csf in csfs]
        #self.save_l2_matrix(csfs)
        return csfs, config_labels

    def save_l2_matrix(self, csfs):
        l2matrix = self.L2projector.L2_matrix(csfs)
        with open('l2_matrix.txt', 'w') as mat_file:
            for row in l2matrix:
                mat_file.write('\t'.join([str(c) for c in row]) + '\n')

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
            return numpy.dot(err_vec, err_vec)/numpy.dot(wf_det_coeffs, wf_det_coeffs)
        elif self.proj_matrix_rep == 'sparse':
            err_op = sparse.identity(wf_det_coeffs.shape[0]) - ovlp.T*ovlp 
            err_vec = err_op*wf_det_coeffs
            err = err_vec.T*err_vec/(wf_det_coeffs.T*wf_det_coeffs)
            return err[0,0]
        else:
            raise Exception('Unknown matrix rep \''+ self.proj_matrix_rep + '\' in get_proj_error')

    def print_important_dets(self, dets, det_coeffs, csfs, csf_coeffs):
        wf_shci, wf_proj = vec.Vec.zero(), vec.Vec.zero()
        for det, coef in zip(dets, det_coeffs):
            wf_shci += coef*det
        for csf, coef in zip(csfs, csf_coeffs):
            wf_proj += coef*csf
        raw_proj = self.L2projector.project(self.target_l2, wf_shci)
        larger_proj_coef = lambda det, coef: -abs(coef)
        all_dets = set(list(wf_proj.dets.keys()) + list(wf_shci.dets.keys()))
        important_dets = sorted(all_dets, key = lambda det: -abs(wf_proj.dets.get(det, 0.)))
        print('\nPrinting most important dets:')
        print("Det:\t\t\tProjected Coef\tSHCI Coef\tDiff1\t Raw SHCI Proj Coef\t Diff2")
        for det in important_dets:
            coef = wf_proj.dets.get(det, 0.)
            shci_coef = wf_shci.dets.get(det, 0.)
            proj_coef = raw_proj.dets.get(det, 0.)
            diff1 = abs(coef - shci_coef)
            diff2 = abs(proj_coef - shci_coef)
            print(det, ": %12.8f\t %12.8f\t %12.8f\t %12.8f\t %12.8f"% 
                  (coef, shci_coef, diff1, proj_coef, diff2))
        wf_proj += -1*wf_shci
        print("WF Norm diff: %12.8f"% wf_proj.norm())
        print()
        return

    def get_error(self, dets, det_coeffs, csfs, csf_coeffs):
        wf_diff = vec.Vec.zero()
        for det, coef in zip(dets, det_coeffs):
            wf_diff += coef*det
        wf_norm = wf_diff.norm()
        for csf, coef in zip(csfs, csf_coeffs):
            wf_diff += -1*coef*csf
        return (wf_diff.norm()/wf_norm)**2

    def get_eigen_error(self, dets, det_coeffs, csfs, csf_coeffs):
        wf_shci, wf_proj = vec.Vec.zero(), vec.Vec.zero()
        for csf, coef in zip(csfs, csf_coeffs):
            wf_proj += coef*csf
        for det, coef in zip(dets, det_coeffs):
            wf_shci += coef*det
        shci_eigen_error = self.L2projector.eigen_error(self.target_l2, wf_shci)
        proj_eigen_error = self.L2projector.eigen_error(self.target_l2, wf_proj)
        return shci_eigen_error, proj_eigen_error
    
    def l2_expectation(self, dets, det_coeffs, csfs, csf_coeffs):
        wf_shci, wf_proj = vec.Vec.zero(), vec.Vec.zero()
        for csf, coef in zip(csfs, csf_coeffs):
            wf_proj += coef*csf
        for det, coef in zip(dets, det_coeffs):
            wf_shci += coef*det
        shci_l2 = self.L2projector.L2_expectation(wf_shci)
        proj_l2 = self.L2projector.L2_expectation(wf_proj)
        return shci_l2, proj_l2

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
    
    def truncate_csfs(self, csfs, config_labels, wf_csf_coeffs):
        csf_tol = self.config['csf_tol']
        idx = set(n for n, coef in enumerate(wf_csf_coeffs) if abs(coef) > csf_tol)
        trunc_csfs = [csf for n, csf in enumerate(csfs) if n in idx]
        trunc_labels = [label for n, label in enumerate(config_labels) if n in idx]
        trunc_coeffs = [coef for n, coef in enumerate(wf_csf_coeffs) if n in idx]
        return trunc_csfs, trunc_labels, trunc_coeffs

    def rotate_csfs(self, csfs, config_labels, wf_csf_coeffs, reduce_csfs = True):
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
            if len(config_csfs) != len(rotated_csfs):
                print('Warning: len(in_csfs) = %d != len(rot_csfs) = %d'% (len(config_csfs), len(rotated_csfs)))
            csfs += rotated_csfs
            wf_csf_coeffs += rotated_coefs
            config_labels += [label for csf in rotated_csfs]
            orth_err += self.check_orthonormal(rotated_csfs, config_csfs)
        print('ORTHONORMAL ERROR: ' + str(orth_err)) 
        return csfs, config_labels, wf_csf_coeffs

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
