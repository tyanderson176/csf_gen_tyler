import sys
import os
import numpy as np
import cProfile
import itertools
from functools import reduce

from pyscf import gto, scf

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib'))
#sys.path.append('/home/tanderson/csfgen_andre/read_gamess')
#sys.path.append('/home/tanderson/shci4qmc/src')
#sys.path.append('/home/tanderson/shci4qmc/lib')

from shci4qmc.src.vec import Det, Vec, Config
from shci4qmc.lib.rel_parity import rel_parity_few_elec as rel_parity
import shci4qmc.src.p2d as p2d
import shci4qmc.src.sym_rhf as sym_rhf

from shci4qmc.lib.andre.determinant import Determinant
from shci4qmc.lib.andre.det_lin_comb import DeterminantLinearCombination
from shci4qmc.lib.andre.orbital import Orbital
from shci4qmc.lib.andre.csf import project_ang_mom
from shci4qmc.lib.andre.csf import build_operators
from shci4qmc.lib.andre.operators import Operator

def get_indices(atomic_orbs):
    indices = {}
    for i, orb in enumerate(atomic_orbs):
        n, l, m = orb.quant_nums
        basis_id, atom_id = orb.bvec.bv_id, orb.ia
        indices[(atom_id, basis_id, l, m)] = i
    return indices

def l_ops_atomic(atomic_orbs):
    nao = len(atomic_orbs)
    lup = np.zeros((nao, nao), dtype = np.complex64) 
    ldn = np.zeros((nao, nao), dtype = np.complex64) 
    lz  = np.zeros((nao, nao), dtype = np.complex64) 
    indices = get_indices(atomic_orbs)
    for n, orb in enumerate(atomic_orbs):
        n, l, m = orb.quant_nums
        basis_id, atom_id = orb.bvec.bv_id, orb.ia
        i = indices[(atom_id, basis_id, l, m)]
        iup = indices.get((atom_id, basis_id, l, m+1), None)
        idn = indices.get((atom_id, basis_id, l, m-1), None)
        if iup:
            lup[iup, i] = np.sqrt(l*(l+1) - m*(m+1))
        if idn:
            ldn[idn, i] = np.sqrt(l*(l+1) - m*(m-1))
        lz[i, i] = m
    #write in a 'sparse' format?
    return lup, ldn, lz

def cmplx_ao2real_ao_matrix(atomic_orbs):
    #complex atomic orbs to molecular orbs
    nao = len(atomic_orbs)
    cmplx_ao2real_ao = np.zeros((nao, nao), dtype = np.complex64)
    indices = get_indices(atomic_orbs)
    for n, orb in enumerate(atomic_orbs):
        n, l, m = orb.quant_nums
        basis_id, atom_id = orb.bvec.bv_id, orb.ia
        i1, i2 = indices[(atom_id, basis_id, l, m)], indices[(atom_id, basis_id, l, -m)]
        if m == 0:
            cmplx_ao2real_ao[i1, i1] = 1
            continue
        cmplx_ao2real_ao[i1, i1] = -1j/np.sqrt(2) if m < 0 else np.power(-1, m)/np.sqrt(2)
        cmplx_ao2real_ao[i2, i1] = 1/np.sqrt(2) if m < 0 else 1j*np.power(-1, m)/np.sqrt(2)
    return cmplx_ao2real_ao

def atomic2mol_matrix(mo_coeffs, atomic_orbs):
    #complex atomic orbs to molecular orbs
    cmplx_ao2real_ao = cmplx_ao2real_ao_matrix(atomic_orbs)
    return np.matmul(mo_coeffs, cmplx_ao2real_ao)

#def expand_orbs(orbs):
#    if not orbs:
#        return [(1., [])]
#    orb, other_orbs = orbs[0], orbs[1:]
#    others_expand = expand_orbs(other_orbs)
#    if isinstance(orb, list):
#        return [(c1*c2, [new_orb] + expanded) for c1, expanded in others_expand
#                                              for c2, new_orb in orb]
#    else:
#        return [(coef, [orb] + expanded) for coef, expanded in others_expand]

def combine_orbs(orbs):
    coef, orb_list = 1, []
    for orb in orbs:
        if isinstance(orb, tuple):
            orb_coef, orb_num = orb
            coef *= orb_coef
            orb_list.append(orb_num)
        elif isinstance(orb, (int, np.int64)):
            orb_list.append(orb)
        else:
            raise TypeError(
                "Bad 'orbs' type in combine_orbs: ", type(orb), "orbs: ", orbs)
    coef *= rel_parity(orb_list)
    return (coef, sorted(orb_list))

def expand_orbs(orbs):
    lists = [orb if isinstance(orb, list) else [orb] for orb in orbs]
    return [combine_orbs(orbs) for orbs in itertools.product(*lists)]

def get_new_orbs(op, up_orbs, dn_orbs):
    new_up_orbs, new_dn_orbs = [], []
    for n, orb in enumerate(up_orbs):
        expanded = expand_orbs(up_orbs[:n] + [op[orb]] + up_orbs[n+1:])
        new_up_orbs += expanded
    for n, orb in enumerate(dn_orbs):
        expanded = expand_orbs(dn_orbs[:n] + [op[orb]] + dn_orbs[n+1:])
        new_dn_orbs += expanded
    return new_up_orbs, new_dn_orbs

def dets_from_new_orbs(new_up_orbs, new_dn_orbs, up_orbs, dn_orbs):
    res = Vec.zero()
    for up_coef, up_occ in new_up_orbs:
        res += Vec({Det(up_occ, dn_orbs): up_coef})
#        p = rel_parity(up_occ)
#        res += up_coef*p*Det(up_occ, dn_orbs)
    for dn_coef, dn_occ in new_dn_orbs:
        res += Vec({Det(up_orbs, dn_occ): dn_coef})
#        p = rel_parity(dn_occ)
#        res += dn_coef*p*Det(up_orbs, dn_occ)
    return res
    
def apply_1body(op, state):
    if isinstance(state, Det):
        #returns a Vec/CSF after applying operator
        up_orbs, dn_orbs = state.up_occ, state.dn_occ
#        new_up_orbs, new_dn_orbs = [], []
#        for n, orb in enumerate(up_orbs):
#            expanded = expand_orbs(up_orbs[:n] + [op[orb]] + up_orbs[n+1:])
#            new_up_orbs += expanded
#        for n, orb in enumerate(dn_orbs):
#            expanded = expand_orbs(dn_orbs[:n] + [op[orb]] + dn_orbs[n+1:])
#            new_dn_orbs += expanded
        new_up_orbs, new_dn_orbs = get_new_orbs(op, up_orbs, dn_orbs)
        return dets_from_new_orbs(new_up_orbs, new_dn_orbs, up_orbs, dn_orbs)
#        res = Vec.zero()
#        for up_coef, up_occ in new_up_orbs:
#            p = rel_parity(up_occ)
#            res += up_coef*p*Det(up_occ, dn_orbs)
#        for dn_coef, dn_occ in new_dn_orbs:
#            p = rel_parity(dn_occ)
#            res += dn_coef*p*Det(up_orbs, dn_occ)
#        return res
    elif isinstance(state, Vec):
        res = Vec.zero()
        for det, coef in state.dets.items():
            res += coef*apply_1body(op, det)
        return res

def make_state(coef, up_occ, dn_occ):
    return Vec({Det(up_occ, dn_occ): coef})

#def add_det(res, up_orbs, dn_orbs, i, j):
#    up_coef, up_occ = up_orbs[i]
#    dn_coef, dn_occ = dn_orbs[j]
#    res += Vec({Det(up_occ, dn_occ): up_coef*dn_coef})
##    res += make_state(up_coef*dn_coef, up_occ, dn_occ)
##    res += (up_coef*dn_coef)*Det(up_occ, dn_occ)
#    return

def construct_state(up_orbs, dn_orbs):
    res = Vec.zero()
    for i, j in np.ndindex(len(up_orbs), len(dn_orbs)):
        up_coef, up_occ = up_orbs[i]
        dn_coef, dn_occ = dn_orbs[j]
        res += Vec({Det(up_occ, dn_occ): up_coef*dn_coef})
    return res
#    for up_orb, dn_orb in itertools.product(up_orbs, dn_orbs):
#        up_coef, up_occ = up_orb
#        dn_coef, dn_occ = dn_orb
#        res += (up_coef*dn_coef)*Det(up_occ, dn_occ)
#    for up_coef, up_occ in up_orbs:
#        for dn_coef, dn_occ in dn_orbs:
#            res += (up_coef*dn_coef)*Det(up_occ, dn_occ)

def change_basis(U, state):
    if isinstance(state, Det):
        up_orbs, dn_orbs = state.up_occ, state.dn_occ
        new_up_orbs = expand_orbs([U[orb] for orb in up_orbs])
        new_dn_orbs = expand_orbs([U[orb] for orb in dn_orbs])
        return construct_state(new_up_orbs, new_dn_orbs)
    elif isinstance(state, Vec):
        res = Vec.zero()
        for det, coef in state.dets.items():
            res += coef*change_basis(U, det)
        return res

def sparse_rep(matrix):
    tol = 1e-12
    sparse_dict = {}
    for n, column in enumerate(matrix.T):
        sparse_col = []
        for m, elem in enumerate(column):
            if abs(elem) > tol:
                sparse_col.append((elem, m+1))
        sparse_dict[n+1] = sparse_col
    return sparse_dict

def convert_op(a2m, op):
    #converts operator to molecular orbital basis (from complex atomic basis)
    #and to sparse format
    return sparse_rep(reduce(np.matmul, (np.linalg.inv(a2m.T), op, a2m.T)))

def apply_l2(lup, ldn, lz, csf):
    res = Vec.zero()
    lz_csf = apply_1body(lz, csf)
    lz2_csf = apply_1body(lz, lz_csf)
    lup_csf = apply_1body(lup, csf)
    ldn_lup_csf = apply_1body(ldn, lup_csf)
    res += ldn_lup_csf
    res += lz2_csf
    res += lz_csf
    return res

#def proj_l2(projected_out_ls, targ, lup, ldn, lz, csf):
#    #TODO: prune small coefs?
#    res = Vec.zero()
#    res += csf
#    for l in projected_out_ls:
#        print('Projecting out l=%d'%l)
#        l2_csf = apply_l2(lup, ldn, lz, res)
#        res *= -l*(l+1)
#        res += l2_csf
#        res /= (targ*(targ+1) - l*(l+1))
#        remove = []
#        for det, coef in res.dets.items():
#            if abs(coef) < 1e-12:
#                remove.append(det)
#        for det in remove:
#            del res.dets[det]
#    return res

def real_part(csf):
    res = Vec.zero()
    for det, coef in csf.dets.items():
        res += coef.real*det
    return res

def memoize_op(op):
    mem = {}
    def memoized(p, state):
        if isinstance(state, Det):
            if state not in mem:
                mem[state] = op(p, state)
            return mem[state]
        return op(p, state)
    return memoized

#TEST
class BVec:
    def __init__(self, bv_id):
        self.bv_id = bv_id

class AtomicOrb:
    def __init__(self, n, l, m, bv_id, ia):
        self.quant_nums = n, l, m
        self.bvec = BVec(bv_id)
        self.ia = ia
#TEST

class L2Projector:
    def __init__(self, mol, mf):
        self.op_tol = 1e-2
        truncate = np.vectorize(lambda num : num if abs(num) > self.op_tol else 0j)

        self.mo_occ = mf.mo_occ
        self.atomic_orbs = p2d.mol2aos(mol, mf, None)
        self.mo_coeffs_full = p2d.aos2mo_coeffs(self.atomic_orbs)
        #self.mo_coeffs = truncate(self.mo_coeffs_full)
        self.mo_coeffs = self.mo_coeffs_full

        self.mo_coeffs_sparse = sparse_rep(self.mo_coeffs.T)
        lup, ldn, lz = l_ops_atomic(self.atomic_orbs)

        #Truncated operators
        self.a2m = atomic2mol_matrix(self.mo_coeffs, self.atomic_orbs)
        self.lup = convert_op(self.a2m, truncate(lup))
        self.ldn = convert_op(self.a2m, truncate(ldn))
        self.lz  = convert_op(self.a2m, truncate(lz))

        #Full operators
        self.a2m_full = atomic2mol_matrix(self.mo_coeffs_full, self.atomic_orbs)
        self.lup_tot = convert_op(self.a2m_full, lup)
        self.ldn_tot = convert_op(self.a2m_full, ldn)
        self.lz_tot  = convert_op(self.a2m_full, lz)
        self.complex_bas = sparse_rep(np.linalg.inv(self.a2m))
        self.mol_bas = sparse_rep(self.a2m)
        self.ang_mom = self.get_ang_mom()

        print('\nLZ: ')
        for orb, coefs in self.lz.items():
            print(orb, ' '.join([str(coef) for coef in coefs]))
        print('\nLUP: ')
        for orb, coefs in self.lup.items():
            print(orb, ' '.join([str(coef) for coef in coefs]))
        print('\nLDN: ')
        for orb, coefs in self.ldn.items():
            print(orb, ' '.join([str(coef) for coef in coefs]))
        print("\nCOEFS: ")
        for orb, coefs in self.mo_coeffs_sparse.items():
            print(orb, ' '.join([str(coef) for coef in coefs]))
        print("\nA2M: ")
        for orb, coefs in sparse_rep(self.a2m.T).items():
            print(orb, ' '.join([str(coef) for coef in coefs]))
        print("\nA2M_FULL: ")
        for orb, coefs in sparse_rep(self.a2m_full.T).items():
            print(orb, ' '.join([str(coef) for coef in coefs]))
        print("\nAtomic Orbs:")
        for n, ao in enumerate(self.atomic_orbs):
            print(n+1, ao)
        print("\n")

    def hf_lz(self):
        res = 0
        for mo, occ in enumerate(self.mo_occ):
            ao_lz = set(self.atomic_orbs[ao-1].m for coef, ao in self.mo_coeffs_sparse[mo+1])
            assert(len(ao_lz) == 1)
            res += int(round(occ))*ao_lz.pop()
        print('occ: ', self.mo_occ)
        print('lz: ', res)
        return res
            
    def lz_of(self, state):
        assert(isinstance(state, Det))
        res = 0
        for orb in state.up_occ:
            lzs = set(self.atomic_orbs[ao-1].m for coef, ao in self.mo_coeffs_sparse[orb])
            #With linear symmetry, mol orbs should be comprised of aos with equal lz vals
            assert(len(lzs) == 1)
            res += lzs.pop()
        for orb in state.dn_occ:
            lzs = set(self.atomic_orbs[ao-1].m for coef, ao in self.mo_coeffs_sparse[orb])
            #With linear symmetry, mol orbs should be comprised of aos with equal lz vals
            assert(len(lzs) == 1)
            res += lzs.pop()
        return res

    @memoize_op
    def apply_lz(self, state):
        return apply_1body(self.lz, state)

    @memoize_op
    def apply_ldn(self, state):
        return apply_1body(self.ldn, state)
        
    @memoize_op
    def apply_lup(self, state):
        return apply_1body(self.lup, state)

    def apply_l2(self, state):
        lz_state = self.apply_lz(state)
        lz2_state = self.apply_lz(lz_state)
        lup_state = self.apply_lup(state)
        ldn_lup_state = self.apply_ldn(lup_state)
        res = Vec.zero()
        res += ldn_lup_state
        res += lz2_state
        res += lz_state
        return res

    def L2(self, state):
        lz_state = apply_1body(self.lz_tot, state)
        lz2_state = apply_1body(self.lz_tot, lz_state)
        lup_state = apply_1body(self.lup_tot, state)
        ldn_lup_state = apply_1body(self.ldn_tot, lup_state)
        res = Vec.zero()
        res += ldn_lup_state
        res += lz2_state
        res += lz_state
        return res

    def L2_expectation(self, state):
        l2_state = self.L2(state)
        expec = state.dot(l2_state)
        assert(expec.imag < 1e-12)
        return expec.real 

    def get_ang_mom(self):
        ang_mom = {}
        for mol_orb, ao_coefs in self.mo_coeffs_sparse.items():
            ang_mom[mol_orb] = set()
            for coef, atom_orb in ao_coefs:
                ang_mom[mol_orb].add(self.atomic_orbs[atom_orb-1].l)
        return ang_mom

    def possible_ang_mom(self, det):
        assert(isinstance(det, Det))
        def update_min_l(min_l, max_l, l):
            if l > max_l:
                return l - max_l 
            elif l < min_l:
                return min_l - l
            else:
                return 0
        min_l, max_l = 0, 0
        occs = det.up_occ + det.dn_occ
        for orb in occs:
            ls = self.ang_mom[orb]
            min_l = min([update_min_l(min_l, max_l, l) for l in ls])
            max_l = max_l + max(ls)
        return [l for l in range(min_l, max_l+1)]
    
    def pretty_print(self, state):
        for det, coef in sorted(state.dets.items(), key = lambda item: -abs(item[1])):
            print(det, ": ", coef)

    def project(self, targ, state):
        if isinstance(state, Det):
            return self.proj_det(targ, state)
        res = Vec.zero()
        for det, coef in state.dets.items():
            res += coef*self.proj_det(targ, det)
        #res = self.prune(targ, res, 1e-8)
        res = self.real_coeffs(res)
        return res

    def proj_det(self, targ, state):
        #TODO: Change proj to iterate over every det in state and add results
        assert(isinstance(state, Det))
        projected_out_ls = [l for l in self.possible_ang_mom(state) if l != targ]
        res = Vec.zero()
        res += state
        #OP IN COMPLEX BAS
        #res = change_basis(self.complex_bas, res)
        for l in projected_out_ls:
#            print("Projecting l = ", l, " with ", len(res.dets), " dets.")
            l2_state = self.apply_l2(res)
            res *= -l*(l+1)
            res += l2_state
            res /= (targ*(targ+1) - l*(l+1))
            #Prune res s.t. some percentage of total magnitude is present
#            print("Ndets before prune: ", len(res.dets))
#            res = self.prune(targ, res, 1e-8)
#            print("Ndets after prune: ", len(res.dets))
            remove = []
            for det, coef in res.dets.items():
                if abs(coef) < 1e-8:
                    remove.append(det)
            for det in remove:
                del res.dets[det]
        #OP IN COMPLEX BAS
        #res = change_basis(self.mol_bas, res)
        return res

    def prune(self, targ, state, err):
        assert(0 < err and err < 1)
        tol = (1 - err)*state.norm()
        res = Vec.zero()
        for det, coef in sorted(state.dets.items(), key = lambda item: -abs(item[1])):
            if res.norm() > tol:
                break
            res += det*(coef.real)
            #if self.eigen_error(targ, res) < err:
            #    break
        return res

    def real_coeffs(self, state):
        real = Vec.zero()
        err = 0
        for det, coef in state.dets.items():
            err += abs(coef.imag)
            real += det*(coef.real)
        if (err > 1e-2):
            print('\n', 'Warning: imag. part of complex coef discarded in L2 projection')
        return real

    def eigen_error(self, targ, state):
        #Calculate err = |targ*(targ+1)*res - L2(res)| / |targ*(targ+1)*res)|
        #If targ = 0, calculate |L2(res)|
        if (state.norm() == 0):
            return 0
        err_vec = self.L2(state)
        err_aux = targ*(targ+1)*state
        err_vec += -1*err_aux
        err = err_vec.norm()/err_aux.norm() if err_aux.norm() > 0 else err_vec.norm()/state.norm()
        return abs(err)

    def debug_proj(self, state):
        float_format = "%12.8f"
        print('START DEBUG: ')
        state = change_basis(self.complex_bas, state)
        lz_state = self.apply_lz(state)
        lz2_state = self.apply_lz(lz_state)
        lup_state = self.apply_lup(state)
        ldn_lup_state = self.apply_ldn(lup_state)
        print('input state: ')
#        for det, coef in change_basis(self.complex_bas, state).dets.items():
        for det, coef in state.dets.items():
            print(det, (float_format + float_format + "i")%(coef.real, coef.imag))
        print('input state (orig basis): ')
#        for det, coef in change_basis(self.mol_bas, change_basis(self.complex_bas, state)).dets.items():
        for det, coef in state.dets.items():
            print(det, (float_format + float_format + "i")%(coef.real, coef.imag))
        print('lz * state: ')
#        for det, coef in change_basis(self.complex_bas, lz_state).dets.items():
        for det, coef in lz_state.dets.items():
            print(det, (float_format + float_format + "i")%(coef.real, coef.imag))
        print('lz * lz * state: ')
#        for det, coef in change_basis(self.complex_bas, lz2_state).dets.items():
        for det, coef in lz2_state.dets.items():
            print(det, (float_format + float_format + "i")%(coef.real, coef.imag))
        print('lup * state: ')
#        for det, coef in change_basis(self.complex_bas, lup_state).dets.items():
        for det, coef in lup_state.dets.items():
            print(det, (float_format + float_format + "i")%(coef.real, coef.imag))
        print('ldn * lup * state: ')
#        for det, coef in change_basis(self.complex_bas, ldn_lup_state).dets.items():
        for det, coef in ldn_lup_state.dets.items():
            print(det, (float_format + float_format + "i")%(coef.real, coef.imag))
        res = Vec.zero()
        res += ldn_lup_state
        res += lz2_state
        res += lz_state
        print('L2 * state: ')
#        for det, coef in change_basis(self.complex_bas, res).dets.items():
        for det, coef in ldn_lup_state.dets.items():
            print(det, (float_format + float_format + "i")%(coef.real, coef.imag))
        print('END DEBUG: ')

    def L2_matrix(self, states):
        mat = np.zeros((len(states), len(states)))
        for i, state1 in enumerate(states):
            l2state = self.apply_l2(state1)
            for j, state2 in enumerate(states):
                if j <= i:
                    mat[j][i] = state2.dot(l2state)
                    mat[i][j] = np.conj(mat[j][i])
        return mat

class AndreProjector():
    def __init__(self, mol, mf):
        self.atomic_orbs = p2d.mol2aos(mol, mf, None)
        self.mo_coeffs = p2d.aos2mo_coeffs(self.atomic_orbs)
        self.a2m = atomic2mol_matrix(self.mo_coeffs, self.atomic_orbs)
        self.op_tol = 1e-8

        self.mo_coeffs_sparse = sparse_rep(self.mo_coeffs.T)
        self.ang_mom = self.get_ang_mom()

        self.l_chars = {0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G', 5: 'H'}  
        self.up_mos, self.dn_mos = self.get_mos()
        self.up_aos, self.dn_aos = self.get_aos()
        self.ao_basis = self.get_ao_basis()
        self.mo_basis = self.get_mo_basis()
        self.max_n = max(ao.n for ao in self.atomic_orbs)

        l_z, l_plus, l_minus = self.build_ops()
#        print('lz: ')
#        for orb, res in l_z.items():
#            print(orb, ': ', res)
#        print('l_plus: ')
#        for orb, res in l_plus.items():
#            print(orb, ': ', res)
#        print('l_minus: ')
#        for orb, res in l_minus.items():
#            print(orb, ': ', res)
        self.l_z = l_z
        self.l_plus = l_plus
        self.l_minus = l_minus

    def get_basis(self, init_orbs, fin_orbs, coeffs):
        #rows of coeffs = the coefs of each init_orb in the fin_orbs
        tol = 1e-15
        basis = {}
        init_up_orbs, init_dn_orbs = init_orbs
        fin_up_orbs, fin_dn_orbs = fin_orbs
        for orb, row in zip(init_up_orbs, coeffs):
            coefs = [coef for coef in row if abs(coef) > tol]
            dets = [Determinant([fin_up_orbs[n]]) for n, coef in enumerate(row) if abs(coef) > tol]
            basis[orb] = DeterminantLinearCombination(coefs, dets)
        for orb, row in zip(init_dn_orbs, coeffs):
            coefs = [coef for coef in row if abs(coef) > tol]
            dets = [Determinant([fin_dn_orbs[n]]) for n, coef in enumerate(row) if abs(coef) > tol]
            basis[orb] = DeterminantLinearCombination(coefs, dets)
        return basis

    def get_mo_basis(self):
        init_orbs = self.up_aos, self.dn_aos
        fin_orbs = self.up_mos, self.dn_mos
        return self.get_basis(init_orbs, fin_orbs, np.linalg.inv(self.a2m))

    def get_ao_basis(self):
        init_orbs = self.up_mos, self.dn_mos
        fin_orbs = self.up_aos, self.dn_aos
        return self.get_basis(init_orbs, fin_orbs, self.a2m)

    def get_ang_mom(self):
        ang_mom = {}
        for mol_orb, ao_coefs in self.mo_coeffs_sparse.items():
            ang_mom[mol_orb] = set()
            for coef, atom_orb in ao_coefs:
                ang_mom[mol_orb].add(self.atomic_orbs[atom_orb-1].l)
        return ang_mom

    def build_ops(self):
        l_minus, l_plus, l_z = {}, {}, {}
        for ao in self.atomic_orbs:
            n, l, m = ao.quant_nums
            bv_id = ao.bvec.bv_id
            for spin_char, s_z in [('-', -0.5), ('+', 0.5)]:
                name = str(n) + self.l_chars[l] + '_' + str(m) + spin_char + '_bv' + str(bv_id)
                orb = Orbital({'name': name, 'n': n, 's': +0.5, 'l': l, 'l_z': m, 's_z': s_z})
                l_z[orb] = m*orb
                if m < l:
                    name = str(n) + self.l_chars[l] + '_' + str(m+1) + spin_char + '_bv' + str(bv_id)
                    new_orb = Orbital({'name': name, 'n': n, 's': +0.5, 'l': l, 'l_z': m+1, 's_z': s_z})
                    l_plus[orb] = np.sqrt(l*(l+1) - m*(m+1)) * new_orb
                else:
                    l_plus[orb] = 0

                if m > -l:
                    name = str(n) + self.l_chars[l] + '_' + str(m-1) + spin_char + '_bv' + str(bv_id)
                    new_orb = Orbital({'name': name, 'n': n, 's': +0.5, 'l': l, 'l_z': m-1, 's_z': s_z})
                    l_minus[orb] = np.sqrt(l*(l+1) - m*(m-1)) * new_orb
                else:
                    l_minus[orb] = 0
        return Operator(l_z), Operator(l_plus), Operator(l_minus)

    def possible_ang_mom(self, det):
#        if isinstance(config, Det):
#            return self.possible_ang_mom(Config(config))
        assert(isinstance(det, Det))
        def update_min_l(min_l, max_l, l):
            if l > max_l:
                return l - max_l 
            elif l < min_l:
                return min_l - l
            else:
                return 0
        min_l, max_l = 0, 0
        occs = det.up_occ + det.dn_occ
        for orb in occs:
            ls = self.ang_mom[orb]
            min_l = min([update_min_l(min_l, max_l, l) for l in ls])
            max_l = max_l + max(ls)
        return min_l, max_l

    def get_wrong_ang_mom(self, target, state):
        min_l = min([self.possible_ang_mom(det)[0] for det in state.dets])
        max_l = max([self.possible_ang_mom(det)[1] for det in state.dets])
        assert(min_l <= target and target <= max_l)
        return [l for l in range(min_l, max_l+1) if l != target]

    def convert_det_lin_comb(self, state):
        dets, coefs = [], []
        for det, coef in state.dets.items():
            orbitals = self.get_orbitals(det)
            dets.append(Determinant(orbitals)) 
            coefs.append(coef)
        return DeterminantLinearCombination(coefs, dets)
    
    def get_orbitals(self, det):
        up_orbitals = [self.up_mos[orb-1] for orb in det.up_occ]
        dn_orbitals = [self.dn_mos[orb-1] for orb in det.dn_occ]
        return up_orbitals + dn_orbitals

    def get_mos(self):
        up_mos = [Orbital({'name': 'mol_orb_up_' + str(n+1), 's_z': +0.5}) 
                  for n in range(len(self.mo_coeffs))]
        dn_mos = [Orbital({'name': 'mol_orb_dn_' + str(n+1), 's_z': -0.5}) 
                  for n in range(len(self.mo_coeffs))]
        return up_mos, dn_mos
        
    def get_aos(self):
        dn_aos, up_aos = [], []
        count = {}
        for atomic_orb in self.atomic_orbs:
            n, l, m = atomic_orb.quant_nums
            bv_id = atomic_orb.bvec.bv_id
            name_up = str(n) + self.l_chars[l] + '_' + str(m) + '+' + '_bv' + str(bv_id)
            name_dn = str(n) + self.l_chars[l] + '_' + str(m) + '-' + '_bv' + str(bv_id)
            up_orb = Orbital({'name': name_up, 'n': n, 's': +0.5, 'l': l, 'l_z': m, 's_z': +0.5})
            dn_orb = Orbital({'name': name_dn, 'n': n, 's': +0.5, 'l': l, 'l_z': m, 's_z': -0.5})
            up_aos.append(up_orb)
            dn_aos.append(dn_orb)
        return up_aos, dn_aos

    def convert_vec(self, state):
        if isinstance(state, DeterminantLinearCombination):
            res = Vec.zero() 
            for det, coef in state.det_coeffs.items():
                p, converted_det = self.convert_vec(det)
                res += coef * p * converted_det
            return res
        elif isinstance(state, Determinant):
            up_orbs, dn_orbs = [], []
            for orbital in state.orbitals:
                name = orbital.labels['name']
                orbs = up_orbs if name[8:10] == 'up' else dn_orbs
                orbs.append(int(name[11:]))
            p = rel_parity(up_orbs)*rel_parity(dn_orbs)
            return p, Det(up_orbs, dn_orbs)

    def proj(self, target, state):
        assert(isinstance(state, Vec))
        wrong_ang_mom = self.get_wrong_ang_mom(target, state)
        #build det_lin_comb object
        state = self.convert_det_lin_comb(state)
        #change basis to complex atomic orbitals
        state.change_basis(self.ao_basis)
        projected = project_ang_mom(state, self.l_z, self.l_plus, self.l_minus, wrong_ang_mom) 
        #change basis to molecular orbitals
        projected.change_basis(self.mo_basis)
        return self.convert_vec(projected)

def main():
    mol = gto.Mole()
    mol.unit = 'bohr'
    mol.atom = 'C 0 0 0'
    mol.spin = 0
    mol.symmetry = 'dooh'
    mol.basis = 'ccpvdz'
    
    mol.output = 'pyscf.txt'
    mol.verbose = 4
    mol.build()
    mol.is_atomic_system = mol.natm == 1

    mf = sym_rhf.RHF(mol)
    mf.max_cycle = 1000
    mf.run()

    p = L2Projector(mol, mf)

    print('Is atomic: ', mol.is_atomic_system)
    print('Type: ', type(mf))
    print('mo_coeffs: ')
    for row in mf.mo_coeff:
        print(''.join(["%8.4f"%c for c in row]))
    print('mo energy: ')
    print(mf.mo_energy)

    d1 = Det([2, 3, 4, 7, 10], [1, 2, 3, 4, 8])
    d2 = Det([1, 4], [1])
    d3 = Det([1, 3], [1])
    d4 = Det([1, 5], [1])

    csf1 = Vec({d1 : 1})
    csf2 = Vec({d2: 1})
    csf3 = Vec({d3: 1, d4: 1})
    #v = Vec({d1: 0.707, d2: 0.707})
    targ = 1
    p1 = p.project(targ, csf1)
    p2 = p.project(targ, csf2)
    p3 = p.project(targ, csf3)
#    p4 = p.project(targ, csf4)
#    p5 = (0.516/0.266) * p.project(targ, csf5)
    print("p1: ")
    for det, coef in sorted(p1.dets.items(), key = lambda det_coef: -abs(det_coef[1])):
        print(det, ": ", coef/p1.norm())
#    print("p2: ")
#    for det, coef in sorted(p2.dets.items(), key = lambda det_coef: -abs(det_coef[1])):
#        print(det, ": ", coef)
#    print("p3: ")
#    for det, coef in sorted(p3.dets.items(), key = lambda det_coef: -abs(det_coef[1])):
#        print(det, ": ", coef)
#    print("p4: ")
#    for det, coef in sorted(p4.dets.items(), key = lambda det_coef: -abs(det_coef[1])):
#        print(det, ": ", coef)
#    print("p5: ")
#    for det, coef in sorted(p5.dets.items(), key = lambda det_coef: -abs(det_coef[1])):
#        print(det, ": ", coef)

    print('Error: ', p.eigen_error(targ, p1))

#    print("\nOrtho:")
#    for n, csf in enumerate(Vec.gram_schmidt([p1, p2, p3], 4, 1e-4)):
#        print(n+1, csf, '\n')

#    print('orbital energies: ', mf.mo_energy)
#    return p

if __name__ == "__main__":
    main()
