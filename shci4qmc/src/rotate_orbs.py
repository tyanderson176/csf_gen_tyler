import numpy as np
import ctypes
import math
from functools import reduce
#import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__), '../lib'))

from shci4qmc.lib.rel_parity import rel_parity
from shci4qmc.src.vec import Vec, Det, Config
import shci4qmc.src.vec as vec

import pyscf
from pyscf import symm
from pyscf.lib import load_library

class OrbRotator():
    def __init__(self, mol, mf):
        self.mol = mol
        self.mf = mf
        self.orbsym = getattr(self.mf.mo_coeff, 'orbsym', None)
        self.xorbs = self.get_xorbs()

    def get_xorbs(self):
        A_irrep_ids = set([0, 1, 4, 5])
        E_irrep_ids = set(self.orbsym).difference(A_irrep_ids)
        Ex_irrep_ids = [ir for ir in E_irrep_ids if (ir % 10) in (0, 2, 5, 7)]
        xorbs = []
        for ir in Ex_irrep_ids:
            orbs ,= np.where(self.orbsym == ir)
            xorbs += list(orbs)
        return set([xorb + 1 for xorb in xorbs])

    def ir_lz(self, ir):
        if ir%10 in (0, 1, 4, 5):
            l = ir//10 * 2
        else:
            l = ir//10 * 2 + 1
        return l if (ir%10 in (0, 2, 5, 7)) else -l

    def lz(self, orb):
        ir = self.orbsym[orb-1]
        return self.ir_lz(ir)

    def rotate_orb(self, orb):
        def partner(orb):
            return self.mf.partner_orbs[orb-1] + 1

        if orb == partner(orb):
            return (1, orb), 
        coef = 1./np.sqrt(2)
        sign = -1 if self.lz(orb)%2 else 1
        if orb in self.xorbs:
            xpart = sign*coef, orb
            ypart = sign*1j*coef, partner(orb)
        else:
            xpart = coef, partner(orb)
            ypart = -1j*coef, orb
        return xpart, ypart

    def rotate_orbs(self, orbs):
        res, real_orbs, skip = [], [], set()
        def helper(i, coef):
            if (i == len(orbs)):
                #append a copy of real_orbs to res
                res.append((coef, [orb for orb in real_orbs]))
            else:
                for c, orb in self.rotate_orb(orbs[i]):
                    if orb in skip:
                        continue
                    real_orbs.append(orb)
                    skip.add(orb)
                    helper(i+1, c*coef)
                    real_orbs.pop()
                    skip.remove(orb)
        helper(0, 1)
        return res
    
#    def real_orbs_old(self, orbs):
#        rot_first = self.real_orbs(orbs[:-1])
#        orb = orbs[-1]
#        if partner(orb) == orb:
#            return [(f, orbs + [orb]) for f, orbs in rot_first]
#        root2inv = 1./math.sqrt(2)
#        xorb, yorb = (orb, partner(orb)) if orb in self.xorbs else (partner(orb), orb)
#        sign = -1 if sign_of(orb) else 1
#        c_y = sign*1j*root2inv if orb in self.xorbs else -1j*root2inv
#        c_x = sign*root2inv if orb in self.xorbs else root2inv
#        rorbs = [(c*c_x, orbs + [xorb]) for c, orbs in rot_first if xorb not in orbs]
#        rorbs += [(c*c_y, orbs + [yorb]) for c, orbs in rot_first if yorb not in orbs]
#        return rorbs

    def rotate_det(self, det):
        real_up = self.rotate_orbs(det.up)
        real_dn = self.rotate_orbs(det.dn)
        real_det = Vec.zero() 
        for rcoef, rup in real_up:
            for dcoef, rdn in real_dn:
                coef = rel_parity(rup)*rel_parity(rdn)*rcoef*dcoef
                real_det += coef * Det(rup, rdn)
        return real_det

    def to_real_orbs(self, wf):
        real_wf = Vec.zero()
        for det, coef in wf.dets.items():
            real_wf += coef * self.rotate_det(det)
        return real_wf

    def partner_config(self, config):
        def partner(orb):
            return self.mf.partner_orbs[orb-1] + 1
            
        partner_occs = {partner(orb): config.occs[orb] for orb in config.occs}
        up_orbs = [orb for orb in partner_occs if partner_occs[orb] > 0]
        dn_orbs = [orb for orb in partner_occs if partner_occs[orb] == 2]
        return vec.Config.fromorbs(up_orbs, dn_orbs) 

ndpointer = np.ctypeslib.ndpointer
shci_lib = load_library('libshciscf')

transformComplex = shci_lib.transformDinfh
transformComplex.restyp = None
transformComplex.argtypes = [
    ctypes.c_int,
    ndpointer(ctypes.c_int32),
    ndpointer(ctypes.c_int32),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double)
]

writeIntNoSymm = shci_lib.writeIntNoSymm
writeIntNoSymm.argtypes = [
    ctypes.c_int,
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double), 
    ctypes.c_double, 
    ctypes.c_int,
    ndpointer(ctypes.c_int32)
]

def writeComplexOrbIntegrals(h1, eri, norb, nelec, ecore, orbsym, partner_orbs):
    coeffs, num_rows, row_index, row_coeffs, orbsym = \
        realToComplex(norb, nelec, orbsym, partner_orbs)

    #self.real2complex_coeffs = coeffs
    new_h1 = reduce(np.dot, (coeffs, h1, coeffs.conj().T)).real
    eri = pyscf.ao2mo.restore(1, eri, norb)
    new_eri = np.zeros_like(eri)

    transformComplex(norb,
                     np.ascontiguousarray(num_rows, np.int32),
                     np.ascontiguousarray(row_index, np.int32),
                     np.ascontiguousarray(row_coeffs, np.float64),
                     np.ascontiguousarray(eri, np.float64),
                     np.ascontiguousarray(new_eri, np.float64))
 
    writeIntNoSymm(norb,
                   np.ascontiguousarray(new_h1, np.float64),
                   np.ascontiguousarray(new_eri, np.float64),
                   ecore,
                   nelec,
                   np.ascontiguousarray(orbsym, np.int32))
    return

def get_real2complex_coeffs(h1, eri, norb, nelec, ecore, orbsym, partner_orbs):
    coeffs, num_rows, row_index, row_coeffs, orbsym = \
        realToComplex(norb, nelec, orbsym, partner_orbs)
    return coeffs
#    self.real2complex_coeffs = coeffs

def realToComplex(norb, nelec, orbsym, partner_orbs):
    coeffs = np.zeros(shape=(norb, norb)).astype(complex)
    num_rows = np.zeros(shape=(norb, ), dtype=int)
    row_index = np.zeros(shape=(2 * norb, ), dtype=int)
    row_coeffs = np.zeros(shape=(2 * norb, ), dtype=float)
    new_orbsym = np.zeros(shape=(norb, ), dtype=int)

    A_irrep_ids = set([0, 1, 4, 5])
    E_irrep_ids = set(orbsym).difference(A_irrep_ids)

    # A1g/A2g/A1u/A2u for Dooh or A1/A2 for Coov
    for ir in A_irrep_ids:
        is_gerade = ir in (0, 1)
        for i in np.where(orbsym == ir)[0]:
            coeffs[i, i] = 1.0
            num_rows[i] = 1
            row_index[2 * i] = i
            row_coeffs[2 * i] = 1.
            if is_gerade:  # A1g/A2g for Dooh or A1/A2 for Coov
                new_orbsym[i] = 1
            else:  # A1u/A2u for Dooh
                new_orbsym[i] = 2

    # See L146 of pyscf/symm/basis.py
    Ex_irrep_ids = [ir for ir in E_irrep_ids if (ir % 10) in (0, 2, 5, 7)]
    for ir in Ex_irrep_ids:
        Ex_ty, = np.where(orbsym == ir)
        Ey_ty = np.array([partner_orbs[orb] for orb in Ex_ty])
        is_gerade = (ir % 10) in (0, 2)
        if is_gerade:
            # See L146 of basis.py
            Ex = np.where(orbsym == ir)[0]
            Ey = np.where(orbsym == ir + 1)[0]
        else:
            Ex = np.where(orbsym == ir)[0]
            Ey = np.where(orbsym == ir - 1)[0]

        # These should be the same, unless Sandeep's script or my method to calculate
        # Ex or Ey is incorrect. Ex_ty, Ey_ty are my labels.
        assert(list(Ex) == list(Ex_ty))
        assert(list(Ey) == list(Ey_ty))

        if ir % 10 in (0, 5):
            l = (ir // 10) * 2
        else:
            l = (ir // 10) * 2 + 1

        for ix, iy in zip(Ex, Ey):
            num_rows[ix] = num_rows[iy] = 2
            if is_gerade:
                new_orbsym[ix], new_orbsym[iy] = 2 * l + 3, -(2 * l + 3)
            else:
                new_orbsym[ix], new_orbsym[iy] = 2 * l + 4, -(2 * l + 4)

            row_index[2 * ix], row_index[2 * ix + 1] = ix, iy
            row_index[2 * iy], row_index[2 * iy + 1] = ix, iy
    
            coeffs[ix, ix], coeffs[ix, iy] = ((-1)**l) * 1.0 / (2.0**0.5), (
                (-1)**l) * 1.0j / (2.0**0.5)
            coeffs[iy, ix], coeffs[iy, iy] = 1.0 / (2.0**0.5), -1.0j / (2.0** 0.5)
            row_coeffs[2 * ix], row_coeffs[2 * ix + 1] = (
                (-1)**l) * 1.0 / (2.0**0.5), ((-1)**l) * 1.0 / (2.0**0.5)
            row_coeffs[2 * iy], row_coeffs[
                2 * iy + 1] = 1.0 / (2.0**0.5), -1.0 / (2.0**0.5)

    return coeffs, num_rows, row_index, row_coeffs, new_orbsym
