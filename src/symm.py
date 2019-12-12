import numpy
import ctypes
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../lib'))

from rel_parity import rel_parity
import vec
from functools import reduce
import pyscf
from pyscf import symm
from pyscf.lib import load_library

ndpointer = numpy.ctypeslib.ndpointer
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

class SymMethods():
    def __init__(self):
        #Symmetry info from Pyscf
        self.symmetry = self.mol.symmetry.upper()
        self.orbsym = getattr(self.mf.mo_coeff, 'orbsym', None)
        self.orb_symm_labels = symm.label_orb_symm(
            self.mol, self.mol.irrep_name, self.mol.symm_orb, self.mf.mo_coeff)
        #Orbital information used for real to complex spherical harmonic conversion
        if self.symmetry in ('DOOH', 'COOV'):
            self.partner_orbs = self.mf.partner_orbs
            self.xorbs = self.get_xorbs()
            self.porbs = {n+1: orb+1 for n, orb in enumerate(self.partner_orbs) }
            print('ORB, PARTNER ORB:')
            for n, (orb, partner) in enumerate(self.porbs.items()):
                print(self.orb_symm_labels[n], orb, partner)
        self.orb2lz = self.get_orb2lz()
        self.use_real_part = None #Use Hartree-Fock det from SHCI
        self.real2complex_coeffs = None

    def real_or_imag_part(self, hf_det):
        rhf, ihf = self.convert_det_helper(hf_det)
        self.use_real_part = (rhf.norm() >= ihf.norm())
        #self.use_real_part = False

    def get_xorbs(self):
        A_irrep_ids = set([0, 1, 4, 5])
        E_irrep_ids = set(self.orbsym).difference(A_irrep_ids)
        Ex_irrep_ids = [ir for ir in E_irrep_ids if (ir % 10) in (0, 2, 5, 7)]
        xorbs = []
        for ir in Ex_irrep_ids:
            orbs ,= numpy.where(self.orbsym == ir)
            xorbs += list(orbs)
        return set([xorb + 1 for xorb in xorbs])

    def rel_parity_old(self, orbs):
        #find parity of (potentially) unsorted orbs relative to sorted orbs
        #Obsolete, use rel_parity module instead (written in c, ~10x faster)
        if not orbs:
            return 1
        first, rest = orbs[0], orbs[1:]
        p = -1 if len([orb for orb in rest if orb < first])%2 else 1
        return p*self.rel_parity_old(rest)

    def partner_config(self, config):
        partner_occs = {self.porbs[orb]: config.occs[orb] for orb in config.occs}
        up_orbs = [orb for orb in partner_occs if partner_occs[orb] > 0]
        dn_orbs = [orb for orb in partner_occs if partner_occs[orb] == 2]
        return vec.Config.fromorbs(up_orbs, dn_orbs) 

    def real_orbs(self, orbs):
        if not orbs:
            return [(1, [])]
        rot_first = self.real_orbs(orbs[:-1])
        orb = orbs[-1]
        if self.porbs[orb] == orb:
            return [(f, orbs + [orb]) for f, orbs in rot_first]
        root2inv = 1./math.sqrt(2)
        xorb, yorb = (orb, self.porbs[orb]) if orb in self.xorbs else (self.porbs[orb], orb)
        sign = -1 if (self.orb2lz[orb-1]%2) else 1
        c_y = sign*1j*root2inv if orb in self.xorbs else -1j*root2inv
        c_x = sign*root2inv if orb in self.xorbs else root2inv
        rorbs = [(c*c_x, orbs + [xorb]) for c, orbs in rot_first if xorb not in orbs]
        rorbs += [(c*c_y, orbs + [yorb]) for c, orbs in rot_first if yorb not in orbs]
        return rorbs

    def convert_det_helper(self, det):
        rdet, idet = vec.Vec.zero(), vec.Vec.zero()
        rup = self.real_orbs(det.up_occ)
        rdn = self.real_orbs(det.dn_occ)
        for ucoef, up in rup:
            for dcoef, dn in rdn:
                coef = rel_parity(up)*rel_parity(dn)*ucoef*dcoef
                rdet += coef.real*vec.Det(up, dn)
                idet += coef.imag*vec.Det(up, dn)
        return rdet, idet

    def convert_det(self, det):
        rdet, idet = self.convert_det_helper(det)
        return rdet if self.use_real_part else idet

    def ir_lz(self, ir):
        if ir%10 in (0, 1, 4, 5):
            l = ir//10 * 2
        else:
            l = ir//10 * 2 + 1
        return l if (ir%10 in (0, 2, 5, 7)) else -l

    def get_orb2lz(self):
        return [self.ir_lz(ir) for ir in self.orbsym]

    def convert_wf(self, dets, wf_coeffs):
        rwf, iwf = vec.Vec.zero(), vec.Vec.zero()
        for det, coef in zip(dets, wf_coeffs):
            rdet, idet = self.convert_det_helper(det)
            rwf += coef*rdet
            iwf += coef*idet
        #assert(self.use_real_part == (rwf.norm() >= iwf.norm()))
        converted_wf = rwf if self.use_real_part else iwf

#        iwf_list = sorted([(det, coef) for det, coef in iwf.dets.items()],
#                          key = lambda det_coef: -abs(det_coef[1]))
#        print('IWF:')
#        for det, coef in iwf_list:
#            print(det, coef)
#        rwf_list = sorted([(det, coef) for det, coef in rwf.dets.items()],
#                          key = lambda det_coef: -abs(det_coef[1]))
#        print('RWF:')
#        for det, coef in rwf_list:
#            print(det, coef)

        dets, coefs = [], []
        for det, coef in converted_wf.dets.items():
            dets.append(det)
            coefs.append(coef)
        return dets, coefs

    def convert_csf(self, csf):
        rcsf = vec.Vec.zero()
        for det, coef in csf.dets.items():
            rdet, idet = self.convert_det(det)
            rcsf += coef*rdet if self.use_real_part else coef*idet
        return rcsf

    def get_real2complex_coeffs(self, h1, eri, norb, nelec, ecore, orbsym, partner_orbs):
        coeffs, num_rows, row_index, row_coeffs, orbsym = \
            self.realToComplex(norb, nelec, orbsym, partner_orbs)
        self.real2complex_coeffs = coeffs
    
    def writeComplexOrbIntegrals(self, h1, eri, norb, nelec, ecore, orbsym, partner_orbs):
        coeffs, num_rows, row_index, row_coeffs, orbsym = \
            self.realToComplex(norb, nelec, orbsym, partner_orbs)
    
        self.real2complex_coeffs = coeffs
        new_h1 = reduce(numpy.dot, (coeffs, h1, coeffs.conj().T)).real
        eri = pyscf.ao2mo.restore(1, eri, norb)
        new_eri = numpy.zeros_like(eri)
    
        transformComplex(norb,
                         numpy.ascontiguousarray(num_rows, numpy.int32),
                         numpy.ascontiguousarray(row_index, numpy.int32),
                         numpy.ascontiguousarray(row_coeffs, numpy.float64),
                         numpy.ascontiguousarray(eri, numpy.float64),
                         numpy.ascontiguousarray(new_eri, numpy.float64))
     
        writeIntNoSymm(norb,
                       numpy.ascontiguousarray(new_h1, numpy.float64),
                       numpy.ascontiguousarray(new_eri, numpy.float64),
                       ecore,
                       nelec,
                       numpy.ascontiguousarray(orbsym, numpy.int32))

    def realToComplex(self, norb, nelec, orbsym, partner_orbs):
        coeffs = numpy.zeros(shape=(norb, norb)).astype(complex)
        num_rows = numpy.zeros(shape=(norb, ), dtype=int)
        row_index = numpy.zeros(shape=(2 * norb, ), dtype=int)
        row_coeffs = numpy.zeros(shape=(2 * norb, ), dtype=float)
        new_orbsym = numpy.zeros(shape=(norb, ), dtype=int)
    
        A_irrep_ids = set([0, 1, 4, 5])
        E_irrep_ids = set(orbsym).difference(A_irrep_ids)
    
        # A1g/A2g/A1u/A2u for Dooh or A1/A2 for Coov
        for ir in A_irrep_ids:
            is_gerade = ir in (0, 1)
            for i in numpy.where(orbsym == ir)[0]:
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
            Ex_ty, = numpy.where(orbsym == ir)
            Ey_ty = numpy.array([partner_orbs[orb] for orb in Ex_ty])
            is_gerade = (ir % 10) in (0, 2)
            if is_gerade:
                # See L146 of basis.py
                Ex = numpy.where(orbsym == ir)[0]
                Ey = numpy.where(orbsym == ir + 1)[0]
            else:
                Ex = numpy.where(orbsym == ir)[0]
                Ey = numpy.where(orbsym == ir - 1)[0]

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
