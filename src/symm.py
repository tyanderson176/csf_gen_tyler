import numpy
import ctypes
import math

import vec
from functools import reduce
import pyscf
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

def realToComplex(norb, nelec, orbsym, partner_orbs):
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
#        Ex, = numpy.where(orbsym == ir)
#        Ey = numpy.array([partner_orbs[orb] for orb in Ex])
        is_gerade = (ir % 10) in (0, 2)
        if is_gerade:
            # See L146 of basis.py
            Ex = numpy.where(orbsym == ir)[0]
            Ey = numpy.where(orbsym == ir + 1)[0]
        else:
            Ex = numpy.where(orbsym == ir)[0]
            Ey = numpy.where(orbsym == ir - 1)[0]

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

def writeComplexOrbIntegrals(h1, eri, norb, nelec, ecore, orbsym, partner_orbs):
    coeffs, num_rows, row_index, row_coeffs, orbsym = realToComplex(norb, nelec, orbsym, partner_orbs)

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

def rel_parity(orbs):
    #find parity of (potentially) unsorted orbs relative to sorted orbs
    if not orbs:
        return 1
    first, rest = orbs[0], orbs[1:]
    p = -1 if len([orb for orb in rest if orb < first])%2 else 1
    return p*rel_parity(rest)

def rel_parity_slow(orbs):
    argsorted, skip = numpy.argsort(orbs), set()
    parity = True
    for n, m in enumerate(argsorted):
        if m in skip: continue
        skip.update([n])
        odd = False
        while m != n:
            skip.update([m])
            m = argsorted[m]
            odd = not odd
        if odd: 
            parity = not parity
    return 1 if parity else -1

def partner_det(dmc, det):
    up_orbs = [dmc.porbs[orb] for orb in det.up_occ]
    dn_orbs = [dmc.porbs[orb] for orb in det.dn_occ]
    pdet = vec.Det(up_orbs, dn_orbs) 
    parity = rel_parity(up_orbs)*rel_parity(dn_orbs)
    return parity, pdet 

def partner_config(dmc, config):
    partner_occs = {dmc.porbs[orb]: config.occs[orb] for orb in config.occs}
    up_orbs = [orb for orb in partner_occs if partner_occs[orb] > 0]
    dn_orbs = [orb for orb in partner_occs if partner_occs[orb] == 2]
    return vec.Config.fromorbs(up_orbs, dn_orbs) 

def real_orbs(porbs, xorbs, l, orbs):
    if not orbs:
        return [(1, [])]
    rot_first = real_orbs(porbs, xorbs, l, orbs[:-1])
    orb = orbs[-1]
    if porbs[orb] == orb:
        return [(f, orbs + [orb]) for f, orbs in rot_first]
    root2inv = 1./math.sqrt(2)
    xorb, yorb = (orb, porbs[orb]) if orb in xorbs else (porbs[orb], orb)
    sign = -1 if (l%2) else 1
    c_y = sign*1j*root2inv if orb in xorbs else -1j*root2inv
    c_x = sign*root2inv if orb in xorbs else root2inv
    rorbs = [(c*c_x, orbs + [xorb]) for c, orbs in rot_first if xorb not in orbs]
    rorbs += [(c*c_y, orbs + [yorb]) for c, orbs in rot_first if yorb not in orbs]
    return rorbs

def convert_det_helper(dmc, det):
    rdet, idet = vec.Vec.zero(), vec.Vec.zero()
    rup = real_orbs(dmc.porbs, dmc.xorbs, dmc.ang_mom, det.up_occ)
    rdn = real_orbs(dmc.porbs, dmc.xorbs, dmc.ang_mom, det.dn_occ)
    for ucoef, up in rup:
        for dcoef, dn in rdn:
            coef = rel_parity(up)*rel_parity(dn)*ucoef*dcoef
            rdet += coef.real*vec.Det(up, dn)
            idet += coef.imag*vec.Det(up, dn)
    return rdet, idet

def convert_det(dmc, det):
    rdet, idet = convert_det_helper(dmc, det)
    return rdet if dmc.real else idet

def xorbs_of(orbsym):
    A_irrep_ids = set([0, 1, 4, 5])
    E_irrep_ids = set(orbsym).difference(A_irrep_ids)
    Ex_irrep_ids = [ir for ir in E_irrep_ids if (ir % 10) in (0, 2, 5, 7)]
    xorbs = []
    for ir in Ex_irrep_ids:
        orbs ,= numpy.where(orbsym == ir)
        xorbs += list(orbs)
    return [xorb + 1 for xorb in xorbs]

def porbs_of(dmc):
    return {n+1: orb+1 for n, orb in enumerate(dmc.partner_orbs) }

def ir_lz(ir):
    if ir%10 in (0, 1, 4, 5):
        l = ir//10 * 2
    else:
        l = ir//10 * 2 + 1
    
    return l if (ir%10 in (0, 2, 5, 7)) else -l

def lz_of(orbsym, det):
    return (sum([ir_lz(orbsym[orb-1]) for orb in det.up_occ]) 
          + sum([ir_lz(orbsym[orb-1]) for orb in det.dn_occ]))

def setup_dmc(dmc, orbsym, dets):
    dmc.xorbs = xorbs_of(orbsym)
    dmc.porbs = porbs_of(dmc) 
    dmc.ang_mom = abs(lz_of(orbsym, dets[0]))
    print("ang_mom: " + str(dmc.ang_mom))
    rhf_det, ihf_det = convert_det_helper(dmc, dets[0])
    dmc.real = (rhf_det.norm() >= ihf_det.norm())

def convert_wf(dmc, dets, wf_coeffs):
    rwf, iwf = vec.Vec.zero(), vec.Vec.zero()
    for det, coef in zip(dets, wf_coeffs):
        rdet, idet = convert_det_helper(dmc, det)
        rwf += coef*rdet
        iwf += coef*idet
    assert(dmc.real == (rwf.norm() >= iwf.norm()))
    converted_wf = rwf if dmc.real else iwf
    dets, coefs = [], []
    for det, coef in converted_wf.dets.items():
        dets.append(det)
        coefs.append(coef)
    return dets, coefs

def convert_csf(dmc, csf):
    rcsf = vec.Vec.zero()
    for det, coef in csf.dets.items():
        rdet, idet = convert_det(dmc, det)
        rcsf += coef*rdet if dmc.real else coef*idet
    return rcsf
