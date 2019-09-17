from pyscf import gto, symm, scf
import vec
import numpy as np

def is_x_symm(ir_name):
  if len(ir_name) < 3:
    return False
  return ir_name[0] == 'E' and ir_name[-1] == 'x'

def is_y_symm(ir_name):
  if len(ir_name) < 3:
    return False
  return ir_name[0] == 'E' and ir_name[-1] == 'y'

mol = gto.Mole(verbose=0)
#mol.atom = 'Ti 0 0 0; O 0 0 1'
mol.atom = 'N 0 0 0; N 0 0 1'
mol.symmetry = True
mol.basis = 'tzp'
#mol.basis = 'sto3g'
mol.spin = 0
mol.build()

mf = scf.ROHF(mol)
mf.run()

def rel_parity(orbs):
    #find parity of (potentially) unsorted orbs relative to sorted orbs
    if not orbs:
        return 1
    first, rest = orbs[0], orbs[1:]
    p = -1 if len([orb for orb in rest if orb < first])%2 else 1
    return p*rel_parity(rest)

def find_orb_pairs(mf):
    mol = mf.mol
    num_orb = len(mf.mo_coeff)
    orb_pairs = [n for n in range(num_orb)]
    orb_symm = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
    x_irreps = np.unique([ir for ir in orb_symm if is_x_symm(ir)])
    y_irreps = np.unique([ir for ir in orb_symm if is_y_symm(ir)])
    for x_ir, y_ir in zip(x_irreps, y_irreps):
        Ex_orbs, = np.where(orb_symm == x_ir)
        Ey_orbs, = np.where(orb_symm == y_ir)
        for xorb, yorb in zip(Ex_orbs, Ey_orbs):
            orb_pairs[xorb] = yorb
            orb_pairs[yorb] = xorb
    return [orb+1 for orb in orb_pairs]

def get_partner_orb(mf):
    orb_pairs = find_orb_pairs(mf)
    def partner_orb(orb):
        return orb_pairs[orb-1]
    return partner_orb

partner_orb = get_partner_orb(mf)

def partner_det(det):
    up_orbs = [partner_orb(orb) for orb in det.up_occ]
    dn_orbs = [partner_orb(orb) for orb in det.dn_occ]
    up, dp = rel_parity(up_orbs), rel_parity(dn_orbs)
    return up*dp, vec.Det(up_orbs, dn_orbs)

def partner_csf(csf):
    partner_dets = {}
    for det in csf.dets:
        parity, pdet = partner_det(det)
        partner_dets[pdet] = parity*csf.dets[det]
    return vec.Vec(partner_dets)

def partner_config(config):
    partner_occs = {partner_orb(orb): config.occs[orb] for orb in config.occs}
    up_orbs = [orb for orb in partner_occs if partner_occs[orb] > 0]
    dn_orbs = [orb for orb in partner_occs if partner_occs[orb] == 2]
    return vec.Config.fromorbs(up_orbs, dn_orbs) 

def datum2vec(det_data, csf_datum):
    index2det = {det_data.index(det): det for det in det_data.indices}
    csf = vec.Vec.zero() 
    for det_ind, coef in csf_datum:
        det = index2det[det_ind]
        csf += coef*det
    return csf    

def update_configs(partner_configs, configs, csf):
    det_configs = set(vec.Config(det) for det in csf.dets)
    config, = det_configs
    pconfig = partner_config(config)
    partner_configs.append(pconfig)
    if config == pconfig:
        return
    if config not in configs:
        configs[config] = []
    configs[config].append(csf)

#def symmetrize(csfs, tol=1e-3):
#    symmetrized = []
#    partner_configs, configs = [], {}
#    for csf in csfs:
#        update_configs(partner_configs, configs, csf)
#    for csf, pconfig in zip(csfs, partner_configs):
#        if not find_partner(symmetrized, configs, pconfig, csf, tol):
#            symmetrized.append(csf)
#    return symmetrized

def find_partner(configs, csf, pconfig, partner, tol=1e-3):
    for cand_partner in configs.get(pconfig, []):
        if (partner - cand_partner).norm() < tol:
            return True
    return False

def symmetrize(csfs, tol=1e-3):
    symmetrized = []
    configs = {}
    for csf in csfs:
        det_configs = set(vec.Config(det) for det in csf.dets)
        config, = det_configs
        pconfig = partner_config(config)
        partner = partner_csf(csf)
        if config == pconfig:
            symmetrized.append(csf)
        elif not find_partner(configs, csf, pconfig, partner, tol):    
            symmetrized.append(csf + partner)
        if config not in configs:
            configs[config] = []
        configs[config].append(csf)
    return symmetrized

def symmetrize_data(csf_data, det_data, wf_csf_coeffs, tol=1e-3):
    csfs = [datum2vec(det_data, datum) for datum in csf_data]
    return symmetrize(csfs, tol)

def check_symm(mol, mf, csf_data, det_data, wf_csf_coeffs, tol=1e-3):
    csfs = [datum2vec(det_data, datum) for datum in csf_data]
    print('Start symm: ')
    symmetrized = symmetrize_data(csf_data, det_data, wf_csf_coeffs)
    print('Done symm.')
    orb_symm = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
    partners = []
    coefs = []
    for n, csf in enumerate(csfs):
        [coef] = wf_csf_coeffs[n]
        partner = partner_csf(csf)
        if (partner_csf(partner) - csf).norm() > tol:
            raise Exception("p(p(csf)) != csf")
#        if (partner + csf).norm() < tol:
#            print("Parter = - CSF:\n" + str(csf))
#            raise Exception("partner = -CSF")
        if (csf - partner).norm() < tol:
            partners.append(csf)
            coefs.append((coef,))
            continue
        found = False
        for m, cand_partner in enumerate(csfs):
            if not m < n:
                continue
            if (cand_partner - partner).norm() < tol:
                found = True
                #partners.append(csf + cand_partner)
                partners.append(csf + partner)
                [partner_coef] = wf_csf_coeffs[m]
                coefs.append((coef, partner_coef))
        if not found:
            partners.append(csf + partner)
            coefs.append((coef, 0.0))
    print("Test length: " + str(len(partners)))
    print("Symmetrized length: " + str(len(symmetrized)))
    for pair in partners:
        found = False
        for symm_pair in symmetrized:
            if (pair - symm_pair).norm() < tol:
                found = True
                break
        if not found:
            print("--------------------------------")
            print("WARNING: Pair not found in symmetrized:\n" + str(pair))
            print("--------------------------------")
    for symm_pair in symmetrized:
        found = False
        for pair in partners:
            if (pair - symm_pair).norm() < tol:
                found = True
                break
        if not found:
            print("--------------------------------")
            print("WARNING: Pair not found in partners:\n" + str(symm_pair))
            print("--------------------------------")
    for partner, coef in zip(partners, coefs):
        print('Partners: ')
        print(partner)
        print(str(coef) + '\n')

def check_configs(mol, mf, csf_data, det_data, wf_csf_coeffs, tol=1e-3):
    csfs = [datum2vec(det_data, datum) for datum in csf_data]
    configs, config_arr = {}, []
    for [coef], csf in zip(wf_csf_coeffs, csfs):
        det_configs = set(vec.Config(det) for det in csf.dets)
        config, = det_configs
        if config not in configs:
            configs[config] = []
        configs[config].append((csf, coef))
        config_arr.append(config)
    for config, csf in zip(config_arr, csfs):
        if config == partner_config(config):
            continue
        if partner_config(config) in configs:
            print('CSF: ' + str(csf))
            print('Partner Candidates:')
            for candidate in configs.get(partner_config(config), []):
                print('    ' + str(candidate))
            print('End\n')
