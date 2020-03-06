import numpy as np

import shci4qmc.src.gamess as gamess
import shci4qmc.src.sym_rhf as sym_rhf
import shci4qmc.src.proj_l2 as l2
from shci4qmc.src.vec import Vec, Det
from shci4qmc.lib.rel_parity import rel_parity

from pyscf import symm, gto

class CsfParser():
    def __init__(self, mol, mf):
        self.mol = mol
        self.mf = mf
        self.orbital_dict = self.get_orbital_dict(mol, mf)

        #symbol tables
        self.l_sym = {'s' : 0, 'p': 1, 'd': 2}
        self.m_sym = {'': 0,  #s
                      'x': 1, 'y': -1, 'z': 0, #p
                      'xy': -2, 'yz': -1, 'r2−z2': 0, 'xz': 1, 'x2−y2': 2} #d
        self.s_sym = {'+': 1, '−': -1}

    def get_orbital_dict(self, mol, mf):
        orbs = {} 
        shell_count = {0: 1, 1: 2, 2: 3}
        self.orb_symm_labels = symm.label_orb_symm(
            self.mol, self.mol.irrep_name, self.mol.symm_orb, self.mf.mo_coeff)
        i = 0
        while i < len(self.orb_symm_labels):
            if self.orb_symm_labels[i] == 'A1g':
                orbs[(shell_count[0], 0, 0)] = i+1
                shell_count[0] += 1
                i += 1
            elif self.orb_symm_labels[i] == 'E1ux':
                orbs[(shell_count[1], 1, 1)] = i+1
                orbs[(shell_count[1], 1, -1)] = i+2
                orbs[(shell_count[1], 1, 0)] = i+3
                shell_count[1] += 1
                i += 3
            elif self.orb_symm_labels[i] == 'E2gy':
                orbs[(shell_count[2], 2, -2)] = i+1
                orbs[(shell_count[2], 2, -1)] = i+2
                orbs[(shell_count[2], 2, 0)] = i+3
                orbs[(shell_count[2], 2, 1)] = i+4
                orbs[(shell_count[2], 2, 2)] = i+5
                shell_count[2] += 1
                i += 5
            else:
                raise Exception("Unexpected symm label in get_orbital_dict: " 
                                + self.orb_symm_labels[i])
        return orbs

    def parse_qnums_str(self, s):
        return (int(s[0]), #n
                self.l_sym[s[1]], #l
                self.m_sym[s[2:-1]], #lz
                self.s_sym[s[-1]]) #sz

    def parse_det_str(self, det_str):
        states = [s.strip() for s in det_str.split(';')]
        qnums = [self.parse_qnums_str(s) for s in states]
        up_orbs = [self.orbital_dict[qn[:-1]] for qn in qnums if qn[-1] == 1]
        dn_orbs = [self.orbital_dict[qn[:-1]] for qn in qnums if qn[-1] != 1]
        sign = rel_parity(up_orbs)*rel_parity(dn_orbs)
        return sign, Det(up_orbs, dn_orbs)

    def csf_string_to_det(self, csf_string):
        str_pairs = [s.split('|') for s in csf_string.split('i') if s != ""]
        coef_det_pairs = [(float(c), *self.parse_det_str(det_str)) 
                          for [c, det_str] in str_pairs]
        csf = Vec.zero()
        for coef, sign, det in coef_det_pairs:
            csf += sign*coef*det
        return csf

if __name__ == "__main__":
    mol = gto.Mole()
    mol.unit = 'bohr'
    mol.atom = 'N 0 0 0.'
    mol.spin = 3
    mol.symmetry = 'dooh'
    
    basis_path = '../src/test/n_sto14g_cvb1_hf.out'
    mol.basis = gamess.get_basis(basis_path).get_pyscf_basis()
    mol.build()
    mol.is_atomic_system = (mol.natm == 1)
    
    mf = sym_rhf.RHF(mol).run()
    
    p = l2.L2_Projector(mol, mf)

    parser = CsfParser(mol, mf)

    csf_str1 = "1.00000000 |2s+; 2py+; 2pz+; 2px+; 2s−i"
    csf1 = parser.csf_string_to_det(csf_str1)
    print(csf1)
    print(p.project(0, csf1))

    csf_str2 = "0.57735027 |2s+; 2pz+; 2px+; 3py+; 2s−i-0.57735027 |2s+; 2py+; 2px+; 3pz+; 2s−i0.57735027 |2s+; 2py+; 2pz+; 3px+; 2s−i"
    csf2 = parser.csf_string_to_det(csf_str2)
    print(csf2)
    print(p.project(0, csf2))

    csf_str3 = "-0.05773503 |2py+; 3dxy+; 3dyz+; 3dr2−z2+; 2s−i-0.05773503 |2px+; 3dxy+; 3dr2−z2+; 3dxz+; 2s−i-0.11547005 |2pz+; 3dyz+; 3dr2−z2+; 3dxz+; 2s−i-0.10000000 |2py+; 3dxy+; 3dyz+; 3dx2−y2+; 2s−i-0.05773503 |2pz+; 3dxy+; 3dr2−z2+; 3dx2−y2+; 2s−i-0.05773503 |2px+; 3dyz+; 3dr2−z2+; 3dx2−y2+; 2s−i-0.10000000 |2px+; 3dxy+; 3dxz+; 3dx2−y2+; 2s−i0.05773503 |2py+; 3dr2−z2+; 3dxz+; 3dx2−y2+; 2s−i-0.23094011 |2s+; 3dxy+; 3dyz+; 3dr2−z2+; 2py−i-0.40000000 |2s+; 3dxy+; 3dyz+; 3dx2−y2+; 2py−i0.23094011 |2s+; 3dr2−z2+; 3dxz+; 3dx2−y2+; 2py−i-0.46188022 |2s+; 3dyz+; 3dr2−z2+; 3dxz+; 2pz−i-0.23094011 |2s+; 3dxy+; 3dr2−z2+; 3dx2−y2+; 2pz−i-0.23094011 |2s+; 3dxy+; 3dr2−z2+; 3dxz+; 2px−i-0.23094011 |2s+; 3dyz+; 3dr2−z2+; 3dx2−y2+; 2px−i-0.40000000 |2s+; 3dxy+; 3dxz+; 3dx2−y2+; 2px−i-0.05773503 |2s+; 2py+; 3dyz+; 3dr2−z2+; 3dxy−i-0.05773503 |2s+; 2px+; 3dr2−z2+; 3dxz+; 3dxy−i-0.10000000 |2s+; 2py+; 3dyz+; 3dx2−y2+; 3dxy−i-0.05773503 |2s+; 2pz+; 3dr2−z2+; 3dx2−y2+; 3dxy−i-0.10000000 |2s+; 2px+; 3dxz+; 3dx2−y2+; 3dxy−i0.05773503 |2s+; 2py+; 3dxy+; 3dr2−z2+; 3dyz−i-0.11547005 |2s+; 2pz+; 3dr2−z2+; 3dxz+; 3dyz−i0.10000000 |2s+; 2py+; 3dxy+; 3dx2−y2+; 3dyz−i-0.05773503 |2s+; 2px+; 3dr2−z2+; 3dx2−y2+; 3dyz−i-0.05773503 |2s+; 2py+; 3dxy+; 3dyz+; 3dr2−z2−i0.05773503 |2s+; 2px+; 3dxy+; 3dxz+; 3dr2−z2−i0.11547005 |2s+; 2pz+; 3dyz+; 3dxz+; 3dr2−z2−i0.05773503 |2s+; 2pz+; 3dxy+; 3dx2−y2+; 3dr2−z2−i0.05773503 |2s+; 2px+; 3dyz+; 3dx2−y2+; 3dr2−z2−i0.05773503 |2s+; 2py+; 3dxz+; 3dx2−y2+; 3dr2−z2−i-0.05773503 |2s+; 2px+; 3dxy+; 3dr2−z2+; 3dxz−i-0.11547005 |2s+; 2pz+; 3dyz+; 3dr2−z2+; 3dxz−i0.10000000 |2s+; 2px+; 3dxy+; 3dx2−y2+; 3dxz−i-0.05773503 |2s+; 2py+; 3dr2−z2+; 3dx2−y2+; 3dxz−i-0.10000000 |2s+; 2py+; 3dxy+; 3dyz+; 3dx2−y2−i-0.05773503 |2s+; 2pz+; 3dxy+; 3dr2−z2+; 3dx2−y2−i-0.05773503 |2s+; 2px+; 3dyz+; 3dr2−z2+; 3dx2−y2−i-0.10000000 |2s+; 2px+; 3dxy+; 3dxz+; 3dx2−y2−i0.05773503 |2s+; 2py+; 3dr2−z2+; 3dxz+; 3dx2−y2−i"
    csf3 = parser.csf_string_to_det(csf_str3)
    prj3 = p.project(0, csf3)
    diff = Vec.zero()
    diff += csf3
    diff += -1*prj3
    print(diff.norm())

    csf_str4 = "0.19720266 |2py+; 3dxy+; 3dyz+; 3dr2−z2+; 2s−i0.19720266 |2px+; 3dxy+; 3dr2−z2+; 3dxz+; 2s−i0.39440532 |2pz+; 3dyz+; 3dr2−z2+; 3dxz+; 2s−i0.34156503 |2py+; 3dxy+; 3dyz+; 3dx2−y2+; 2s−i0.19720266 |2pz+; 3dxy+; 3dr2−z2+; 3dx2−y2+; 2s−i0.19720266 |2px+; 3dyz+; 3dr2−z2+; 3dx2−y2+; 2s−i0.34156503 |2px+; 3dxy+; 3dxz+; 3dx2−y2+; 2s−i-0.19720266 |2py+; 3dr2−z2+; 3dxz+; 3dx2−y2+; 2s−i-0.09759001 |2s+; 2px+; 3dxy+; 3dyz+; 3dxy−i0.02817181 |2s+; 2py+; 3dyz+; 3dr2−z2+; 3dxy−i0.09759001 |2s+; 2py+; 3dxy+; 3dxz+; 3dxy−i0.02817181 |2s+; 2px+; 3dr2−z2+; 3dxz+; 3dxy−i-0.14638501 |2s+; 2py+; 3dyz+; 3dx2−y2+; 3dxy−i-0.14085904 |2s+; 2pz+; 3dr2−z2+; 3dx2−y2+; 3dxy−i-0.14638501 |2s+; 2px+; 3dxz+; 3dx2−y2+; 3dxy−i-0.09759001 |2s+; 2pz+; 3dxy+; 3dyz+; 3dyz−i0.14085904 |2s+; 2py+; 3dxy+; 3dr2−z2+; 3dyz−i-0.09759001 |2s+; 2py+; 3dyz+; 3dxz+; 3dyz−i-0.11268723 |2s+; 2pz+; 3dr2−z2+; 3dxz+; 3dyz−i0.04879500 |2s+; 2py+; 3dxy+; 3dx2−y2+; 3dyz−i-0.14085904 |2s+; 2px+; 3dr2−z2+; 3dx2−y2+; 3dyz−i0.09759001 |2s+; 2pz+; 3dxz+; 3dx2−y2+; 3dyz−i-0.08451543 |2s+; 2py+; 3dxy+; 3dyz+; 3dr2−z2−i-0.09759001 |2s+; 2px+; 3dyz+; 3dr2−z2+; 3dr2−z2−i0.08451543 |2s+; 2px+; 3dxy+; 3dxz+; 3dr2−z2−i0.16903085 |2s+; 2pz+; 3dyz+; 3dxz+; 3dr2−z2−i-0.09759001 |2s+; 2py+; 3dr2−z2+; 3dxz+; 3dr2−z2−i-0.08451543 |2s+; 2pz+; 3dxy+; 3dx2−y2+; 3dr2−z2−i0.08451543 |2s+; 2px+; 3dyz+; 3dx2−y2+; 3dr2−z2−i0.08451543 |2s+; 2py+; 3dxz+; 3dx2−y2+; 3dr2−z2−i-0.14085904 |2s+; 2px+; 3dxy+; 3dr2−z2+; 3dxz−i-0.11268723 |2s+; 2pz+; 3dyz+; 3dr2−z2+; 3dxz−i0.09759001 |2s+; 2pz+; 3dxy+; 3dxz+; 3dxz−i-0.09759001 |2s+; 2px+; 3dyz+; 3dxz+; 3dxz−i0.04879500 |2s+; 2px+; 3dxy+; 3dx2−y2+; 3dxz−i0.09759001 |2s+; 2pz+; 3dyz+; 3dx2−y2+; 3dxz−i-0.14085904 |2s+; 2py+; 3dr2−z2+; 3dx2−y2+; 3dxz−i-0.14638501 |2s+; 2py+; 3dxy+; 3dyz+; 3dx2−y2−i-0.14085904 |2s+; 2pz+; 3dxy+; 3dr2−z2+; 3dx2−y2−i0.02817181 |2s+; 2px+; 3dyz+; 3dr2−z2+; 3dx2−y2−i-0.14638501 |2s+; 2px+; 3dxy+; 3dxz+; 3dx2−y2−i-0.02817181 |2s+; 2py+; 3dr2−z2+; 3dxz+; 3dx2−y2−i0.09759001 |2s+; 2px+; 3dyz+; 3dx2−y2+; 3dx2−y2−i-0.09759001 |2s+; 2py+; 3dxz+; 3dx2−y2+; 3dx2−y2−i"
    csf4 = parser.csf_string_to_det(csf_str4)
    prj4 = p.project(0, csf4)
    diff = Vec.zero()
    diff += csf4
    diff += -1*prj4
    print(diff.norm())

    csf_str5 = "0.10540926 |2py+; 3dxy+; 3dyz+; 3dr2−z2+; 2s−i0.10540926 |2px+; 3dxy+; 3dr2−z2+; 3dxz+; 2s−i0.21081851 |2pz+; 3dyz+; 3dr2−z2+; 3dxz+; 2s−i0.18257419 |2py+; 3dxy+; 3dyz+; 3dx2−y2+; 2s−i0.10540926 |2pz+; 3dxy+; 3dr2−z2+; 3dx2−y2+; 2s−i0.10540926 |2px+; 3dyz+; 3dr2−z2+; 3dx2−y2+; 2s−i0.18257419 |2px+; 3dxy+; 3dxz+; 3dx2−y2+; 2s−i-0.10540926 |2py+; 3dr2−z2+; 3dxz+; 3dx2−y2+; 2s−i0.18257419 |2s+; 2px+; 3dxy+; 3dyz+; 3dxy−i-0.21081851 |2s+; 2py+; 3dyz+; 3dr2−z2+; 3dxy−i-0.18257419 |2s+; 2py+; 3dxy+; 3dxz+; 3dxy−i-0.21081851 |2s+; 2px+; 3dr2−z2+; 3dxz+; 3dxy−i0.10540926 |2s+; 2pz+; 3dr2−z2+; 3dx2−y2+; 3dxy−i0.18257419 |2s+; 2pz+; 3dxy+; 3dyz+; 3dyz−i-0.10540926 |2s+; 2py+; 3dxy+; 3dr2−z2+; 3dyz−i0.18257419 |2s+; 2py+; 3dyz+; 3dxz+; 3dyz−i-0.10540926 |2s+; 2pz+; 3dr2−z2+; 3dxz+; 3dyz−i0.18257419 |2s+; 2py+; 3dxy+; 3dx2−y2+; 3dyz−i0.10540926 |2s+; 2px+; 3dr2−z2+; 3dx2−y2+; 3dyz−i-0.18257419 |2s+; 2pz+; 3dxz+; 3dx2−y2+; 3dyz−i0.18257419 |2s+; 2px+; 3dyz+; 3dr2−z2+; 3dr2−z2−i0.18257419 |2s+; 2py+; 3dr2−z2+; 3dxz+; 3dr2−z2−i0.31622777 |2s+; 2pz+; 3dxy+; 3dx2−y2+; 3dr2−z2−i0.10540926 |2s+; 2px+; 3dxy+; 3dr2−z2+; 3dxz−i-0.10540926 |2s+; 2pz+; 3dyz+; 3dr2−z2+; 3dxz−i-0.18257419 |2s+; 2pz+; 3dxy+; 3dxz+; 3dxz−i0.18257419 |2s+; 2px+; 3dyz+; 3dxz+; 3dxz−i0.18257419 |2s+; 2px+; 3dxy+; 3dx2−y2+; 3dxz−i-0.18257419 |2s+; 2pz+; 3dyz+; 3dx2−y2+; 3dxz−i0.10540926 |2s+; 2py+; 3dr2−z2+; 3dx2−y2+; 3dxz−i0.10540926 |2s+; 2pz+; 3dxy+; 3dr2−z2+; 3dx2−y2−i-0.21081851 |2s+; 2px+; 3dyz+; 3dr2−z2+; 3dx2−y2−i0.21081851 |2s+; 2py+; 3dr2−z2+; 3dxz+; 3dx2−y2−i-0.18257419 |2s+; 2px+; 3dyz+; 3dx2−y2+; 3dx2−y2−i0.18257419 |2s+; 2py+; 3dxz+; 3dx2−y2+; 3dx2−y2−i"
    csf5 = parser.csf_string_to_det(csf_str5)
    prj5 = p.project(0, csf5)
    diff = Vec.zero()
    diff += csf5
    diff += -1*prj5
    print(diff.norm())
