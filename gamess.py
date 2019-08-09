import numpy as np
import numpy.linalg
from pyscf import gto

class GamessBasisVec():
    def __init__(self, atom, l_label, l, n, slater_exp, coeffs, gauss_exps):
        self.atom = atom
        self.l_label = l_label
        self.l = l
        self.n = n
        self.slater_exp = slater_exp
        self.coeffs = np.array(coeffs)
        self.gauss_exps = np.array(gauss_exps)

    def get_n(self):
        return self.n
   
    def get_slater_exponent(self):
        return self.slater_exp

class GamessBasis():
    def __init__(self):
        self.l_dict = {'S':0, 'P':1, 'D':2, 'F':3, 'G':4, 'H':5}
        self.basis_vecs = {}

    def add(self, atom, l_label, n, slater_exp, coeffs, gauss_exps):
        l = self.l_dict[l_label]
        if atom not in self.basis_vecs:
            self.basis_vecs[atom] = []
        self.basis_vecs[atom].append(
            GamessBasisVec(atom, l_label, l, n, slater_exp, coeffs, gauss_exps))

    def same_coeffs(self, coeffs, bv):
        if coeffs.shape != bv.coeffs.shape:
            return False
        coeffs = sorted(coeffs)
        bv_coeffs = sorted(bv.coeffs)
        for coef, my_coef in zip(coeffs, bv_coeffs):
            if np.abs((coef - my_coef)/my_coef) > 1e-2:
                return False
        return True

    def same_exps(self, exps, bv):
        if exps.shape != bv.gauss_exps.shape:
            return False
        exps = sorted(exps)
        bv_exps = sorted(bv.gauss_exps)
        for exp, my_exp in zip(exps, bv_exps):
            if np.abs((exp-my_exp)/my_exp) > 1e-2:
                return False
        return True

    def find(self, atom, l, coeffs, exps):
        for bv in self.basis_vecs[atom]:
            if bv.l != l:
                continue
            if not self.same_coeffs(coeffs, bv):
                continue
            if not self.same_exps(exps, bv):
                continue
            return bv
        raise Exception("Could not find basis vector from PySCF in GAMESS input")

    def get_basis_str(self, atom):
        basis_str = "" 
        bvs = self.basis_vecs[atom]
        for bv in bvs:
            atom, l_label = bv.atom, bv.l_label
            basis_str += "%s\t%s\n" % (atom, l_label)
            coeffs, gauss_exps = bv.coeffs, bv.gauss_exps
            for coef, exp in zip(coeffs, gauss_exps):
                basis_str += "%20.10f\t%15.10f\n" % (exp, coef)
            basis_str += '\n'
        return basis_str

    def get_pyscf_basis(self):
        pyscf_basis = {}
        for atom in self.basis_vecs:
            pyscf_basis[atom] = gto.basis.parse(self.get_basis_str(atom))
        return pyscf_basis

    def __len__(self):
        tot = 0
        for atom in self.basis_vecs:
            tot += len(self.basis_vecs[atom])
        return tot

    def __repr__(self):
        basis_str = ""
        for atom in self.basis_vecs:
            basis_str += self.get_basis_str(atom)
        return basis_str

def not_basis_chunk(chunk):
    if 'TOTAL NUMBER OF BASIS SET SHELLS' in chunk:
        return True
    return False

def get_gamess_basis(filename):
    f = open(filename, 'r')
    read_data = f.read()
    ao_start = read_data.index('SHELL TYPE PRIMITIVE')
    data_chunks = read_data[ao_start:].split('\n\n')[1:]
    basis = GamessBasis()
    atom = ""
    for chunk in data_chunks:
        if not_basis_chunk(chunk):
            break
        if not chunk:
            continue
        elif chunk.strip().isalpha():
            atom = chunk.strip()
        else:
            n, l_label, slater_exp = 0, 0, 0
            coeffs, gauss_exps = [], []
            for i, line in enumerate(chunk.split('\n')):
                data = line.strip().split()
                if i == 0:
                    [num, orb_label, num2, slater_exp, exp, coef] = data
                    l_start = min(
                        [i for i,c in enumerate(orb_label) if c.isalpha()])
                    n, l_label = int(orb_label[:l_start]), orb_label[l_start:]
                else:
                    [num, orb_label, num2, exp, coef] = data
                try:
                    coeffs.append([float(coef)])
                    gauss_exps.append(float(exp))
                except:
                    raise Exception("Couldn't convert coef or exp to float")
            basis.add(atom, l_label, n, float(slater_exp), coeffs, gauss_exps)
    f.close()
    return basis

def get_basis(filename):
    basis = get_gamess_basis(filename)
    return basis

if __name__ == "__main__":
    basis = get_gamess_basis('n2_sto14g_cvb1_ci_cas108_spher.out')
    print(str(basis))
    print('PYSCF BASIS\n\n')
    print(basis.get_pyscf_basis()['N'])
