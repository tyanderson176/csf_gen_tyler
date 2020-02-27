import numpy as np
import shci4qmc.lib.load_wf as load_wf

'''
This code was originally authored by George Booth for the MOLSSI stochastic summer school. 
It was modified by Tyler Anderson.
'''

class Ham:
    def __init__(self, filename = 'FCIDUMP', real_orbs = True):
        ''' Define a hamiltonian to sample, as well as its quantum numbers.
        In addition, it defines the probability of generating a single excitation, rather than double
        excitation in the random excitation generator (which is a method of this class).
        Finally, it also defines a reference determinant energy.'''

        # All these quantities are defined by the 'read_in_fcidump' method
        self.nelec = None       # Number of electrons
        self.ms = None          # 2 x spin-polarization
        self.nbasis = None      # Number of spatial orbitals
        self.spin_basis = None  # Number of spin orbitals (= 2 x self.nbasis)
        self.real_orbs = real_orbs

        # The one-electron hamiltonian in the spin-orbital basis 
        self.h1 = None
        # The two-electron hamiltonian in the spin-orbital basis
        # Note that eri[i,j,k,l] = < phi_i(r_1) phi_k(r_2) | 1/r12 | phi_j(r_1) phi_l(r_2) >
        # This ordering is called 'chemical ordering', and means that the first two indices of the array
        # define the charge density for electron 1, and the second two for electron two.
        self.h2 = None
        # The (scalar) nuclear-nuclear repulsion energy
        self.nn = None

        self.read_in_fcidump(filename)
        return

    def read_in_fcidump(self, filename):
        '''
        self.nelec          # Number of electrons
        self.ms             # 2 x spin-polarization
        self.nbasis         # Number of spatial orbitals
        self.spin_basis     # Number of spin orbitals (= 2 x self.nbasis)

        as well as the integrals defining the hamiltonian terms:
        self.h1[:,:]        # A self.spin_basis x self.spin_basis matrix of one-electron terms
        self.h2[:,:,:,:]    # A rank-4 self.spin_basis array for the two electron terms
        self.nn             # The (scalar) nuclear repulsion energy

        eri[i,j,k,l] = < phi_i(r_1) phi_k(r_2) | 1/r12 | phi_j(r_1) phi_l(r_2) >
        This ordering is called 'chemical ordering', and means that the first two indices of the array
        define the charge density for electron 1, and the second two for electron two.'''
        import os
        import re

        assert(os.path.isfile(os.path.join('./', filename)))

        finp = open(filename, 'r')
        dat = re.split('[=,]', finp.readline())
        while not 'FCI' in dat[0].upper():
            dat = re.split('[=,]', finp.readline())
        self.nbasis = int(dat[1])
        self.nelec = int(dat[3])
        self.ms = int(dat[5])

        # Read in symmetry information, but we are not using it
        sym = []
        dat = finp.readline().strip()
        while not 'END' in dat:
            sym.append(dat)
            dat = finp.readline().strip()

        isym = [x.split('=')[1] for x in sym if 'ISYM' in x]
        if len(isym) > 0:
            isym_out = int(isym[0].replace(',','').strip())
        symorb = ','.join([x for x in sym if 'ISYM' not in x and 'KSYM' not in x]).split('=')[1]
        orbsym = [int(x.strip()) for x in symorb.replace(',', ' ').split()]

        # Read in integrals, but immediately transform them into a spin-orbital basis.
        # We order things with alpha, then beta spins
        self.spin_basis = 2*self.nbasis
        self.h1 = np.zeros((self.spin_basis, self.spin_basis))
        # Ignore permutational symmetry
        self.h2 = np.zeros((self.spin_basis, self.spin_basis, self.spin_basis, self.spin_basis))
        dat = finp.readline().split()
        while dat:
            ii, jj, kk, ll = [int(x) for x in dat[1:5]] # Note these are 1-indexed
            i = ii-1
            j = jj-1
            k = kk-1
            l = ll-1
            if kk != 0:
                # Two electron integral - 8 spatial permutations x 4 spin (=32) allowed permutations!
                # alpha, alpha, alpha, alpha
                self.h2[i, j, k, l] = float(dat[0])
                self.h2[j, i, l, k] = float(dat[0])
                self.h2[k, l, i, j] = float(dat[0])
                self.h2[l, k, j, i] = float(dat[0])
                if self.real_orbs:
                    self.h2[j, i, k, l] = float(dat[0])
                    self.h2[i, j, l, k] = float(dat[0])
                    self.h2[l, k, i, j] = float(dat[0])
                    self.h2[k, l, j, i] = float(dat[0])

                # beta, beta, beta, beta
                self.h2[i+self.nbasis, j+self.nbasis, k+self.nbasis, l+self.nbasis] = float(dat[0])
                self.h2[j+self.nbasis, i+self.nbasis, l+self.nbasis, k+self.nbasis] = float(dat[0])
                self.h2[k+self.nbasis, l+self.nbasis, i+self.nbasis, j+self.nbasis] = float(dat[0])
                self.h2[l+self.nbasis, k+self.nbasis, j+self.nbasis, i+self.nbasis] = float(dat[0])
                if self.real_orbs:
                    self.h2[j+self.nbasis, i+self.nbasis, k+self.nbasis, l+self.nbasis] = float(dat[0])
                    self.h2[i+self.nbasis, j+self.nbasis, l+self.nbasis, k+self.nbasis] = float(dat[0])
                    self.h2[l+self.nbasis, k+self.nbasis, i+self.nbasis, j+self.nbasis] = float(dat[0])
                    self.h2[k+self.nbasis, l+self.nbasis, j+self.nbasis, i+self.nbasis] = float(dat[0])

                # alpha, alpha, beta, beta
                self.h2[i, j, k+self.nbasis, l+self.nbasis] = float(dat[0])
                self.h2[j, i, l+self.nbasis, k+self.nbasis] = float(dat[0])
                self.h2[k, l, i+self.nbasis, j+self.nbasis] = float(dat[0])
                self.h2[l, k, j+self.nbasis, i+self.nbasis] = float(dat[0])
                if self.real_orbs:
                    self.h2[j, i, k+self.nbasis, l+self.nbasis] = float(dat[0])
                    self.h2[i, j, l+self.nbasis, k+self.nbasis] = float(dat[0])
                    self.h2[l, k, i+self.nbasis, j+self.nbasis] = float(dat[0])
                    self.h2[k, l, j+self.nbasis, i+self.nbasis] = float(dat[0])

                # beta, beta, alpha, alpha
                self.h2[i+self.nbasis, j+self.nbasis, k, l] = float(dat[0])
                self.h2[j+self.nbasis, i+self.nbasis, l, k] = float(dat[0])
                self.h2[k+self.nbasis, l+self.nbasis, i, j] = float(dat[0])
                self.h2[l+self.nbasis, k+self.nbasis, j, i] = float(dat[0])
                if self.real_orbs:
                    self.h2[j+self.nbasis, i+self.nbasis, k, l] = float(dat[0])
                    self.h2[i+self.nbasis, j+self.nbasis, l, k] = float(dat[0])
                    self.h2[l+self.nbasis, k+self.nbasis, i, j] = float(dat[0])
                    self.h2[k+self.nbasis, l+self.nbasis, j, i] = float(dat[0])

            elif kk == 0:
                if jj != 0:
                    # One electron term
                    self.h1[i,j] = float(dat[0])
                    self.h1[j,i] = float(dat[0])
                    self.h1[i+self.nbasis, j+self.nbasis] = float(dat[0])
                    self.h1[j+self.nbasis, i+self.nbasis] = float(dat[0])
                else:
                    # Nuclear repulsion term
                    self.nn = float(dat[0])
            dat = finp.readline().split()

        finp.close()
        return

    def slater_condon(self, det, excited_det, excit_mat, parity):
        ''' Calculate the hamiltonian matrix element between two determinants, det and excited_det.
        In:
            det:            A list of occupied orbitals in the original det (note should be ordered)
            excited_det:    A list of occupied orbitals in the excited det (note should be ordered)
            excit_mat:      A list of two tuples, giving the orbitals excited from and to respectively
                                (i.e. [(3, 6), (0, 12)] means we have excited from orbitals 3 and 6 to orbitals 0 and 12
                                    for a single excitation, the tuples will just be of length 1)
                                Note: For a diagonal matrix element (i.e. det == excited_det), the excit_mat should be 'None'.
            parity:         The parity of the excitation
        Out: 
            The hamiltonian matrix element'''

        hel = 0.0
        if excit_mat == None or len(excit_mat[0]) == 0:
            assert(det == excited_det)

            hel += self.nn

            for i in range(self.nelec):
                hel += self.h1[det[i],det[i]]
                for j in range(i+1, self.nelec):
                    dir_2b = self.h2[det[i],det[i],det[j],det[j]]
                    exc_2b = self.h2[det[i],det[j],det[j],det[i]]
                    hel += dir_2b - exc_2b
        elif len(excit_mat[0]) == 1:
            # Single excitation
            hel += self.h1[excit_mat[0][0], excit_mat[1][0]]
            for i in det:
                dir_2b = self.h2[excit_mat[0][0], excit_mat[1][0], i, i]
                exc_2b = self.h2[excit_mat[0][0], i, i, excit_mat[1][0]]
                hel += dir_2b -  exc_2b
            hel *= parity
        elif len(excit_mat[0]) == 2:
            # Double excitation
            dir_2b = self.h2[excit_mat[0][0], excit_mat[1][0], excit_mat[0][1], excit_mat[1][1]]
            exc_2b = self.h2[excit_mat[0][0], excit_mat[1][1], excit_mat[0][1], excit_mat[1][0]]
            hel += dir_2b - exc_2b
            hel *= parity
        return hel

    def get_e(self, dets, coefs):
        energy, norm = 0, 0
        for i in range(len(dets)):
            det, d_coef = dets[i], coefs[i]
            norm += d_coef**2
            energy += (d_coef**2)*self.slater_condon(det, det, None, 1)
            for j in range(i+1, len(dets)):
                excited_det, e_coef = dets[j], coefs[j]
                excit_mat, parity = calc_excit_mat_parity(det, excited_det)
                mat_elem = self.slater_condon(det, excited_det, excit_mat, parity)
                energy += 2*d_coef*e_coef*mat_elem
        return energy/norm

    def det_to_list(self, det):
        up_orbs = [orb - 1 for orb in det.up_occ]
        dn_orbs = [orb - 1 + self.nbasis for orb in det.dn_occ] 
        return up_orbs + dn_orbs

    def expectation(self, wf):
        dets = [self.det_to_list(det) for det in wf.dets.keys()]
        coefs = [coef for coef in wf.dets.values()]
        return self.get_e(dets, coefs)

def elec_exchange_ops(det, ind):
    ''' Given a determinant defined by a list of occupied orbitals
    which is ordered apart from one element (ind), find the number of
    local (nearest neighbour) electron exchanges required to order the 
    list of occupied orbitals.
    
    We can assume that there are no repeated elements of the list, and that
    the list is ordered apart from one element on entry.
    
    Return: The number of pairwise permutations required.'''

    a_orb = det[ind]
    det_sort = sorted(det)
    newind = det_sort.index(a_orb)
    perm = abs(newind - ind)
    return perm

def calc_excit_mat_parity(det, excited_det):
    ''' Given two determinants (excitations of each other), calculate and return 
    the excitation matrix (see the definition in the slater-condon function), 
    and parity of the excitation'''

    # First, we have to compute the indices which are different in the two orbital lists.
    excit_mat = []
    excit_mat.append(tuple(set(det) - set(excited_det)))    # These are the elements in det which are not in excited_det
    excit_mat.append(tuple(set(excited_det) - set(det)))    # These are the elements in excited_det which are not in det
    assert(len(excit_mat[0]) == len(excit_mat[1]))

    # Now find the parity
    new_det = det[:]
    perm = 0
    for elec in range(len(excit_mat[0])):

        # Find the index of the electron we want to remove
        ind_elec = new_det.index(excit_mat[0][elec])
        # Substitute the new electron
        new_det[ind_elec] = excit_mat[1][elec]
        # Find the permutation of this replacement
        perm += elec_exchange_ops(new_det, ind_elec)
        # Reorder the determinant
        new_det.sort()
    assert(new_det == excited_det)

    return excit_mat, (-1)**perm

if __name__ == '__main__':
    # Test for hamiltonian matrix elements
    # Read in System 
    ham = Ham(filename='FCIDUMP_n2', real_orbs = True)
    wf = load_wf.load("wf_eps1_1.00e-01_n2.dat")
    dets = [[orb for orb in up] + [orb + ham.nbasis for orb in dn] 
            for [up, dn] in wf['dets']]
    coefs = [coef for coef in wf['coefs']]
    e = ham.get_e(dets[:1], coefs[:1])
    e = ham.get_e(dets, coefs)
