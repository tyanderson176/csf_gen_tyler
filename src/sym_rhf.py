import numpy
from pyscf.scf.atom_hf import AtomSphericAverageRHF
from pyscf.scf import hf, rohf
from pyscf.scf import hf_symm
from pyscf.scf import chkfile
from pyscf import scf
from pyscf import symm
from pyscf import lib

def get_partner_orbs(self):
    def is_x_symm(ir_label):
      if len(ir_label) < 3:
        return False
      return ir_label[0] == 'E' and ir_label[-1] == 'x'
    
    def is_y_symm(ir_label):
      if len(ir_label) < 3:
        return False
      return ir_label[0] == 'E' and ir_label[-1] == 'y'

    mol = self.mol
    num_orb = len(self.mo_coeff)
    partner_orbs = numpy.arange(num_orb)#[n for n in range(num_orb)]
    orb_symm = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, self.mo_coeff)
    x_irreps = numpy.unique([ir for ir in orb_symm if is_x_symm(ir)])
    y_irreps = numpy.unique([ir for ir in orb_symm if is_y_symm(ir)])
    for x_ir, y_ir in zip(x_irreps, y_irreps):
        Ex_orbs, = numpy.where(orb_symm == x_ir)
        Ey_orbs, = numpy.where(orb_symm == y_ir)
        for xorb, yorb in zip(Ex_orbs, Ey_orbs):
            partner_orbs[xorb] = yorb
            partner_orbs[yorb] = xorb
    return partner_orbs

def _finalize(self):
    if isinstance(self, rohf.ROHF):
        rohf.ROHF._finalize(self)
    elif isinstance(self, hf.RHF):
        hf.RHF._finalize(self)

    if (self.mol.groupname in ('Dooh', 'Coov')):
        self.partner_orbs = self.get_partner_orbs()
    # sort MOs wrt orbital energies, it should be done last.
    # Using mergesort because it is stable. We don't want to change the
    # ordering of the symmetry labels when two orbitals are degenerated.

    if isinstance(self, rohf.ROHF):
        c_sort = numpy.argsort(self.mo_energy[self.mo_occ==2].round(9), kind='mergesort')
        o_sort = numpy.argsort(self.mo_energy[self.mo_occ==1].round(9), kind='mergesort')
        v_sort = numpy.argsort(self.mo_energy[self.mo_occ==0].round(9), kind='mergesort')
        idx = numpy.arange(self.mo_energy.size)
        idx = numpy.hstack((idx[self.mo_occ==2][c_sort],
                            idx[self.mo_occ==1][o_sort],
                            idx[self.mo_occ==0][v_sort]))
        if hasattr(self.mo_energy, 'mo_ea'):
            mo_ea = self.mo_energy.mo_ea[idx]
            mo_eb = self.mo_energy.mo_eb[idx]
            self.mo_energy = lib.tag_array(self.mo_energy[idx],
                                           mo_ea=mo_ea, mo_eb=mo_eb)
        else:
            self.mo_energy = self.mo_energy[idx]
    elif isinstance(self, hf.RHF):
        o_sort = numpy.argsort(self.mo_energy[self.mo_occ> 0].round(9), kind='mergesort')
        v_sort = numpy.argsort(self.mo_energy[self.mo_occ==0].round(9), kind='mergesort')
        idx = numpy.arange(self.mo_energy.size)
        idx = numpy.hstack((idx[self.mo_occ> 0][o_sort],
                            idx[self.mo_occ==0][v_sort]))
        self.mo_energy = self.mo_energy[idx]

    orbsym = hf_symm.get_orbsym(self.mol, self.mo_coeff)
    self.mo_coeff = lib.tag_array(self.mo_coeff[:,idx], orbsym=orbsym[idx])
    self.mo_occ = self.mo_occ[idx]

    if hasattr(self, 'partner_orbs'):
        idx_inv = numpy.argsort(idx)
        self.partner_orbs = [idx_inv[orb] for orb in self.partner_orbs[idx]]

    if self.chkfile:
        chkfile.dump_scf(self.mol, self.chkfile, self.e_tot, self.mo_energy,
                         self.mo_coeff, self.mo_occ, overwrite_mol=False)
    return self

#class AtomicRHF(AtomSphericAverageRHF):
#    scf = SCF.scf
#
#    _finalize = _finalize
#    
#    get_partner_orbs = get_partner_orbs

#class MolecularRHF(RHF):
#    _finalize = _finalize
#    
#    get_partner_orbs = get_partner_orbs

def partner_irrep(self, ir, gpname):
    if gpname not in ('Dooh', 'Coov'):
        return ir
    elif ir in (0, 1, 4, 5): #A irreps
        return ir
    elif ir%10 in (0, 2, 5, 7): #Ex irreps
        gerade = ir%10 in (0, 2)
        return ir + 1 if gerade else ir - 1
    else: #Ey irreps
        gerade = ir%10 in (1, 3)
        return ir - 1 if gerade else ir + 1

def eig(self, h, s):
    '''Solve generalized eigenvalue problem, for each irrep.  The
    eigenvalues and eigenvectors are not sorted to ascending order.
    Instead, they are grouped based on irreps.
    '''
    mol = self.mol
    if not mol.symmetry:
        return self._eigh(h, s)

    nirrep = mol.symm_orb.__len__()
    h = symm.symmetrize_matrix(h, mol.symm_orb)
    s = symm.symmetrize_matrix(s, mol.symm_orb)
    partner_ir = [self.partner_irrep(ir, mol.groupname) for ir in mol.irrep_id]
    for ir in range(nirrep):
        if ir > 0 and mol.irrep_id[ir-1] == partner_ir[ir]:
            h_av = (h[ir] + h[ir-1])/2
            h[ir] = h[ir-1] = h_av
    cs = []
    es = []
    orbsym = []
    for ir in range(nirrep):
        e, c = self._eigh(h[ir], s[ir])
        cs.append(c)
        es.append(e)
        orbsym.append([mol.irrep_id[ir]] * e.size)
    e = numpy.hstack(es)
    c = hf_symm.so2ao_mo_coeff(mol.symm_orb, cs)
    c = lib.tag_array(c, orbsym=numpy.hstack(orbsym))
    return e, c

#def create(mol):
#    if mol.is_atomic_system:
#        return AtomicRHF(mol).run()
#    else:
#        return MolecularRHF(mol).run()

def RHF(mol, *args):
    if mol.is_atomic_system:
        rhf = AtomSphericAverageRHF(mol)
        rhf.scf = lambda dm0=None, **kwargs: hf.SCF.scf(rhf, dm0, **kwargs)
    else:
        rhf = scf.RHF(mol, *args) 
        #Force partner orbs to be equal
        rhf.eig = lambda h, s: eig(rhf, h, s)
        rhf.partner_irrep = lambda ir, gp: partner_irrep(rhf, ir, gp)

    #Add partner_orbs utility
    rhf._finalize = lambda : _finalize(rhf)
    rhf.get_partner_orbs = lambda : get_partner_orbs(rhf)
    return rhf

if __name__ == "__main__":
    from pyscf import gto

    mol = gto.Mole()
    mol.unit = 'bohr'
    #mol.atom = 'H 0 0 2.15; C 0 0 0.'
    mol.atom = 'C 0 0 0'
    mol.spin = 0
    #mol.symmetry = 'coov'
    mol.symmetry = 'dooh'
    mol.basis = 'ccpvdz'
    
    mol.output = "pyscf.log"
    mol.verbose = 5
    mol.build(0, 0)

    mol.is_atomic_system = mol.natm == 1
    rhf = RHF(mol).run()

    print(rhf.partner_orbs)
    for row in rhf.mo_coeff:
        print(''.join(["%8.4f"%c for c in row]))
