import numpy
import subprocess
import os
import json
from functools import reduce
from decimal import Decimal

import shci4qmc.src.p2d as p2d
import shci4qmc.src.gamess as gamess
import shci4qmc.src.vec as vec
import shci4qmc.src.sym_rhf as sym_rhf
import shci4qmc.src.make_input as inp

from shci4qmc.src.ham import Ham
from shci4qmc.src.vec import Vec, Det, Config
from shci4qmc.src.csf import get_csfs
from shci4qmc.src.rotate_orbs import writeComplexOrbIntegrals, get_real2complex_coeffs

from pyscf.tools import fcidump
from pyscf import scf, ao2mo, gto, symm

#TODO: Print helpful output during calculation
#TODO: Test cases/examples for each module
#TODO: Make pyscf chkfile: FCIDUMP + wf.dat + chkfile = exactly same output

class ChampInputFiles:
    def __init__(self):
        self.config = None
        self.shci_cmd = None
        self.dir_reuse = '.'
        self.dir_qmc_inp = None
        self.bas_path = None

        self.tol_det = 1e-3
        self.tol_csf = [5e-2, 2e-2, 1e-1]

        self.opt_orbs = False
        self.rot_matrix = None

        self.target_l2 = None
        self.mol = None
        self.basis = None

        self.out_file = None

    def make(self):
        self.dat_file_path = os.path.join(self.dir_reuse, 'champ.dat')
        if os.path.exists(self.dat_file_path):
            self.make_inp_file()
            return

        self.wf_filename = 'wf_eps1_%.2e.dat'%min(self.config['eps_vars'])
        self.wf_path = os.path.join(self.dir_reuse, self.wf_filename)
        self.rot_matrix_path = os.path.join(self.dir_reuse, 'rotation_matrix')

        assert(self.mol.unit == 'bohr' and self.mol.symmetry)
        assert(self.mol.spin == abs(self.config['n_up'] - self.config['n_dn']))
        assert(self.mol.symmetry.lower() == self.config['chem']['point_group'].lower())
        assert(self.opt_orbs == self.config['optorb'])

        self.symmetry = self.mol.symmetry.lower()

        #Use analytic basis external to pyscf
        if self.bas_path:
            self.basis = gamess.get_basis(self.bas_path)
            self.mol.basis = self.basis.get_pyscf_basis()
        self.mol.build()
        self.mol.is_atomic_system = (self.mol.natm == 1)

        #Calculate molecular orbitals
        print("Starting RHF...")
        self.mf = sym_rhf.RHF(self.mol).run()
        #self.mf = scf.RHF(self.mol).run()
        print("Finished RHF.") 
        
        self.orb_symm_labels = symm.label_orb_symm(
            self.mol, self.mol.irrep_name, self.mol.symm_orb, self.mf.mo_coeff)

        #Get atomic orbitals
        self.aos = p2d.mol2aos(self.mol, self.mf, self.basis)
        self.mo_coeffs = p2d.aos2mo_coeffs(self.aos)
        self.atoms = self.get_atom_types() 
        self.n_up, self.n_down = self.mol.nelec
        self.hf_energy = self.mf.energy_tot()

        #Updates self.dets, self.csfs, self.coefs 
        self.get_shci_output()

        #make dat files
        self.make_dat_file() 
        
        #make input files
        self.make_inp_file()
        return

    def make_inp_file(self):
        for tol in self.tol_csf:
            inp.make_input(self.dat_file_path, self.dir_qmc_inp, tol)
        return

    def make_dat_file(self):
        self.out_file = open(self.dat_file_path, 'a')

        #Output data
        self.print_header()
        self.print_radial_bfs()
        self.print_orbs()
        self.print_shci()
        self.print_jastrow()
        self.print_opt()

        self.out_file.close()
        return

    def print_header(self):
        self.print_qmc_header()
        self.print_geometry_header()
        self.print_determinant_header()
        return

    def print_qmc_header(self):
        format_str = '{:40}'
        self.out_file.write(
            format_str.format('\'ncsf=%d ndet=%d norb=%d\''% 
            (len(self.csfs), len(self.dets), len(self.mo_coeffs))) 
            + 'title\n')
        self.out_file.write(format_str.format('1837465927472523') + 'irn\n')
        self.out_file.write(
            format_str.format('0  1 slater') 
            + 'iperiodic,ibasis,which_analytical_basis\n')
        self.out_file.write(
            format_str.format('0.5  %10.3f  \'Hartrees\''% self.proj_e) 
            + 'hb,etrial,eunit\n')
        self.out_file.write(
            format_str.format('100  100  1  100  0') 
            + 'nstep,nblk,nblkeq,nconf,nconf_new\n')
        self.out_file.write(
            format_str.format('0  0  1  -2') 
            + 'idump,irstar,isite,ipr\n')
        self.out_file.write(
            format_str.format('6  1.  5.  1.  1.') 
            + 'imetro delta,deltar,deltat fbias\n')
        self.out_file.write( 
            format_str.format('2  1  1  1  1  0  0  0  0')
            + 'idmc,ipq,itau_eff,iacc_rej,icross,icuspg,idiv_v,icut_br,icut_e\n')
        self.out_file.write(format_str.format('50  .01') + 'nfprod, tau\n')
        self.out_file.write(format_str.format('0  -3  1   0') + 'nloc,numr,nforce,nefp\n')
        self.out_file.write(
            format_str.format('%d %d'% (self.n_up + self.n_down, self.n_up)) 
            + 'nelec,nup\n')
        return
    
    def print_determinant_header(self):
        num_dets = len(self.dets)
        self.out_file.write('\'* Determinantal section\'\n')
        self.out_file.write('0 0                    inum_orb\n')
        self.out_file.write(
            '%d %d %d               ndet, nbasis, norb \n\n'% 
            (num_dets, len(self.aos), len(self.mo_coeffs)))
        return
     
    def print_geometry_header(self):
        ''' Atoms written in order of increasing ia
        '''
        #Write number of atom types, number of atoms
        self.out_file.write('\' * Geometry section\'\n')
        self.out_file.write('3\t\t\tndim\n')
        self.out_file.write(
            '%d %d\t\t\tnctypes, ncent\n'% 
            (len(self.atoms), self.mol.natm))
    
        #For each atom, write the atom type
        ia2species = [
            self.atoms.index(self.mol.atom_symbol(ia))+1 
            for ia in range(self.mol.natm)]
        self.out_file.write(' '.join([str(species) for species in ia2species]))
        self.out_file.write('\t\t\t(iwctype(i), i= 1,ncent)\n')
    
        #For each atom type, write Z
        self.out_file.write(' '.join([str(gto.charge(atom)) for atom in self.atoms]))
        self.out_file.write('\t\t\t(znuc(i), i=1, nctype)\n')
    
        #For each atom, write the coordinates/atom type
        for ia in range(self.mol.natm):
            self.out_file.write(
                '%E\t%E\t%E\t%d\n'% 
                (tuple(self.mol.atom_coord(ia)) + (ia2species[ia],)))
        self.out_file.write('\n')
        return
    
    def get_atom_types(self):
        atoms = []
        for ia in range(self.mol.natm):
            atom = self.mol.atom_symbol(ia)
            if atom not in atoms:
                atoms += [atom]
        return atoms
    
    #WRITE NUMERICAL RADIAL BASIS FUNCTIONS
    #--------------------------------------
    def print_radial_bfs(self):
        #IF ANALYTIC, SKIP THIS
        for atom in self.atoms:
            self.make_radial_file(atom)
        return
    
    def make_radial_file(self, atom):
        filename = os.path.join(self.dir_qmc_inp, str(atom) + ".out")
        r0, rf, num_pts, x = 0, 7., 100, 1.03
        grid = p2d.radial_grid(r0, rf, num_pts, x)
        vals = p2d.radial_bf_vals(self.aos, atom, grid)
        self.write_radial_format(filename, grid, vals, num_pts, x, r0, rf) 
        return
    
    def write_radial_format(self, filename, grid, vals, num_pts, x, r0, rf):
        atom_file = open(filename, 'w')
        atom_file.write(
            ('%d %d %d %6.5f %6.5f %d (num_bfs, ?, num_points, h, rf, ?)\n'% 
            (len(vals), 3, len(grid), x, rf, 0)))
        data = [numpy.insert(val, 0, r) for r, val in zip(grid, vals.T)]
        for row in data:
            for entry in row:
                atom_file.write('%15.8E'% (Decimal(entry)))
            atom_file.write('\n')
        atom_file.close()
        return
    
    #WRITE ORBITAL INFORMATION
    #-------------------------
    def get_orb_coeffs(self):
        if self.opt_orbs:
            return numpy.matmul(self.rotation_matrix, self.mo_coeffs)
        return self.mo_coeffs
    
    def print_orbs(self):
        #Number of (non-empty OR empty) shells printed in output
        num_shells = 7
        for atom in self.atoms:
            self.out_file.write(
                p2d.occ_orbs_str(self.aos, atom, num_shells)[2:] 
                + '\tn1s,n2s,n2px,...\n')
            self.out_file.write(
                p2d.bf_str(self.aos, atom) + '\t(iwrwf(ib),ib=1,nbastyp)\n')
        orb_coeffs = self.get_orb_coeffs()
        for n, row in enumerate(orb_coeffs):
            for orb_coeff in row:
                self.out_file.write(
                    '%15.8E\t'% (orb_coeff if abs(orb_coeff) > 1e-12 else 0)) 
            if n == 0:
                self.out_file.write(
                    '\t((coef(ibasis, iorb), ibasis=1, nbasis) iorb=1, norb)')
            self.out_file.write('\n')
        for ao in self.aos:
            self.out_file.write('%15.8E\t'% ao.slater_exp())
        self.out_file.write(' (zex(ibas), ibas=1, nbas)\n')
        return
    
    #SETUP SHCI CALCULATION
    #----------------------
    def setup_shci(self):
        fcidump_path = os.path.join(self.dir_qmc_inp, 'FCIDUMP')
        if not os.path.exists(fcidump_path):
            self.make_fcidump(fcidump_path)
        if self.symmetry in ('dooh', 'coov'):
            self.make_real2complex_coeffs()

        complex_ints = self.symmetry in ('dooh', 'coov')
        self.ham = Ham(fcidump_path + '_real_orbs' if complex_ints else fcidump_path)

        config_path = os.path.join(self.dir_qmc_inp, 'config.json')
        with open(config_path, 'a') as f:
            json.dump(self.config, f)
        return

    def make_real2complex_coeffs(self):
        mo_coeff = self.mf.mo_coeff
        h1 = reduce(
            numpy.dot, 
            (mo_coeff.T, self.mf.get_hcore(), mo_coeff))
        if self.mf._eri is None:
            eri = ao2mo.full(self.mol, mo_coeff)
        else:
            eri = ao2mo.full(self.mf._eri, mo_coeff)
        nuc = self.mf.energy_nuc()
        orbsym = getattr(mo_coeff, 'orbsym', None) 
        self.real2complex_coeffs = \
            get_real2complex_coeffs(h1, eri, h1.shape[0], self.n_up + self.n_down, nuc, orbsym, 
                                    self.mf.partner_orbs) 
        return

    def make_fcidump(self, filename):
        print('filename: ', filename) 
        mo_coeff = self.mf.mo_coeff
        h1 = reduce(numpy.dot, (mo_coeff.T, self.mf.get_hcore(), mo_coeff))
        if self.mf._eri is None:
            eri = ao2mo.full(self.mol, mo_coeff)
        else:
            eri = ao2mo.full(self.mf._eri, mo_coeff)
        nuc = self.mf.energy_nuc()
        orbsym = getattr(mo_coeff, 'orbsym', None) 
        if self.symmetry in ('dooh', 'coov'):
            writeComplexOrbIntegrals(h1, eri, h1.shape[0], self.n_up + self.n_down, nuc, 
                                     orbsym, self.mf.partner_orbs)
            os.rename('FCIDUMP', filename)
            real_fcidump_path = filename + '_real_orbs'
            fcidump.from_integrals(real_fcidump_path, h1, eri, h1.shape[0], self.mol.nelec, nuc, 0, 
                                   orbsym)
        else:
            orbsym = [sym+1 for sym in orbsym]
            fcidump.from_integrals(filename, h1, eri, h1.shape[0], self.mol.nelec, nuc, 0, orbsym, 
                                   tol=1e-15, float_format=' %.16g')

    def test_fcidump(self, filename):
        mo_coeff = self.mf.mo_coeff
        orbsym = getattr(mo_coeff, 'orbsym', None) 
        mo_coeff = reduce(numpy.dot, (mo_coeff, self.real2complex_coeffs.conj().T))
        h1 = reduce(numpy.dot, (mo_coeff.conj().T, self.mf.get_hcore(), mo_coeff))
        if self.mf._eri is None:
            eri = ao2mo.full(self.mol, mo_coeff)
        else:
            eri = ao2mo.full(self.mf._eri, mo_coeff)
        nuc = self.mf.energy_nuc()
        orbsym = [sym+1 for sym in orbsym]
        fcidump.from_integrals(
            filename, h1, eri, h1.shape[0], self.mol.nelec, nuc, 0, orbsym,
            tol=1e-15, float_format=' %.16g')
    
    #GET SHCI DATA
    #-------------
    def get_shci_output(self):
        self.setup_shci()
        wf_file_exists = os.path.exists(self.wf_path)
        mat_file_exists = os.path.exists(self.rot_matrix_path)
        if (not wf_file_exists) or (self.opt_orbs and not mat_file_exists):
            print('wf_file: ', self.wf_path, wf_file_exists)
            print('self.rot_matrix_path: ', self.rot_matrix_path, mat_file_exists)
            print('Running shci...\n')
            self.run_shci()
            if self.opt_orbs:
                os.rename(os.path.join(self.dir_qmc_inp, 'rotation_matrix'), self.rot_matrix_path)

        #write FCIDUMP in real basis


        if self.opt_orbs:
                self.load_rotation_matrix()
        print('Loading WF from: ' + self.wf_path)
        print('Starting CSF calculation...')


        self.csfs, self.coefs, self.shci_wf = get_csfs(self.wf_path, self.tol_det, 
                                                       self.mol, self.mf, self.target_l2)
        self.proj_wf = Vec.add(self.csfs, self.coefs)
        self.sort_dets()
        self.sort_csfs()
        self.shci_e = self.ham.expec_val(self.shci_wf)
        self.proj_e = self.ham.expec_val(self.proj_wf)
        print("CSF calculation complete.")
        print("Error = %.8f" % self.error())
        print('Energies = %.8f, %.8f'% (self.shci_e, self.proj_e))
        return

    def run_shci(self):
        shci_output_path = os.path.join(self.dir_qmc_inp, 'shci.out')
        shci_output_file = open(shci_output_path, 'w')
        subprocess.run(self.shci_cmd.split(), stdout=shci_output_file, cwd = self.dir_qmc_inp)
        self.wf_path_init = os.path.join(self.dir_qmc_inp, self.wf_filename) #wf placed here by shci
        os.rename(self.wf_path_init, self.wf_path)
        return

    def load_rotation_matrix(self):
        rot_matrix_file = open(self.rot_matrix_path, 'r')
        self.rotation_matrix = []
        for line in rot_matrix_file:
            row = numpy.array([float(elmt) for elmt in line.strip().split()])
            self.rotation_matrix.append(row)
        self.rotation_matrix = numpy.array(self.rotation_matrix).T
        if self.symmetry in ('dooh', 'coov'):
            R = self.real2complex_coeffs
            self.rotation_matrix = reduce(numpy.dot, (R.conj().T, self.rotation_matrix, R)).real

    def sort_dets(self):
        class Pair:
            def __init__(self, det, coef):
                self.det, self.coef = det, coef
            def __lt__(self, other):
                if self.coef != other.coef: #deterministic det order
                    return abs(other.coef) < abs(self.coef)
                else:
                    return str(self.det) < str(other.det)

        sorted_pairs = sorted(Pair(det, coef) for det, coef in self.proj_wf.dets.items())
        self.dets = [p.det for p in sorted_pairs]

        zero_dets = set(det for csf in self.csfs for det in csf.dets) - set(self.dets)
        for det in zero_dets: #add dets with zero coef
            self.dets.append(det)
        return

    def sort_csfs(self):
        class Pair:
            def __init__(self, csf, coef):
                self.csf, self.coef = csf, coef
            def __lt__(self, other):
                if self.coef != other.coef: #deterministic csf order
                    return abs(other.coef) < abs(self.coef)
                else:
                    return str(self.csf) < str(other.csf)
        
        sorted_pairs = sorted(Pair(csf, coef) for csf, coef in zip(self.csfs, self.coefs))
        self.csfs, self.coefs = zip(*[(p.csf, p.coef) for p in sorted_pairs])
        return

    def error(self):
        err_wf = Vec.zero()
        err_wf += self.proj_wf
        err_wf += -1*(self.shci_wf)
        return (err_wf.norm()/self.shci_wf.norm())**2
    
    #OUTPUT SHCI DATA
    #----------------
    def print_shci(self):
        det_ind = {det : n+1 for n, det in enumerate(self.dets)}
        det_labels = self.label_by_energy(self.dets)
        for n, det in enumerate(self.dets):
            det_str = det.qmc_str() + '\t\t' + str(det_ind[det]) + '\t' + str(det_labels[det]) + '\n'
            if n == len(self.dets) - 1:
                det_str = det_str.strip('\n')
                det_str += ' (iworbd(iel,idet), iel=1, nelec)\n'
            self.out_file.write(det_str)

        #print ncsf
        ncsf_str = '%d ncsf\n'% len(self.csfs)
        self.out_file.write(ncsf_str)

        #print coefs of csfs in wf
        coef_str = '\t'.join(['%.10f'%coef for coef in self.coefs])
        coef_str += ' (csf_coef(icsf), icsf=1, ncsf)\n'
        self.out_file.write(coef_str)

        #print ndets per csf
        ndet_str = '\t'.join([str(len(csf.dets)) for csf in self.csfs])
        ndet_str += ' (ndet_in_csf(icsf), icsf1, ncsf)\n'
        self.out_file.write(ndet_str)

        #print csfs
        csf_labels = self.label_by_config(self.csfs)
        for csf in self.csfs:
            config = csf.config_label
            csf_str = ' '.join('%d'%det_ind[det] for det in csf.dets.keys())
            csf_str += ' (iwdet_in_csf(idet_in_csf,icsf),idet_in_csf=1,ndet_in_csf(icsf))\n'
            csf_str += ' '.join('%.8f'%coef for coef in csf.dets.values())
            csf_str += (' (cdet_in_csf(idet_in_csf,icsf),idet_in_csf=1,ndet_in_csf(icsf))'
                        + ' (%d)\n'%csf_labels[config]) 
            self.out_file.write(csf_str)
        return
    
    def label_by_energy(self, dets): #label dets by sum of orb energies
        def energy(det): #sum of orbital energies
            return sum([self.mf.mo_energy[orb-1] for orb in det.up + det.dn])

        sorted_dets = sorted([det for det in dets], key = lambda det: energy(det))
        labels, energy_set = {}, set()
        for det in sorted_dets:
            energy_str = '%.4f'% energy(det) #truncate to four decimal places
            if energy_str not in energy_set:
                energy_set.add(energy_str)
            labels[det] = len(energy_set)
        return labels

    def label_by_config(self, csfs): #label csfs by config
        labels = {}
        for config in [csf.config_label for csf in csfs]:
            if config not in labels:
                labels[config] = len(labels) + 1
        return labels

    def print_shci_old(self):
        csf_coeffs_str = '\t'.join(['%.10f' % coeff for coeff in self.wf_csf_coeffs])
        ndets_str = '\t'.join([str(len(csf_datum)) for csf_datum in self.csf_data])
        sorted_dets = sorted(
            [det for det in self.det_data.indices], key=lambda d: self.det_data.index(d))
        dets_str = '\n'.join([self.det_row_str(det, n) for n, det in enumerate(sorted_dets)])
        self.out_file.write(dets_str + ' (iworbd(iel,idet), iel=1, nelec)\n')
        self.out_file.write(str(len(self.csf_data)) + ' ncsf\n')
        self.out_file.write(csf_coeffs_str + ' (csf_coef(icsf), icsf=1, ncsf)\n')
        self.out_file.write(ndets_str + ' (ndet_in_csf(icsf), icsf=1, ncsf)\n')
 
        #'csf_data' is a list of 'csf's
        #'csf' is a list of (index, coeff) pairs for each det in the csf
        for csf, config in zip(self.csf_data, self.config_data):
            index_str = (' '.join([str(pair[0] + 1) for pair in csf]) +
                ' (iwdet_in_csf(idet_in_csf,icsf),idet_in_csf=1,ndet_in_csf(icsf))\n')
            coeff_str = (' '.join(['%.8f'%pair[1] for pair in csf]) +
                ' (cdet_in_csf(idet_in_csf,icsf),idet_in_csf=1,ndet_in_csf(icsf)) (%d)\n'%(config+1))
            self.out_file.write(index_str)
            self.out_file.write(coeff_str)
        return

    def det_row_str(self, det, n):
        label = self.det_config_labels[det]
        return det.qmc_str() + '\t\t' + str(n+1) + '\t' + str(label)

    #PRINT JASTROW/OPTIMIZATION SECTIONS
    #-----------------------------------

    def print_jastrow(self):
        self.out_file.write('\n\'* Jastrow section\'\n')
        self.out_file.write('1             ianalyt_lap\n')
        self.out_file.write('4 4 1 1 5 0   ijas,isc,nspin1,nspin2,nord,ifock\n')
        self.out_file.write('5 5 5         norda,nordb,nordc\n')
        self.out_file.write('1. 0. scalek,a21\n')
        self.out_file.write('0. 0. 0. 0. 0. 0. (a(iparmj),iparmj=1,nparma)\n')
        self.out_file.write('.5 1. 0. 0. 0. 0. (b(iparmj),iparmj=1,nparmb)\n')
        self.out_file.write('0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ' 
                            + '(c(iparmj),iparmj=1,nparmc)\n')
    
    def print_opt(self):
        self.out_file.write('\n\'* Optimization section\'\n')
        self.out_file.write('12 10000 1.d-8 0.05 1.d-4     ' 
                            + 'nopt_iter,nblk_max,add_diag(1),p_var,tol_energy\n') 
        self.out_file.write('INSERT NDATA LINE NDATA,NPARM,icusp,icusp2,NSIG,NCALLS,iopt,ipr\n')
        self.out_file.write('0 0 0 0 i3body,irewgt,iaver,istrech\n')
        self.out_file.write('0 0 0 0 0 0 0 0 0 0 ' 
                            + 'ipos,idcds,idcdr,idcdt,id2cds,id2cdr,id2cdt,idbds,idbdr,idbdt\n')
        self.out_file.write('1 '*len(self.mo_coeffs) + '(lo(iorb),iorb=1,norb)\n')
#        self.out_file.write('0  4  5  15  0  0 0 0  ' 
#                            + 'nparml,nparma,nparmb,nparmc,nparmf,nparmcsf,nparms,nparmg\n')
        self.out_file.write('INSERT NPARAM LINE '
                            + 'nparml,nparma,nparmb,nparmc,nparmf,nparmcsf,nparms,nparmgp\n')
        self.out_file.write('  (iworb(iparm),iwbasi(iparm),iparm=1,nlarml)\n')
        self.out_file.write('  (iwbase(iparm),iparm=1,nparm-nparml)\n')
        self.out_file.write('INSERT PARMCSF LINE  (iwcsf(iparm),iparm=1,nparmcsf)\n')
        self.out_file.write('    3 4 5 6 (iwjasa(iparm),iparm=1,nparma)\n')
        self.out_file.write('  2 3 4 5 6 (iwjasb(iparm),iparm=1,nparmb)\n')
        self.out_file.write('    3   5   7 8 9    11    13 14 15 16 17 18    20 21    23 '
                            + '(iwjasc(iparm),iparm=1,nparmc)\n')
        self.out_file.write('0 0       necn,nebase\n')
        self.out_file.write('          ((ieorb(j,i),iebasi(j,i),j=1,2),i=1,necn)\n')
        self.out_file.write('          ((iebase(j,i),j=1,2),i=1,nebase)\n')
        self.out_file.write('0 '*len(self.mo_coeffs) + '(ipivot(j),j=1,norb)\n')
        self.out_file.write('%6.2f'%self.hf_energy + ' eave\n')
        self.out_file.write('1.d-6 5. 1 15 4 pmarquardt,tau,noutput,nstep,ibold\n')
        self.out_file.write('T F analytic,cholesky\n')
        self.out_file.write('end\n\n')
        self.out_file.write('basis\n')
        self.out_file.write('which_analytical_basis = slater\n')
        self.out_file.write('optimized_exponents ' 
                            + ' '.join([str(n+1) for n in range(len(self.mo_coeffs))]) + ' end\n')
        self.out_file.write('end\n\n')
        self.out_file.write('orbitals\n')
        self.out_file.write(' energies\n')
        self.out_file.write(' '.join(["%.8f"%energy for energy in self.mf.mo_energy]) + '\n')
        self.out_file.write(' end\n')
        self.out_file.write(' symmetry\n')
        self.out_file.write(' '.join(self.orb_symm_labels) + '\n')
        self.out_file.write(' end\n')
        self.out_file.write('end\n\n')
        self.out_file.write('optimization\n')
        self.out_file.write(' parameters jastrow csfs orbitals end\n')
        self.out_file.write('  method = linear\n')
        self.out_file.write('!linear renormalize=true end\n')
        self.out_file.write(' increase_blocks=true\n')
        self.out_file.write(' increase_blocks_factor=1.4\n')
        self.out_file.write('casscf=true\n')
        self.out_file.write('check_redundant_orbital_derivative=false\n')
        self.out_file.write('!do_add_diag_mult_exp=.true.\n')
        self.out_file.write('end\n')

    #AUX
    #---
    def clear_file(self, filename):
        try:
            open(filename, 'r')
            raise Exception(filename + ' already exists. Remove it to continue.\n')
        except FileNotFoundError:
            open(filename, 'w').close()
        return
