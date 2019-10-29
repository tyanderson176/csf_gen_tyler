from functools import reduce
import numpy
import subprocess
from decimal import Decimal

import p2d
import gamess
import vec
from symm import SymMethods
from gen import GenMethods
from csf import CsfMethods

from pyscf.tools import fcidump
from pyscf import scf, ao2mo, gto, symm

class InputMaker(SymMethods, GenMethods, CsfMethods):
    #config = {eps_vars, eps_vars_schedule, num_dets}
    def __init__(self, mol, config, shci_cmd, wf_filename = None, basis_path = None,
                 optimize_orbs = False):
        assert(mol.unit == 'bohr')
        assert(mol.symmetry)
        self.out_file = None
        self.wf_filename = wf_filename if wf_filename else 'wf_eps1_%.2e.dat'%config['eps_vars'][-1]
        self.shci_cmd = shci_cmd

        #Use analytic basis external to pyscf
        self.mol = mol
        self.basis = gamess.get_basis(basis_path) if basis_path else None
        if self.basis:
            self.mol.basis = self.basis.get_pyscf_basis()
            self.mol.build()

        #Calculate molecular orbitals
        print("Starting RHF...")
        self.mf = scf.RHF(self.mol).run()
        print("Finished RHF.") 

        #Get atomic orbitals
        self.aos = p2d.mol2aos(self.mol, self.mf, self.basis)
        self.mo_coeffs = p2d.aos2mo_coeffs(self.aos)
        self.atoms = self.get_atom_types() 
        self.n_up, self.n_down = self.mol.nelec
        self.hf_energy = self.mf.energy_tot()
        #Optimized_orbs
        self.optimize_orbs = optimize_orbs
        self.rotation_matrix = None
        print('HF ENERGY: ' + str(self.hf_energy))

        #Import symmetry methods
        SymMethods.__init__(self)
        #Import csf generation methods
        GenMethods.__init__(self)
        #Import csf projection/formatting methods
        CsfMethods.__init__(self)
        
        #SHCI variables & output data
        self.config = config
        self.wf_csf_coeffs = None
        self.csf_data = None
        self.det_data = None

    def make_input(self, filename):
        #Updates wf_csfs_coeffs, csf_data, and det_data
        self.get_shci_output()

        #Set output file
        self.clear_file(filename)
        self.out_file = open(filename, 'a')

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
            (len(self.csf_data), len(self.det_data), len(self.mo_coeffs))) 
            + 'title\n')
        self.out_file.write(format_str.format('1837465927472523') + 'irn\n')
        self.out_file.write(
            format_str.format('0  1 slater') 
            + 'iperiodic,ibasis,which_analytical_basis\n')
        self.out_file.write(
            format_str.format('0.5  %10.3f  \'Hartrees\''% self.hf_energy) 
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
        num_dets = len(self.det_data)
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
        filename = str(atom) + ".out"
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
        if self.optimize_orbs:
            return numpy.dot(self.rotation_matrix, self.mo_coeffs)
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
        try:
            attempt = open('FCIDUMP', 'r')
            print("FCIDUMP found.\n")
        except FileNotFoundError:
            print("FCIDUMP not found. Making new FCIDUMP...\n")
            h1 = reduce(
                numpy.dot, 
                (self.mf.mo_coeff.T, self.mf.get_hcore(), self.mf.mo_coeff))
            if self.mf._eri is None:
                eri = ao2mo.full(self.mol, self.mf.mo_coeff)
            else:
                eri = ao2mo.full(self.mf._eri, self.mf.mo_coeff)
            nuc = self.mf.energy_nuc()
            orbsym = getattr(self.mf.mo_coeff, 'orbsym', None) 
            if self.symmetry in ('DOOH', 'COOV'):
                self.writeComplexOrbIntegrals(
                    h1, eri, h1.shape[0], self.n_up + self.n_down, nuc, orbsym, 
                    self.partner_orbs)
            else:
                orbsym = [sym+1 for sym in orbsym]
                fcidump.from_integrals(
                    'FCIDUMP', h1, eri, h1.shape[0], self.mol.nelec, nuc, 0, orbsym,
                    tol=1e-15, float_format=' %.16g')
        try:
            attempt = open('config.json', 'r')
            print("config.json found.\n")
        except FileNotFoundError:
            print("config.json not found. Making new config.json...\n")
            self.make_config()
        return
    
    def make_config(self):
        #Get variables
        eps_vars = self.config['eps_vars']
        eps_vars_sched = self.config['eps_vars_sched']
        n_up = (self.mol.nelectron + self.mol.spin)//2
        n_dn = (self.mol.nelectron - self.mol.spin)//2
        if not self.mol.symmetry:
            raise Exception(
                "Point group for molecule is required to run SHCI.\n"
                + "SHCI supports `C1`, `C2`, `Cs`, `Ci`, `C2v`, `C2h`, `Coov`," 
                + "`D2`, `D2h`, and `Dooh`.")

        symmetry = self.mol.symmetry
        sym = '\"' + str(symmetry) + '\"'
    
        #Write config file
        config = open('config.json', 'w')
        config.write('{\n')
        self.write_config_var(config, 'system', '\"chem\"')
        self.write_config_var(config, 'n_up', n_up)
        self.write_config_var(config, 'n_dn', n_dn)
        self.write_config_var(config, 'var_only', 'true')
        self.write_config_var(config, 'eps_vars', eps_vars)
        self.write_config_var(config, 'eps_vars_schedule', eps_vars_sched)

        if self.optimize_orbs:
            self.write_config_var(config, 'optorb', 'true')
            opt_orbs_vars = [
                '\"rotation_matrix\": true,', 
                '\"method\", \"appnewton\"',
                '\"accelerate\": true']
            self.write_config_var(config, 'optimization', opt_orbs_vars, curly = True)

        sym_var = ('{\n' +
                   '\t\t\"point_group\": ' + sym + '\n' +
                   '\t}')
        self.write_config_var(config, 'chem', sym_var, end = '')
        config.write('}')
        config.close()
        return
    
    def write_config_var(self, config_file, var_name, vals, end = ",", curly = False):
        config_file.write('\t"'+var_name+'": ')
        if not isinstance(vals, list):
            config_file.write(str(vals) + end + '\n')
            return
        elif len(vals) == 1:
            config_file.write(str(vals[0]) + end + '\n')
            return
        elif len(vals) > 1:
            open_paren, closed_paren = '{', '}' if curly else '[', ']' 
            config_file.write(open_paren + '\n')
            for val in vals[:-1]:
                config_file.write('\t\t' + str(val) + ',\n')
            config_file.write('\t\t' + str(vals[-1]) + '\n')
            config_file.write('\t' + closed_paren + end + '\n')
            return
        raise Exception('Passed null list as vals to write_config_var.')
        return

    #GET SHCI DATA
    #-------------
    def get_shci_output(self):
        self.setup_shci()
        try:
            wf_file = open(self.wf_filename, 'r')
            if self.optimize_orbs:
                rot_matrix_file = open('rotation_matrix', 'r')
        except FileNotFoundError:
            print("Running shci...\n")
            self.run_shci()
        print('Loading WF from: ' + self.wf_filename)
        print('Starting CSF calculation...')
        self.wf_csf_coeffs, self.csf_data, self.det_data, err = \
            self.get_csf_info(self.wf_filename)
        if self.optimize_orbs:
            self.rotation_matrix = self.load_rotation_matrix()
        print("CSF calculation complete.")
        print("Projection error = %10.5f %%" % (100*err))
        return

    def run_shci(self):
        process = subprocess.Popen(
            self.shci_cmd, shell=True, stdout=subprocess.PIPE,
            universal_newlines = True)
        shci_output_file = open('shci.out', 'w')
        for line in iter(process.stdout.readline, ''):
            shci_output_file.write(line)
        return

    def load_rotation_matrix(self):
        rot_matrix_file = open('rotation_matrix', 'r')
        self.rotation_matrix = numpy.array([])
        for line in rot_matrix_file:
            row = np.array([float(elmt) for elmt in line.strip().split(' ')])
            self.rotation_matrix.append(row)
        return

    #OUTPUT SHCI DATA
    #----------------
    def print_shci(self):
        csf_coeffs_str = '\t'.join(['%.10f' % coeff for coeff in self.wf_csf_coeffs])
        ndets_str = '\t'.join([str(len(csf_datum)) for csf_datum in self.csf_data])
        sorted_dets = sorted(
            [det for det in self.det_data.indices], 
            key=lambda d: self.det_data.index(d))
        dets_str = '\n'.join([det.qmc_str() for det in sorted_dets])
        self.out_file.write(dets_str + ' (iworbd(iel,idet), iel=1, nelec)\n')
        self.out_file.write(str(len(self.csf_data)) + ' ncsf\n')
        self.out_file.write(csf_coeffs_str + ' (csf_coef(icsf), icsf=1, ncsf)\n')
        self.out_file.write(ndets_str + ' (ndet_in_csf(icsf), icsf=1, ncsf)\n')
 
        #'csf_data' is a list of 'csf's
        #'csf' is a list of (index, coeff) pairs for each det in the csf
        for csf in self.csf_data:
            index_str = (' '.join([str(pair[0] + 1) for pair in csf]) +
                ' (iwdet_in_csf(idet_in_csf,icsf),idet_in_csf=1,ndet_in_csf(icsf))\n')
            coeff_str = (' '.join(['%.8f'%pair[1] for pair in csf]) +
                ' (cdet_in_csf(idet_in_csf,icsf),idet_in_csf=1,ndet_in_csf(icsf))\n')
            self.out_file.write(index_str)
            self.out_file.write(coeff_str)
        return

    #PRINT JASTROW/OPTIMIZATION SECTIONS
    #-----------------------------------

    def print_jastrow(self):
        self.out_file.write('\n\'* Jastrow section\'\n')
        self.out_file.write('1             ianalyt_lap\n')
        self.out_file.write('4 4 1 1 5 0   ijas,isc,nspin1,nspin2,nord,ifock\n')
        self.out_file.write('5 5 5         norda,nordb,nordc\n')
        self.out_file.write('1. 0. scalek,a21\n')
        self.out_file.write('0. 0. 0. 0. 0. 0. (a(iparmj),iparmj=1,nparma)\n')
        self.out_file.write('0. 1. 0. 0. 0. 0. (b(iparmj),iparmj=1,nparmb)\n')
        self.out_file.write('0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ' 
                            + '(c(iparmj),iparmj=1,nparmc)\n')
    
    def print_opt(self):
        self.out_file.write('\n\'* Optimization section\'\n')
        self.out_file.write('0 10000 1.d-8 0.05 1.d-4     ' 
                            + 'nopt_iter,nblk_max,add_diag(1),p_var,tol_energy\n') 
        self.out_file.write('1000 24 1 1 5 1000 21101 1 NDATA,NPARM,icusp,icusp2,NSIG,NCALLS,iopt,ipr\n')
        self.out_file.write('0 0 0 0 i3body,irewgt,iaver,istrech\n')
        self.out_file.write('0 0 0 0 0 0 0 0 0 0 ' 
                            + 'ipos,idcds,idcdr,idcdt,id2cds,id2cdr,id2cdt,idbds,idbdr,idbdt\n')
        self.out_file.write('1 '*len(self.mo_coeffs) + '(lo(iorb),iorb=1,norb)\n')
        self.out_file.write('0  4  5  15  0  0 0 0  ' 
                            + 'nparml,nparma,nparmb,nparmc,nparmf,nparmcsf,nparms,nparmg\n')
        self.out_file.write('  (iworb(iparm),iwbasi(iparm),iparm=1,nlarml)\n')
        self.out_file.write('  (iwbase(iparm),iparm=1,nparm-nparml)\n')
        self.out_file.write('  (iwcsf(iparm),iparm=1,nparmcsf)\n')
        self.out_file.write('    3 4 5 6 (iwjasa(iparm),iparm=1,nparma)\n')
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
        self.out_file.write('exit\n\n')
        self.out_file.write('optimization\n')
        self.out_file.write(' parameters jastrow end\n')
        self.out_file.write('  method = linear\n')
        self.out_file.write('!linear renormalize=true end\n')
        self.out_file.write(' increase_blocks=true\n')
        self.out_file.write(' increase_blocks_factor=1.4\n')
        self.out_file.write('!casscf=true\n')
        self.out_file.write('!check_redundant_orbital_derivative=false\n')
        self.out_file.write('!do_add_diag_mult_exp=.true.\n')
        self.out_file.write('end\n')

    #AUX
    #---
    def clear_file(self, filename):
        open(filename, 'w').close()
        return
