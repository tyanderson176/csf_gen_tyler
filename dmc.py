from functools import reduce
import numpy
import subprocess
import p2d
import csf
from pyscf.tools import fcidump
from pyscf import ao2mo, gto

from decimal import Decimal

def print_header(fname, mol, aos, num_dets):
    f = open(fname, 'a')
    f.write('TODO: REST OF HEADER\n\n')
    print_geometry_header(f, mol)
    print_determinant_header(f, aos, num_dets)
    f.close()

def print_determinant_header(f, aos, num_dets):
    #TODO: num_dets should match csf diagonalization output
    f.write('\'* Determinantal section\'\n')
    f.write('0 0                    inum_orb\n')
    f.write('%d %d %d               ndet, nbasis, norb \n\n'% 
            (num_dets, len(aos), len(p2d.aos2mo_coeffs(aos))))
 
def print_geometry_header(f, mol):
    ''' Atoms written in order of increasing ia
    '''
    #Write number of atom types, number of atoms
    atoms = atom_types(mol) 
    f.write('\' * Geometry section\'\n')
    f.write('3\t\t\tndim\n')
    f.write('%d %d\t\t\tnctypes, ncent\n'% (len(atoms), mol.natm))
    #For each atom, write the atom type
    ia2species = [atoms.index(mol.atom_symbol(ia))+1 for ia in range(mol.natm)]
    f.write(' '.join([str(species) for species in ia2species]))
    f.write('\t\t\t(iwctype(i), i= 1,ncent)\n')
    #For each atom type, write Z
    f.write(' '.join([str(gto.charge(atom)) for atom in atoms]))
    f.write('\t\t\t(znuc(i), i=1, nctype)\n')
    #For each atom, write the coordinates/atom type
    for ia in range(mol.natm):
        f.write('%E\t%E\t%E\t%d\n'% (tuple(mol.atom_coord(ia)) + (ia2species[ia],)))
    f.write('\n')

def atom_types(mol):
    atoms = []
    for ia in range(mol.natm):
        atom = mol.atom_symbol(ia)
        if atom not in atoms:
            atoms += [atom]
    return atoms

def print_radial_bfs(mol, aos):
    atoms = atom_types(mol)
    for atom in atoms:
        fname = str(atom) + ".out"
        make_radial_file(aos, atom, fname)

def make_radial_file(aos, atom, fname):
    r0, rf, num_pts, x = 0, 7., 100, 1.03
    grid = p2d.radial_grid(r0, rf, num_pts, x)
    vals = p2d.radial_bf_vals(aos, atom, grid)
    write_radial_format(fname, grid, vals, num_pts, x, r0, rf) 

def write_radial_format(fname, grid, vals, num_pts, x, r0, rf):
    f = open(fname, 'w')
    f.write(('%d %d %d %6.5f %6.5f %d (num_bfs, ?, num_points, h, rf, ?)\n'% 
            (len(vals), 3, len(grid), x, rf, 0)))
    data = [numpy.insert(val, 0, r) for r, val in zip(grid, vals.T)]
    for row in data:
        for entry in row:
            f.write('%15.8E'% (Decimal(entry)))
        f.write('\n')
    f.close()

def get_orb_coeffs(aos, opt_orbs):
    if opt_orbs:
        return opt_orbs
    return p2d.aos2mo_coeffs(aos)

def print_orbs(fname, mol, aos, opt_orbs = False):
    f = open(fname, 'a')
    atoms = atom_types(mol)
    for atom in atoms:
        f.write(p2d.occ_orbs_str(aos, atom)[2:] + '\tn1s,n2s,n2px,...\n')
        f.write(p2d.basis_func_str(aos, atom) + '\tiorb=1, natom_orbs\n')
    orb_coeffs = get_orb_coeffs(aos, opt_orbs)
    for n, row in enumerate(orb_coeffs):
        for orb_coeff in row:
            f.write('%15.8E \t'% (orb_coeff if orb_coeff > 1e-15 else 0)) 
        if n == 0:
            f.write('\t((coef(ibasis, iorb), ibasis=1, nbasis) iorb=1, norb)')
        f.write('\n')
    #print dummy exponents
    for _ in range(mol.nbas):
        f.write('%15.8E\t'% 0)
    f.write('(bas_exp(ibas), ibas=1, nbas)\n')
    f.close()

def get_shci_output(mol, mf, eps_vars, eps_vars_sched,
        shci_path, shci_cmd, num_dets, cache):
    try:
        fh = open('FCIDUMP', 'r')
        print("FCIDUMP found.\n")
    except FileNotFoundError:
        print("FCIDUMP not found. Making new FCIDUMP...\n")
        h1 = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
        h2 = ao2mo.full(mf._eri, mf.mo_coeff)
        fcidump.from_integrals('FCIDUMP', h1, h2, h1.shape[0], 
                       mol.nelectron, tol=1e-15)
    try:
        fh = open('config.json', 'r')
        print("config.json found.\n")
    except FileNotFoundError:
        print("config.json not found. Making new config.json...\n")
        make_config(mol, mf, eps_vars, eps_vars_sched, num_dets)
    print("Running shci...\n")
    output = subprocess.run(shci_cmd.split(' '), capture_output = True)
    if shci_path == 'stdout':
        out = output.stdout.decode('ascii')
    else:
        f = open(shci_path)
        out = f.read()
    shci_out, opt_orbs = parse_output(out)
    print("Starting CSF calculation...")
    wf_csf_coeffs, csfs_info, det_indices, err = \
            csf.get_det_info(shci_out, cache)
    print("CSF calculation complete.")
    print("Projection error = %10.5f %%" % (100*err))
    return wf_csf_coeffs, csfs_info, det_indices, opt_orbs

def parse_output(out):
#    dmc_start = out.find('START DMC\n') 
#    dmc_end = out.find('END DMC')
#    shci_out = out[dmc_start+len('START DMC\n'):dmc_end]
    lines = out.split('\n')
    start_index = lines.index("START DMC")
    end_index = lines.index("END DMC")
    dmc_lines = lines[start_index: end_index+1]
    coef_index = dmc_lines.index("DET COEFFS:")
    s2_index = dmc_lines.index("S^2 VAL:") 
    S, det_strs, coeffs = -1, [], []
    for n, line in enumerate(dmc_lines):
        if 0 < n < coef_index: 
            det_strs.append(line)
        elif n == coef_index + 1:
            coeffs = line
        elif n == s2_index + 1:
            S = float(line) 
    shci_out = (S, det_strs, coeffs)
    return shci_out, False

def make_config(mol, mf, eps_vars, eps_vars_sched, num_dets):
    #TODO: inc error checking for symmetry (should be valid, like d2h, etc)
    n_up, n_dn = (mol.nelectron + mol.spin)//2, (mol.nelectron-mol.spin)//2
    sym = '\"' + str(mol.symmetry) + '\"'
    f = open('config.json', 'w')
    f.write('{\n')
    write_config_var(f, 'system', '\"chem\"')
    write_config_var(f, 'n_up', n_up)
    write_config_var(f, 'n_dn', n_dn)
    write_config_var(f, 'dmc_num_dets', num_dets)
    write_config_var(f, 'var_only', 'true')
    write_config_var(f, 'eps_vars', eps_vars)
    write_config_var(f, 'eps_vars_schedule', eps_vars_sched)
    sym_var = ('{\n' +
               '\t\t\"point_group\": ' + sym + '\n' +
               '\t}')
    write_config_var(f, 'chem', sym_var, end = '')
    f.write('}')
    f.close()

def write_config_var(f, var_name, vals, end = ","):
    f.write('\t"'+var_name+'": ')
    if not isinstance(vals, list):
        f.write(str(vals) + end + '\n')
        return
    elif len(vals) == 1:
        f.write(str(vals[0]) + end + '\n')
        return
    elif len(vals) > 1:
        f.write('[\n')
        for val in vals[:-1]:
            f.write('\t\t' + str(val) + ',\n')
        f.write('\t\t' + str(vals[-1]) + '\n')
        f.write('\t]' + end + '\n')
        return
    raise Exception('Passed null list as vals to put_config_var.')
    return

def print_shci(fname, wf_csf_coeffs, csfs_info, det_indices):
    csf_coeffs_str = '\t'.join(['%.10f' % coeff for coeff in wf_csf_coeffs])
    ndets_str = '\t'.join([str(len(csf_info)) for csf_info in csfs_info])
    f = open(fname, 'a')
    sorted_dets = sorted([det for det in det_indices.indices], 
            key=lambda d: det_indices.index(d))
    dets_str = '\n'.join([det.dmc_str() for det in sorted_dets])
    f.write(dets_str + '(iworbd(iel,idet), iel=1, nelec)\n')
    f.write(str(len(csfs_info)) + ' ncsf\n')
    f.write(csf_coeffs_str + '(csf_coef(icsf), icsf=1, ncsf)\n')
    f.write(ndets_str + '(ndet_in_csf(icsf), icsf=1, ncsf)\n')
    for csf_info in csfs_info:
        index_str = (' '.join([str(pair[0] + 1) for pair in csf_info]) +
            ' (iwdet_in_csf(idet_in_csf,icsf),idet_in_csf=1,ndet_in_csf(icsf))\n')
        coeff_str = (' '.join([str(pair[1]) for pair in csf_info]) +
            ' (cdet_in_csf(idet_in_csf,icsf),idet_in_csf=1,ndet_in_csf(icsf))\n')
        f.write(index_str)
        f.write(coeff_str)
    f.close()
    return

def clear_file(fname):
    open(fname, 'w').close()
    return

def make_dmc(mol, mf, eps_vars, eps_vars_schedule, shci_cmd,
        shci_path, num_dets, fname):
    wf_csf_coeffs, csfs_info, det_indices, opt_orbs = \
            get_shci_output(mol, mf, eps_vars, eps_vars_schedule, 
                    shci_path, shci_cmd, num_dets, cache = True)
    aos = p2d.mol2aos(mol, mf)
    clear_file(fname)
    print_header(fname, mol, aos, len(det_indices))
    print_radial_bfs(mol, aos)
    print_orbs(fname, mol, aos, opt_orbs)
    print_shci(fname, wf_csf_coeffs, csfs_info, det_indices)
