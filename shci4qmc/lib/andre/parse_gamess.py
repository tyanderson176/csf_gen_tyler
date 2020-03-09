#!/usr/bin/env/python3

import re

from shci4qmc.lib.andre.orbital import Orbital
from shci4qmc.lib.andre.determinant import Determinant
from shci4qmc.lib.andre.det_lin_comb import DeterminantLinearCombination


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_gamess_input(gamess_input_file):
	"""Read GAMESS input parameters."""
	gamess_input_file.seek(0)
	params = {
				'is_spher': False,
				'ecp': False,
				'atom_symbols': [],
				'atomic_numbers': [],
				'coordinates': [],
			}

	for line in gamess_input_file:
		if re.search('ISPHER\s*=\s*1', line):
			params['is_spher'] = True
		
		elif ' ATOM      ATOMIC' in line:
			next(gamess_input_file)
			new_line = next(gamess_input_file)

			while new_line.split() != []:
				params['atom_symbols'].append(new_line.split()[0])
				params['atomic_numbers'].append(new_line.split()[1])
				params['coordinates'].append(new_line.split()[2::])
				new_line = next(gamess_input_file)

		#	TODO: Read internuclear distances

		elif 'ECP POTENTIALS' in line:
			params['ecp'] = 1

		elif re.search('FROZEN CORE ENERGY =\s*([-]?[0-9]+[.][0-9]+)\s*$', line):
			params['fzc_energy'] = float(re.search('FROZEN CORE ENERGY =\s*([-]?[0-9]+[.][0-9]+)\s*$', line).group(1))

		elif re.search('^\s*THE NUCLEAR REPULSION ENERGY IS\s*([-]?[0-9]+[.][0-9]+)\s*$', line):
			params['nuc_repulsion'] = float(re.search('^\s*THE NUCLEAR REPULSION ENERGY IS\s*([-]?[0-9]+[.][0-9]+)\s*$', line).group(1))

		elif re.search('^\s*THE ADJUSTED NUCLEAR REPULSION ENERGY=\s*([-]?[0-9]+[.][0-9]+)\s*$', line):
			params['adj_nuc_repulsion'] = float(re.search('^\s*THE ADJUSTED NUCLEAR REPULSION ENERGY=\s*([-]?[0-9]+[.][0-9]+)\s*$', line).group(1))

		elif 'NUMBER OF OCCUPIED ORBITALS (ALPHA)' in line:
			params['alpha'] = int(re.search('=(.+)', line).group(1))

		elif 'NUMBER OF OCCUPIED ORBITALS (BETA )' in line:
			params['beta'] = int(re.search('=(.+)', line).group(1))

		elif 'STATE #' in line:
			params['total_energy'] = round(float(re.search('=(.+)', line).group(1)), 2)

	return params


def read_basis_orbitals(gamess_file):
	"""Outputs list of Orbital objects corresponding to GAMESS basis."""
	gamess_file.seek(0)
	l_vals = {'S': 0, 'P': 1, 'D': 2, 'F': 3}
	l_vals_exceptions = {
							'XX': 0,
							'XXX': 1,
							'YYY': 1,
							'ZZZ': 1,
							'XXXX': 0,
							'XXYY': 2,
							'XXZZ': 2,
							'XXXY': 2,
							'XXXZ': 2,
							'XXYZ': 2
						}
	orbs_by_shell = {
						'S': ['S'],
						'L': ['L'],
						'P': ['X', 'Y', 'Z'],
						'D': ['XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ'],
						'F': ['XXX', 'YYY', 'ZZZ', 'XXY', 'XXZ', 'YYX', 'YYZ', 'ZZX', 'ZZY', 'XYZ'],
						'G': ['XXXX', 'YYYY', 'ZZZZ', 'XXXY', 'XXXZ', 'YYYX', 'YYYZ', 'ZZZX', 'ZZZY', 'XXYY', 'XXZZ','YYZZ','XXYZ','YYXZ','ZZXY']
					}

	gamess_basis = []

	shells = []
	searching_shells = False

	for line in gamess_file:
		if ' SHELL TYPE PRIMITIVE    SLATER EXPONENT' in line:
			searching_shells = True
			#	Skip 4 lines
			for i in range(3):
				next(gamess_file)

		elif 'TOTAL NUMBER OF BASIS SET SHELLS' in line:
			break

		elif searching_shells and len(line.split()) == 6:
			shells.append([line.split()[1], float(line.split()[3])])

	for shell in shells:
		n = int(shell[0][0])
		l_char = shell[0][1]
		slater_exp = shell[1]

		for orb_orientation in orbs_by_shell[l_char]:
			if orb_orientation in l_vals_exceptions:
				l = l_vals_exceptions[orb_orientation]
			else:
				l = l_vals[l_char]

			orient_char = '' if orb_orientation == 'S' else '_' + orb_orientation

			name_up = str(n) + l_char + orient_char + '_(' + str(slater_exp)[0:4] + ')+'
			name_dn = str(n) + l_char + orient_char + '_(' + str(slater_exp)[0:4] + ')-'
			new_orb_up = Orbital({'name': name_up, 'n': n, 's': +0.5, 'orientation': orb_orientation, 'l': l, 's_z': +0.5, 'slater_exp': slater_exp})
			new_orb_dn = Orbital({'name': name_dn, 'n': n, 's': +0.5, 'orientation': orb_orientation, 'l': l, 's_z': -0.5, 'slater_exp': slater_exp})
			gamess_basis.append(new_orb_up)
			gamess_basis.append(new_orb_dn)

	return gamess_basis


def read_alpha_beta_electrons(gamess_file):
	gamess_file.seek(0)
	for line in gamess_file:
		if 'NUMBER OF ALPHA ELECTRONS' in line:
			alpha = int(line.strip()[-1])
		elif 'NUMBER OF  BETA ELECTRONS' in line:
			beta = int(line.strip()[-1])
			break
	return alpha, beta


def read_csfs(gamess_file):
	"""
	Reads CSFs in GAMESS format.

	Returns
	-------
	csfs: list
		List of CSFs with format:
		[
			[
				[
					[det1, det2, ...],
					...
				],
				[
				 	coeff1, 
				 	...
				],
			],
			...
		]
		det1, ... are integers (GAMESS format).

	"""
	gamess_file.seek(0)

	csfs = []
	reading_CSF = False
	
	for line in gamess_file:
		if 'CSF' in line  and 'C( ' in line:
			reading_CSF = True
			current_csf_coeffs = []
			current_csf_dets = []

		#	Line with coefficient
		if 'C( ' in line:
			#	In some lines there are no spaces between negative numbers in 
			#	the determinant	so we add them here
			line = line.replace('-', ' -')

			trimmed_line = line[line.index('=')+1::]
			det = []
			for i in trimmed_line.split():
				if is_number(i):
					if isinstance(eval(i), float):
						current_csf_coeffs.append(float(i))
					elif isinstance(eval(i), int):
						det.append(int(i))
			current_csf_dets.append(det)

		if reading_CSF and is_number(line):
			reading_CSF = False
			csfs.append([current_csf_dets, current_csf_coeffs])

	return csfs


def read_hf_orbitals(gamess_file, basis_orbitals):
	"""
	Reads HF orbitals and outputs them in GAMESS basis, as well as a list of 
	their energies.
	"""
	gamess_file.seek(0)

	orb_coeffs = []
	orb_energies = []
	searching_orbs = False
	line_coeffs = []
	hf_orbs = []

	#	Generate HF orbitals in terms of GAMESS basis
	for line in gamess_file:
		if 'INITIAL GUESS ORBITALS' in line and '----------------------' in next(gamess_file):
			searching_orbs = True

		if 'END OF INITIAL ORBITAL SELECTION' in line:
			orb_coeffs.extend(zip(*line_coeffs))
			break

		if searching_orbs:
			if is_number((''.join(line.split()))):
				if line_coeffs:
					orb_coeffs.extend(zip(*line_coeffs))
				line_coeffs = []
				energies = next(gamess_file)
				orb_energies.extend([float(en) for en in energies.split()])
				next(gamess_file)
			else:
				coeffs = [float(i) for i in line.split()[4::]]
				if coeffs:
					line_coeffs.append(coeffs)
	
	orb_coeffs = [list(coeffs) for coeffs in orb_coeffs]

	basis_orbitals_up = [orb for orb in basis_orbitals if orb.labels['s_z'] == 0.5]
	basis_orbitals_dn = [orb for orb in basis_orbitals if orb.labels['s_z'] == -0.5]

	for coeffs in orb_coeffs:
		new_orb_up = sum([coeff * orb for coeff, orb in zip(coeffs, basis_orbitals_up)])
		new_orb_dn = sum([coeff * orb for coeff, orb in zip(coeffs, basis_orbitals_dn)])
		hf_orbs.append(new_orb_up)
		hf_orbs.append(new_orb_dn)	

	return hf_orbs, orb_energies


def read_ground_state(gamess_file):
	"""Outputs list of ground state CSF coefficents and its energy."""

	gamess_file.seek(0)

	#	Find number of CSFs:
	for line in gamess_file:
		if ' COMPUTING THE HAMILTONIAN FOR THE' in line:
			csf_num = int(line.split()[5])
			
	csf_coeffs = csf_num*[0]
	reading_coeffs = False

	gamess_file.seek(0)
	for line in gamess_file:
		if ' STATE #' in line:
			gs_energy = round(float(line.split()[-1]), 2)

		if ' ...... END OF CI-MATRIX DIAGONALIZATION ......' in line:
			break
		elif reading_coeffs:
			csf_idx =  int(line.split()[0]) - 1
			coeff = float(line.split()[1])
			csf_coeffs[csf_idx] = coeff
		elif 'CSF      COEF    OCCUPANCY (IGNORING CORE)' in line:
			reading_coeffs = True
			next(gamess_file)

	return csf_coeffs, gs_energy
