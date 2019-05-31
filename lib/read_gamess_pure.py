#!/usr/bin/env/python3
# -*- coding: utf-8 -*-

import numpy as np
from operator import mul
from output_champ import *
from parse_gamess import *
from functools import reduce

from csf import *
from orbital import Orbital
from operators import Operator
from determinant import Determinant
from det_lin_comb import DeterminantLinearCombination


l_to_char = {0: 'S', 1: 'P', 2: 'D', 3: 'F'}
warnings = []

def multiply_all(iterable):
	return reduce(mul, iterable)


def gamess_to_rh_basis(basis_orbitals, gamess_input_params):
	"""
	Build change of basis dictionary to go from GAMESS basis to CHAMP 
	real harmonics.
	"""
	gamess_to_rh = {}

	#	TODO: Implement F and G orbitals

	for orb in basis_orbitals:
		n = orb.labels['n']
		s = orb.labels['s']
		s_z = orb.labels['s_z']
		l = orb.labels['l']
		slater_exp = orb.labels['slater_exp']
		orient = orb.labels['orientation']

		#	These orbitals are the same as in GAMESS.
		if orient in ['S', 'X' ,'Y', 'Z']:
			gamess_to_rh[orb] = orb.to_det_lin_comb()
			if orb.labels['orientation'] in ['S', 'Z']:
				orb.labels['l_z'] = 0

		#	GAMESS P orbitals
		elif len(orient) == 2:
			#	These orbitals are the same as in GAMESS.
			if orient in ['XY', 'XZ', 'YZ']:
				gamess_to_rh[orb] = orb.to_det_lin_comb()
				continue

			rh_orient = ['2Z^2-X^2-Y^2', 'X^2-Y^2']
			orient_to_l = {'2Z^2-X^2-Y^2': 2, 'X^2-Y^2': 2}

			#	If contaminant cartesian functions are present, create extra
			#	orbital.
			if not gamess_input_params['is_spher']:
				rh_orient.append('S')
				orient_to_l['S'] = 0

			#	Instantiate all P real harmonics
			rh = {o:[] for o in rh_orient}
			for o in rh:
				spin_char = '+' if s_z > 0 else '-'
				l_rh = orient_to_l[o]
				l_char_rh = l_to_char[l_rh]
				orient_char_rh = '' if o == 'S' else '_' + o
				name = str(n) + l_char_rh + orient_char_rh + '_(' + str(slater_exp) + ')' + spin_char
				rh[o] = Orbital({'name': name, 'n': n, 's': s, 's_z': s_z, 'l': l_rh, 'slater_exp': slater_exp, 'orientation': o})
				
				if o == '2Z^2-X^2-Y^2':
					rh[o].labels['l_z'] = 0

			if not gamess_input_params['is_spher']:
				if orient == 'XX':
					gamess_to_rh[orb] = np.sqrt(5)/3 * rh['S'] - 1/3 * rh['2Z^2-X^2-Y^2'] + 1/np.sqrt(3) * rh['X^2-Y^2']
				elif orient == 'YY':
					gamess_to_rh[orb] = np.sqrt(5)/3 * rh['S'] - 1/3 * rh['2Z^2-X^2-Y^2'] - 1/np.sqrt(3) * rh['X^2-Y^2']
				elif orient == 'ZZ':
					gamess_to_rh[orb] = np.sqrt(5)/3 * rh['S'] + 2/3* rh['2Z^2-X^2-Y^2']
			else:
				if orient == 'XX':
					gamess_to_rh[orb] = -1/3 * rh['2Z^2-X^2-Y^2'] + 1/np.sqrt(3) * rh['X^2-Y^2']
				elif orient == 'YY':
					gamess_to_rh[orb] = -1/3 * rh['2Z^2-X^2-Y^2'] - 1/np.sqrt(3) * rh['X^2-Y^2']
				elif orient == 'ZZ':
					gamess_to_rh[orb] = 2/3* rh['2Z^2-X^2-Y^2']

	#	Find all champ basis orbitals used and sort them
	champ_basis_orbitals = []	
	for _, det_lin_comb in gamess_to_rh.items():
		for det in list(det_lin_comb.dets.keys()):
			champ_basis_orbitals.extend([orb for orb in det.orbitals if orb not in champ_basis_orbitals])

	sorted_champ_orbs = []

	for orb in basis_orbitals:
		n = orb.labels['n']
		l = orb.labels['l']
		s_z = orb.labels['s_z']
		slater_exp = orb.labels['slater_exp']
		for champ_orb in champ_basis_orbitals:
			champ_n = champ_orb.labels['n']
			champ_l = champ_orb.labels['l']
			champ_s_z = champ_orb.labels['s_z']
			champ_slater_exp = champ_orb.labels['slater_exp']

			if (champ_n, champ_l, champ_s_z, champ_slater_exp) == (n, l, s_z, slater_exp) and champ_orb not in sorted_champ_orbs:
				sorted_champ_orbs.append(champ_orb)
				break

	return gamess_to_rh, sorted_champ_orbs


def angular_part(hf_orbs):
	"""
	Take Hartree Fock orbitals from CHAMP and extract the angular part of the
	orbital. Note this assumes the orbitals are pure!
	"""

	#	Look for impure orbitals and keep only largest pure part.
	impure_orbs = 0
	for i, orb in enumerate(hf_orbs):
		orient_vals = [gamess_orb.orbitals[0].labels['orientation'] for gamess_orb in list(orb.dets.keys())]
		
		#	if impure
		if len(set(orient_vals)) != 1:
			impure_orbs += 1

			new_orb = DeterminantLinearCombination([], [])
			abs_coeff = [abs(coeff) for coeff in orb.det_coeffs.values()]
			max_coeff = abs_coeff.index(max(abs_coeff))
			max_orient = list(orb.det_coeffs.keys())[max_coeff].orbitals[0].labels['orientation']
			for det, coeff in orb.det_coeffs.items():
				if det.orbitals[0].labels['orientation'] == max_orient:
					new_orb += coeff * det

			print('---------')
			print("WARNING: found impure orbital. Keeping largest component.")
			print("Impure orbital:")
			print(orb)
			print('\n')
			print("New pure orbital:")
			print(new_orb)
			
			orb = new_orb

	#	Divide by two so we don't count spin up and spin down orbs
	impure_orbs = int(round(impure_orbs/2))
	if impure_orbs > 0:
		warnings.append("Found " + str(impure_orbs) + " impure orbitals.")

	#	Count orbitals for each value of orbital angular momentum
	n_vals = []
	current_n = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
	current_orbs_by_l = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
	for i, orb in enumerate(hf_orbs[::2]):
		l = list(orb.dets.keys())[0].orbitals[0].labels['l']
		if current_orbs_by_l[l] == 2*l+1:
			current_orbs_by_l[l] = 0
			current_n[l] += 1
		current_orbs_by_l[l] += 1
		n_vals.extend(2*[current_n[l]])

	#	Generate pure angular orbitals
	angular_orbitals = []
	ang_rh_to_full_orbs = {}
	gamess_idx = {}
	for i, orb in enumerate(hf_orbs):
		orientation = list(orb.dets.keys())[0].orbitals[0].labels['orientation']
		s_z = list(orb.dets.keys())[0].orbitals[0].labels['s_z']
		l = list(orb.dets.keys())[0].orbitals[0].labels['l']
		l_char = l_to_char[l]
		orientation = list(orb.dets.keys())[0].orbitals[0].labels['orientation']
		spin_char = '+' if s_z > 0 else '-'
		name = str(n_vals[i]) + l_char + '_' + orientation + spin_char
		
		new_orb = Orbital({'name': name, 's': +0.5, 'orientation': orientation, 'l': l, 's_z': s_z, 'n': n_vals[i]})
		angular_orbitals.append(new_orb)
		ang_rh_to_full_orbs[new_orb] = orb
		gamess_idx[new_orb] = int(i/2) + 1 if i%2 == 0 else int((i-1)/2) + 1

	return angular_orbitals, ang_rh_to_full_orbs, gamess_idx


def build_csfs(csf_gamess_format, ang_orbs):
	"""
	Take CSFs in GAMESS format and construct DeterminantLinearCombination 
	objects.
	"""
	ang_orbs_up = [orb for i, orb in enumerate(ang_orbs) if i%2 == 0]
	ang_orbs_dn = [orb for i, orb in enumerate(ang_orbs) if i%2 == 1]

	csfs = []
	for i, (det_list, coeffs) in enumerate(csf_gamess_format):
		csf = DeterminantLinearCombination([], [])
		for orb_nums, coeff in zip(det_list, coeffs):
			det = []
			for num in orb_nums:
				if num < 0:
					det.append(ang_orbs_dn[abs(num)-1])
				else:
					det.append(ang_orbs_up[abs(num)-1])
			csf += coeff * multiply_all(det)

		csf.normalize()
		csfs.append(csf)

	return csfs


def read_configurations_in_ground_state(ground_state):
	"""Find all configurations in CSFs that make up GAMESS ground state."""
	configs_str = []
	ground_state_config_proj = []

	for det, coeff in ground_state.det_coeffs.items():
		occupations = {}
		for orb in det.orbitals:
			n = orb.labels['n']
			l_char = l_to_char[orb.labels['l']]
			shell = str(n) + l_char
			if shell in occupations:
				occupations[shell] += 1
			else:
				occupations[shell] = 1
		config_str = '_'.join(sorted([shell + str(occup) for shell, occup in occupations.items()]))
		if config_str not in configs_str:
			configs_str.append(config_str)
			ground_state_config_proj.append(coeff * det)
		else:
			config_idx = configs_str.index(config_str)
			ground_state_config_proj[config_idx] += coeff * det

	for i in range(len(ground_state_config_proj)):
		ground_state_config_proj[i] = ground_state_config_proj[i].sorted()
		ground_state_config_proj[i].normalize()

	return configs_str, ground_state_config_proj

########################################
N = 3
desired_L = 2
desired_L_z = [2, -2]
desired_S = 0.5
desired_parity = 1

gamess_cutoff = 0	    # cutoff on GAMESS wave function
sym_csf_cutoff = 0.01   # cutoff on CSF components of GAMESS wave function
config_tol = 1	        # cutoff to stop generating CSFs for given config

analytic_basis = True
no_jastrow = False
norda = 5
nordb = 5
nordc = 5
basis_functions = 'slater'

# gamess_input_file = 'runs/ci_oh_orbs_cas4-8.inp'
# gamess_output_file = 'runs/ci_oh_orbs_cas4-8.out'
# champ_input_file = 'ci_oh_orbs_cas4-8_vmc_melo.inp2'
gamess_input_file = 'runs/ci_oh_orbs_cas3-9.inp'
gamess_output_file = 'runs/ci_oh_orbs_cas3-9.out'
champ_input_file = 'ci_oh_orbs_cas4-9_vmc_melo.inp2'
# gamess_input_file = 'runs/ci_pbe_oh_orbs_cas4-13.inp'
# gamess_output_file = 'runs/ci_pbe_oh_orbs_cas4-13.out'
# champ_input_file = 'ci_pbe_oh_orbs_cas4-13_vmc_melo.inp2'
# gamess_input_file = 'runs/ci_oh_orbs_ras5-21-3.inp'
# gamess_output_file = 'runs/ci_oh_orbs_ras5-21-3.out'
# champ_input_file = 'ci_oh_orbs_ras5-21-3.out_vmc_melo.inp2'

with open(gamess_output_file, 'r') as gamess_file, open(gamess_input_file, 'r') as input_file:
	input_params = read_gamess_input(gamess_file)
	input_params['analytic_basis'] = analytic_basis
	input_params['basis_functions'] = basis_functions
	input_params['no_jastrow'] = no_jastrow
	input_params['norda'] = norda
	input_params['nordb'] = nordb
	input_params['nordc'] = nordc

	basis_orbitals = read_basis_orbitals(gamess_file)
	gamess_to_champ, champ_orbs = gamess_to_rh_basis(basis_orbitals, input_params)

	hf_orbs, orb_energies = read_hf_orbitals(gamess_file, basis_orbitals)

	for i, orb in enumerate(hf_orbs):
		orb.change_basis(gamess_to_champ)

	ang_orbs_rh, ang_rh_to_full_orbs, gamess_idx = angular_part(hf_orbs)

	csfs_coeffs, gs_energy = read_ground_state(gamess_file)
	csf_gamess_format = read_csfs(gamess_file)
	csf_gamess_format = [csf for i, csf in enumerate(csf_gamess_format) if np.abs(csfs_coeffs[i]) > gamess_cutoff]
	csfs_coeffs = np.array([coeff for coeff in csfs_coeffs if np.abs(coeff) > gamess_cutoff])
	csfs_coeffs = csfs_coeffs/np.sqrt(np.sum(np.abs(csfs_coeffs)**2))
	csfs = build_csfs(csf_gamess_format, ang_orbs_rh)

	#	Build ground state
	ground_state = sum([coeff * csf for coeff, csf in zip(csfs_coeffs, csfs)])

	#	Build symmetrized CSFs through configurations
	configs_str, ground_state_config_proj = read_configurations_in_ground_state(ground_state)
	symmetrized_csfs = get_csfs_by_configs(N, desired_L, desired_S, desired_parity, configs_str, config_targets=ground_state_config_proj, config_tol=config_tol, desired_L_z=desired_L_z)

	#	Filter CSFs with coefficient smaller than the cutoff
	symmetrized_csfs_coeffs = [sym_csf.dot(ground_state) for sym_csf in symmetrized_csfs]
	symmetrized_csfs = [csf for i, csf in enumerate(symmetrized_csfs) if np.abs(symmetrized_csfs_coeffs[i]) > sym_csf_cutoff]
	symmetrized_csfs_coeffs = [coeff for coeff in symmetrized_csfs_coeffs if np.abs(coeff) > sym_csf_cutoff]

	#	Sort CSFs in descending order of coefficients
	symmetrized_csfs = [csf for (coeff, csf) in sorted(zip(np.abs(symmetrized_csfs_coeffs), symmetrized_csfs), key=lambda tup: tup[0])][::-1]
	symmetrized_csfs_coeffs = [coeff for (abs_coeff, coeff) in sorted(zip(np.abs(symmetrized_csfs_coeffs), symmetrized_csfs_coeffs))][::-1]
	symmetrized_csfs_coeffs = np.array(symmetrized_csfs_coeffs)

	unique_dets = set([det for csf in symmetrized_csfs for det in list(csf.dets.keys())])
	print(len(symmetrized_csfs), 'CSFs left after filtering with', len(unique_dets), 'unique determinants.')

	if symmetrized_csfs:
		with open(champ_input_file, 'w+') as output_gamess_file:
			write_csfs(input_params, orb_energies, champ_orbs, hf_orbs, symmetrized_csfs, symmetrized_csfs_coeffs, gamess_idx, output_gamess_file)

	if warnings:
		print("-------")
		print("RAN PROGRAM SUCCESSFULLY BUT WITH WARNINGS:")
		print('\n'.join(warnings))
	else:
		print("-------")
		print("RAN PROGRAM SUCCESSFULLY.")