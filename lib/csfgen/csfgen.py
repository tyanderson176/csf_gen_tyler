#!/usr/bin/env/python3
# -*- coding: utf-8 -*-

import itertools 
import numpy as np 
from copy import copy
from operator import mul
from functools import reduce

from orbital import Orbital 
from operators import Operator
from determinant import Determinant
from det_lin_comb import DeterminantLinearCombination


l_chars = {0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G', 5: 'H'}
l_vals = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5}

def multiply_all(iterable):
	"""Returns the product of all elements in an iterable."""
	return reduce(mul, iterable)


def gram_schmidt(csfs, orthonormal_csfs=[]):
	"""
	Creates an orthonormal basis of states.

	Parameters
	---------
		csfs: list 
			A list of CSFs to orthonormalize.
		orthonormal_csfs: list (optional)
			A list of CSFs that are already orthonormal. If supplied, CSFs will 
			be orthonormalized amongst themselves but also 	with respect to 
			all CSFs in orthonormal_csfs.
	"""
	
	N = len(csfs)
	orth_csfs = list(orthonormal_csfs)
	new_orthonormal_csfs = list(orthonormal_csfs)

	for csf in csfs:
		proj = 0 
		for orth_csf in orth_csfs:
			proj += orth_csf.dot(csf) * orth_csf *(1/orth_csf.dot(orth_csf))

		new_orth_csf = csf - proj
		if new_orth_csf:
			orth_csfs.append(new_orth_csf)
			onorm_new_csf = copy(new_orth_csf)
			onorm_new_csf.normalize()
			new_orthonormal_csfs.append(onorm_new_csf)

	return new_orthonormal_csfs


def configs_from_active_space(electrons_to_dist, current_configs):
	"""
	Recursively generates all possible configurations in an active space. Each 
	time the function is called it distributes one more electron in the current 
	configurations (current_configs). The recursion stops when all electrons 
	have been distributed, i.e. electrons_to_dist = 0.

	Parameters
	---------
	electrons_to_dist: int
		Number of electrons left to distribute among the current configurations 
		in current_configs.
	current_configs: list
		Configurations where a new electron will be distributed. A configuration 
		is a tuple of integers in the format (n, l, e_num), where n and l are 
		respectively the principal quantum number and angular momentum and e_num 
		is and the number of electrons on the n-l shell.

	Returns
	-------
		unique_configs: list 
			All configurations in the active space in the (n, l, e_num) format.
	"""

	if electrons_to_dist == 0:
		return current_configs
	
	new_configs = []

	#	Generate all configs
	for config in current_configs:
		for i, (n, l, e_num) in enumerate(config):
			if e_num < 2*(2*l + 1):
				new_config = list(config)
				new_config[i] = (n, l, e_num + 1)
				new_configs.extend(configs_from_active_space(electrons_to_dist-1, [new_config]))

	#	Remove duplicates
	unique_configs = []
	for config in new_configs:
		if config not in unique_configs:
			unique_configs.append(config)

	return unique_configs


def dets_from_configs(configs, desired_S, desired_parity, desired_L_z=[0], desired_S_z = None):
	"""
	Generates all determinants with a given L, S, parity and L_z.

	Parameters
	---------
		configs: list 
			List of configurations to generate determinants from. A configuration 
			is specified by a tuple of integers in the format (n, l, e_num), where 
			n and l are respectively the principal quantum number and angular 
			momentum and  e_num is and the number of electrons on the n-l shell.
		desired_S: int
		desired_parity: int (1 or -1)
		desired_L_z: list (optional)
			List of allowed L_z values.

	Returns
	-------
		all_dets: list
			List of Determinant objects with all the possible determinants 
			generated from configs.
		filtered_dets: list
			List of Determinant objects with the specified S_z, parity and L_z.
	"""

	#	Generate all possible determinants
	if desired_S_z == None:
		desired_S_z = [desired_S]

	all_dets = []
	for (n, l, num) in configs:
		if num == 0:
			continue

		possible_orbs = []
		for l_z in range(-l, l + 1):
			name_up = str(n) + l_chars[l] + '_' + str(l_z) + '+' 
			name_dn = str(n) + l_chars[l] + '_' + str(l_z) + '-' 
			new_orb_up = Orbital({'name': name_up, 'n': n, 's': +0.5, 'l': l, 'l_z': l_z, 's_z': +0.5})
			new_orb_dn = Orbital({'name': name_dn, 'n': n, 's': +0.5, 'l': l, 'l_z': l_z, 's_z': -0.5})
			possible_orbs += [new_orb_up, new_orb_dn]

		all_orb_combs = list(itertools.combinations(possible_orbs, num))
		all_orb_combs = [multiply_all(orbs) for orbs in all_orb_combs]

		if all_dets:
			all_dets = list(itertools.product(all_dets, all_orb_combs))
			all_dets = [det1 * det2 for det1, det2 in all_dets]
		else:
			all_dets = all_orb_combs

	#	Now filter determinants to those that have desired L_z and S_z
	filtered_dets = []
	for det in all_dets:
		tot_l_z = sum([orbital.labels['l_z'] for orbital in det.orbitals])
		tot_s_z = sum([orbital.labels['s_z'] for orbital in det.orbitals])
		parity = (-1)**sum([orbital.labels['l'] for orbital in det.orbitals])
		if parity == desired_parity and tot_l_z in desired_L_z and tot_s_z in desired_S_z:
			filtered_dets.append(det)

	return all_dets, filtered_dets


def range_angular_momentum(ang_mom_list, current_min=0, current_max=0):
	"""	
	Recursively determine maximum and minimum value of the sum of angular momentum 
	values.

	Arguments
	---------
		ang_mom_list: list
			List of angular momentum values to sum
	
	Notes
	-----
	current_min and current_max are only used internally for the  part and should 
	_not_ be specified by the user.

	Returns
	-------
		l_min: int
		l_max: int
	"""

	l_1 = ang_mom_list[0]
	l_min = min([np.abs(l_1 - l) for l in np.arange(current_min, current_max + 1)])
	l_max = l_1 + current_max
	if len(ang_mom_list) == 1:
		return l_min, l_max
	else:
		return range_angular_momentum(ang_mom_list[1::], current_min=l_min, current_max=l_max)


def build_operators(max_n):
	"""
	Generate orbital angular momentum operators (L_z, L_+, L_-) and spin operators 
	(S_z, S_+, S_-) for complex harmonic orbitals up to principal quantum number 
	max_n.
	"""

	l_minus_transf = {}
	l_plus_transf = {}
	l_z_transf = {}
	s_minus_transf = {}
	s_plus_transf = {}
	s_z_transf = {}

	for n in range(1, max_n+1):
		for l in range(0, n):
			for l_z in range(-l, l+1):
				for s_z in [-0.5, 0.5]:
					spin_char = '+' if s_z == 0.5 else '-'
					#print("l, lz: ", l, l_z)
					name = str(n) + l_chars[l] + '_' + str(l_z) + spin_char
					orb = Orbital({'name': name, 'n': n, 's': +0.5, 'l': l, 'l_z': l_z, 's_z': s_z})

					#	Generate orbital angular momentum operators
					l_z_transf[orb] = l_z*orb
					if l_z < l:
						new_name = str(n) + l_chars[l] + '_' + str(l_z+1) + spin_char
						new_orb = Orbital({'name': new_name, 'n': n, 's': +0.5, 'l': l, 'l_z': l_z+1, 's_z': s_z})
						l_plus_transf[orb] = np.sqrt(l*(l+1) - l_z*(l_z+1)) * new_orb
					else:
						l_plus_transf[orb] = 0

					if l_z > -l:
						new_name = str(n) + l_chars[l] + '_' + str(l_z-1) + spin_char
						new_orb = Orbital({'name': new_name, 'n': n, 's': +0.5, 'l': l, 'l_z': l_z-1, 's_z': s_z})
						l_minus_transf[orb] = np.sqrt(l*(l+1) - l_z*(l_z-1)) * new_orb
					else:
						l_minus_transf[orb] = 0

					#	Generate spin operators
					s_z_transf[orb] = s_z * orb
					if s_z == 0.5:
						s_plus_transf[orb] = 0
						name = str(n) + l_chars[l] + '_' + str(l_z) + '-'
						s_minus_transf[orb] = Orbital({'name': name, 'n': n, 's': +0.5, 'l': l, 'l_z': l_z, 's_z': -0.5})
					elif s_z == -0.5:
						s_minus_transf[orb] = 0
						name = str(n) + l_chars[l] + '_' + str(l_z) + '+'
						s_plus_transf[orb] = Orbital({'name': name, 'n': n, 's': +0.5, 'l': l, 'l_z': l_z, 's_z': +0.5})

	return Operator(l_z_transf), Operator(l_plus_transf), Operator(l_minus_transf), \
			Operator(s_z_transf), Operator(s_plus_transf), Operator(s_minus_transf)

def build_operators2(max_n):
	"""
	Generate orbital angular momentum operators (L_z, L_+, L_-) and spin operators 
	(S_z, S_+, S_-) for complex harmonic orbitals up to principal quantum number 
	max_n.
	"""

	l_minus_transf = {}
	l_plus_transf = {}
	l_z_transf = {}
	s_minus_transf = {}
	s_plus_transf = {}
	s_z_transf = {}

	for n in range(1, max_n+1):
		for l in range(0, 1): #WARNING
			for l_z in range(-l, l+1):
				for s_z in [-0.5, 0.5]:
					spin_char = '+' if s_z == 0.5 else '-'
					#print("l, lz: ", l, l_z)
					name = str(n) + l_chars[l] + '_' + str(l_z) + spin_char
					orb = Orbital({'name': name, 'n': n, 's': +0.5, 'l': l, 'l_z': l_z, 's_z': s_z})

					#	Generate orbital angular momentum operators
					l_z_transf[orb] = l_z*orb
					if l_z < l:
						new_name = str(n) + l_chars[l] + '_' + str(l_z+1) + spin_char
						new_orb = Orbital({'name': new_name, 'n': n, 's': +0.5, 'l': l, 'l_z': l_z+1, 's_z': s_z})
						l_plus_transf[orb] = np.sqrt(l*(l+1) - l_z*(l_z+1)) * new_orb
					else:
						l_plus_transf[orb] = 0

					if l_z > -l:
						new_name = str(n) + l_chars[l] + '_' + str(l_z-1) + spin_char
						new_orb = Orbital({'name': new_name, 'n': n, 's': +0.5, 'l': l, 'l_z': l_z-1, 's_z': s_z})
						l_minus_transf[orb] = np.sqrt(l*(l+1) - l_z*(l_z-1)) * new_orb
					else:
						l_minus_transf[orb] = 0

					#	Generate spin operators
					s_z_transf[orb] = s_z * orb
					if s_z == 0.5:
						s_plus_transf[orb] = 0
						name = str(n) + l_chars[l] + '_' + str(l_z) + '-'
						s_minus_transf[orb] = Orbital({'name': name, 'n': n, 's': +0.5, 'l': l, 'l_z': l_z, 's_z': -0.5})
					elif s_z == -0.5:
						s_minus_transf[orb] = 0
						name = str(n) + l_chars[l] + '_' + str(l_z) + '+'
						s_plus_transf[orb] = Orbital({'name': name, 'n': n, 's': +0.5, 'l': l, 'l_z': l_z, 's_z': +0.5})

	return Operator(l_z_transf), Operator(l_plus_transf), Operator(l_minus_transf), \
			Operator(s_z_transf), Operator(s_plus_transf), Operator(s_minus_transf)

def project_ang_mom(state, l_z, l_plus, l_minus, wrong_ang_mom_vals):
	"""
	Project out components of state with angular momentum values in wrong_ang_mom_vals 
	list. Works for both orbital angular momentum and spin.


	Parameters
	---------
	state: Orbital, Determinant or DeterminantLinearCombination
		State to project angular momentum out
	l_z: Operator
	l_plus: Operator
	l_minus: Operator
	wrong_ang_mom_vals: list
		Values of angular momentum to be projected out
	"""

	if wrong_ang_mom_vals == []:
		return state

	previous_state = state
	for l in wrong_ang_mom_vals:
		l_z_times_state = l_z*state
		state = l_minus*(l_plus*state) + l_z*(l_z_times_state) + l_z_times_state - (l*(l+1))*state
	state.normalize()

	return state


def real_harmon_basis(orbitals):	
	"""
	Given a list of complex harmonic orbitals, generate a dictionary to change basis to 
	real harmonics.

	Parameters
	---------
	orbitals: list
		List containing Orbital objects. These orbitals must represent 	complex harmonics
		and have the following labels:
			n - principal quantum number
			s - total spin
			s_z - z component of spin
			l - total orbital angular momentum
			l_z - z component of orbital ang. mom.

	Returns
	-------
	rh_basis: dict
		Dictionary that can be used with change_basis() method in Orbital, Determinant and 
		DeterminantLinearCombination objects.
	"""
	rh_basis = {}

	for orb in orbitals:
		n = orb.labels['n']
		s = orb.labels['s']
		s_z = orb.labels['s_z']
		l = orb.labels['l']
		l_z = orb.labels['l_z']
		spin_char = '+' if s_z > 0 else '-'

		if l == 0:
			orientations = ['S']
			l_z_to_orient = {0: 'S'}

		elif l == 1:
			orientations = ['X', 'Y', 'Z']
			l_z_to_orient = {0: 'Z', 1: 'X', -1: 'Y'}

		elif l == 2:
			orientations = ['2Z^2-X^2-Y^2', 'X^2-Y^2', 'XZ', 'XY', 'YZ']
			l_z_to_orient = {-2: 'XY', -1: 'YZ', 0: '2Z^2-X^2-Y^2', 1: 'XZ', 2: 'X^2-Y^2'}

		rh = {orient:[] for orient in orientations}
		for orient in rh:
			new_name = str(n) + l_chars[l] + '_' + orient  + spin_char
			new_labels = dict(orb.labels)
			new_labels.pop('l_z')
			new_labels['name'] = new_name
			new_labels['orientation'] = orient
			rh[orient] = Orbital(new_labels)

		if l_z == 0:
			rh_basis[orb] = rh[l_z_to_orient[l_z]]
		elif l_z > 0:
			rh_basis[orb] = (-1)**(l_z)/(np.sqrt(2)) * (rh[l_z_to_orient[np.abs(l_z)]] + 1j*rh[l_z_to_orient[-np.abs(l_z)]])
		elif l_z < 0:
			rh_basis[orb] = 1/(np.sqrt(2)) * (rh[l_z_to_orient[np.abs(l_z)]] - 1j*rh[l_z_to_orient[-np.abs(l_z)]])

	return rh_basis


def get_csfs_by_active_space(N, desired_L, desired_S, desired_parity, active_space_str, desired_L_z=[0]):
	"""
	Generate all CSFs in active space with desired orbital angular 
	momentum, spin and parity.

	Parameters
	---------
		N: int
			Number of electrons in active space.
		desired_L: int
		desired_S: int
		desired_parity: int (1 or -1)
		active_space_str: string
			String representing active space. Shells are specified by their principal quantum 
			number, orbital angular momentum letter (in caps) and separated by underscores.
			Examples of valid strings:
				1S_2S_2P
				2P_4S_5D
		desired_L_z: list (optional)
			List of desired values of z component of orbital angular momentum 

	Returns
	-------
		all_csfs: list 
			List of states with desired symmetries.
	"""

	configs = [(int(shell[0:-1]), l_vals[shell[-1]], 0) for shell in active_space_str.split('_')]
	print("GENERATING CONFIGS FROM ACTIVE SPACE")
	configs = configs_from_active_space(N, [configs])
	print("Generated", len(configs), "configs.\n\n")
	configs = ['_'.join([str(n) + l_chars[l] +str(e_num) for (n,l,e_num) in config if e_num]) for config in configs]

	return get_csfs_by_configs(N, desired_L, desired_S, desired_parity, configs, desired_L_z=desired_L_z)


def get_csfs_by_configs(N, desired_L, desired_S, desired_parity, configs_str, config_targets=[], config_tol=1, desired_L_z=[0], desired_S_z = None):

	"""
	Generate all CSFs in a set of configurations with desired orbital angular momentum, spin 
	and parity.

	Parameters
	---------
		N: int
			Number of electrons in active space.
		desired_L: int
		desired_S: int
		desired_parity: int (1 or -1)
		configs_str: list
			List of configurations to generate CSFs from. The configurations must be 
			specified in string form with underscores seperating the different shells. 
			Examples of valid strings:
				'1S2_2P2_3D1'
				'1S2_2P6'
		config_targets: list
			List of same length as config_str containing DeterminanLinearCombination
			objects (targets). For a given configuration, each time a new CSF is 
			generated the program will compute the projection of the target onto the 
			current CSFs for that configuration. Once the sum of the square modulus 
			of the coefficients exceeds config_tol, no more CSFs are generated.
		config_tol: float
			See config_targets.
		desired_L_z: list (optional)
			List of desired values of z component of orbital angular momentum 

	Returns
	-------
		all_csfs: list 
			List of states with desired symmetries.

	"""

	if desired_S_z == None:
		desired_S_z = [desired_S]
	#	Convert configs to (n, l, e_num) tuples
	configs = []
	for c_str in configs_str:
		c = []
		for shell_occup_str in c_str.split('_'):
			n = shell_occup_str[0:len(shell_occup_str)-2]
			l_char = shell_occup_str[-2]
			e_num = shell_occup_str[-1]
			c.append((int(n), l_vals[l_char], int(e_num)))
		configs.append(c)

	#	Generate CSFs for each CSF
	csfs = []
	for i, config in enumerate(configs):
		config_csfs = []

		print('------------------------------')
		print("CONFIGURATION:", configs_str[i], '(' + str(i+1) + ' of ' + str(len(configs)) + ')')
		print('------------------------------')

		max_n = np.max(np.array(config)[:, 0])
		l_z, l_plus, l_minus, s_z, s_plus, s_minus = build_operators2(max_n)	

		ang_mom_list = []
		for n, l, e_num in config:
			ang_mom_list += e_num*[l]
		l_min, l_max = range_angular_momentum(ang_mom_list)

		spin_list = N*[0.5]
		s_min, s_max = range_angular_momentum(spin_list)
		#print(s_min, s_max)

		if (l_min <= desired_L <= l_max) and (s_min <= desired_S <= s_max):
			wrong_ang_mom_vals = list(range(l_min, l_max + 1))
			wrong_spin_vals = list(np.arange(s_min, s_max + 1, 0.5))
			wrong_ang_mom_vals.remove(desired_L)
			wrong_spin_vals.remove(desired_S)
			
			print("GENERATING DETERMINANTS WITH S_z = ", desired_S, "AND L_z = ", ', '.join([str(l_z) for l_z in desired_L_z]) + "...")
			all_dets, filtered_dets = dets_from_configs(config, desired_S, desired_parity, desired_L_z=desired_L_z, desired_S_z=desired_S_z)
			print("Generated", len(filtered_dets), "determinants.")

			print("\n\nITERATING OVER DETERMINANTS TO GENERATE CSFS")

			if config_targets and config_tol:
				target = config_targets[i]
				current_dot_product = 0
			else:
				target = None

			for j, det in enumerate(filtered_dets):

				print('\n')
				print("Current determinant:", det, "(", j+1, "of", len(filtered_dets), ")")
				#print("Projecting L^2...")
				#projected_state = project_ang_mom(det, l_z, l_plus, l_minus, wrong_ang_mom_vals)
				print("Projecting S^2...")
				projected_state = project_ang_mom(det, s_z, s_plus, s_minus, wrong_spin_vals)
				#print("Changing basis to real harmonics...")
				#orbitals = [orbital for new_det in projected_state.dets for orbital in new_det.orbitals]
				#real_basis = real_harmon_basis(orbitals)
				#projected_state.change_basis(real_basis)

				if projected_state:
					config_csfs = gram_schmidt([projected_state.sorted()], orthonormal_csfs=config_csfs)
				if target:
					current_dot_product = sum([np.abs(target.dot(csf))**2 for csf in config_csfs])
					#print("Current dot product:", current_dot_product)
					if current_dot_product >= config_tol:
						#print("BREAKING CURRENT CONFIGURATION.")
						break

		print("\n\nGenerated ", len(config_csfs), "CSFs.")
		
		if config_csfs:
			print('Found', len(config_csfs), 'linearly independent states.')
			csfs.append(config_csfs)
		else:
			csfs.append([])

	print('------------------------------')
	#print("\n\nCALCULATED ALL CSFS, PRINTING RESULTS")
	
	unique_dets = []
	all_csfs = []
	for i, config in enumerate(configs):
		#	Create name for config for printing
		name = '_'.join([str(n) + l_chars[l] + str(e_num) for n,l,e_num in config if e_num>0])

		#print('\n\n------------------------------')
		#print("CONFIGURATION:", name)
		#print('------------------------------')

		if csfs[i]:
			for j, csf in enumerate(csfs[i]):
				#	If coefficients are pure imaginary numbers, make them real
				csf_coeffs = np.array(list(csf.det_coeffs.values()))
				if np.allclose(csf_coeffs, 1j*csf_coeffs.imag):
					csf.det_coeffs = {det:coeff.imag for det, coeff in csf.det_coeffs.items()}

				#print('\n\n')
				#print('CSF #', j+1, '(with', len(csf.dets), 'determinants)')
				csf.coeffs_to_integers()

				#if np.all(np.array(list(csf.det_coeffs.values())) < 0):
					#print(-1 * csf)
				#else:
					#print(csf)
				#print('\n\n')

				for det in csf.dets:
					if det not in unique_dets:
						unique_dets.append(det)
				all_csfs.append(csf)
		#else:
			#print("No CSFs were found for this configuration.")

	#print("\n------")
	print("Generated", len(all_csfs), "CSFs with", len(unique_dets), "unique determinants.")

	return all_csfs

def compute_csfs(config_str, S, Sz):
    nelec = lambda orb_str : int(orb_str[orb_str.index('S')+1:])
    N = sum([nelec(orb_str) for orb_str in config_str.split('_')])
    L = 0
    Lzs = [0]
    Szs = [Sz]
    parity = 1
    return get_csfs_by_configs(N, L, S, parity, [config_str], desired_L_z=Lzs, desired_S_z=Szs)

if __name__ == '__main__':
	N = 4
	desired_L = 0
	desired_S = 0.5 
	desired_parity = 1
	desired_L_z = [0]
	desired_S_z = [0.5]
	l_z, l_plus, l_minus, s_z, s_plus, s_minus = build_operators2(N)	

	#	Generate CSFs in set of configurations
	#configs_str = ['2S2_2P2', '2P4', '2S2_3D2', '2S1_2P2_3D1']
	configs_str = ['1S1_4S1_5S1_6S2_7S2']
	configs_str2 = ['1S1_4S1_5S1']
	csfs = get_csfs_by_configs(N, desired_L, desired_S, desired_parity, configs_str, desired_L_z=desired_L_z, desired_S_z=desired_S_z)
	print(csfs)
	csfs = get_csfs_by_configs(N, desired_L, desired_S, desired_parity, configs_str2, desired_L_z=desired_L_z, desired_S_z=desired_S_z)
	print(csfs)
	#print(csfs[0])	
	#csf = csfs[0]
	#for det in csf.dets:
	#	print(det)
	#	for orb in det.orbitals:
	#		print(orb.labels)
	#	Generate CSFs in active space
	# active_space_str = '2S_2P_3D'
	# get_csfs_by_active_space(N, desired_L, desired_S, desired_parity, active_space_str, desired_L_z=desired_L_z)
