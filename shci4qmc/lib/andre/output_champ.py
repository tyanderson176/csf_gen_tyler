#!/usr/bin/env/python3

import math
import numpy as np


def parity(str1, str2):
	list1 = list(str1)
	list2 = list(str2)

	trans_count = 0
	for loc in range(len(list1) - 1):
		p0 = list1[loc]
		p1 = list2[loc]
		if p0 != p1:
			# Find position in list2
			sloc = list2[loc:].index(p0)+loc
			# Swap in list2
			list2[loc], list2[sloc] = p0, p1
			trans_count += 1

	if (trans_count % 2) == 0:
		return 1
	else:
		return -1


def write_csfs(input_params, orb_energies, champ_orbs, hf_orbs, csfs, ground_state_coeffs, gamess_idx, output_file):
	champ_inp_order_orb = ['1S', '2S', '2P_X', '2P_Y', '2P_Z', '3S', '3D_2Z^2-x^2-Y^2', '4S',
						   '3D_X^2-Y^2', '3D_XY', '3D_XZ', '3D_YZ', '4P_X', '4P_Y', '4P_Z',
						   '5S', '4D_X^2-Y^2', '4D_XY', '4D_XZ', '4D_YZ', '5P_X', '5P_Y',
						   '5P_Z', '6S']

	n_orb = len(hf_orbs[::2])
	dets = []
	dets_energies = []
	csf_det_coeffs = []
	csf_det_nums = []

	for i, csf in enumerate(csfs):
		csf.coeffs_to_integers()

	for i, csf in enumerate(csfs):
		temp_coeffs = []
		temp_dets = []

		for det, coeff in csf.det_coeffs.items():
			spin_up_elecs = []
			spin_dn_elecs = []

			for orb in det.orbitals:
				print("ORB: ", orb, "\nto: ", gamess_idx[orb])
				if orb.labels['s_z'] > 0:
					spin_up_elecs.append(gamess_idx[orb])
				else:
					spin_dn_elecs.append(gamess_idx[orb])

			alpha_up_elecs = list(range(1, math.ceil(input_params['alpha']/2)))
			alpha_dn_elecs = list(range(1, math.floor(input_params['alpha']/2)))
			print("ALPHA UP ELECS: ", alpha_up_elecs)
			sorted_det = [sorted(alpha_up_elecs + spin_up_elecs), sorted(alpha_dn_elecs + spin_dn_elecs)]
			print("SORTED_DET: ", sorted_det)
			
			if sorted_det not in dets:
				dets.append(sorted_det)
				det_energy = sum([orb_energies[abs(orb_num)-1] for orb_num in spin_up_elecs + spin_dn_elecs])
				det_energy = round(det_energy, 4)
				dets_energies.append(det_energy)
				det_idx = len(dets)
			else:
				det_idx = dets.index(sorted_det) + 1

			sorted_det_str = ''.join(map(str, sorted(spin_up_elecs))) + ''.join(['-' + str(spin) for spin in sorted(spin_dn_elecs)])
			unsorted_det_str = ''.join([str(gamess_idx[orb]) if orb.labels['s_z']>0 else str(-gamess_idx[orb]) for orb in det.orbitals ]) 
			sorted_det_list = sorted(spin_up_elecs) + [-spin for spin in sorted(spin_dn_elecs)]
			unsorted_det_list = [int(2*orb.labels['s_z']*gamess_idx[orb]) for orb in det.orbitals]
			temp_dets.append(det_idx)	
			temp_coeffs.append(parity(sorted_det_list, unsorted_det_list) * coeff)
		
		csf_det_nums.append(temp_dets)
		csf_det_coeffs.append(temp_coeffs)

	#	Write header 
	output_file.write("'ncsf=" + str(len(csfs)) + ' ndet=' + str(len(dets)) + ' norb=' + str(n_orb) + ' csf_sum= ' + str(np.sum(np.abs(ground_state_coeffs)**2)) + "'  title")
	output_file.write("\n1837465927472523                         irn")
	output_file.write("\n0 1 " + input_params['basis_functions'] + "                               iperiodic,ibasis,which_analytical_basis")
	output_file.write("\n0.5   " + str(input_params['total_energy']) + "  '  Hartrees'               hb,etrial,eunit")
	output_file.write('\n100   100  1   100   0                   nstep,nblk,nblkeq,nconf,nconf_new')
	output_file.write('\n0    0    1    -2                        idump,irstar,isite,ipr')
	output_file.write('\n6  1.  5.  1.  1.                        imetro delta,deltar,deltat fbias')
	output_file.write('\n2 1 1 1 1 0 0 0 0                        idmc,ipq,itau_eff,iacc_rej,icross,icuspg,idiv_v,icut_br,icut_e')
	
	if input_params['ecp']:
		output_file.write('\n50  .1                                   nfprod,tau')
	else:
		output_file.write('\n50  .01                                  nfprod,tau')

	if input_params['ecp']:
		output_file.write('\n6  -1   1  0                             nloc,numr,nforce,nefp')
	elif input_params['analytic_basis']:
		output_file.write('\n0  -3   1  0                             nloc,numr,nforce,nefp')
	else:
		output_file.write('\n0  1   1  0                              nloc,numr,nforce,nefp')
		
	output_file.write("\n" + str(input_params['alpha'] + input_params['beta']) + ' ' +  str(input_params['alpha']) + " 	 	 	 	 	 nelec,nup")

	#	Geometry section
	output_file.write("\n\n'* Geometry section'")
	output_file.write("\n3 	 	 	 	 	 ndim")

	unique_atom_symbols = list(set(input_params['atom_symbols']))
	nctype = len(unique_atom_symbols)
	ncent = len(input_params['atom_symbols'])
	output_file.write("\n" + str(nctype) + " " + str(ncent) + " 	 	 	 	 	 nctype,ncent\n")

	#	iwctype
	for atom in input_params['atom_symbols']:
		output_file.write(str(unique_atom_symbols.index(atom)+1) + ' ')
	output_file.write("(iwctype(i),i=1,ncent)")

	#	znuc
	#	TODO: MISSING ECP!!!
	unique_atomic_numbers = list(set(input_params['atomic_numbers']))
	for atom_number in unique_atomic_numbers:
		output_file.write('\n' + str(atom_number))
	output_file.write(" (znuc(i),i=1,nctype)")

	#	TODO: MISSING ECP!!!

	#	Coordinates
	for i, coordinates in enumerate(input_params['coordinates']):
		unique_atom_idx = unique_atom_symbols.index(input_params['atom_symbols'][i]) + 1
		output_file.write('\n  ' + '   '.join(coordinates) + ' ' + str(unique_atom_idx))
	output_file.write(" ((cent(k,i),k=1,3),i=1,ncent)")

	#	Write orbitals
	output_file.write("\n\n'* Determinantal section'")
	output_file.write("\n0 0 'tm' 	 	 	 	 inum_orb,iorb_used,iorb_format")
	output_file.write("\n  " + str(len(dets)) + ' ' + str(int(len(hf_orbs)/2)) + ' ' + str(int(len(hf_orbs)/2)) + '             ndet,nbasis,norb\n')

	#	Write out orbitals in order that CHAMP reads
	#	TODO: IMPLEMENT F!!!!!
	l_z_to_orient = {0: {0: 'S'},
					 1: {-1: 'Y', 0: 'Z', 1: 'X'},
					 2: {-2: 'XY', -1: 'YZ', 0: '2Z^2-X^2-Y^2', 1: 'XZ', 2: 'X^2-Y^2'},
					 3: {-3: '', -2: '', -1: '', 0: '', 1: '', 2: '', 3: ''},
					 4: {-4: '', -3: '', -2: '', -1: '', 0: '', 1: '', 2: '', 3: '', 4: ''}
					}

	l_to_char = {0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G'}

	#	Count each type of orbitals
	champ_orb_count = {}
	for orb in champ_orbs[::2]:
		n = orb.labels['n']
		l_char = l_to_char[orb.labels['l']]
		orientation = orb.labels['orientation']
		orb_str = str(n) + l_char + '_' + orientation
		if orb_str in champ_orb_count:
			champ_orb_count[orb_str] += 1
		else:
			champ_orb_count[orb_str] = 1

	#	Give indices to each of the orbitals
	champ_orb_idx = {}
	current_idx = 0
	output_file.write(' ')
	for i, n in enumerate(range(6)):
		for l in range(n):
			#	Ordering depends on value of L
			if l == 1:
				l_z_ordering = [1, -1, 0]
			elif l == 2:
				l_z_ordering = [0, 2, -2, 1, -1]
			else:
				l_z_ordering = list(range(-l, l+1))

			for l_z in l_z_ordering:
				orb_str = str(n) + l_to_char[l] + '_' + l_z_to_orient[l][l_z]

				#	Create indexes for orbitals
				for orb in champ_orbs[::2]:
					orb_n = orb.labels['n']
					orb_l = orb.labels['l']
					orb_s_z = orb.labels['s_z']
					orb_orient = orb.labels['orientation']

					#	We only label spin up orbitals to avoid duplication
					if (n, l, l_z_to_orient[l][l_z]) == (orb_n, orb_l, orb_orient):
						champ_orb_idx[orb] = current_idx
						current_idx += 1

				if orb_str in champ_orb_count:
					output_file.write(str(champ_orb_count[orb_str]) + ' ')
				else:
					output_file.write('0' + ' ')

			output_file.write(' ')
		output_file.write(' ')

	output_file.write(' n1s,n2s,n2px,n2py,np2z,...,n5g,sa,pa,da\n')

	sorted_champ_orbs = [orb for _, orb in sorted([(champ_orb_idx[orb], orb) for orb in champ_orbs[::2]])]
	all_slater_exps = [orb.labels['slater_exp'] for orb in sorted_champ_orbs]
	unique_slater_exps = sorted(set(all_slater_exps), key=all_slater_exps.index)
	unique_slater_exps_idx = [all_slater_exps.index(exp) for exp in unique_slater_exps]
	output_file.write(' '.join([str(unique_slater_exps.index(orb.labels['slater_exp']) + 1) for orb in sorted_champ_orbs]))
	output_file.write(' (iwrwf(ib),ib=1,nbastyp)\n')
	
	for i, orb in enumerate(hf_orbs[::2]):
		sorted_coeffs = len(champ_orbs)*[0]
		for det, coeff in orb.det_coeffs.items():
			champ_orb = det.orbitals[0]
			rounded_coeff = round(coeff.real, 7)
			if rounded_coeff == 1:
				rounded_coeff = int(rounded_coeff)
			sorted_coeffs[champ_orb_idx[champ_orb]] = rounded_coeff
		
		if i == 0:
			champ_comment = '((coef(ibasis,iorb),ibasis=1,nbasis),iorb=1,norb)'
			output_file.write(' '.join([str(coeff) for coeff in sorted_coeffs]) + ' ' + champ_comment + '\n')
		else:
			output_file.write(' '.join([str(coeff) for coeff in sorted_coeffs]) + '\n')

	#	Write Slater exponents
	slater_exp_and_idx = [(idx, orb.labels['slater_exp']) for orb, idx in champ_orb_idx.items()]
	sorted_slater_exp = [exp for idx, exp in sorted(slater_exp_and_idx)]
	output_file.write(' '.join([str(exp) for exp in sorted_slater_exp]))
	output_file.write(' (zex(i),i=1,nbasis)')

	#	Write determinants
	sorted_dets_energies = sorted(list(set(dets_energies)))

	for i, det in enumerate(dets):
		det[0] = [str(i) for i in det[0]]
		det[1] = [str(i) for i in det[1]]
		det_energy_idx = sorted_dets_energies.index(dets_energies[i]) + 1
		if i != len(dets) - 1:
			output_file.write('\n  ' + '   '.join(det[0]) + '      ' + '   '.join(det[1]) + '    ' + str(det_energy_idx) + ' ' + str(i+1))
		else:
			champ_str = ' (iworbd(iel,idet),iel=1,nelec), label_det(idet)'
			output_file.write('\n  ' + '   '.join(det[0]) + '      ' + '   '.join(det[1]) + '    ' + str(det_energy_idx) + ' ' + str(i+1) + champ_str + '\n')

	#	If first determinant coefficient is negative, 'transfer' that to CSF coeff
	for i in range(len(csf_det_coeffs)):
		coeffs_are_real = np.allclose(csf_det_coeffs[i], np.real(csf_det_coeffs[i]))
		if coeffs_are_real and csf_det_coeffs[i][0] < 0:
			csf_det_coeffs[i] = [-coeff for coeff in csf_det_coeffs[i]]
			ground_state_coeffs[i] = -ground_state_coeffs[i]

	#	If largest coefficient is negative, flip all signs
	if ground_state_coeffs[0] == ground_state_coeffs[0].real and ground_state_coeffs[0] < 0:
		ground_state_coeffs = [-coeff for coeff in ground_state_coeffs]
	
	#	Write CSFs
	output_file.write('   ' + str(len(csfs)) + ' ncsf\n')
	output_file.write('  ' +  ' '.join([str(coeff) for coeff in ground_state_coeffs]) + ' (csf_coef(icsf),icsf=1,ncsf)' + '\n')
	output_file.write('  ' +  '  '.join([str(len(output_csf)) for output_csf in csf_det_nums]) + ' (ndet_in_csf(icsf),icsf=1,ncsf)' '\n')

	for i, (coeffs, dets) in enumerate(zip(csf_det_coeffs, csf_det_nums)):
		det_line_str = '(iwdet_in_csf(idet_in_csf,' + str(i+1) + '),idet_in_csf=1,ndet_in_csf(' + str(i+1) + '))'
		coeff_line_str = '(cdet_csf(idet_in_csf,' + str(i+1) + '),idet_in_csf=1,ndet_in_csf(' + str(i+1) + '))'
		output_file.write('   ' + ' '.join([str(det) for det in dets]) + '  ' + det_line_str + '\n')
		output_file.write('   '  + ' '.join([str(coeff) for coeff in coeffs]) + '  ' + coeff_line_str + '\n')

	#	Write Jastrow section
	output_file.write("\n'* Jastrow section'")
	output_file.write("\n1             ianalyt_lap")
	output_file.write("\n4 4 1 1 5 0   ijas,isc,nspin1,nspin2,nord,ifock")

	if input_params['no_jastrow']:
		output_file.write("\n0 0 0         norda,nordb,nordc")
	else:
		jas_vars_str = [str(i) for i in [input_params['norda'], input_params['nordb'], input_params['nordc']]]
		output_file.write("\n" + ' '.join(jas_vars_str) + "         norda,nordb,nordc")

	if input_params['ecp']:
		output_file.write("\n0.4 0. scalek,a21")
	else:
		output_file.write("\n0.8 0. scalek,a21")
	
	for i in range(nctype):
		output_file.write("\n0. 0. 0. 0. 0. 0. (a(iparmj),iparmj=1,nparma)")
	
	if input_params['no_jastrow']:
		output_file.write("\n0. 0. 0. 0. 0. 0. (b(iparmj),iparmj=1,nparmb)")
	else:
		output_file.write("\n0.5 1. 0. 0. 0. 0. (b(iparmj),iparmj=1,nparmb)")

	for i in range(nctype):
		output_file.write("\n0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. (c(iparmj),iparmj=1,nparmc)")

	#	Write optimization section
	output_file.write("\n\n'* Optimization section'")
	if input_params['no_jastrow']:
		output_file.write("\n0 10000 1.d-8 0.05 1.d-4     nopt_iter,nblk_max,add_diag(1),p_var,tol_energy")
	else:
		output_file.write("\n10 10000 1.d-8 0.05 1.d-4     nopt_iter,nblk_max,add_diag(1),p_var,tol_energy")
	
	aux = 4*nctype + 5 + 15*nctype + len(csfs) - 1
	if input_params['ecp']:
		output_file.write('\n1000 ' + str(aux) + ' -1 1 5 1000 21101 1 NDATA,NPARM,icusp,icusp2,NSIG,NCALLS,iopt,ipr')		
	else:
		output_file.write('\n1000 ' + str(aux) + ' 1 1 5 1000 21101 1 NDATA,NPARM,icusp,icusp2,NSIG,NCALLS,iopt,ipr')		
	output_file.write("\n0 0 0 0 i3body,irewgt,iaver,istrech")
	output_file.write("\n0 0 0 0 0 0 0 0 0 0 ipos,idcds,idcdr,idcdt,id2cds,id2cdr,id2cdt,idbds,idbdr,idbdt\n")

	output_file.write(' '.join([str(list(orb.dets.keys())[0].orbitals[0].labels['l']) for orb in hf_orbs[::2]]))
	output_file.write('(lo(iorb),iorb=1,norb)\n')

	#	nparml
	output_file.write("0  ")

	#	nparma
	for i in range(nctype):
		output_file.write("4 ")

	output_file.write(" 5  ")

	#	nparmc
	for i in range(nctype):
		output_file.write("15 ")

	#	nparmf
	for i in range(nctype):
		output_file.write(" 0")

	#	nparmcsf/s/g
	output_file.write("  " + str(len(csfs)-1) + " 0 0  nparml,nparma,nparmb,nparmc,nparmf,nparmcsf,nparms,nparmg\n")
	output_file.write(" (iworb(iparm),iwbasi(iparm),iparm=1,nlarml)\n")
	output_file.write(' '.join([str(idx+1) for idx in unique_slater_exps_idx]))
	output_file.write(" (iwbase(iparm),iparm=1,nparm-nparml)\n")

	output_file.write('  ')
	for i in range(2, len(csfs) +1):
		output_file.write(str(i) + ' ')
	output_file.write('(iwcsf(iparm),iparm=1,nparmcsf)\n')

	for i in range(nctype):
		output_file.write('    3 4 5 6 (iwjasa(iparm),iparm=1,nparma)\n')

	output_file.write("2 3 4 5 6 (iwjasb(iparm),iparm=1,nparmb)\n")

	for i in range(nctype):
		output_file.write("    3   5   7 8 9    11    13 14 15 16 17 18    20 21    23 (iwjasc(iparm),iparm=1,nparmc)\n")

	output_file.write("0 0       necn,nebase\n")
	output_file.write("          ((ieorb(j,i),iebasi(j,i),j=1,2),i=1,necn)\n")

	identical_slater_exps = []
	for i in range(len(all_slater_exps)):
		if i in unique_slater_exps_idx:
			continue
		slater_exp = all_slater_exps[i]
		slater_exp_idx = unique_slater_exps_idx[unique_slater_exps.index(slater_exp)]
		identical_slater_exps.append(str(i+1) + ' ' + str(slater_exp_idx+1))

	output_file.write('  '.join(identical_slater_exps))
	output_file.write(" ((iebase(j,i),j=1,2),i=1,nebase)\n")

	output_file.write(str(input_params['total_energy']) + ' eave\n')
	output_file.write("1.d-6 5. 1 15 4 pmarquardt,tau,noutput,nstep,ibold\n")
	output_file.write("T F analytic,cholesky\n")
	output_file.write('end')

	#	Write basis
	output_file.write('\n\nbasis')
	output_file.write('\n which_analytical_basis = ' + str(input_params['basis_functions']))
	output_file.write('\noptimized_exponents ' + ' '.join([str(i+1) for i in range(int(len(hf_orbs)/2))]))
	output_file.write(' end')
	output_file.write('\nend')

	#	Write energies
	output_file.write('\n\norbitals \n energies\n')
	output_file.write(' '.join([str(energy) for energy in orb_energies]))
	output_file.write('\n end\n')
	output_file.write(' symmetry\n')

	#	Write symmetries
	orb_symmetries = []
	for orb in hf_orbs[::2]:
		orb_symmetries.append(list(orb.dets.keys())[0].orbitals[0].labels['orientation'])

	output_file.write(' '.join([symmetry for symmetry in orb_symmetries]))
	output_file.write('\n end\n')
	output_file.write('end\n')

	#	Optimization part
	output_file.write('\noptimization')
	output_file.write('\n parameters jastrow end')
	output_file.write('\n method = linear')
	output_file.write('\n!linear renormalize=true end')
	output_file.write('\n increase_blocks=true')
	output_file.write('\n increase_blocks_factor=1.4')
	output_file.write('\n!casscf=true')
	output_file.write('\n!check_redundant_orbital_derivative=false')
	output_file.write('\n!do_add_diag_mult_exp=.true.')
	output_file.write('\nend')
