#!/usr/bin/env/python3
# -*- coding: utf-8 -*-

import numpy as np
from operator import mul
from functools import reduce

import shci4qmc.lib.andre.orbital as orbital
import shci4qmc.lib.andre.determinant as determinant


tol = 1e-12

def product(iterable):
	return reduce(mul, iterable, 1)


class DeterminantLinearCombination:
	"""
	Linear combination of Slater determinants.

	Attributes
	----------
	det_coeffs: dict
		Dictionary containing determinants as keys and coefficients as values
	dets: dict
		Dictionary where determinants are both keys and values. Used in order to preserve 
		original order of determinants (determinants with the same orbitals and 
		different order have the same hash).
	N: int
		Number of electrons
	"""

	def __init__(self, coeffs, dets):
		if not all([isinstance(i, (int, float, complex)) for i in coeffs]):
			raise TypeError('coeffs list must only contain numbers.')
		elif not all([isinstance(i, determinant.Determinant) for i in dets]):
			raise TypeError('dets list must only contain instances of Determinant class.')

		self.det_coeffs = {}
		self.dets = {}

		for det, coeff in zip(dets, coeffs):
			if det:
				if det in self.det_coeffs:
					self.det_coeffs[det] += coeff * self.dets[det].parity(det)
				else:
					self.det_coeffs[det] = coeff
					self.dets[det] = det

		#	Eliminate determinants with zero coefficient
		for det, coeff in list(self.det_coeffs.items()):
			if np.abs(coeff) < tol:
				del self.det_coeffs[det]
				del self.dets[det]

		electron_numbers = [len(det.orbitals) for det in self.dets]
		if len(set(electron_numbers)) > 1:
			raise RuntimeError("Can't sum determinants with different number of electrons")

		self.N = electron_numbers[0] if electron_numbers else 0

	def get_det_coeff(self, det):
		"""Get determinant coefficient with appropriate Fermi anti-commutation factor"""
		if det in self.det_coeffs:
			return self.det_coeffs[det] * self.dets[det].parity(det)
		return 0

	def sorted(self):
		"""Sort each determinant"""
		new_det_coeffs = []
		new_dets = []
		for det in self.det_coeffs:
			sorted_det, new_parity = det.sorted()
			new_dets.append(sorted_det)
			new_det_coeffs.append(self.det_coeffs[det] * new_parity)

		return DeterminantLinearCombination(new_det_coeffs, new_dets)

	def dot(self, other):
		if self.dets == {} or other == 0:
			return 0

		if not self.N == other.N:
			raise RuntimeError("Can't take dot product of states with different number of electrons.")

		if not isinstance(other, (determinant.Determinant, orbital.Orbital, DeterminantLinearCombination)):
			raise RuntimeError("Can't take dot product of DeterminantLinearCombination with " + str(type(other)))

		if isinstance(other, (determinant.Determinant, orbital.Orbital)):
			return self.dot(other.to_det_lin_comb())

		dot_product = sum([coeff.conjugate() * other.get_det_coeff(det) for det, coeff in self.det_coeffs.items()])

		return dot_product

	def normalize(self):
		if self.dets:
			norm_factor = np.sqrt(sum([np.abs(coeff)**2 for coeff in self.det_coeffs.values()]))
			self.det_coeffs = {det:coeff/norm_factor for det, coeff in self.det_coeffs.items()}

	def change_basis(self, new_basis):
		new_det_lin_comb = DeterminantLinearCombination([], [])
		for det, coeff in self.det_coeffs.items():  
			new_det_lin_comb += coeff * product([new_basis[orb] for orb in det.orbitals])

		self.det_coeffs = new_det_lin_comb.det_coeffs
		self.dets = new_det_lin_comb.dets

	def coeffs_to_integers(self):
		"""Multiply determinants by a factor such that all coefficients are integers"""
		max_int = 50
		new_coeffs = list(self.det_coeffs.values())
		smallest = min([abs(i) for i in new_coeffs])
		new_coeffs = [coeff/smallest for coeff in new_coeffs]
		factor = 1

		for i in range(1, max_int):
			temp = [coeff * i for coeff in new_coeffs]
			integer_temp = [np.abs(np.around(i)) for i in temp]
			if np.allclose(integer_temp, temp):
				factor = i
				break 

		new_coeffs = [coeff * factor for coeff in new_coeffs]
		integer_new_coeffs = [np.rint(coeff) for coeff in new_coeffs] 

		try:
			if not np.allclose(integer_new_coeffs, new_coeffs):
				raise RuntimeError
			self.det_coeffs = {det:coeff for det, coeff in zip(self.det_coeffs.keys(), integer_new_coeffs)}
		except RuntimeError:
			print("Converting coefficients to integers failed.")
			pass

	def __mul__(self, other):
		if isinstance(other, (orbital.Orbital, determinant.Determinant)):
			return self * other.to_det_lin_comb()

		elif isinstance(other, DeterminantLinearCombination):
			new_coeffs = []
			new_dets = []
			for det1, coeff1 in self.det_coeffs.items():
				for det2, coeff2 in other.det_coeffs.items():
					new_coeffs.append(coeff1 * coeff2)
					new_dets.append(det1 * det2)

			return DeterminantLinearCombination(new_coeffs, new_dets)

		elif isinstance(other, (int, float, complex)):
			new_det_coeffs = {det:coeff*other for det, coeff in self.det_coeffs.items()}
			return DeterminantLinearCombination(new_det_coeffs.values(), new_det_coeffs.keys()) 
		
		else:
	 		raise Exception(type(other), '*', type(self), ' - Operation not supported')

	__rmul__ = __mul__

	def __truediv__(self, other):
		if isinstance(other, (int, float, complex)):
			return self * (1/other)
		else:
	 		raise Exception(type(other), '/', type(self), ' - Operation not supported')

	__rtruediv__ = __truediv__

	def __add__(self, other):
		if isinstance(other, (orbital.Orbital, determinant.Determinant)):
			return self + other.to_det_lin_comb()
		elif isinstance(other, (DeterminantLinearCombination)):
			if not ((self.N == other.N) or self.N == 0 or other.N == 0):
				raise RuntimeError("Can't sum determinants with different number of electrons.")
			return DeterminantLinearCombination(list(self.det_coeffs.values()) + list(other.det_coeffs.values()), list(self.det_coeffs.keys()) + list(other.det_coeffs.keys()))
		elif isinstance(other, (int, float, complex)) and other == 0:
			return self
		else:
			raise Exception(type(self), ' + ', type(other), ' - Operation not supported')

	__radd__ = __add__

	def __sub__(self, other):
		if isinstance(other, (orbital.Orbital, determinant.Determinant)):
			return self + other.to_det_lin_comb(coeff=-1)
		elif isinstance(other, (DeterminantLinearCombination)):
			if other.dets == {}:
				return self
			else:
				new_det_coeffs = [-coeff for coeff in other.det_coeffs.values()]
				dets = other.det_coeffs.keys()
				if self.dets == {}:
					return DeterminantLinearCombination(new_det_coeffs, dets)
				return self + DeterminantLinearCombination(new_det_coeffs, dets)
		elif isinstance(other, (int, float, complex)) and other == 0:
			return self
		else:
			raise Exception(type(self), ' + ', type(other), ' - Operation not supported')

	__rsub__ = __sub__

	def __bool__(self):
		return (self.det_coeffs != {} and self.dets != {})

	def __eq__(self, other):
		if other == 0:
			return self.det_coeffs == {}
		elif isinstance(other, (orbital.Orbital, determinant.Determinant)):
			return self == other.to_det_lin_comb()
		elif isinstance(other, DeterminantLinearCombination):
			return self.det_coeffs == other.det_coeffs
		else:
			return False
			
	def __str__(self):
		if self.det_coeffs:
			string_to_print = ''
			for i, (det, coeff) in enumerate(self.det_coeffs.items()):
				if coeff.real == coeff:
					sign = ' + ' if coeff.real > 0 else ' - '
					if abs(coeff) == 1:
						if i == 0 and coeff.real > 0:
							string_to_print += str(det)
						else:
							string_to_print += sign + str(det)
					else:
						string_to_print += sign + str(abs(coeff)) + ' ' + str(det)
				else:
					if i == 0:
						string_to_print += str(coeff) + ' ' + str(det)
					else:
						string_to_print += ' + ' + str(coeff) + ' ' + str(det)

			return string_to_print

		return '0'

	__repr__ = __str__
