#!/usr/bin/env/python3
# -*- coding: utf-8 -*-

import numpy as np
from operator import mul
from functools import reduce

import shci4qmc.lib.andre.orbital as orbital
import shci4qmc.lib.andre.det_lin_comb as det_lin_comb


class Determinant:
	"""
	A Slater determinant.

	Attributes
	----------
	orbitals_set: set
		Unordered set with unique orbitals in determinant.
	orbitals: list
		Ordered list of determinants.
	N: int
		Number of electrons in determinant.
	hash: int
		Hash is precomputed and stored during initialization for speed
	"""

	def __init__(self, orbitals):
		if not all([isinstance(i, orbital.Orbital) for i in orbitals]):
			raise TypeError('orbitals list must only contain instances of Orbital class.')

		self.orbitals_set = set(orbitals)
		if len(orbitals) != len(self.orbitals_set):
			self.orbitals = []
			self.N = 0
			self.orbitals_set = set()
		else:
			self.orbitals = orbitals
			self.N = len(self.orbitals)

		self.hash = hash('_'.join(sorted([orbital.labels['name'] for orbital in self.orbitals])))

	def to_det_lin_comb(self, coeff=1):
		"""Convert to DeterminantLinearCombination with one determinant and coefficient 1."""
		if self.orbitals:
			return det_lin_comb.DeterminantLinearCombination([coeff], [self])
		else:
			return det_lin_comb.DeterminantLinearCombination([], [])

	def parity(self, other):
		"""
		Check whether two Determinants are even permutations (returns 1) or odd 
		(returns -1) permutations of each other.
		"""
		if self.orbitals_set != other.orbitals_set:
			raise RuntimeError('Determinants are not permutations of each other.')

		other_orbitals = list(other.orbitals)

		trans_count = 0
		for loc in range(len(self.orbitals) - 1):
			p0 = self.orbitals[loc]
			p1 = other_orbitals[loc]
			if p0 != p1:
				sloc = other_orbitals[loc:].index(p0)+loc
				other_orbitals[loc], other_orbitals[sloc] = p0, p1
				trans_count += 1

		if (trans_count % 2) == 0:
			return 1
		else:
			return -1

	def dot(self, state):
		if isinstance(state, orbital.Orbital):
			if self.N == 1:
				if self.orbitals == [state]:
					return 1
				return 0
			raise TypeError("Can't take dot product of many-body determinant with Orbital.")

		if isinstance(state, Determinant):
			if self.N != state.N:
				raise ValueError("Can't take dot product of determinants with different number of electrons.")

			elif self.orbitals_set == state.orbitals_set:
				return self.parity(other)
			
			return 0

		elif isinstance(state, det_lin_comb.DeterminantLinearCombination):
			return self.to_det_lin_comb().dot(state)

		else:
			raise Exception(type(self), ' dot ', type(other), ' - Operation not supported')

	def sorted(self):
		sorted_orbitals = sorted(self.orbitals)
		new_det = Determinant(sorted_orbitals)
		new_parity = self.parity(new_det)

		return Determinant(sorted_orbitals), new_parity

	def change_basis(self, new_basis):
		det_lin_comb = self.to_det_lin_comb()
		det_lin_comb.change_basis(new_basis)
		
		return det_lin_comb

	def __mul__(self, other):
		if isinstance(other, orbital.Orbital):
			if self.orbitals == []:
				return Determinant([])
			return Determinant(self.orbitals + [other])

		elif isinstance(other, Determinant):
			if other.orbitals == [] or self.orbitals == []:
				return Determinant([])
			return Determinant(self.orbitals + other.orbitals)

		elif isinstance(other, det_lin_comb.DeterminantLinearCombination):
			return self.to_det_lin_comb() * other

		elif isinstance(other, (float, int, complex)):
			return self.to_det_lin_comb(coeff=other)

		else:
			raise Exception(type(self), ' * ', type(other), ' - Operation not supported')

	__rmul__ = __mul__

	def __add__(self, other):
		return self.to_det_lin_comb() + other

	__radd__ = __add__

	def __sub__(self, other):
		return self.to_det_lin_comb() - other

	__rsub__ = __sub__

	def __eq__(self, other):
		if other == 0:
			return self.orbitals == []
		if isinstance(other, orbital.Orbital):
			return self == other.to_det()
		elif isinstance(other, Determinant):
			return self.orbitals_set == other.orbitals_set
		elif isinstance(other, det_lin_comb.DeterminantLinearCombination):
			return self.to_det_lin_comb() == other
		else:
			return False

	def __bool__(self):
		return self.orbitals != []

	def __str__(self):
		list_to_print = [str(orb) for  i, orb in enumerate(self.orbitals)]
		if list_to_print:
			return '|' + ', '.join(list_to_print) + ' 〉'

		return '0'

	__repr__ = __str__

	def __hash__(self):
		return self.hash
