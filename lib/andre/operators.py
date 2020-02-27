#!/usr/bin/env/python3

import numpy as np
from operator import mul
from functools import reduce

from shci4qmc.lib.andre.orbital import Orbital
from shci4qmc.lib.andre.determinant import Determinant
from shci4qmc.lib.andre.det_lin_comb import DeterminantLinearCombination


def multiply_all(iterable):
	return reduce(mul, iterable, 1)


class Operator:
	"""
	Represents many-body operators. Currently only supports many-body operators 
	written as a sum of single-body operators.

	Attributes
	----------
	transformations: dict
		Dictionary containing effect of acting with operator on different 
		orbitals. If we denote an operator as O, it is of the form:
		{
			orb1: O(orb1)
			....
		}
	cache: dict
		Stores the result of acting with operator on different determinants.
		Again denoting an operator as O, the dictionary is of the form:
		{
			det1: O(det1)
			...
		}
	cached_dets: dict
		Stores determinants in cache as both keys and values so we can keep
		track of the ordering of the orbitals.
		{
			det1: det1
			...
		}
	"""

	def __init__(self, transformations):
		self.transformations = transformations
		self.cache = {}
		self.cached_dets = {}

	def apply(self, state):
		if isinstance(state, (Orbital, Determinant)):
			state = state.to_det_lin_comb()

		elif not isinstance(state, (DeterminantLinearCombination)):
			return 0

		if state == DeterminantLinearCombination([], []):
			return 0

		new_det_lin_comb = DeterminantLinearCombination([], [])

		for det, coeff in state.det_coeffs.items():
			if det in self.cache:
				new_det_lin_comb += coeff * self.cache[det] * self.cached_dets[det].parity(det)
			else:
				new_det = DeterminantLinearCombination([], [])
				
				#	Apply operator to each orbital in determinant
				for i, orb in enumerate(det.orbitals):
					if i > 0:
						previous_orbs = Determinant(det.orbitals[0:i])
					else:
						previous_orbs = 1

					if i < len(det.orbitals) - 1:
						next_orbs = Determinant(det.orbitals[i+1::])
					else:
						next_orbs = 1

					if orb in self.transformations:
						aux = previous_orbs * self.transformations[orb] * next_orbs
						new_det_lin_comb += coeff * aux
						new_det += aux
					else:
						raise RuntimeError('Orbital', orb, 'is not defined in operator self.transformations.')

				self.cache[det] = new_det
				self.cached_dets[det] = det

		return new_det_lin_comb

	def __mul__(self, other):
		if isinstance(other, (Orbital, Determinant, DeterminantLinearCombination)):
			return self.apply(other)
		elif isinstance(other, (int, float, complex)) and other == 0:
			return self.apply(other)
		else:
			raise Exception("Not implemented")
