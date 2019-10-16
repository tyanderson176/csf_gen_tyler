#!/usr/bin/env/python3

import determinant 
import det_lin_comb 


class Orbital:
	"""
	Class describing a single body state (orbital).

	Attributes
	----------
		labels: dict
			Labels that identify quantum state. The only required label 
			is 'name', which must be a string.
	"""

	def __init__(self, labels):
		if 'name' not in labels:
			raise ValueError('labels dict must contain name entry.')
		if not isinstance(labels['name'], str):
			raise TypeError("labels['name'] must be a string.")

		self.labels = labels

	def to_det(self):
		"""Convert orbital to single orbital determinant."""
		return determinant.Determinant([self])

	def to_det_lin_comb(self, coeff=1):
		"""Convert to DeterminantLinearCombination with one determinant and coefficient 1."""
		return det_lin_comb.DeterminantLinearCombination([coeff], [self.to_det()])

	def change_basis(self, rules):
		"""
		Convert self to DeterminantLinearCombination and apply change of basis.

		Arguments
		---------
		rules: dict
			Dictionary containing rules for basis change. rules must be of the
			form:
				{
					orb1: new_orb1
					...
				}
			orb1 must be Orbital object. new_orb1 can be Orbital, Determinant
			or DeterminantLinearCombination object. self must be one of the 
			keys in rules.

		"""
		det_lin_comb = self.to_det_lin_comb()
		det_lin_comb.change_basis(rules)
		return det_lin_comb

	def __mul__(self, other):
		if isinstance(other, Orbital):
			if other == self:
				return determinant.Determinant([], [])
			return determinant.Determinant([self, other])
		elif isinstance(other, determinant.Determinant):
			return self.to_det() * other
		elif isinstance(other, det_lin_comb.DeterminantLinearCombination):
			return self.to_det_lin_comb() * other
		elif isinstance(other, (int, float, complex)):
			return self.to_det_lin_comb(coeff=other)
		else:
			raise Exception(type(self), ' * ', type(other), ' - Operation not supported')

	__rmul__ = __mul__

	def __truediv__(self, other):
		return self.to_det_lin_comb() / other

	__rtruediv__ = __truediv__

	def __add__(self, other):
		return self.to_det_lin_comb() + other

	__radd__ = __add__

	def __sub__(self, other):
		return self.to_det_lin_comb() - other

	__rsub__ = __sub__

	def __eq__(self, other):
		if isinstance(other, Orbital):
			return (self.labels == other.labels)
		elif isinstance(other, determinant.Determinant):
			return self.to_det() == other
		elif isinstance(other, det_lin_comb.DeterminantLinearCombination):
			return self.to_det_lin_comb() == other
		else:
			return False

	def __ne__(self, other):
		return (not self == other)

	def __lt__(self, other):
		if 's_z' in self.labels and 's_z' in other.labels:
			if self.labels['s_z'] > other.labels['s_z']:
				return True
			elif self.labels['s_z'] < other.labels['s_z']:
				return False
		elif 'n' in self.labels and 'n' in other.labels:
			if self.labels['n'] > other.labels['n']:
				return True
			elif self.labels['n'] < other.labels['n']:
				return False
		elif 'l' in self.labels and 'l' in other.labels:
			if self.labels['l'] > other.labels['l']:
				return True
			elif self.labels['l'] < other.labels['l']:
				return False
		sorted_names = sorted([self.labels['name'], other.labels['name']])
		return (sorted_names[0] == self.labels['name'])

	def __str__(self):
		return self.labels['name']

	__repr__ = __str__

	def __hash__(self):
		return hash(self.labels['name'])
