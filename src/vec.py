import copy
import numpy
import bisect

tol = 1e-15

class Vec:
    '''
    Represents a vector in Hilbert space, written in terms of Slater 
    determinants.  Initialized using a dict whose keys are Slater 
    determinants.
    '''
    def __init__(self, det_dict={}):
        self.dets = {key: det_dict[key] for key in det_dict 
                if abs(det_dict[key]) > tol}
        self.config_label = None

    def norm(self):
        if (not self.dets): 
            return 0.
        coeffs = numpy.array([self.dets[det] for det in self.dets])
        return numpy.sqrt(numpy.vdot(coeffs, coeffs))

    def dot(self, other):
        scalar_product = 0
        for det, coef in self.dets.items():
            if det in other.dets:
                scalar_product += numpy.conj(coef)*other.dets[det]
        return scalar_product

    def real_part(self):
        real_part = Vec.zero()
        for det, coef in self.dets.items():
            real_part += coef.real * det 
        return real_part

    def imag_part(self):
        imag_part = Vec.zero()
        for det, coef in self.dets.items():
            imag_part += coef.imag * det
        return imag_part

    @staticmethod
    def zero():
        return Vec({})

    @staticmethod
    def gram_schmidt(vecs, dim, tol=tol):
        orthonormal_basis = []
        for vec in vecs: #for vec in numpy.copy(vecs):
            new_vec = Vec.zero()
            new_vec += vec #vs. numpy.copy(vecs)
            if len(orthonormal_basis) == dim:
                return orthonormal_basis
            for basis_vec in orthonormal_basis:
                new_vec += -basis_vec.dot(new_vec) * basis_vec
            if new_vec.norm() <= tol:
                continue
            new_vec /= new_vec.norm()
            new_vec.config_label = vec.config_label
            orthonormal_basis.append(new_vec)
        return orthonormal_basis

    def __repr__(self):
        if len(self.dets) == 0:
            return '0'
        vec_str = ' + '.join([self._coeff_str(self.dets[det]) + str(det) 
                             for det in self.dets])
        return vec_str

    def _coeff_str(self, coeff):
        return '' if coeff == 1 else "%14.4e"%coeff

    def __iadd__(self, other):
        if isinstance(other, Vec):
            for det in other.dets:
                if det in self.dets:
                    self.dets[det] += other.dets[det]
                    if (abs(self.dets[det]) < tol): 
                        del self.dets[det]
                else:
                    self.dets[det] = other.dets[det]
            return self
        elif isinstance(other, Det):
            self += 1*other
            return self
        else:
            raise TypeError("Cannot add type " + str(type(other)) + " to Vec")

    def __imul__(self, scalar):
        for det, coef in self.dets.items():
            self.dets[det] *= scalar
        return self 

    def __rmul__(self, scalar):
        mul_dict = {}
        for det in self.dets:
            mul_dict[det] = scalar*self.dets[det]
        return Vec(mul_dict)

    def __truediv__(self, scalar):
        return (1./scalar)*self

    def __eq__(self, other):
        return self.dets == other.dets

class Det:
    def __init__(self, up_occ, dn_occ):
        self.up_occ = sorted(up_occ)
        self.dn_occ = sorted(dn_occ)
        self.up = self.up_occ
        self.dn = self.dn_occ
        self.my_hash = None #self._get_hash()

    def get_Sz(self):
        return (len(self.up_occ) - len(self.dn_occ))/2

    def qmc_str(self):
        qmc_str = ['%4d' % orb for orb in self.up_occ]
        qmc_str += '     '
        qmc_str += ['%4d' % orb for orb in self.dn_occ]
        return "".join(qmc_str)

    def __mul__(self, other):
        if isinstance(other, (int, float, numpy.int64, numpy.float64, complex)):
            return Vec({self: other})
        raise Exception("Unknown type \'" + str(type(other)) +
                "\' in Det.__mul__")

    __rmul__ = __mul__

    def __hash__(self):
        if not self.my_hash:
            self.my_hash = self._get_hash()
        return self.my_hash

    def _get_hash(self):
        #return hash(self.__repr__())
        return hash((tuple(self.up_occ), tuple(self.dn_occ)))

    def __eq__(self, other):
        return (self.up_occ == other.up_occ 
                and self.dn_occ == other.dn_occ)

    def __repr__(self):
        return self._get_repr()

    def _get_repr(self):
        det_str = '|'
        det_str += ' '.join([str(orb) for orb in self.up_occ])
        det_str += '; '
        det_str += ' '.join([str(orb) for orb in self.dn_occ])
        det_str += ')'
        return det_str

class Config:
    #Data structure for orbital configuration.
    #Gives occupation of each spacial orbital (0, 1 or 2)
    def __init__(self, det):
        self.occs = {}
        for up_orb in det.up_occ:
            self.occs[up_orb] = 1
        for dn_orb in det.dn_occ:
            self.occs[dn_orb] = 1 if dn_orb not in self.occs else 2
        self.orbs = self.occs
        self.num_open = \
            sum([1 if self.occs[orb] == 1 else 0 for orb in self.occs])
        self.config_str = self.make_config_str()

    @classmethod
    def fromorbs(cls, up_orbs, dn_orbs):
        return cls(Det(up_orbs, dn_orbs))

    def make_config_str(self):
        orbs = sorted([orb for orb in self.occs])
        return '_'.join([str(orb)+'S'+str(self.occs[orb]) for orb in orbs])

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        return self.config_str

    def __eq__(self, other):
        return self.occs == other.occs 

    def __lt__(self, other):
        return self.config_str < other.config_str
