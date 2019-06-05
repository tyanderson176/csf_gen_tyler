import copy
import numpy

tol = 1e-15

class Vec:
    '''
    Represents a vector in Hilbert space, written in terms of Slater 
    determinants.  Initialized using a dict whose keys are Slater 
    determinants.
    '''
    def __init__(self, det_dict={}):
        self.det_dict = {key: det_dict[key] for key in det_dict 
                if abs(det_dict[key]) > tol}
        self.dets = [det for det in self.det_dict]
        self.coeffs = [det_dict[det] for det in self.dets] 

    def norm(self):
        coeffs = numpy.array(self.coeffs)
        return numpy.sqrt(numpy.dot(coeffs, coeffs))

    @staticmethod
    def zero():
        return Vec({})

    def __repr__(self):
        if len(self.dets) == 0:
            return '0'
        vec_str = ' + '.join([self._coeff_str(coeff) + str(det) 
            for det, coeff in zip(self.dets, self.coeffs)])
        return vec_str

    def _coeff_str(self, coeff):
        return '' if coeff == 1 else str(coeff)

    def __add__(self, other):
        sum_dict = copy.deepcopy(self.det_dict)
        for det in other.det_dict:
            if det in sum_dict:
                sum_dict[det] += other.det_dict[det]
            else:
                sum_dict[det] = other.det_dict[det]
        return Vec(sum_dict)

    def __rmul__(self, scalar):
        mul_dict = {}
        for det in self.det_dict:
            mul_dict[det] = scalar*self.det_dict[det]
        return Vec(mul_dict)

    def __hash__(self):
        #TODO: Compute only once; calc in __init__, return result here
        #Safe to add hash values? Should order dets/coeffs and
        #simply use hash(self.__repr__()).
        det_strs = [str(coeff) + str(det) 
                for det, coeff in zip(self.dets, self.coeffs)]
        return sum([hash(det_str) for det_str in det_strs])

    def __eq__(self, other):
        return self.det_dict == other.det_dict

class Det:
    def __init__(self, up_occ, dn_occ):
        #TODO: Handle permutation?
        self.up_occ = sorted(up_occ)
        self.dn_occ = sorted(dn_occ)
        parity = (self._parity(up_occ, self.up_occ)
                *self._parity(dn_occ, self.dn_occ))
        if parity != 1:
            raise Exception("Occs supplied to Det have wrong parity.")

    def get_Sz(self):
        #TODO: Is this safe?
        return (len(self.up_occ) - len(self.dn_occ))/2

    def dmc_str(self):
        dmc_str = ['%4d' % orb for orb in self.up_occ]
        dmc_str += '     '
        dmc_str += ['%4d' % orb for orb in self.dn_occ]
        return "".join(dmc_str)

    def __mul__(self, other):
        if isinstance(other, (int, float, numpy.int64, numpy.float64)):
            return Vec({self: other})
        raise Exception("Unknown type \'" + str(type(other)) +
                "\' in Det.__mul__")

    __rmul__ = __mul__

    def _parity(self, occ1, occ2):
        occ1, occ2 = copy.deepcopy(occ1), copy.deepcopy(occ2)
        num_perms = 0
        for n, orb in enumerate(occ1):
            index = occ2.index(orb)
            if index == -1:
                raise Exception(
                        "Occupation strings not permutations in _parity")
            if index != n:
                occ2[index], occ2[n] = occ2[n], occ2[index]
                num_perms += 1
        return (-2)*num_perms%2 + 1 

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return (self.up_occ == other.up_occ 
                and self.dn_occ == other.dn_occ)

    def __repr__(self):
        det_str = '|'
        det_str += ' '.join([str(orb) for orb in self.up_occ])
        det_str += '; '
        det_str += ' '.join([str(orb) for orb in self.dn_occ])
        det_str += ')'
        return det_str

class Config:
    def __init__(self, det):
        self.occs = {}
        for up_orb in det.up_occ:
            self.occs[up_orb] = 1
        for dn_orb in det.dn_occ:
            self.occs[dn_orb] = (1 if dn_orb not in self.occs else 2)
        self.num_open = \
            sum([1 if self.occs[orb] == 1 else 0 for orb in self.occs])

    @classmethod
    def fromorbs(cls, up_orbs, dn_orbs):
        return cls(Det(up_orbs, dn_orbs))

    def make_config_str(self):
        orbs = sorted([orb for orb in self.occs])
        return '_'.join([str(orb)+'S'+str(self.occs[orb]) for orb in orbs])

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        return self.make_config_str()

    def __eq__(self, other):
        return self.occs == other.occs 

if __name__ == "__main__":
    v = Vec()
    det = Det([1, 3, 2], [1, 2, 4])
    det2 = Det([10, 3, 2], [1, 2, 4])
    v += Vec({det : 1})
    w = Vec({det2: 2})
    v2 = v + w
    w += v
    d = {v2}
    print(w)
    print(w + (-2)*v)
