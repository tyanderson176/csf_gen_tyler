import numpy
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import csfgen as proj
import vec

tol = 1e-15

def get_det_info(shci_out):
    S2, det_strs, wf_coeffs = shci_out
    #estimate S; use <S^2> = s(s+1); S = 1/2 +- sqrt(<S^2 + 1/4>)
    s = numpy.rint(numpy.sqrt(S2 + 0.25) - 0.5)
    dets = det_strs2dets(det_strs)
    csfs = get_csfs(dets, wf_coeffs, s, 'projection')
    det_indices, ovlp = csf_matrix(csfs)
    wf_det_coeffs = get_det_coeffs(det_indices, wf_coeffs, dets)
    wf_csf_coeffs = numpy.dot(ovlp, wf_det_coeffs)
    csf_info = [
        [(det_indices.index(d), c) for d, c in zip(csf.dets, csf.coeffs)] 
        for csf in csfs]
    err = get_proj_error(ovlp, wf_det_coeffs)
    return wf_csf_coeffs, csf_info, det_indices, err
    
def get_det_coeffs(det_indices, wf_coeffs, dets):
    wf_coeffs = [float(coeff) for coeff in wf_coeffs.split()]
    det_coeffs = numpy.zeros(len(det_indices))
    for det, coeff in zip(dets, wf_coeffs):
        index = det_indices.index(det)
        det_coeffs[index] = coeff
    return det_coeffs

def get_csfs(dets, wf_coeffs, s, method='projection', cache=False):
    sz = get_Sz(dets)
    configs = set(vec.Config(det) for det in dets)
    max_open = max([config.num_open for config in configs])
    print("Max open: %d", % max_open)
    csfs = []
    if cache:
        csf_data = gen.load_csf_info(max_open, s, sz)
        for config in configs:
            csfs += gen.config2csfs(config, csf_data, rel_parity=False)
    else:
        for config in configs: 
            csfs += compute_csfs(config, s, sz, method)
    return csfs

def get_proj_error(ovlp, wf_det_coeffs):
    err_op = numpy.identity(len(wf_det_coeffs)) - numpy.dot(ovlp.T, ovlp)
    err_vec = numpy.dot(err_op, wf_det_coeffs)
    return numpy.dot(err_vec, err_vec)/numpy.dot(wf_det_coeffs, wf_det_coeffs)

def det_strs2dets(det_strs):
    #TODO: give a format string i.e. "up_occs     dn_occs"
    #or "up_occs\t\t\tdn_occs"
    dets = []
    for det_str in det_strs:
        up_str, dn_str = tuple(det_str.split('     '))
        up_occs = [int(orb)+1 for orb in up_str.split()]
        dn_occs = [int(orb)+1 for orb in dn_str.split()]
        dets.append(vec.Det(up_occs, dn_occs))
    return dets

def get_Sz(dets):
    sz_vals = set(det.get_Sz() for det in dets)
    if len(sz_vals) > 1:
        raise Exception("Different Sz values in dets")
    for sz in sz_vals:
        return sz

def csf_matrix(csfs):
    det_indices = IndexList()
    for csf in csfs:
        for det in csf.dets:
            det_indices.add(det)
    matrix = numpy.array([get_coeffs(csf, det_indices) for csf in csfs])
    return det_indices, matrix

def get_coeffs(csf, det_indices):
    coeffs = numpy.zeros(len(det_indices))
    for det, coeff in zip(csf.dets, csf.coeffs):
        index = det_indices.index(det)
        coeffs[index] = coeff
    return coeffs/csf.norm()

def compute_csfs(config, S, Sz, method):
    '''
    Use projection method to compute configurations
    '''
    if method == 'projection':
        config_str = config.make_config_str()
        #TODO: lookup the appropriate csf from the table and convert
        proj_csfs = proj.compute_csfs(config_str, S, Sz)
        csfs = convert_proj_csfs(proj_csfs)
    else:
        raise Exception('Invalid method \'' + method + '\' in compute_csfs')
    return csfs

def convert_proj_csfs(proj_csfs):
    csfs = []
    for proj_csf in proj_csfs:
        csf_dets = {} 
        for proj_det in proj_csf.dets:
            coeff = proj_csf.get_det_coeff(proj_det)
            up_occs, dn_occs = [], []
            for orb in proj_det.orbitals:
                sz = orb.labels['s_z']
                n = orb.labels['n']
                if sz == 0.5: 
                    up_occs.append(n)
                else: 
                    dn_occs.append(n)
            csf_dets[vec.Det(up_occs, dn_occs)] = coeff
        csfs.append(vec.Vec(csf_dets))
    return csfs

class IndexList:
    def __init__(self):
        self.indices = {}

    def add(self, obj):
        sz = len(self)
        if obj not in self.indices:
            self.indices[obj] = sz

    def index(self, obj):
        return self.indices[obj]

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return str(self.indices)

if __name__ == '__main__':
    out = "START DMC\n0   1   2   3      0   1\n0   2   3   7      0   7\n0   1   2  10      0   7\n0   1   3   9      0   7\n0   1   3  13      1   3\n0   1   2  13      1   2\n0   1   2  12      1   3\n0   1   3  12      1   2\n0   2   3  14      1   2\n0   2   3  15      1   3\n0   2   7  10      0   1\n0   3   7   9      0   1\n0   1   9  10      0   1\n0   2   3  16      0  16\n0   1   2   3      1  11\n0   1   2  11      1   3\n0   1   3  11      1   2\n0   2   3   7      1  11\n1   2   3  11      0   7\n0   1   2  10      1  11\nDET COEFFS:\n 0.97958444    -0.05930949    -0.04187026     0.04187026     0.03630456    -0.03630456    -0.03630456    -0.03630456    -0.03492338    -0.03492338     0.03245056    -0.03245056    -0.03139306    -0.02819201    -0.02761537    -0.02653736     0.02653736    -0.02641846    -0.02607992    -0.02601047\nS^2 VAL:\n 2.00000006\nEND DMC"
    lines = out.split('\n')
    start_index = lines.index("START DMC")
    det_index = lines.index("DET COEFFS:")
    s2_index = lines.index("S^2 VAL:")
    end_index = lines.index("END DMC")
    S2 = float(lines[s2_index+1])
    det_strs = lines[start_index+1:det_index]
    det_coefs = lines[det_index+1]
    shci_out = (S2, det_strs, det_coefs)
#    shci_out = "1 2     1 3\n1 3     1 2\n2 3     2 4\n0.99 -0.98 0.01"
    coeffs, info, dets = get_det_info(shci_out)
#    print(coeffs)
#    print(info)
    print([det.dmc_str() for det in dets.indices])
