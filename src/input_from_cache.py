import sys
import re
import numpy as np

from shci4qmc.src.ham import Ham
from shci4qmc.src.vec import Det, Vec, Config

def input_from_cache(cache_filename, csf_tol):
    qmc_filename = 'qmc_tol%7.1e'%csf_tol + '.in'
    qmc_file = open(qmc_filename, 'w+')
    with open(cache_filename, 'r') as qmc_cache:
        before_csfs = copy_before_csfs(qmc_cache, qmc_file)
        wf_energy, ncsf, ndet, csf_section = \
            write_csf_section(qmc_cache, qmc_file, csf_tol, reduce_csfs = False)
        after_csfs = copy_after_csfs(qmc_cache, qmc_file)
        before_csfs = update_before_csfs(before_csfs, ncsf, ndet, wf_energy)
#        qmc_file.write("\'ncsf=%d ndet=%d norb=%d\'\t\t\t\ttitle\n"% (ncsf, ndet, norb))
        qmc_file.write(''.join(before_csfs))
        qmc_file.write(''.join(csf_section))
        qmc_file.write(''.join(after_csfs))
    qmc_file.close()

def copy_before_csfs(qmc_cache, qmc_file):
    output_lines = []
    for n, line in enumerate(qmc_cache):
        output_lines.append(line)
        if (line.split() and line.split()[-1] == 'nbas)'):
            break
    return output_lines

def update_before_csfs(before_csfs, ncsf, ndet, wf_energy):
    updated_before_csfs = []
    for n, line in enumerate(before_csfs):
        if n == 0:
            norb = int(re.search("\'ncsf=(.*) ndet=(.*) norb=(.*)\'(.*)", line).group(3))
            updated_before_csfs.append(
                "\'ncsf=%d ndet=%d norb=%d\'\t\t\ttitle\n"% (ncsf, ndet, norb))
        elif n == 3:
            updated_before_csfs.append("0.5    %.4f  'Hartrees'\t\t\thb,etrial,eunit\n"% wf_energy)
        elif line.split() and line.split().pop() == 'norb':
            match = re.search("([0-9]+) ([0-9]+) ([0-9]+)([ ]+)ndet, nbasis, norb", line)
            nbasis, norb = int(match.group(2)), int(match.group(3))
            updated_before_csfs.append("%d %d %d \t\t\tndet, nbasis, norb"% (ndet, nbasis, norb))
        else:
            updated_before_csfs.append(line)
    return updated_before_csfs

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def is_int(string):
    try:
        int(string)
        return True
    except ValueError:
        return False

def add_csfs(coefs, csfs):
    res = Vec.zero()
    for coef, csf in zip(coefs, csfs):
        res += coef*csf
    return res

def write_csf_section(qmc_cache, qmc_file, csf_tol, reduce_csfs):
    config_labels, dets = {}, []
    config_csfs = {}
    for line in qmc_cache:
        det_str, empty, index, config_label = line.strip().split('\t')
        up_det_string, dn_det_string = det_str.strip().split('      ')
        up_det = [int(orb) for orb in up_det_string.split()]
        dn_det = [int(orb) for orb in dn_det_string.split()]
        det = Det(up_det, dn_det)
        dets.append(det)
        config_labels[det] = int(config_label.split()[0])
        if (config_label.split().pop() == 'nelec)'):
            break
    ncsf = int(qmc_cache.readline().split()[0])
    csf_coefs = [float(coef) for coef in qmc_cache.readline().split() if is_float(coef)]
    csf_sizes = [int(size) for size in qmc_cache.readline().split() if is_int(size)]

    count = 0
    det_line = qmc_cache.readline()
    coef_line = qmc_cache.readline()
    while det_line and coef_line:
        csf_dets = [dets[int(det_num)-1] for det_num in det_line.split() if det_num.isdigit()]
        csf_det_coefs = [float(coef) for coef in coef_line.split() if is_float(coef)]
        config_label = int(coef_line.split().pop()[1:-1])

        csf = add_csfs(csf_det_coefs, csf_dets)
        if config_label not in config_csfs:
            config_csfs[config_label] = []
        config_csfs[config_label].append((csf_coefs[count], csf))

        count += 1
        if count == ncsf:
            break
        det_line = qmc_cache.readline()
        coef_line = qmc_cache.readline()
    return write_csfs(qmc_file, config_csfs, config_labels, csf_tol, reduce_csfs)

def det_row_str(det, n, config_label):
    return (det.qmc_str() + '\t\t' + str(n+1) + '\t' + str(config_label+1))

def write_csfs(qmc_file, config_csfs, config_labels, csf_tol, reduce_csfs):
    det_indices, accepted_csfs = {}, []
    output_lines = []
    for config, csfs in config_csfs.items():
        config_sum = add_csfs([coef for coef, csf in csfs], [csf for coef, csf in csfs])
        n_dets = len(config_sum.dets)
        #n_dets = 1
        if (config_sum.norm()/np.sqrt(n_dets) < csf_tol): continue
        for coef, csf in csfs:
#            for det in csf.dets:
#                if det not in det_indices:
#                    det_indices[det] = len(det_indices)
            accepted_csfs.append((coef, csf))
    
    decreasing_coefs = lambda coef_and_csf: -abs(coef_and_csf[0])
    sorted_csfs = sorted([(coef, csf) for coef, csf in accepted_csfs], key = decreasing_coefs)
    sorted_dets, reindex, det_indices = [], {}, {}
    for coef, csf in sorted_csfs:
        for det in csf.dets:
            if det not in det_indices:
                sorted_dets.append(det)
                det_indices[det] = len(det_indices)
                if config_labels[det] not in reindex:
                    reindex[config_labels[det]] = len(reindex)
#    for det, index in det_indices.items():
#        rep = config_rep(Config(det))
#        if rep not in config_labels:
#            config_labels[rep] = len(config_labels)
#        sorted_dets[index] = det
    
    dets_str = '\n'.join(
        [det_row_str(det, n, reindex[config_labels[det]]) for n, det in enumerate(sorted_dets)])
    output_lines.append(dets_str + ' (iworbd(iel,idet), iel=1, nelec)\n')
    csf_coeffs_str = '\t'.join(['%.10f' % coeff for coeff, csf in sorted_csfs])
    ndets_str = '\t'.join([str(len(csf.dets)) for coef, csf in sorted_csfs])
    output_lines.append(str(len(sorted_csfs)) + ' ncsf\n')
    output_lines.append(csf_coeffs_str + ' (csf_coef(icsf), icsf=1, ncsf)\n')
    output_lines.append(ndets_str + ' (ndet_in_csf(icsf), icsf=1, ncsf)\n')
    wf = Vec.zero()
    for coef, csf in sorted_csfs:
        index_str = (' '.join([str(det_indices[det] + 1) for det, coef in csf.dets.items()]) +
            ' (iwdet_in_csf(idet_in_csf,icsf),idet_in_csf=1,ndet_in_csf(icsf))\n')
        coeff_str = (' '.join(['%.8f'%coef for det, coef in csf.dets.items()]) +
            ' (cdet_in_csf(idet_in_csf,icsf),idet_in_csf=1,ndet_in_csf(icsf))\n')
        output_lines.append(index_str)
        output_lines.append(coeff_str)
        wf += coef*csf
    wf_energy = Ham("FCIDUMP_real_orbs").expectation(wf)
    return wf_energy, len(sorted_csfs), len(sorted_dets), output_lines

def copy_after_csfs(qmc_cache, qmc_file):
    output_lines = []
    for line in qmc_cache:
        output_lines.append(line)
    return output_lines
