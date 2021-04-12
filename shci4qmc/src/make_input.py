import sys
import re
import os
import numpy as np

from shci4qmc.src.ham import Ham
from shci4qmc.src.vec import Det, Vec, Config

def make_input(cache_filename, dir_out, csf_key, csf_tol, 
               reduce_csfs, rotate_csfs):
    qmc_filename = os.path.join(dir_out, 'champ_tol%7.1e'%csf_tol + '.in')
    qmc_file = open(qmc_filename, 'w+')
    with open(cache_filename, 'r') as qmc_cache:
        before_csfs = copy_before_csfs(qmc_cache, qmc_file)
        wf_energy, ncsf, ndet, csf_section = \
            write_csf_section(qmc_cache, qmc_file, dir_out, 
                              csf_key, csf_tol, reduce_csfs, rotate_csfs)
        after_csfs = after_csfs_section(qmc_cache, qmc_file, ncsf)
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
            updated_before_csfs.append("0.5    %.8f  'Hartrees'\t\t\thb,etrial,eunit\n"% wf_energy)
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

def write_csf_section(qmc_cache, qmc_file, dir_out, csf_key, csf_tol, 
                      reduce_csfs, rotate_csfs):
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
    return write_csfs(qmc_file, dir_out, config_csfs, config_labels, 
                      csf_key, csf_tol, reduce_csfs, rotate_csfs)

def det_row_str(det, n, config_label):
    return (det.qmc_str() + '\t\t' + str(n+1) + '\t' + str(config_label+1))

def write_csfs(qmc_file, dir_out, config_csfs, config_labels, 
               csf_key, csf_tol, reduce_csfs, rotate_csfs):
    det_indices, accepted_csfs = {}, []
    output_lines = []
    for config, csfs in config_csfs.items():
        if rotate_csfs:
            cnfg_sum = add_csfs([coef for coef, csf in csfs], [csf for coef, csf in csfs])
#            if (config_sum.norm()/np.sqrt(n_dets) < csf_tol): 
#                continue
            cnfg_norm = cnfg_sum.norm()
            cnfg_sum *= 1./cnfg_norm
            if (csf_key(cnfg_norm, cnfg_sum) < csf_tol):
                continue
            if reduce_csfs:
                accepted_csfs.append((cnfg_norm, cnfg_sum))
            else:
                for coef, csf in csfs:
                    accepted_csfs.append((coef, csf))
        else:
            for coef, csf in csfs:
                if csf_key(coef, csf) < csf_tol:
                    continue
                accepted_csfs.append((coef, csf))
    
#    decreasing_coefs = lambda coef_and_csf: -abs(coef_and_csf[0])/np.sqrt(len(coef_and_csf[1].dets))
    sorted_csfs = sorted(accepted_csfs, key = lambda p: -csf_key(*p))
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
    fcidump_path = os.path.join(dir_out, 'FCIDUMP')
    if os.path.exists(fcidump_path + '_real_orbs'):
        fcidump_path += '_real_orbs'
    if not os.path.exists(fcidump_path):
        raise Exception("FCIDUMP/FCIDUMP_real_orbs file not found")
#    wf_energy = Ham(fcidump_path).expectation(wf)
    wf_energy = 0.0
    return wf_energy, len(sorted_csfs), len(sorted_dets), output_lines

def after_csfs_section(qmc_cache, qmc_file, ncsf):
    output_lines = []
    for line in qmc_cache:
        if 'INSERT NPARAM LINE' in line:
            nparam_line = ('0  4  5  15  0 ' + str(ncsf-1) + ' 0 0' 
                          + ' nparml,nparma,nparmb,nparmc,nparmf,nparmcsf,nparms,nparmg\n')
            output_lines.append(nparam_line) 
        elif 'INSERT PARMCSF LINE' in line:
            parmcsf_line = (' '.join(str(n+2) for n in range(ncsf-1))
                           + ' (iwcsf(iparm),iparm=1,nparmcsf)\n')
            output_lines.append(parmcsf_line)
        elif 'INSERT NDATA LINE' in line:
            ndata_line = ('1000 ' + str(24 + ncsf-1) + ' 1 1 5 1000 21101 1' 
                         + ' NDATA,NPARM,icusp,icusp2,NSIG,NCALLS,iopt,ipr\n')
            output_lines.append(ndata_line)
        else:
            output_lines.append(line)
    return output_lines
