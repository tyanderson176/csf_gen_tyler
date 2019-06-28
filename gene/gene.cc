#include <unordered_map>
#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "det.h"

#define TOL 1E-15

class CSF {
public:
    CSF() {}

    CSF(const CSF& vec):
        dets (vec.dets) {}

    CSF(const std::unordered_map<Det, double, DetHasher> ds):
        dets (ds) {}

    CSF(const Det& det) { dets[det] = 1.; }

    CSF& operator+=(const CSF& other) {
        for(auto &pair : other.dets) {
            const Det& det = pair.first;
            double coef = pair.second;
            dets[det] += coef;
            if (std::abs(dets[det]) < TOL)
                dets.erase(det);
        }
        return *this;
    }

    CSF& operator*=(const double& scalar) {
        for(auto &pair : dets) {
            const Det& det = pair.first;
            double& coef = pair.second;
            coef *= scalar;
            if (std::abs(dets[det]) < TOL)
                dets.erase(det);
        }
        return *this;
    }

    CSF operator*(const double& scalar) {
        CSF prod(*this);
        prod *= scalar;
        return prod;
    }

    CSF operator+(const CSF& other) {
        CSF sum(*this);
        sum += other;
        return sum;
    }

    CSF add_up_elec(unsigned orb) { 
        std::unordered_map<Det, double, DetHasher> new_dets;
        for(const auto& pair : dets) {
            Det det = pair.first;
            det.up.set(orb); 
            new_dets[det] = pair.second;
        }
        return CSF(new_dets);
    }

    CSF add_dn_elec(unsigned orb) {
        std::unordered_map<Det, double, DetHasher> new_dets;
        for(const auto& pair : dets) {
            Det det = pair.first;
            det.dn.set(orb); 
            new_dets[det] = pair.second;
        }
        return CSF(new_dets);
    }

    std::unordered_map<Det, double, DetHasher> dets;
};

CSF operator*(const double& scalar, const CSF& f) {
    CSF scaled(f);
    scaled *= scalar;
    return scaled;
}

int perm_factor(const Det &det) {
    auto ups = det.up.get_occupied_orbs();
    auto dns = det.dn.get_occupied_orbs();
    int j = ups.size()-1;
    int num_greater = 0, tot = 0;
    for(int i = dns.size()-1; i != -1; --i) {
        while(j != -1 and ups[j] > dns[i]) {
            --j;
            ++num_greater;
        }
        tot += num_greater;
    }
    return tot%2 ? -1 : 1;
}

std::ostream & operator << (std::ostream &out, const CSF& csf) {
    for(auto &pair : csf.dets) {
        const Det& det = pair.first;
        double coef = perm_factor(det)*pair.second;
        out << coef << "*|";
        std::vector<unsigned> orbs = det.up.get_occupied_orbs();
        for(const unsigned &orb : orbs)
            out << orb << " ";
        orbs = det.dn.get_occupied_orbs();
        out << ";";
        for(const unsigned &orb : orbs)
            out << orb << " "; 
        out << ") + ";
    }
    return out;
}

//set of csfs corresponding to the same representation
using CSFSet = std::vector<CSF>;
using SpinSet = std::vector<CSFSet>;
//using csf_vec = std::vector<csf>;
//using csf_vec_2D = std::vector<csf_vec>

CSFSet inc_spin(CSFSet& csfs, double s, int orb) {
    assert(csfs.size() == 2*s+1);
    CSFSet new_csfs;
    double sz = -s, coef1, coef2;
    new_csfs.push_back(csfs[0].add_dn_elec(orb));
    for(int i=0; i<csfs.size()-1; ++i, ++sz) {
        coef1 = sqrt((s + sz + 1)/(2*s+1));
        coef2 = sqrt((s-sz)/(2*s+1));
        auto up = csfs[i].add_up_elec(orb);
        auto dn = csfs[i+1].add_dn_elec(orb);
        new_csfs.push_back(up*coef1 + dn*coef2); 
    }
    new_csfs.push_back(csfs.back().add_up_elec(orb));
    return new_csfs;
}


CSFSet dec_spin(CSFSet& csfs, double s, int orb) {
    assert(csfs.size() == 2*s+1);
    CSFSet new_csfs;
    double sz = -s, coef1, coef2;
    for(int i=0; i<csfs.size()-1; ++i, ++sz) {
        coef1 = -sqrt((s-sz)/(2*s+1));
        coef2 = sqrt((s+sz+1)/(2*s+1));
        auto up = csfs[i].add_up_elec(orb);
        auto dn = csfs[i+1].add_dn_elec(orb);
        new_csfs.push_back(up*coef1 + dn*coef2); 
    }
    return new_csfs;
}

void inc_electron(std::vector<SpinSet> &spins, int nelecs) {
    double s = 0.5*(nelecs%2);
//    tw_spin = nelecs%2;
    std::vector<SpinSet> new_spins(spins.size()+nelecs%2);
    //Iterate over spin values
    for(int i = 0; i < spins.size(); ++i, ++s) {
        int j = nelecs%2 ? i : i-1;
        SpinSet &spin_set = spins[i];
        for(CSFSet& csfs : spin_set) {
            CSFSet add = inc_spin(csfs, s, nelecs);
            new_spins[j+1].push_back(add);
            if (s == 0) continue;
            assert(j >= 0);
            CSFSet sub = dec_spin(csfs, s, nelecs);
            new_spins[j].push_back(sub);
        }
    }
    spins = new_spins;
}

void print_csfs(std::vector<SpinSet> &spins, int nelecs) {
    std::cout << '\n';
    for(int s=0; s<spins.size(); s++) {
        auto csf_sets = spins[s];
        std::cout << "Spin: " << s + 0.5*(nelecs%2) << '\n';
//        std::cout << "Size: " << csf_sets.size() << '\n';
        for(int i = 0; i < csf_sets.size(); i++) {
            std::cout << "  Set #: " << i << '\n';
            auto set = csf_sets[i];
            for(int sz = 0; sz < set.size(); ++sz) {
                std::cout << "    " <<  sz << ": ";
                std::cout << set[sz] << '\n';
            }
        }
    }
} 

Det string_to_det(std::string str) {
    Det det;
    for(int i=0; i<str.size(); i++) {
        if (str[i] == '0')
            det.dn.set(i);
        else
            det.up.set(i);
    }
    return det;
}

void generate_dets(std::vector<Det> & dets, 
        std::vector<std::string> &det_strings, int nelecs, int nup) {
    assert(nelecs >= nup);
    dets.clear(), det_strings.clear();
    std::string seed(nelecs, '0');
    for(int i=nelecs-nup; i < nelecs; ++i)
        seed[i] = '1';
    dets.push_back(string_to_det(seed));
    det_strings.push_back(seed);
    while(std::next_permutation(seed.begin(), seed.end())) {
        dets.push_back(string_to_det(seed));
        det_strings.push_back(seed);
    }
}

void fill_index_of(std::unordered_map<Det, int, DetHasher> &index_of, 
        std::vector<std::string> & det_strings, int nelecs, int two_s) {
    assert((two_s + nelecs)%2 == 0);
    int nup = (two_s + nelecs)/2;
    std::vector<Det> dets;
    generate_dets(dets, det_strings, nelecs, nup);
    for(int i=0; i<dets.size(); ++i) {
        Det& det = dets[i];
        index_of[det] = i;
    }
}

void save_csfs(std::vector<SpinSet> &spins, int nelecs) {
    int two_s = nelecs%2;
    for(int i=0; i<spins.size(); ++i, two_s += 2) {
        std::vector<std::vector<CSF>> sets = spins[i];
        std::unordered_map<Det, int, DetHasher> index_of;
        std::vector<std::string> det_strings;
        fill_index_of(index_of, det_strings, nelecs, two_s);
        printf("\nSTART\nN = %d S = %3.1f\n", nelecs, two_s/2.);
        for(std::string str : det_strings)
            printf("%s ", str.c_str());
        printf("\n");
        for(auto& set : sets) {
            std::vector<double> coefs(index_of.size(), 0.);
            CSF& max_sz_csf = set[set.size()-1];
            for(const auto& pair : max_sz_csf.dets) {
                const Det& det = pair.first;
                coefs[index_of[det]] = perm_factor(det)*max_sz_csf.dets[det];
            }
            //print the coefs in appropriate order
            for(int i=0; i<coefs.size(); i++)
                printf("%f\t", coefs[i]);
            printf("\n");
        }
        printf("END\n");
    }
}

std::vector<SpinSet> spin_one_half() {
    Det up_det, dn_det;
    up_det.up.set(0);
    dn_det.dn.set(0);
    std::vector<CSF> one_half;
    one_half.push_back(CSF(dn_det));
    one_half.push_back(CSF(up_det));
    return std::vector<SpinSet>(1, SpinSet(1, one_half));
}

void compute_csfs(int max_nelecs) {
    std::vector<SpinSet> spins = spin_one_half();
//    print_csfs(spins, 1);
    std::cout << '\n';
    save_csfs(spins, 1);
    for(int nelecs = 1; nelecs < max_nelecs; ++nelecs) {
        inc_electron(spins, nelecs);
//        print_csfs(spins, nelecs+1);
        save_csfs(spins, nelecs+1);
    }
}

int main(int argc, char **argv) {
    assert(argc > 1);
    char *nelecs_str = argv[1];
    compute_csfs(atoi(nelecs_str));
    return 1;
}
