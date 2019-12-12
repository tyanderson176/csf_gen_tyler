#include <Python.h>
#include "det.h"
#include "../lib/hps/src/hps.h"
#include <fstream>
#include <vector>

class Wavefunction {
public:
  unsigned n_up = 0;

  unsigned n_dn = 0;

  double energy_hf = 0.0;

  double energy_var = 0.0;

  std::vector<Det> dets;

  std::vector<double> coefs;

  template <class B>
  void parse(B& buf) {
    buf >> n_up >> n_dn >> dets >> coefs >> energy_hf >> energy_var;
  }
};

PyObject* HalfDet2PyList(const HalfDet& hdet) {
    std::vector<unsigned> orbs = hdet.get_occupied_orbs();
    PyObject *py_orbs = PyList_New(orbs.size());
    for(size_t i = 0; i<orbs.size(); ++i)
        PyList_SetItem(py_orbs, i, PyLong_FromUnsignedLong(orbs[i]));
    return py_orbs;
}

PyObject* Det2PyList(const Det& det) {
    PyObject *py_det = PyList_New(2);
    PyList_SetItem(py_det, 0, HalfDet2PyList(det.up));
    PyList_SetItem(py_det, 1, HalfDet2PyList(det.dn));
    return py_det;
}

PyObject* Dets2PyList(const std::vector<Det>& dets) {
    PyObject *py_dets = PyList_New(dets.size());
    for(size_t i=0; i<dets.size(); ++i) {
        const Det& det = dets[i];
        PyList_SetItem(py_dets, i, Det2PyList(det));
    }
    return py_dets;
}

PyObject* Coefs2PyList(const std::vector<double>& coefs) {
    PyObject *py_coefs = PyList_New(coefs.size());
    for(size_t i=0; i<coefs.size(); ++i)
        PyList_SetItem(py_coefs, i, PyFloat_FromDouble(coefs[i]));
    return py_coefs;
}

PyObject* wf2PyDict(const Wavefunction& wf) {
    PyObject *py_wf = PyDict_New();
    PyDict_SetItemString(py_wf, "dets", Dets2PyList(wf.dets));
    PyDict_SetItemString(py_wf, "coefs", Coefs2PyList(wf.coefs));
    PyDict_SetItemString(py_wf, "energy_var", PyFloat_FromDouble(wf.energy_var));
    PyDict_SetItemString(py_wf, "energy_hf", PyFloat_FromDouble(wf.energy_hf));
    PyDict_SetItemString(py_wf, "n_dn", PyLong_FromUnsignedLong(wf.n_dn)); 
    PyDict_SetItemString(py_wf, "n_up", PyLong_FromUnsignedLong(wf.n_up)); 
    return py_wf; 
}

static PyObject* load(PyObject *self, PyObject *args) {
    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename))
        return NULL;
    
    std::ifstream wf_file(filename);
    if (!wf_file) {
        PyErr_SetString(PyExc_FileNotFoundError, "File not found");
        return NULL;
    }
    std::stringstream buffer;
    buffer << wf_file.rdbuf();
    
    //Check buffer was correctly parsed?
    Wavefunction wf;
    hps::from_stream(buffer, wf);
    return wf2PyDict(wf);
}

static PyMethodDef load_wf_methods[] = { 
    {
        "load", load, METH_VARARGS,
        "Loads dets of wf"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef load_wf_definition = { 
    PyModuleDef_HEAD_INIT,
    "load_wf",
    "A Python module that loads a wavefunctions serialized by FGPL.",
    -1, 
    load_wf_methods
};

PyMODINIT_FUNC PyInit_load_wf(void) {
    Py_Initialize();
    return PyModule_Create(&load_wf_definition);
}
