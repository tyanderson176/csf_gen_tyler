#include <Python.h>
#include <fstream>
#include <iostream>
#include <vector>

long par_merge(std::vector<long> &array, size_t start, size_t mid, size_t end) {
    long parity = 0;
    size_t len = end - start;
    std::vector<long> merged(len, 0);
    size_t p1 = start, p2 = mid;
    for(size_t i = 0; i < len; i++) {
        if (p1 == mid or (p2 != end and array[p2] < array[p1])) {
            merged[i] = array[p2++];
            parity += mid - p1;
        }
        else merged[i] = array[p1++];
    }
    for(size_t i = 0; i < len; i++)
        array[start + i] = merged[i];
    return parity;
}

long par_sort(std::vector<long> &array, size_t start, size_t end) {
    if (end - start <= 1)
        return 0;
    size_t mid = (end + start)/2; 
    long rp_left = par_sort(array, start, mid);
    long rp_right = par_sort(array, mid, end);
    return (rp_left + rp_right + par_merge(array, start, mid, end))%2;
}

long rel_parity(std::vector<long> &array) {
    /**
      Divide and conquer algorithm to find parity of an array.
      It should be asymptotically faster than rel_parity, but
      a small amount of testing indicates that the simplicity
      of rel_parity_few_elec makes it about as fast as rel_parity
      for arrays of size ~10
    **/
    return par_sort(array, 0, array.size()) == 0 ? 1 : -1;
}

long rel_parity_few_elec(std::vector<long> &array) {
    int count = 0;
    for(int start = 0; start < array.size(); ++start) {
        for(int i=start+1; i < array.size(); ++i)
            //count += (array[i] < array[start]) ? 1 : 0;
            if (array[i] < array[start])
                count++;
            else if (array[i] == array[start])
                return 0;
    }
    return (count%2 == 0) ? 1 : -1;
}

static PyObject* rel_parity(PyObject *self, PyObject *args) {
    PyObject *listObj;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &listObj))
        return NULL;

    size_t len = PyList_Size(listObj);
    if (len < 0) return NULL; 

    std::vector<long> list;
    for (size_t i=0; i<len; i++) {
        PyObject *entry = PyList_GetItem(listObj, i);
        list.push_back(PyLong_AsLong(entry));
    }

    return PyLong_FromLong(rel_parity(list));
}

static PyObject* rel_parity_few_elec(PyObject *self, PyObject *args) {
    PyObject *listObj;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &listObj))
        return NULL;

    size_t len = PyList_Size(listObj);
    if (len < 0) return NULL; 

    std::vector<long> list;
    for (size_t i=0; i<len; i++) {
        PyObject *entry = PyList_GetItem(listObj, i);
        list.push_back(PyLong_AsLong(entry));
    }

    return PyLong_FromLong(rel_parity_few_elec(list));
}

static PyMethodDef rel_parity_methods[] = { 
    {
        "rel_parity", rel_parity, METH_VARARGS,
        "Calculates parity of a list"
    },
    {
        "rel_parity_few_elec", rel_parity_few_elec, METH_VARARGS,
        "Calculates parity of a list (2)"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef rel_parity_definition = { 
    PyModuleDef_HEAD_INIT,
    "rel_parity",
    "A Python module that calculates parity of a list (rel. to sorted).",
    -1, 
    rel_parity_methods,
};

PyMODINIT_FUNC PyInit_rel_parity(void) {
    Py_Initialize();
    return PyModule_Create(&rel_parity_definition);
}

