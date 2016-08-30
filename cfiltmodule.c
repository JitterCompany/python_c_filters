#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdbool.h>
#include <stdint.h>

#include "Python.h"
#include "arm_math.h"
#include "filter.h"

#define MAX_FILTERS 25

static unsigned int num_filters64 = 0;
static unsigned int num_filters32 = 0;
static Filter64 filters64[MAX_FILTERS];
static Filter32 filters32[MAX_FILTERS];

static PyObject *error;
static PyObject* test(PyObject *self, PyObject *args);
static PyObject* filter(PyObject *self, PyObject *args);

static PyObject* filter64_init(PyObject *self, PyObject *args);
static PyObject* filter32_init(PyObject *self, PyObject *args);

static PyObject* filter64_apply(PyObject *self, PyObject *args);
static PyObject* filter32_apply(PyObject *self, PyObject *args);

static PyMethodDef methods[] = {
    {"test", test, METH_VARARGS, "Test this module"},
    {"filter64_init", filter64_init, METH_VARARGS, 
        "Init a filter object that uses doubles internally."},
    {"filter32_init", filter32_init, METH_VARARGS, 
        "Init a filter object that uses floats internally."},
    {"filter64_apply", filter64_apply, METH_VARARGS, 
        "Apply filter on input"},
    {"filter32_apply", filter32_apply, METH_VARARGS, 
        "Apply filter on input"},
    {NULL,NULL,0,NULL}
};

static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "cfilt",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   methods
};

PyMODINIT_FUNC
PyInit_cfilt(void)
{
    PyObject *m;

    m = PyModule_Create(&module);
    if (m == NULL)
        return NULL;

    error = PyErr_NewException("cfilt.error", NULL, NULL);
    Py_INCREF(error);
    PyModule_AddObject(m, "error", error);

    import_array();

    return m;
}

static PyObject* filter32_init(PyObject *self, PyObject *args)
{
    PyObject *data_obj;
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "O", &data_obj))
        return NULL;

    PyArrayObject *data_array = (PyArrayObject *)PyArray_FROM_OTF(data_obj, 
                                NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    npy_intp *dims = PyArray_DIMS(data_array);
    npy_intp *dim1 = PyArray_DIM(data_array, 1);
    if (!dim1 || dims[1] != 6) {
        printf("Error, dim 1 should have size 6: [b0 b1 b2 a1 a2]");
        Py_DECREF(data_array);
        return Py_BuildValue("");
    }

    Filter32 *filter = &filters32[num_filters32];
    filter->num_stages = dims[0];

    // loop over stages
    for (int i = 0; i < filter->num_stages; i++) {
        int offset = i * COEFFS_PER_STAGE;
        npy_intp index[] = {i,0};
        float *row = PyArray_GetPtr(data_array, index);
        // copy b coeffs
        filter->coeffs[offset+0] = row[0];
        filter->coeffs[offset+1] = row[1];
        filter->coeffs[offset+2] = row[2];

        // copy a coeffs, skip first a since it is always 1
        filter->coeffs[offset+3] = -row[4];
        filter->coeffs[offset+4] = -row[5];
    }

    if (num_filters32 >= MAX_FILTERS) {
        printf("Error, to many filters initialized. Only %d are allowed", 
                MAX_FILTERS);
        Py_DECREF(data_array);
        return Py_BuildValue("");
    }

    filter_init_32(&filters32[num_filters32]);
    int index = num_filters32;
    num_filters32++;

    Py_DECREF(data_array);
    return Py_BuildValue("i", index);
}

static PyObject* filter64_init(PyObject *self, PyObject *args)
{
    PyObject *data_obj;
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "O", &data_obj))
        return NULL;

    PyArrayObject *data_array = (PyArrayObject *)PyArray_FROM_OTF(data_obj, 
                                NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    npy_intp *dims = PyArray_DIMS(data_array);
    npy_intp *dim1 = PyArray_DIM(data_array, 1);
    if (!dim1 || dims[1] != 6) {
        printf("Error, dim 1 should have size 6: [b0 b1 b2 a1 a2]");
        Py_DECREF(data_array);
        return Py_BuildValue("");
    }

    Filter64 *filter = &filters64[num_filters64];
    filter->num_stages = dims[0];

    // loop over stages
    for (int i = 0; i < filter->num_stages; i++) {
        int offset = i * COEFFS_PER_STAGE;
        npy_intp index[] = {i,0};
        double *row = PyArray_GetPtr(data_array, index);
        // copy b coeffs
        filter->coeffs[offset+0] = row[0];
        filter->coeffs[offset+1] = row[1];
        filter->coeffs[offset+2] = row[2];

        // copy a coeffs, skip first a since it is always 1
        filter->coeffs[offset+3] = -row[4];
        filter->coeffs[offset+4] = -row[5];
    }

    if (num_filters64 >= MAX_FILTERS) {
        printf("Error, to many filters initialized. Only %d are allowed", 
                MAX_FILTERS);
        Py_DECREF(data_array);
        return Py_BuildValue("");
    }

    filter_init_64(&filters64[num_filters64]);
    int index = num_filters64;
    num_filters64++;

    Py_DECREF(data_array);
    return Py_BuildValue("i", index);
}

static PyObject* filter32_apply(PyObject *self, PyObject *args)
{
    int filter_index;
    PyObject *data_obj;
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "iO", &filter_index, &data_obj))
        return NULL;

    if (filter_index >= num_filters32) 
        return NULL;

    PyArrayObject *data_array = (PyArrayObject *)PyArray_FROM_OTF(data_obj, 
                                NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    npy_intp *shape = PyArray_SHAPE(data_array);
    int n = PyArray_NDIM(data_array);
    npy_intp *dims = PyArray_DIMS(data_array);
    int N = 0; // num samples in data

    if (n == 1) {
        N = dims[0];
    } else if (n == 2 && dims[0] == 1) {
        N = dims[1];
    } else {
       return NULL; 
    }
    
    npy_intp index[] = {0,0};
    float *data_row = PyArray_GetPtr(data_array, index);
    float y[N];
    Filter32 filter = filters32[filter_index];
    filter_apply_32(&filter, data_row, y, N);
    Py_DECREF(data_array);

    npy_intp dimLength[1] = {N};
	PyArrayObject *r = (PyArrayObject *)PyArray_SimpleNew(1, 
            dimLength, NPY_FLOAT32);

    float *buffer = PyArray_DATA(r);
    for (int i =0; i < N; i++) {
        buffer[i] = y[i];
    }

    return PyArray_Return(r);
}
static PyObject* filter64_apply(PyObject *self, PyObject *args)
{
    int filter_index;
    PyObject *data_obj;
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "iO", &filter_index, &data_obj))
        return NULL;

    if (filter_index >= num_filters64) 
        return NULL;

    PyArrayObject *data_array = (PyArrayObject *)PyArray_FROM_OTF(data_obj, 
                                NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    npy_intp *shape = PyArray_SHAPE(data_array);
    int n = PyArray_NDIM(data_array);
    npy_intp *dims = PyArray_DIMS(data_array);
    int N = 0; // num samples in data

    if (n == 1) {
        N = dims[0];
    } else if (n == 2 && dims[0] == 1) {
        N = dims[1];
    } else {
       return NULL; 
    }
    
    npy_intp index[] = {0,0};
    double *data_row = PyArray_GetPtr(data_array, index);
    double y[N];
    Filter64 filter = filters64[filter_index];
    filter_apply_64(&filter, data_row, y, N);
    Py_DECREF(data_array);

    npy_intp dimLength[1] = {N};
	PyArrayObject *r = (PyArrayObject *)PyArray_SimpleNew(1, 
            dimLength, NPY_FLOAT64);

    double *buffer = PyArray_DATA(r);
    for (int i =0; i < N; i++) {
        buffer[i] = y[i];
    }

    return PyArray_Return(r);
}

static PyObject * test(PyObject *self, PyObject *args)
{
    printf("Test function\n");

    return Py_BuildValue("");
}

