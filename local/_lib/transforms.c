// #define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"

static inline double arr_diff(PyObject *sequence, long idxA, long idxB)
{
    /* A helper that computes the difference between two elements of an array */
    const double valA = *(double *)PyArray_GETPTR1(sequence, idxA);
    const double valB = *(double *)PyArray_GETPTR1(sequence, idxB);

    return valA - valB;
}

static PyObject *sample_entropy(PyObject *self, PyObject *args)
{
    PyObject *sequence;
    long int window_size;
    double tolerance;

    if (!PyArg_ParseTuple(args, "Old", &sequence, &window_size, &tolerance))
        return NULL;

    if (!PyArray_Check(sequence))
    {
        PyErr_SetString(
            PyExc_ValueError,
            "The first argument to sample_entropy must be a numpy array.");
        return NULL;
    }

    if (PyArray_TYPE(sequence) != NPY_FLOAT64)
    {
        PyErr_SetString(
            PyExc_ValueError,
            "The input array must have `dtype=float`.");
        return NULL;
    }

    long size = PyArray_SIZE(sequence);

    double numerator = 0;
    double denominator = 0;
    for (int offset = 1; offset < (size - window_size); ++offset)
    {
        long n_denominator = 0;
        long n_numerator = (abs(arr_diff(sequence, window_size, window_size + offset)) >= tolerance);

        for (int idx = 0; idx < window_size; ++idx)
        {
            n_numerator += abs(arr_diff(sequence, idx, idx + offset)) >= tolerance;
            n_denominator += abs(arr_diff(sequence, idx, idx + offset)) >= tolerance;
        }

        if (n_numerator == 0)
            numerator += 1;
        if (n_denominator == 0)
            denominator += 1;

        int prev_in_diff = abs(arr_diff(sequence, window_size, window_size + offset)) >= tolerance;
        for (int idx = 1; idx < size - offset - window_size; ++idx)
        {
            int out_diff = abs(arr_diff(sequence, idx - 1, idx + offset - 1)) >= tolerance;
            int in_diff = abs(arr_diff(sequence, idx + window_size, idx + offset + window_size)) >= tolerance;
            n_numerator += in_diff - out_diff;
            n_denominator += prev_in_diff - out_diff;
            prev_in_diff = in_diff;

            if (n_numerator == 0)
                numerator += 1;
            if (n_denominator == 0)
                denominator += 1;
        }

        // to match antropy's implementation we exclude counts for the last window
        // int idx = size - offset - window_size;
        // int out_diff = abs(arr_diff(sequence, idx - 1, size - window_size - 1)) >= tolerance;
        // n_denominator += prev_in_diff - out_diff;
        // if (n_denominator == 0)
        //     denominator += 1;
    }

    // to match antropy's implementation we exclude counts for the last window
    // long offset = size - window_size;
    // long n_denominator = 0;
    // for (long idx = 0; idx < window_size; ++idx)
    // {
    //     n_denominator += abs(arr_diff(sequence, idx, idx + offset)) >= tolerance;
    // }
    // if (n_denominator == 0)
    //     denominator += 1;

    if (denominator == 0)
        return PyFloat_FromDouble(0); // use 0/0 == 0
    else if (numerator == 0)
        return PyFloat_FromDouble(INFINITY);
    else
        return PyFloat_FromDouble(-log(numerator / denominator));
}

static PyMethodDef Transformers[] = {
    {"sample_entropy", sample_entropy, METH_VARARGS, "Hello World."},
    {NULL, NULL, 0, NULL} /* End of module attributes */
};

static struct PyModuleDef TransformsModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "transforms",
    .m_doc = NULL,
    .m_size = -1,
    .m_methods = Transformers};

PyMODINIT_FUNC
PyInit_transforms(void)
{
    PyObject *module;

    // Initialise Numpy
    import_array();
    if (PyErr_Occurred())
    {
        return NULL;
    }

    module = PyModule_Create(&TransformsModule);
    if (module == NULL)
        return NULL;

    return module;
}
