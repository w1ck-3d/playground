# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# Fabian Pedregosa <fabian.pedregosa@inria.fr>
# Olivier Grisel <olivier.grisel@ensta.org>
#         Vincent Michel <vincent.michel@inria.fr>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
#         Maryan Morel <maryan.morel@polytechnique.edu>
#         Giorgio Patrini <giorgio.patrini@anu.edu.au>
#         Maria Telenczuk <https://github.com/maikia>
# License: BSD 3 clause

import numpy as np
from scipy.linalg import _flapack


def check_array(array, *, dtype="numeric", copy=False):
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, "kind"):
        dtype_orig = None

    if dtype_numeric:
        dtype = None

    if dtype is not None:
        dtype = np.dtype(dtype)

    array = np.array(array, copy=copy, dtype=dtype)

    return array


def _preprocess_data(X, y):
    X = check_array(X, copy=True)
    y = check_array(y, dtype=X.dtype, copy=True)

    X_offset = np.average(X, axis=0)
    X_offset = X_offset.astype(X.dtype, copy=False)
    X -= X_offset

    X_scale = np.ones(X.shape[1], dtype=X.dtype)

    y_offset = np.average(y, axis=0)
    y -= y_offset

    return X, y, X_offset, y_offset, X_scale


_type_score = {x: 1 for x in "?bBhHef"}
_type_score.update({x: 2 for x in "iIlLqQd"})
_type_score.update({"F": 3, "D": 4, "g": 2, "G": 4})


def find_best_blas_type(arrays=(), dtype=None):
    dtype = np.dtype(dtype)
    max_score = _type_score.get(dtype.char, 5)
    prefer_fortran = False

    if arrays:
        # In most cases, single element is passed through, quicker route
        if len(arrays) == 1:
            max_score = _type_score.get(arrays[0].dtype.char, 5)
            prefer_fortran = arrays[0].flags["FORTRAN"]
        else:
            # use the most generic type in arrays
            scores = [_type_score.get(x.dtype.char, 5) for x in arrays]
            max_score = max(scores)
            ind_max_score = scores.index(max_score)
            # safe upcasting for mix of float64 and complex64 --> prefix 'z'
            if max_score == 3 and (2 in scores):
                max_score = 4

            if arrays[ind_max_score].flags["FORTRAN"]:
                # prefer Fortran for leading array with column major order
                prefer_fortran = True

    return "d", np.dtype("float64"), prefer_fortran


def _get_funcs(
    names,
    arrays,
    dtype,
    lib_name,
    fmodule,
    cmodule,
    fmodule_name,
    cmodule_name,
    alias,
    ilp64=False,
):
    """
    Return available BLAS/LAPACK functions.

    Used also in lapack.py. See get_blas_funcs for docstring.
    """

    funcs = []
    unpack = False
    dtype = np.dtype(dtype)
    module1 = (cmodule, cmodule_name)
    module2 = (fmodule, fmodule_name)

    if isinstance(names, str):
        names = (names,)
        unpack = True

    prefix, dtype, prefer_fortran = find_best_blas_type(arrays, dtype)

    if prefer_fortran:
        module1, module2 = module2, module1

    for name in names:
        func_name = prefix + name
        func_name = alias.get(func_name, func_name)
        func = getattr(module1[0], func_name, None)
        module_name = module1[1]
        if func is None:
            func = getattr(module2[0], func_name, None)
            module_name = module2[1]
        if func is None:
            raise ValueError(f"{lib_name} function {func_name} could not be found")
        func.module_name, func.typecode = module_name, prefix
        func.dtype = dtype
        if not ilp64:
            func.int_dtype = np.dtype(np.intc)
        else:
            func.int_dtype = np.dtype(np.int64)
        func.prefix = prefix  # Backward compatibility
        funcs.append(func)

    if unpack:
        return funcs[0]
    else:
        return funcs


_lapack_alias = {
    "corghr": "cunghr",
    "zorghr": "zunghr",
    "corghr_lwork": "cunghr_lwork",
    "zorghr_lwork": "zunghr_lwork",
    "corgqr": "cungqr",
    "zorgqr": "zungqr",
    "cormqr": "cunmqr",
    "zormqr": "zunmqr",
    "corgrq": "cungrq",
    "zorgrq": "zungrq",
}


def get_lapack_funcs(names, arrays=(), dtype=None, ilp64=False):
    if not ilp64:
        return _get_funcs(
            names,
            arrays,
            dtype,
            "LAPACK",
            _flapack,
            None,
            "flapack",
            "clapack",
            _lapack_alias,
            ilp64=False,
        )


def _check_work_float(value, dtype, int_dtype):
    if dtype == np.float32 or dtype == np.complex64:
        value = np.nextafter(value, np.inf, dtype=np.float32)

    value = int(value)
    return value


def _compute_lwork(routine, *args, **kwargs):
    dtype = getattr(routine, "dtype", None)
    int_dtype = getattr(routine, "int_dtype", None)
    ret = routine(*args, **kwargs)
    return tuple(_check_work_float(x.real, dtype, int_dtype) for x in ret[:-1])


# TODO: taken from https://github.com/scipy/scipy/blob/main/scipy/linalg/_basic.py
# Linear Least Squares
def lstsq(a, b):
    if len(a.shape) != 2:
        raise ValueError("Input array a should be 2D")
    m, n = a.shape
    if len(b.shape) == 2:
        nrhs = b.shape[1]
    else:
        nrhs = 1
    if m != b.shape[0]:
        raise ValueError(
            "Shape mismatch: a and b should have the same number"
            " of rows ({} != {}).".format(m, b.shape[0])
        )
    if m == 0 or n == 0:  # Zero-sized problem, confuses LAPACK
        x = np.zeros((n,) + b.shape[1:], dtype=np.common_type(a, b))
        if n == 0:
            residues = np.linalg.norm(b, axis=0) ** 2
        else:
            residues = np.empty((0,))
        return x, residues, 0, np.empty((0,))

    driver = "gelsd"
    lapack_func, lapack_lwork = get_lapack_funcs((driver, "%s_lwork" % driver), (a, b))
    real_data = True if (lapack_func.dtype.kind == "f") else False

    if m < n:
        # need to extend b matrix as it will be filled with
        # a larger solution matrix
        if len(b.shape) == 2:
            b2 = np.zeros((n, nrhs), dtype=lapack_func.dtype)
            b2[:m, :] = b
        else:
            b2 = np.zeros(n, dtype=lapack_func.dtype)
            b2[:m] = b
        b = b2

    cond = np.finfo(lapack_func.dtype).eps

    if driver in ("gelss", "gelsd"):
        if driver == "gelss":
            lwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
            v, x, s, rank, work, info = lapack_func(a, b, cond, lwork)

        elif driver == "gelsd":
            if real_data:
                lwork, iwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
                x, s, rank, info = lapack_func(a, b, lwork, iwork, cond, False, False)
            else:  # complex data
                lwork, rwork, iwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
                x, s, rank, info = lapack_func(
                    a, b, lwork, rwork, iwork, cond, False, False
                )

        resids = np.asarray([], dtype=x.dtype)
        if m > n:
            x1 = x[:n]
            if rank == n:
                resids = np.sum(np.abs(x[n:]) ** 2, axis=0)
            x = x1
        return x, resids, rank, s

    elif driver == "gelsy":
        lwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
        jptv = np.zeros((a.shape[1], 1), dtype=np.int32)
        v, x, j, rank, info = lapack_func(a, b, jptv, cond, lwork, False, False)
        if info < 0:
            raise ValueError(
                "illegal value in %d-th argument of internal " "gelsy" % -info
            )
        if m > n:
            x1 = x[:n]
            x = x1
        return x, np.array([], x.dtype), rank, None


class LinearRegression:
    def predict(self, X):
        X = check_array(X)
        return np.dot(X, self.coef_.T) + self.intercept_

    def _set_intercept(self, X_offset, y_offset, X_scale):
        self.coef_ = np.divide(self.coef_, X_scale, dtype=X_scale.dtype)
        self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)

    def fit(self, X, y):
        X = check_array(X, dtype="numeric", copy=False)

        y = check_array(y, dtype=None)
        y = np.array(np.reshape(y, (-1,)))

        X, y, X_offset, y_offset, X_scale = _preprocess_data(X, y)

        self.coef_, _, self.rank_, self.singular_ = lstsq(X, y)
        self.coef_ = self.coef_.T

        self.coef_ = np.ravel(self.coef_)

        self._set_intercept(X_offset, y_offset, X_scale)
        return self
