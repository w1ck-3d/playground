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
from scipy import linalg

from sklearn.utils.extmath import safe_sparse_dot

from sklearn.utils._array_api import (
    _asarray_with_order,
    _is_numpy_namespace,
    get_namespace,
)


def column_or_1d(y, *, dtype=None, warn=False):
    xp, _ = get_namespace(y)
    y = check_array(
        y,
        dtype=dtype,
    )

    shape = y.shape
    if len(shape) == 1:
        return _asarray_with_order(xp.reshape(y, (-1,)), order="C", xp=xp)
    if len(shape) == 2 and shape[1] == 1:
        return _asarray_with_order(xp.reshape(y, (-1,)), order="C", xp=xp)


def check_X_y(X, y, *, dtype="numeric", order=None, copy=False):
    X = check_array(X, dtype=dtype, order=order, copy=copy)
    y = column_or_1d(y, warn=True)

    return X, y


def check_array(array, *, dtype="numeric", order=None, copy=False):
    xp, is_array_api_compliant = get_namespace(array)

    array_orig = array

    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not is_array_api_compliant and not hasattr(dtype_orig, "kind"):
        dtype_orig = None

    if dtype_numeric:
        if (
            dtype_orig is not None
            and hasattr(dtype_orig, "kind")
            and dtype_orig.kind == "O"
        ):
            dtype = xp.float64
        else:
            dtype = None

    if dtype is not None and _is_numpy_namespace(xp):
        dtype = np.dtype(dtype)

    if dtype is not None and _is_numpy_namespace(xp):
        dtype = np.dtype(dtype)

    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)

    if copy:
        array = _asarray_with_order(array, dtype=dtype, order=order, copy=True, xp=xp)

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


class LinearRegression:
    def predict(self, X):
        X = check_array(X)
        return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_

    def _set_intercept(self, X_offset, y_offset, X_scale):
        self.coef_ = np.divide(self.coef_, X_scale, dtype=X_scale.dtype)
        self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X, y, X_offset, y_offset, X_scale = _preprocess_data(X, y)

        self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)
        self.coef_ = self.coef_.T

        self.coef_ = np.ravel(self.coef_)

        self._set_intercept(X_offset, y_offset, X_scale)
        return self