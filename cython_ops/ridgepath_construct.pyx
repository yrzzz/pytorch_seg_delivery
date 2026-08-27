# cython: cdivision=True
# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=True

from libc.math cimport sqrt, abs, cos, sin, acos
from libc.math cimport M_PI as pi
import numpy as np
cimport numpy as cnp
import cython
from cython.parallel import parallel, prange
import array

DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t

IJ = np.uint16
ctypedef cnp.uint16_t IJ_t

INDX = np.intp
ctypedef Py_ssize_t INDX_t


cdef _find_direction(
    IJ_t [:,:] target_map,
    IJ_t [:,:] label_map,
    IJ_t [:,:] dist_map,
    DTYPE_t [:,:,:] row_prob,
    DTYPE_t [:,:,:] col_prob,
):
    cdef INDX_t n_rows, n_cols, row, col, d_row, d_col, neigh_row, neigh_col
    cdef IJ_t dist, max_neigh_dist, lb
    n_rows = target_map.shape[0]
    n_cols = target_map.shape[1]

    with nogil:
        for row in prange(n_rows):
            for col in prange(n_cols):
                dist = dist_map[row, col]
                if target_map[row, col] > 0:
                    # this is the target of the flow. stay
                    row_prob[row, col, 1] = 1
                    col_prob[row, col, 1] = 1
                elif dist == 0:
                    # this is background, set to background
                    row_prob[row, col, 3] = 1
                    col_prob[row, col, 3] = 1
                else:
                    max_neigh_dist = dist
                    lb = label_map[row, col]
                    # find the maximum distance among 8 neighbors
                    for d_row in range(3):
                        for d_col in range(3):
                            neigh_row = row + d_row - 1
                            neigh_col = col + d_col - 1
                            if neigh_row >= n_rows or neigh_col >= n_cols or neigh_row < 0 or neigh_col < 0:
                                continue
                            if lb != label_map[neigh_row, neigh_col]:
                                continue
                            if dist_map[neigh_row, neigh_col] > max_neigh_dist:
                                max_neigh_dist = dist_map[neigh_row, neigh_col]
                    if max_neigh_dist == dist:
                        # the current pixel is local maximum, equal prob to all direction
                        row_prob[row, col, 0] = 1
                        col_prob[row, col, 0] = 1
                        row_prob[row, col, 1] = 1
                        col_prob[row, col, 1] = 1
                        row_prob[row, col, 2] = 1
                        col_prob[row, col, 2] = 1
                    else:
                        # accumulate 
                        for d_row in range(3):
                            for d_col in range(3):
                                neigh_row = row + d_row - 1
                                neigh_col = col + d_col - 1
                                if neigh_row >= n_rows or neigh_col >= n_cols or neigh_row < 0 or neigh_col < 0:
                                    continue
                                if dist_map[neigh_row, neigh_col] == max_neigh_dist:
                                    row_prob[row, col, d_row] += 1
                                    col_prob[row, col, d_col] += 1


cdef _smooth_prob(
    DTYPE_t [:,:,:] row_prob,
    DTYPE_t [:,:,:] col_prob,
    DTYPE_t [:,:,:] row_prob_smooth,
    DTYPE_t [:,:,:] col_prob_smooth,
    IJ_t [:,:] label_map,
    IJ_t [:,:] target_map,
    INDX_t smooth_range,
):
    cdef INDX_t n_rows, n_cols, row, col, d_row, d_col, new_row, new_col
    cdef IJ_t dist, max_neigh_dist, lb, count
    cdef DTYPE_t tot_prob_0, tot_prob_1, tot_prob_2

    n_rows = row_prob.shape[0]
    n_cols = row_prob.shape[1]
    with nogil:
        for row in prange(n_rows):
            for col in prange(n_cols):
                lb = label_map[row, col]
                # do not smooth background
                if lb == 0:
                    continue
                # do not smooth ridge
                if target_map[row, col] > 0:
                    row_prob_smooth[row, col, 1] = 1
                    col_prob_smooth[row, col, 1] = 1
                    continue
                count = 0
                tot_prob_0 = 0
                tot_prob_1 = 0
                tot_prob_2 = 0
                for d_row in range(smooth_range):
                    new_row = row + d_row - smooth_range // 2
                    if new_row >= 0 and new_row < n_rows:
                        if label_map[new_row, col] == lb:
                            count = count + 1
                            tot_prob_0 = tot_prob_0 + row_prob[new_row, col, 0]
                            tot_prob_1 = tot_prob_1 + row_prob[new_row, col, 1]
                            tot_prob_2 = tot_prob_2 + row_prob[new_row, col, 2]
                if count > 0:
                    row_prob_smooth[row, col, 0] = tot_prob_0 / count
                    row_prob_smooth[row, col, 1] = tot_prob_1 / count
                    row_prob_smooth[row, col, 2] = tot_prob_2 / count

                count = 0
                tot_prob_0 = 0
                tot_prob_1 = 0
                tot_prob_2 = 0
                for d_col in range(smooth_range):
                    new_col = col + d_col - smooth_range // 2
                    if new_col >= 0 and new_col < n_cols:
                        if label_map[row, new_col] == lb:
                            count = count + 1
                            tot_prob_0 = tot_prob_0 + col_prob[row, new_col, 0]
                            tot_prob_1 = tot_prob_1 + col_prob[row, new_col, 1]
                            tot_prob_2 = tot_prob_2 + col_prob[row, new_col, 2]
                if count > 0:
                    col_prob_smooth[row, col, 0] = tot_prob_0 / count
                    col_prob_smooth[row, col, 1] = tot_prob_1 / count
                    col_prob_smooth[row, col, 2] = tot_prob_2 / count


cdef _smooth_stationary(
    DTYPE_t [:,:,:] row_prob,
    DTYPE_t [:,:,:] col_prob,
    IJ_t [:,:] target_map,
    IJ_t [:,:] label_map,
):
    cdef INDX_t n_rows, n_cols, row, col, new_row, new_col, i
    cdef IJ_t lb, count
    cdef INDX_t steps[2][8]
    steps[0][:] = [-1, -1, -1, 0, 0, 1, 1, 1]
    steps[1][:] = [-1, 0, 1, -1, 1, -1, 0, 1]

    n_rows = row_prob.shape[0]
    n_cols = row_prob.shape[1]
    
    with nogil:
        for row in prange(n_rows):
            for col in prange(n_cols):
                if row_prob[row, col, 1] == 1 and col_prob[row, col, 1] == 1:
                    if target_map[row, col] > 0:
                        continue
                    count = 0
                    lb = label_map[row, col]
                    row_prob[row, col, 0] = 0
                    row_prob[row, col, 1] = 0
                    row_prob[row, col, 2] = 0
                    col_prob[row, col, 0] = 0
                    col_prob[row, col, 1] = 0
                    col_prob[row, col, 2] = 0
                    for i in range(8):
                        new_row = row + steps[0][i]
                        new_col = col + steps[1][i]
                        if new_row >= 0 and new_row < n_rows and new_col >= 0 and new_col < n_cols and label_map[new_row, new_col] == lb:
                            count = count + 1
                            row_prob[row, col, 0] = row_prob[row, col, 0] + row_prob[new_row, new_col, 0]
                            row_prob[row, col, 1] = row_prob[row, col, 1] + row_prob[new_row, new_col, 1]
                            row_prob[row, col, 2] = row_prob[row, col, 2] + row_prob[new_row, new_col, 2]
                            col_prob[row, col, 0] = col_prob[row, col, 0] + col_prob[new_row, new_col, 0]
                            col_prob[row, col, 1] = col_prob[row, col, 1] + col_prob[new_row, new_col, 1]
                            col_prob[row, col, 2] = col_prob[row, col, 2] + col_prob[new_row, new_col, 2]
                    if count > 0:
                        row_prob[row, col, 0] = row_prob[row, col, 0] / count
                        row_prob[row, col, 1] = row_prob[row, col, 1] / count
                        row_prob[row, col, 2] = row_prob[row, col, 2] / count
                        col_prob[row, col, 0] = col_prob[row, col, 0] / count
                        col_prob[row, col, 1] = col_prob[row, col, 1] / count
                        col_prob[row, col, 2] = col_prob[row, col, 2] / count


def label_to_ridge_direct_prob(
    cnp.ndarray[IJ_t, ndim=2] ridge_map,
    cnp.ndarray[IJ_t, ndim=2] label_map,
    cnp.ndarray[IJ_t, ndim=2] dist_map,
):
    n_rows = ridge_map.shape[0]
    n_cols = ridge_map.shape[1]

    cdef cnp.ndarray row_prob = np.zeros((n_rows, n_cols, 4), dtype=DTYPE)
    cdef cnp.ndarray col_prob = np.zeros((n_rows, n_cols, 4), dtype=DTYPE)

    _find_direction(ridge_map, label_map, dist_map, row_prob, col_prob)
    row_prob = row_prob / np.sum(row_prob, axis=2, keepdims=True)
    col_prob = col_prob / np.sum(col_prob, axis=2, keepdims=True)
    return row_prob, col_prob


def smooth_stationary_point(
    cnp.ndarray[DTYPE_t, ndim=3] row_prob,
    cnp.ndarray[DTYPE_t, ndim=3] col_prob,
    cnp.ndarray[IJ_t, ndim=2] ridge_map,
    cnp.ndarray[IJ_t, ndim=2] label_map,
):
    _smooth_stationary(row_prob, col_prob, ridge_map, label_map)


def smooth_direct_prob(
    cnp.ndarray[DTYPE_t, ndim=3] row_prob,
    cnp.ndarray[DTYPE_t, ndim=3] col_prob,
    cnp.ndarray[IJ_t, ndim=2] label_map,
    cnp.ndarray[IJ_t, ndim=2] ridge_map,
    IJ_t smooth_range,
):
    assert smooth_range % 2 == 1
    n_rows = row_prob.shape[0]
    n_cols = row_prob.shape[1]

    cdef cnp.ndarray row_prob_smooth = np.zeros((n_rows, n_cols, 4), dtype=DTYPE)
    cdef cnp.ndarray col_prob_smooth = np.zeros((n_rows, n_cols, 4), dtype=DTYPE)
    _smooth_prob(row_prob, col_prob, row_prob_smooth, col_prob_smooth, label_map, ridge_map, smooth_range)
    row_prob_smooth[:,:,3] = row_prob[:,:,3]
    col_prob_smooth[:,:,3] = col_prob[:,:,3]
    return row_prob_smooth, col_prob_smooth
