# cython: cdivision=True
# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=True

import cython

import numpy as np
cimport numpy as cnp

from scipy import ndimage

DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t

INDX = np.intp
ctypedef Py_ssize_t INDX_t

LBL = np.uint32
ctypedef cnp.uint32_t LBL_t


@cython.profile(False)
cdef inline INDX_t fround(DTYPE_t x):
    return <INDX_t>(x+.5) if x>=0. else <INDX_t>(x-.5)


cdef void step_in_flow(
    DTYPE_t [:,:,:] final_position,
    DTYPE_t [:,:,:] flow,
    INDX_t [:,:] candidate_idx,
    INDX_t niter,
) except *:

    cdef:
        INDX_t t, j, row, col, n_candidate, p0, p1

    max_row = final_position.shape[1]
    max_col = final_position.shape[2]
    n_candidate = candidate_idx.shape[0]
    for t in range(niter):
        for j in range(n_candidate):
            row = candidate_idx[j, 0]
            col = candidate_idx[j, 1]
            p0, p1 = fround(final_position[0, row, col]), fround(final_position[1, row, col])
            final_position[0,row,col] = min(max_row-1, max(0, final_position[0,row,col] - flow[0,p0,p1]))
            final_position[1,row,col] = min(max_col-1, max(0, final_position[1,row,col] - flow[1,p0,p1]))
    return


cdef void extend_center(
    DTYPE_t [:] diffuse_map,
    INDX_t [:] row,
    INDX_t [:] col,
    INDX_t med_row,
    INDX_t med_col,
    INDX_t tot_col,
    INDX_t niter,
) except *:

    cdef:
        INDX_t t, n, n_point
    n_point = row.shape[0]

    for t in range(niter):
        diffuse_map[med_row * tot_col + med_col] += 1
        for n in range(n_point):
            diffuse_map[row[n] * tot_col + col[n]] = (
                diffuse_map[row[n] * tot_col + col[n]] +
                diffuse_map[row[n] * tot_col + col[n] - 1] +
                diffuse_map[row[n] * tot_col + col[n] + 1] +
                diffuse_map[(row[n] - 1) * tot_col + col[n]] +
                diffuse_map[(row[n] - 1) * tot_col + col[n] - 1] +
                diffuse_map[(row[n] - 1) * tot_col + col[n] + 1] +
                diffuse_map[(row[n] + 1) * tot_col + col[n]] +
                diffuse_map[(row[n] + 1) * tot_col + col[n] - 1] +
                diffuse_map[(row[n] + 1) * tot_col + col[n] + 1]
            ) / 9.0
    return


def follow_flows_cpy(flow: np.ndarray, niter: int =200):
    """follow flow to recover masks in 2D

    Parameters
    ----------------
    flow: float32 np.ndarray, (2, row, col) 
    niter: int (optional, default 200)
        number of iterations to follow the flow

    Returns
    ---------------
    p: float32, 3D array
        final locations of each pixel after dynamics
    """
    shape = np.array(flow.shape[1:])
    assert len(shape) == 2

    final_position = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    final_position = np.array(final_position).astype(np.float32)

    inds = np.logical_or(np.abs(flow[0])>1e-3, np.abs(flow[1]>1e-3))
    inds = np.array(np.nonzero(inds)).astype(np.intp).T
    if inds.ndim < 2 or inds.shape[0] < 5:
        print('WARNING: no mask pixels found')
        return final_position
    step_in_flow(final_position, flow, inds, niter)
    return final_position


def masks_to_flows(masks: np.ndarray):
    """convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined to be the
    closest pixel to the median of all pixels that is inside the
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map.

    Parameters
    -------------
    masks: int, 2D array, any non zero integer represent a unique object

    Returns
    -------------
    mu: float, 3D or 4D array
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].
    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask
        in which it resides

    """

    if masks.ndim != 2:
        raise ValueError("masks should have dimension of 2")
    row, col = masks.shape
    flow = np.zeros((2, row, col), np.float32)
    num_objects = masks.max()
    object_list = ndimage.find_objects(masks)

    for i, object_slice in enumerate(object_list):
        if object_slice is not None:
            slice_row, slice_col = object_slice
            window_row = slice_row.stop - slice_row.start + 1
            window_col = slice_col.stop - slice_col.start + 1
            row_idx, col_idx = np.nonzero(masks[slice_row, slice_col] == (i + 1))
            row_idx += 1
            col_idx += 1
            med_row, med_col = find_center(row_idx, col_idx)
            niter = 2 * (np.ptp(row_idx) + np.ptp(col_idx))
            diffuse_map = np.zeros((window_row + 2) * (window_col + 2), DTYPE)
            extend_center(diffuse_map, row_idx, col_idx, med_row, med_col, window_col, niter)
            diffuse_map[(row_idx + 1) * window_col + col_idx + 1] = np.log(1 + diffuse_map[(row_idx + 1) * window_col + col_idx + 1])
            flow_row = diffuse_map[(row_idx + 1) * window_col + col_idx] - diffuse_map[(row_idx - 1) * window_col + col_idx]
            flow_col = diffuse_map[row_idx * window_col + col_idx + 1] - diffuse_map[row_idx * window_col + col_idx - 1]
            flow[:, slice_row.start + row_idx - 1, slice_col.start + col_idx - 1] = np.stack((flow_row, flow_col))
    flow = flow / (np.sqrt(np.sum(flow ** 2, axis=0)) + 1e-20)
    return flow


def find_center(row: np.ndarray, col: np.ndarray):
    med_row = np.median(row)
    med_col = np.median(col)
    closest_idx = np.argmin((row - med_row)**2 + (col - med_col)**2)
    return row[closest_idx], col[closest_idx]
