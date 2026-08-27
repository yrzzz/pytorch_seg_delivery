"""
"""

from libc.stdint cimport (
  uint8_t, uint16_t, uint32_t, uint64_t,
   int8_t,  int16_t,  int32_t,  int64_t
)
from libcpp cimport bool as native_bool

import multiprocessing

from cpython cimport array 
cimport numpy as cnp
import numpy as np


cdef extern from "tenxnet/vision/post_analysis/lb_dist_transform.hpp" namespace "label_edt":
    cdef void squared_edt_1d_multi_seg[T](
        T *labels,
        float *dest,
        int n,
        int stride,
    ) nogil

    cdef void squared_edt_1d_parabolic(
        float *field,
        float *dist,
        int n,
        int, stride,
    ) nogil

    cdef void squared_edt_1d_parabolic_multi_seg[T](
        T *labels,
        float *field,
        float *dist,
        int n,
        int stride,
    ) nogil

    cdef float* _edt2dsq[T](
        T* labels,
        size_t sx,
        size_t sy,
        int parallel,
        float* output
    ) nogil


def label_edtsq_1d(data):
    cdef uint8_t[:] arr_memview8
    cdef uint16_t[:] arr_memview16
    cdef uint32_t[:] arr_memview32
    cdef uint64_t[:] arr_memview64

    cdef size_t n_pixels = data.size
    cdef cnp.ndarray[float, ndim=1] output = np.zeros((n_pixels,), dtype=np.float32)
    cdef float[:] outputview = output

    if data.dtype in (np.uint8, np.int8):
        arr_memview8 = data.astype(np.uint8)
        squared_edt_1d_multi_seg[uint8_t](
            <uint8_t*>&arr_memview8[0],
            &outputview[0],
            data.size,
            1,
        )
    elif data.dtype in (np.uint16, np.int16):
        arr_memview16 = data.astype(np.uint16)
        squared_edt_1d_multi_seg[uint16_t](
            <uint16_t*>&arr_memview16[0],
            &outputview[0],
            data.size,
            1,
        )
    elif data.dtype in (np.uint32, np.int32):
        arr_memview32 = data.astype(np.uint32)
        squared_edt_1d_multi_seg[uint32_t](
            <uint32_t*>&arr_memview32[0],
            &outputview[0],
            data.size,
            1,
        )
    elif data.dtype in (np.uint64, np.int64):
        arr_memview64 = data.astype(np.uint64)
        squared_edt_1d_multi_seg[uint64_t](
            <uint64_t*>&arr_memview64[0],
            &outputview[0],
            data.size,
            1
        )
    else:
        raise ValueError("input data has to be one of the integer type")

    return output


def label_sqedt_1d_parabola(data, field):
    cdef uint8_t[:] arr_memview8
    cdef uint16_t[:] arr_memview16
    cdef uint32_t[:] arr_memview32
    cdef uint64_t[:] arr_memview64

    cdef float[:] field_memviewfloat = field.astype(np.float32)

    cdef size_t n_pixels = data.size
    cdef cnp.ndarray[float, ndim=1] output = np.zeros((n_pixels,), dtype=np.float32)
    cdef float[:] outputview = output

    if data.dtype in (np.uint8, np.int8):
        arr_memview8 = data.astype(np.uint8)
        squared_edt_1d_parabolic_multi_seg[uint8_t](
            <uint8_t*>&arr_memview8[0],
            &field_memviewfloat[0],
            &outputview[0],
            data.size,
            1,
        )
    elif data.dtype in (np.uint16, np.int16):
        arr_memview16 = data.astype(np.uint16)
        squared_edt_1d_parabolic_multi_seg[uint16_t](
            <uint16_t*>&arr_memview16[0],
            &field_memviewfloat[0],
            &outputview[0],
            data.size,
            1,
        )
    elif data.dtype in (np.uint32, np.int32):
        arr_memview32 = data.astype(np.uint32)
        squared_edt_1d_parabolic_multi_seg[uint32_t](
            <uint32_t*>&arr_memview32[0],
            &field_memviewfloat[0],
            &outputview[0],
            data.size,
            1,
        )
    elif data.dtype in (np.uint64, np.int64):
        arr_memview64 = data.astype(np.uint64)
        squared_edt_1d_parabolic_multi_seg[uint64_t](
            <uint64_t*>&arr_memview64[0],
            &field_memviewfloat[0],
            &outputview[0],
            data.size,
            1
        )
    else:
        raise ValueError("input data has to be one of the integer type")

    return output


# dongyao: seem to have bug for supposed padding at row direction.
def label_edtsq_2d(data, order='C', parallel=1):
    cdef uint8_t[:,:] arr_memview8
    cdef uint16_t[:,:] arr_memview16
    cdef uint32_t[:,:] arr_memview32
    cdef uint64_t[:,:] arr_memview64

    cdef size_t sx = data.shape[1]
    cdef size_t sy = data.shape[0]

    if order == 'F':
        sx = data.shape[0]
        sy = data.shape[1]

    cdef size_t voxels = sx * sy
    cdef cnp.ndarray[float, ndim=1] output = np.zeros( (voxels,), dtype=np.float32 )
    cdef float[:] outputview = output

    if data.dtype in (np.uint8, np.int8):
        arr_memview8 = data.astype(np.uint8)
        _edt2dsq[uint8_t](
            <uint8_t*>&arr_memview8[0,0],
            sx, sy,
            parallel,
            &outputview[0]
        )
    elif data.dtype in (np.uint16, np.int16):
        arr_memview16 = data.astype(np.uint16)
        _edt2dsq[uint16_t](
            <uint16_t*>&arr_memview16[0,0],
            sx, sy,
            parallel,
            &outputview[0]      
        )
    elif data.dtype in (np.uint32, np.int32):
        arr_memview32 = data.astype(np.uint32)
        _edt2dsq[uint32_t](
            <uint32_t*>&arr_memview32[0,0],
            sx, sy,
            parallel,
            &outputview[0]      
        )
    elif data.dtype in (np.uint64, np.int64):
        arr_memview64 = data.astype(np.uint64)
        _edt2dsq[uint64_t](
            <uint64_t*>&arr_memview64[0,0],
            sx, sy,
            parallel,
            &outputview[0]      
        )
    else:
        raise ValueError("input data has to be one of the integer type")

    return output


def skeletonize_label(image):
    """Skeletonize image with integer labels.

    Optimized parts of the Zhang-Suen [1]_ skeletonization.

    Iteratively, pixels meeting removal criteria are removed,
    till only the skeleton remains (that is, no further removable pixel
    was found).

    Performs a hard-coded correlation to assign every neighborhood of 8 a
    unique number, which in turn is used in conjunction with a look up
    table to select the appropriate thinning criteria.

    This is a modification from scikit-image implementation of
    skeletonization of binary image. The modified algorithm can accept image with
    integer labels.

    The original implementation can be found here:

    https://github.com/scikit-image/scikit-image/blob/812e21e44129258d8637fbcf665e590556ea96d1/skimage/morphology/_skeletonize_cy.pyx#L11

    Parameters
    ----------
    image : numpy.ndarray
        A image containing the objects to be skeletonized. any integer
        larger than 0 represents objects, and '0' represents background.

    Returns
    -------
    skeleton : ndarray
        A matrix containing the thinned image. The skeleton keeps the original
        integer label.

    References
    ----------
    .. [1] A fast parallel algorithm for thinning digital patterns,
           T. Y. Zhang and C. Y. Suen, Communications of the ACM,
           March 1984, Volume 27, Number 3.
    """

    # look up table - there is one entry for each of the 2^8=256 possible
    # combinations of 8 binary neighbours. 1's, 2's and 3's are candidates
    # for removal at each iteration of the algorithm.
    cdef int *lut = \
      [0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0,
       0, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0,
       0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1,
       0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 0, 0,
       1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3,
       0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0]

    cdef int pixel_removed, first_pass, neighbors
    cdef cnp.uint16_t lbl

    # indices for fast iteration
    cdef Py_ssize_t row, col, nrows = image.shape[0]+2, ncols = image.shape[1]+2

    # we copy over the image into a larger version with a single pixel border
    # this removes the need to handle border cases below
    _skeleton = np.zeros((nrows, ncols), dtype=np.uint16)
    _skeleton[1:nrows-1, 1:ncols-1] = image

    _cleaned_skeleton = _skeleton.copy()

    # cdef'd numpy-arrays for fast, typed access
    cdef cnp.uint16_t [:, ::1] skeleton, cleaned_skeleton

    skeleton = _skeleton
    cleaned_skeleton = _cleaned_skeleton

    pixel_removed = True

    # the algorithm reiterates the thinning till
    # no further thinning occurred (variable pixel_removed set)
    while pixel_removed:
        pixel_removed = False

        # there are two phases, in the first phase, pixels labeled (see below)
        # 1 and 3 are removed, in the second 2 and 3

        # nogil can't iterate through `(True, False)` because it is a Python
        # tuple. Use the fact that 0 is Falsy, and 1 is truthy in C
        # for the iteration instead.
        # for first_pass in (True, False):
        for pass_num in range(2):
            first_pass = (pass_num == 0)
            for row in range(1, nrows-1):
                for col in range(1, ncols-1):
                    # all set pixels ...
                    if skeleton[row, col]:
                        # are correlated with a kernel (coefficients spread around here ...)
                        # to apply a unique number to every possible neighborhood ...

                        # which is used with the lut to find the "connectivity type"

                        lbl = skeleton[row, col]
                        key = 1 * int(skeleton[row - 1, col - 1] == lbl) +\
                                2 * int(skeleton[row - 1, col] == lbl) +\
                                4 * int(skeleton[row - 1, col + 1] == lbl) +\
                                8 * int(skeleton[row, col + 1] == lbl) +\
                                16 * int(skeleton[row + 1, col + 1] == lbl) +\
                                32 * int(skeleton[row + 1, col] == lbl) +\
                                64 * int(skeleton[row + 1, col - 1] == lbl) +\
                                128 * int(skeleton[row, col - 1] == lbl)
                        neighbors = lut[key]

                        # if the condition is met, the pixel is removed (unset)
                        if ((neighbors == 1 and first_pass) or
                                (neighbors == 2 and not first_pass) or
                                (neighbors == 3)):
                            cleaned_skeleton[row, col] = 0
                            pixel_removed = True

            # once a step has been processed, the original skeleton
            # is overwritten with the cleaned version
            skeleton[:, :] = cleaned_skeleton[:, :]

    return _skeleton[1:nrows-1, 1:ncols-1]


def remove_low_weight_points(
    uint16_t[:, :] label_map,
    uint16_t[:, :] candidates,
    uint16_t[:, :] edt_map, 
    float threshold
):

    num_lbls = np.max(label_map)
    cdef cnp.uint16_t [:] max_dist_lookup = np.zeros(num_lbls + 1, dtype=np.uint16)
    cdef cnp.uint16_t lb, max_edt
    cdef Py_ssize_t i, j, nrows = label_map.shape[0], ncols = label_map.shape[1]

    for i in range(nrows):
        for j in range(ncols):
            lb = label_map[i, j]
            if lb > 0:
                max_dist_lookup[lb] = max(max_dist_lookup[lb], edt_map[i, j])

    for i in range(nrows):
        for j in range(ncols):
            if candidates[i, j] > 0:
                max_edt = max_dist_lookup[label_map[i, j]]
                if edt_map[i, j] < max_edt * threshold:
                    candidates[i, j] = 0
    return
