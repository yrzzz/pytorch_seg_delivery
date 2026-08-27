# distutils: language = c
# distutils: sources = maskApi.c

#**************************************************************************
# Original code from Microsoft COCO Toolbox:
# https://github.com/cocodataset/cocoapi/tree/master/common
#
# Modified significantly by 10x Genomics
#
# a python RLE object is represented by (such definition does not seem to work in cython):
#
# class RleDataDict(TypedDict):
#     """Dict representation of uncompressed run length encoding.
#
#     Note that the related RLE implementation is based on column major ordering.
#
#     size (Tuple[int, int]): (height, width) of the window used to calculate run length encoding
#     counts (np.ndarray[int, np.dtype[np.uint32]]): the run length encoding
#     """
#
#     size: tuple[int, int]
#     counts: np.ndarray[int, np.dtype[np.uint32]]
#
#**************************************************************************

# import both Python-level and C-level symbols of Numpy
# the API uses Numpy to interface C and Python
from cpython cimport PyObject, Py_INCREF
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libcpp cimport bool

# intialized Numpy. must do.
np.import_array()

# import numpy C function
# we use PyArray_ENABLEFLAGS to make Numpy ndarray responsible to memoery management
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

# this is needed for bool identifier for c99
cdef extern from "<stdbool.h>":
    pass

# Declare the prototype of the C functions in MaskApi.h
cdef extern from "tenxnet/vision/models/metrics/maskApi.h":
    ctypedef unsigned int uint
    ctypedef unsigned long siz
    ctypedef unsigned char byte
    ctypedef unsigned int uint32
    ctypedef double* BB
    ctypedef struct RLE:
        siz h,
        siz w,
        siz m,
        uint* cnts,
    void rleInit( RLE *R, siz h, siz w, siz m, uint *cnts )
    void rlesInit( RLE **R, siz n )
    void rleEncode( RLE *R, const byte *M, siz h, siz w, siz n )
    void rleEncodeLabels(RLE *R, const uint32 *Rlabels, const uint32 *M, siz h, siz w, siz n )
    void rleDecode( const RLE *R, byte *mask, siz n )
    void rleMerge( const RLE *R, RLE *M, siz n, int intersect )
    void rleArea( const RLE *R, siz n, uint *a )
    void rleIou( RLE *dt, RLE *gt, siz m, siz n, byte *iscrowd, double *o )
    void bbIou( BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o )
    void rleToBbox( const RLE *R, BB bb, siz n )
    void rleFrBbox( RLE *R, const BB bb, siz h, siz w, siz n )
    void rleFrPoly( RLE *R, const double *xy, siz k, siz h, siz w )
    char* rleToString( const RLE *R )
    void rleFrString( RLE *R, char *s, siz h, siz w )
    void rlesOffset(const RLE *orig_rle, RLE *new_rle, siz n, siz new_h, siz new_w, siz offset_h, siz offset_w)
    void rleIntersectPercent( RLE *dt, RLE *gt, siz m, siz n, double *o )
    void rleIntersectExclude(const RLE *ref_rle, const RLE *compare_rle, RLE *intersect_rle, RLE *exclude_rle)

# python class to wrap RLE array in C
# the class handles the memory allocation and deallocation
cdef class RLEs:
    cdef RLE *_R
    cdef siz _n

    def __cinit__(self, siz n =0):
        rlesInit(&self._R, n)
        self._n = n

    # free the RLE array here
    def __dealloc__(self):
        if self._R is not NULL:
            for i in range(self._n):
                free(self._R[i].cnts)
            free(self._R)

    def __getattr__(self, key):
        if key == 'n':
            return self._n
        raise AttributeError(key)

    def _output_counts(self, i):
        """Output counts in uncompressed fashion.

        For debugging purposes.
        """
        if i >= self._n:
            raise IndexError(i)
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self._R[i].m
        cnts = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, <void*> self._R[i].cnts)
        # increase reference count is necessary so that cnts won't point to freed memory
        Py_INCREF(self)
        PyArray_ENABLEFLAGS(cnts, np.NPY_OWNDATA)
        return cnts

# python class to wrap Mask array in C
# the class handles the memory allocation and deallocation
cdef class Masks:
    cdef byte *_mask
    cdef siz _h
    cdef siz _w
    cdef siz _n

    def __cinit__(self, h, w, n):
        self._mask = <byte*> malloc(h*w*n* sizeof(byte))
        self._h = h
        self._w = w
        self._n = n

    # def __dealloc__(self):
        # the memory management of _mask has been passed to np.ndarray
        # it doesn't need to be freed here

    # called when passing into np.array() and return an np.ndarray in column-major order
    def __array__(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self._h*self._w*self._n
        # Create a 1D array, and reshape it to fortran/Matlab column-major array
        ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT8, self._mask).reshape((self._h, self._w, self._n), order='F')
        # The _mask allocated by Masks is now handled by ndarray
        PyArray_ENABLEFLAGS(ndarray, np.NPY_OWNDATA)
        return ndarray

# internal conversion from Python RLEs object to compressed RLE format
def _toString(RLEs Rs):
    cdef siz n = Rs.n
    cdef bytes py_string
    cdef char* c_string
    objs = []
    for i in range(n):
        c_string = rleToString( <RLE*> &Rs._R[i] )
        py_string = c_string
        objs.append({
            'size': [Rs._R[i].h, Rs._R[i].w],
            'counts': py_string
        })
        free(c_string)
    return objs

# convert from RLEs to list of dictionary representing RLEs
def _toPyObj(RLEs Rs, bool compressed=True) -> list:
    cdef siz n = Rs.n
    cdef bytes py_string
    cdef char* c_string
    cdef np.npy_intp shape[1]
    objs = []
    for i in range(n):
        if compressed:
            c_string = rleToString( <RLE*> &Rs._R[i] )
            py_string = c_string
            objs.append({
                'size': [Rs._R[i].h, Rs._R[i].w],
                'counts': py_string
            })
            free(c_string)
        else:
            shape[0] = <np.npy_intp> Rs._R[i].m
            cnts = np.zeros(shape, dtype=np.uint)
            # This only creates a wrapper around the c array
            cnts = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, <void*> Rs._R[i].cnts)
            # increase the reference count so that when RLEs is gc'd, cnts doesn't point at
            # freed memory. This is to avoid duplication of Rs._R[i].cnts
            Py_INCREF(Rs)
            objs.append({
                'size': [Rs._R[i].h, Rs._R[i].w],
                'counts': cnts
            })
    return objs

# internal conversion from compressed or uncompressed RLE objects to RLEs
def _frPyObj(list rleObjs, bool compressed):
    cdef siz n = len(rleObjs)
    cdef np.ndarray[np.uint32_t, ndim=1] cnts
    Rs = RLEs(n)
    cdef bytes py_string
    cdef char* c_string
    if compressed:
        for i, obj in enumerate(rleObjs):
            py_string = str.encode(obj['counts']) if type(obj['counts']) == str else obj['counts']
            c_string = py_string
            rleFrString( <RLE*> &Rs._R[i], <char*> c_string, obj['size'][0], obj['size'][1] )
    else:
        for i, obj in enumerate(rleObjs):
            cnts = np.array(obj['counts'], dtype=np.uint32)
            rleInit( <RLE*> &Rs._R[i], obj['size'][0], obj['size'][1], len(cnts), <uint*> &cnts[0])
    return Rs

# encode mask to RLEs objects
# list of RLE string can be generated by RLEs member function
def encode(
    np.ndarray[np.uint8_t, ndim=3, mode='fortran'] mask,
    bool compressed=True
) -> list[dict]:
    h, w, n = mask.shape[0], mask.shape[1], mask.shape[2]
    cdef RLEs Rs = RLEs(n)
    rleEncode(Rs._R,<byte*>&mask[0,0,0],h,w,n)
    objs = _toPyObj(Rs, compressed=compressed)
    return objs

# encode 2D labelmap to RLEs objects
def encode_labelmap(
    np.ndarray[np.uint32_t, ndim=2, mode='fortran'] mask,
    bool compressed=True
) -> list[dict]:
    h, w, n = mask.shape[0], mask.shape[1], mask.shape[2]
    cdef RLEs Rs = RLEs(n)
    rleEncode(Rs._R,<byte*>&mask[0,0],h,w,n)
    objs = _toPyObj(Rs, compressed=compressed)
    return objs

# real encode 2D labelmap to RLEs objects - not sure what the stuff above is
def encode_labelmap_fast(
    np.ndarray[np.uint32_t, ndim=2, mode='fortran'] labelmap,
    np.ndarray[np.uint32_t, ndim=1] labels,
    bool compressed=True
) -> list[dict]:
    h, w = labelmap.shape[0], labelmap.shape[1]
    n = len(labels)
    cdef RLEs Rs = RLEs(n)
    rleEncodeLabels(Rs._R,<uint32*>&labels[0],<uint32*>&labelmap[0,0],h,w,n)
    objs = _toPyObj(Rs, compressed = compressed)
    return objs


# decode mask from compressed list of RLE string or RLEs object
def decode(list rleObjs, bool compressed=True) -> np.ndarray:
    cdef RLEs Rs = _frPyObj(rleObjs, compressed)
    h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n
    masks = Masks(h, w, n)
    rleDecode(<RLE*>Rs._R, masks._mask, n)
    return np.array(masks)

def merge(list rleObjs, int intersect=0, bool compressed=True) -> dict:
    cdef RLEs Rs = _frPyObj(rleObjs, compressed)
    cdef RLEs R = RLEs(1)
    rleMerge(<RLE*>Rs._R, <RLE*> R._R, <siz> Rs._n, intersect)
    obj = _toPyObj(R, compressed=compressed)[0]
    return obj

def area(list rleObjs, bool compressed=True) -> np.ndarray:
    cdef RLEs Rs = _frPyObj(rleObjs, compressed)
    cdef uint* _a = <uint*> malloc(Rs._n * sizeof(uint))
    rleArea(Rs._R, Rs._n, _a)
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> Rs._n
    a = np.array((Rs._n, ), dtype=np.uint8)
    a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, <void*> _a)
    PyArray_ENABLEFLAGS(a, np.NPY_OWNDATA)
    return a

def offsetRLE(
    rleObjs,
    new_height: np.int16,
    new_width: np.int16,
    offset_height: np.int16,
    offset_width: np.int16,
    compressed: bool =True
) -> list[dict]:
    if len(rleObjs) > 0:
        if rleObjs[0]["size"][0] + offset_height > new_height:
            raise ValueError("New height is not large enough to fit offset.")
        if rleObjs[0]["size"][1] + offset_width > new_width:
            raise ValueError("New width is not large enough to fit offset.")

    cdef RLEs original_rles = _frPyObj(rleObjs, compressed)
    n = original_rles._n
    cdef RLEs offset_rles = RLEs(n)
    rlesOffset(
        <RLE*> original_rles._R,
        <RLE*> offset_rles._R,
        <siz> n,
        <siz> new_height,
        <siz> new_width,
        <siz> offset_height,
        <siz> offset_width,
    )
    objs = _toPyObj(offset_rles, compressed=compressed)
    return objs

def intersect_exclude_rle(ref_rle_obj, compare_rle_obj, bool compressed) -> tuple[dict, dict]:
    """Calculation intersection and exclusion of two RLEs.
    
    Note the exclusion rle means the part that is in compare_rle but not in ref_rle.
    """
    cdef RLEs ref_rle = _frPyObj([ref_rle_obj], compressed=compressed)
    cdef RLEs compare_rle = _frPyObj([compare_rle_obj], compressed=compressed)
    cdef RLEs intersect_rle = RLEs(1)
    cdef RLEs exclude_rle = RLEs(1)
    rleIntersectExclude(<RLE*> ref_rle._R, <RLE*> compare_rle._R, <RLE*> intersect_rle._R, <RLE*> exclude_rle._R)
    inter_rle_obj = _toPyObj(intersect_rle, compressed=compressed)[0]
    exclude_rle_obj = _toPyObj(exclude_rle, compressed=compressed)[0]
    return inter_rle_obj, exclude_rle_obj

# iou computation. support function overload (RLEs-RLEs and bbox-bbox).
# Remove iscrowd tag and make it always False for both rleIou and bbIou, since we won't have such label.
# the underlying c functions are not changed.
def iou(dt, gt, compressed=True):
    def _preproc(objs):
        if len(objs) == 0:
            return objs
        if type(objs) == np.ndarray:
            if len(objs.shape) == 1:
                objs = objs.reshape((objs[0], 1))
            # check if it's Nx4 bbox
            if not len(objs.shape) == 2 or not objs.shape[1] == 4:
                raise Exception('numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension')
            objs = objs.astype(np.double)
        elif type(objs) == list:
            # check if list is in box format and convert it to np.ndarray
            isbox = np.all(np.array([(len(obj)==4) and ((type(obj)==list) or (type(obj)==np.ndarray)) for obj in objs]))
            isrle = np.all(np.array([type(obj) == dict for obj in objs]))
            if isbox:
                objs = np.array(objs, dtype=np.double)
                if len(objs.shape) == 1:
                    objs = objs.reshape((1,objs.shape[0]))
            elif isrle:
                objs = _frPyObj(objs, compressed)
            else:
                raise Exception('list input can be bounding box (Nx4) or RLEs ([RLE])')
        else:
            raise Exception('unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.')
        return objs
    def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):
        rleIou( <RLE*> dt._R, <RLE*> gt._R, m, n, <byte*> &iscrowd[0], <double*> &_iou[0] )
    def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):
        bbIou( <BB> &dt[0,0], <BB> &gt[0,0], m, n, <byte*> &iscrowd[0], <double*> &_iou[0] )
    def _len(obj):
        cdef siz N = 0
        if type(obj) == RLEs:
            N = obj.n
        elif len(obj)==0:
            pass
        elif type(obj) == np.ndarray:
            N = obj.shape[0]
        return N

    # simple type checking
    cdef siz m, n
    dt = _preproc(dt)
    gt = _preproc(gt)
    m = _len(dt)
    n = _len(gt)
    if m == 0 or n == 0:
        return []
    if not type(dt) == type(gt):
        raise Exception('The dt and gt should have the same data type, either RLEs, list or np.ndarray')

    # define local variables
    cdef double* _iou = <double*> 0
    cdef np.npy_intp shape[1]
    # set iscrowd to be false
    cdef np.ndarray[np.uint8_t, ndim=1] iscrowd = np.zeros(n, dtype=np.uint8)
    # check type and assign iou function
    if type(dt) == RLEs:
        _iouFun = _rleIou
    elif type(dt) == np.ndarray:
        _iouFun = _bbIou
    else:
        raise Exception('input data type not allowed.')
    _iou = <double*> malloc(m*n* sizeof(double))
    iou = np.zeros((m*n, ), dtype=np.double)
    shape[0] = <np.npy_intp> m*n
    iou = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, <void*> _iou)
    PyArray_ENABLEFLAGS(iou, np.NPY_OWNDATA)
    _iouFun(dt, gt, iscrowd, m, n, iou)
    return iou.reshape((m,n), order='F')

def toBbox( rleObjs, bool compressed=True):
    cdef RLEs Rs = _frPyObj(rleObjs, compressed)
    cdef siz n = Rs.n
    cdef BB _bb = <BB> malloc(4*n* sizeof(double))
    rleToBbox( <const RLE*> Rs._R, _bb, n )
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> 4*n
    # this is not needed
    # bb = np.array((1,4*n), dtype=np.double)
    bb = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, <void*> _bb).reshape((n, 4))
    PyArray_ENABLEFLAGS(bb, np.NPY_OWNDATA)
    return bb

def frBbox(np.ndarray[np.double_t, ndim=2] bb, siz h, siz w ):
    cdef siz n = bb.shape[0]
    Rs = RLEs(n)
    rleFrBbox( <RLE*> Rs._R, <const BB> &bb[0,0], h, w, n )
    objs = _toString(Rs)
    return objs

def frPoly( poly, siz h, siz w, bool compressed=True):
    cdef np.ndarray[np.double_t, ndim=1] np_poly
    n = len(poly)
    Rs = RLEs(n)
    for i, p in enumerate(poly):
        np_poly = np.array(p, dtype=np.double, order='F')
        rleFrPoly( <RLE*>&Rs._R[i], <const double*> &np_poly[0], int(len(p)/2), h, w )
    if compressed:
        objs = _toString(Rs)
    else:
        objs = _toPyObj(Rs, compressed=False)
    return objs

def frUncompressedRLE(ucRles, siz h, siz w):
    cdef np.ndarray[np.uint32_t, ndim=1] cnts
    cdef RLE R
    cdef uint *data
    n = len(ucRles)
    objs = []
    for i in range(n):
        Rs = RLEs(1)
        cnts = np.array(ucRles[i]['counts'], dtype=np.uint32)
        # time for malloc can be saved here but it's fine
        data = <uint*> malloc(len(cnts)* sizeof(uint))
        for j in range(len(cnts)):
            data[j] = <uint> cnts[j]
        R = RLE(ucRles[i]['size'][0], ucRles[i]['size'][1], len(cnts), <uint*> data)
        Rs._R[0] = R
        objs.append(_toString(Rs)[0])
    return objs

# deprecated original function
def frPyObjects(pyobj, h, w):
    # encode rle from a list of python objects
    if type(pyobj) == np.ndarray:
        objs = frBbox(pyobj, h, w)
    elif type(pyobj) == list and len(pyobj[0]) == 4:
        objs = frBbox(pyobj, h, w)
    elif type(pyobj) == list and len(pyobj[0]) > 4:
        objs = frPoly(pyobj, h, w)
    elif type(pyobj) == list and type(pyobj[0]) == dict \
        and 'counts' in pyobj[0] and 'size' in pyobj[0]:
        objs = frUncompressedRLE(pyobj, h, w)
    # encode rle from single python object
    elif type(pyobj) == list and len(pyobj) == 4:
        objs = frBbox([pyobj], h, w)[0]
    elif type(pyobj) == list and len(pyobj) > 4:
        objs = frPoly([pyobj], h, w)[0]
    elif type(pyobj) == dict and 'counts' in pyobj and 'size' in pyobj:
        objs = frUncompressedRLE([pyobj], h, w)[0]
    else:
        raise Exception('input type is not supported.')
    return objs


def overlap_max_percent(dt, gt, compressed=True):
    def _preproc(objs):
        if len(objs) == 0:
            return objs
        if type(objs) == np.ndarray:
            if len(objs.shape) == 1:
                objs = objs.reshape((objs[0], 1))
            # check if it's Nx4 bbox
            if not len(objs.shape) == 2 or not objs.shape[1] == 4:
                raise Exception('numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension')
            objs = objs.astype(np.double)
        elif type(objs) == list:
            # check if list is in box format and convert it to np.ndarray
            isbox = np.all(np.array([(len(obj)==4) and ((type(obj)==list) or (type(obj)==np.ndarray)) for obj in objs]))
            isrle = np.all(np.array([type(obj) == dict for obj in objs]))
            if isbox:
                objs = np.array(objs, dtype=np.double)
                if len(objs.shape) == 1:
                    objs = objs.reshape((1,objs.shape[0]))
            elif isrle:
                objs = _frPyObj(objs, compressed)
            else:
                raise Exception('list input can be bounding box (Nx4) or RLEs ([RLE])')
        else:
            raise Exception('unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.')
        return objs
    def _rleIou(RLEs dt, RLEs gt, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):
        rleIntersectPercent( <RLE*> dt._R, <RLE*> gt._R, m, n, <double*> &_iou[0] )
    def _len(obj):
        cdef siz N = 0
        if type(obj) == RLEs:
            N = obj.n
        elif len(obj)==0:
            pass
        elif type(obj) == np.ndarray:
            N = obj.shape[0]
        return N

    # simple type checking
    cdef siz m, n
    dt = _preproc(dt)
    gt = _preproc(gt)
    m = _len(dt)
    n = _len(gt)
    if m == 0 or n == 0:
        return []
    if not type(dt) == type(gt):
        raise Exception('The dt and gt should have the same data type, either RLEs, list or np.ndarray')

    # define local variables
    cdef double* _iou = <double*> 0
    cdef np.npy_intp shape[1]
    # check type and assign iou function

    _iouFun = _rleIou

    _iou = <double*> malloc(m*n* sizeof(double))
    iou = np.zeros((m*n, ), dtype=np.double)
    shape[0] = <np.npy_intp> m*n
    iou = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, <void*> _iou)
    PyArray_ENABLEFLAGS(iou, np.NPY_OWNDATA)
    _iouFun(dt, gt, m, n, iou)
    return iou.reshape((m,n), order='F')
