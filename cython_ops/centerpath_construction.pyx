ctypedef double VALUE_T
ctypedef Py_ssize_t REFERENCE_T
ctypedef REFERENCE_T INDEX_T
ctypedef unsigned char BOOL_T
ctypedef unsigned char LEVELS_T

# -*- python -*-

"""Cython implementation of a binary min heap.

Original author: Almar Klein
Modified by: Zachary Pincus

License: BSD

Copyright 2009 Almar Klein

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# cython specific imports
import cython
from libc.stdlib cimport malloc, free

cdef extern from "pyport.h":
  double Py_HUGE_VAL

cdef VALUE_T inf = Py_HUGE_VAL

# this is handy
cdef inline INDEX_T index_min(INDEX_T a, INDEX_T b) nogil:
    return a if a <= b else b


cdef class BinaryHeap:
    """BinaryHeap(initial_capacity=128)

    A binary heap class that can store values and an integer reference.

    A binary heap is an object to store values in, optimized in such a way
    that the minimum (or maximum, but a minimum in this implementation)
    value can be found in O(log2(N)) time. In this implementation, a reference
    value (a single integer) can also be stored with each value.

    Use the methods push() and pop() to put in or extract values.
    In C, use the corresponding push_fast() and pop_fast().

    Parameters
    ----------
    initial_capacity : int
        Estimate of the size of the heap, if known in advance. (In any case,
        the heap will dynamically grow and shrink as required, though never
        below the `initial_capacity`.)

    Attributes
    ----------
    count : int
        The number of values in the heap
    levels : int
        The number of levels in the binary heap (see Notes below). The values
        are stored in the last level, so 2**levels is the capacity of the
        heap before another resize is required.
    min_levels : int
        The minimum number of levels in the heap (relates to the
        `initial_capacity` parameter.)

    Notes
    -----
    This implementation stores the binary heap in an array twice as long as
    the number of elements in the heap. The array is structured in levels,
    starting at level 0 with a single value, doubling the amount of values in
    each level. The final level contains the actual values, the level before
    it contains the pairwise minimum values. The level before that contains
    the pairwise minimum values of that level, etc. Take a look at this
    illustration:

    level: 0 11 2222 33333333 4444444444444444
    index: 0 12 3456 78901234 5678901234567890
                        1          2         3

     The actual values are stored in level 4. The minimum value of position 15
    and 16 is stored in position 7. min(17,18)->8, min(7,8)->3, min(3,4)->1.
    When adding a value, only the path to the top has to be updated, which
    takesO(log2(N)) time.

     The advantage of this implementation relative to more common
    implementations that swap values when pushing to the array is that data
    only needs to be swapped once when an element is removed. This means that
    keeping an array of references along with the values is very inexpensive.
    Th disadvantage is that if you pop the minimum value, the tree has to be
    traced from top to bottom and back. So if you only want values and no
    references, this implementation will probably be slower. If you need
    references (and maybe cross references to be kept up to date) this
    implementation will be faster.

    """

    ## Basic methods
    # The following lines are always "inlined", but documented here for
    # clarity:
    #
    # To calculate the start index of a certain level:
    # 2**l-1 # LevelStart
    # Note that in inner loops, this may also be represented as (1<<l)-1,
    # because code of the form x**y goes via the python pow operations and
    # can thus be a bit slower.
    #
    # To calculate the corresponding ABSOLUTE index at the next level:
    # i*2+1 # CalcNextAbs
    #
    # To calculate the corresponding ABSOLUTE index at the previous level:
    # (i-1)/2 # CalcPrevAbs
    #
    # To calculate the capacity at a certain level:
    # 2**l
    cdef readonly INDEX_T count
    cdef readonly LEVELS_T levels, min_levels
    cdef VALUE_T *_values
    cdef REFERENCE_T *_references
    cdef REFERENCE_T _popped_ref

    def __cinit__(self, INDEX_T initial_capacity=128, *args, **kws):
        # calc levels from the default capacity
        cdef LEVELS_T levels = 0
        while 2**levels < initial_capacity:
            levels += 1
        # set levels
        self.min_levels = self.levels = levels

        # we start with 0 values
        self.count = 0

        # allocate arrays
        cdef INDEX_T number = 2**self.levels
        self._values = <VALUE_T *>malloc(2 * number * sizeof(VALUE_T))
        self._references = <REFERENCE_T *>malloc(number * sizeof(REFERENCE_T))

    def __init__(self, INDEX_T initial_capacity=128):
        """__init__(initial_capacity=128)

        Class constructor.

        Takes an optional parameter 'initial_capacity' so that
        if the required heap capacity is known or can be estimated in advance,
        there will need to be fewer resize operations on the heap.

        """
        if self._values is NULL or self._references is NULL:
            raise MemoryError()
        self.reset()

    def reset(self):
        """reset()

        Reset the heap to default, empty state.

        """
        cdef INDEX_T number = 2**self.levels
        cdef INDEX_T i
        cdef VALUE_T *values = self._values
        for i in range(number * 2):
            values[i] = inf

    def __dealloc__(self):
        free(self._values)
        free(self._references)

    def __str__(self):
        s = ''
        for level in range(1, self.levels + 1):
            i0 = 2**level - 1  # LevelStart
            s += 'level %i: ' % level
            for i in range(i0, i0 + 2**level):
                s += '%g, ' % self._values[i]
            s = s[:-1] + '\n'
        return s

    ## C Maintenance methods

    cdef void _add_or_remove_level(self, LEVELS_T add_or_remove) nogil:
        # init indexing ints
        cdef INDEX_T i, i1, i2, n

        # new amount of levels
        cdef LEVELS_T new_levels = self.levels + add_or_remove

        # allocate new arrays
        cdef INDEX_T number = 2**new_levels
        cdef VALUE_T *values
        cdef REFERENCE_T *references
        values = <VALUE_T *>malloc(number * 2 * sizeof(VALUE_T))
        references = <REFERENCE_T *>malloc(number * sizeof(REFERENCE_T))
        if values is NULL or references is NULL:
            free(values)
            free(references)
            with gil:
                raise MemoryError()

        # init arrays
        for i in range(number * 2):
            values[i] = inf
        for i in range(number):
            references[i] = -1

        # copy data
        cdef VALUE_T *old_values = self._values
        cdef REFERENCE_T *old_references = self._references
        if self.count:
            i1 = 2**new_levels - 1  # LevelStart
            i2 = 2**self.levels - 1  # LevelStart
            n = index_min(2**new_levels, 2**self.levels)
            for i in range(n):
                values[i1+i] = old_values[i2+i]
            for i in range(n):
                references[i] = old_references[i]

        # make current
        free(self._values)
        free(self._references)
        self._values = values
        self._references = references

        # we need a full update
        self.levels = new_levels
        self._update()

    cdef void _update(self) nogil:
        """Update the full tree from the bottom up.

        This should be done after resizing.

        """
        # shorter name for values
        cdef VALUE_T *values = self._values

        # Note that i represents an absolute index here
        cdef INDEX_T i0, i, ii, n
        cdef LEVELS_T level

        # track tree
        for level in range(self.levels, 1, -1):
            i0 = (1 << level) - 1  # 2**level-1 = LevelStart
            n = i0 + 1  # 2**level
            for i in range(i0, i0+n, 2):
                ii = (i-1) // 2  # CalcPrevAbs
                if values[i] < values[i+1]:
                    values[ii] = values[i]
                else:
                    values[ii] = values[i+1]

    cdef void _update_one(self, INDEX_T i) nogil:
        """Update the tree for one value."""
        # shorter name for values
        cdef VALUE_T *values = self._values

        # make index uneven
        if i % 2 == 0:
            i -= 1

        # track tree
        cdef INDEX_T ii
        cdef LEVELS_T level
        for level in range(self.levels, 1, -1):
            ii = (i-1) // 2  # CalcPrevAbs

            # test
            if values[i] < values[i+1]:
                values[ii] = values[i]
            else:
                values[ii] = values[i+1]
            # next
            if ii % 2:
                i = ii
            else:
                i = ii - 1

    cdef void _remove(self, INDEX_T i1) nogil:
        """Remove a value from the heap. By index."""
        cdef LEVELS_T levels = self.levels
        cdef INDEX_T count = self.count
        # get indices
        cdef INDEX_T i0 = (1 << levels) - 1  # 2**self.levels - 1 # LevelStart
        cdef INDEX_T i2 = i0 + count - 1

        # get relative indices
        cdef INDEX_T r1 = i1 - i0
        cdef INDEX_T r2 = count - 1

        cdef VALUE_T *values = self._values
        cdef REFERENCE_T *references = self._references

        # swap with last
        values[i1] = values[i2]
        references[r1] = references[r2]

        # make last Null
        values[i2] = inf

        # update
        self.count -= 1
        count -= 1
        if (levels > self.min_levels) and (count < (1 << (levels-2))):
            self._add_or_remove_level(-1)
        else:
            self._update_one(i1)
            self._update_one(i2)

    ## C Public methods

    cdef INDEX_T push_fast(self, VALUE_T value, REFERENCE_T reference) nogil:
        """The c-method for fast pushing.

        Returns the index relative to the start of the last level in the heap.

        """
        # We need to resize if currently it just fits.
        cdef LEVELS_T levels = self.levels
        cdef INDEX_T count = self.count
        if count >= (1 << levels):  # 2**self.levels:
            self._add_or_remove_level(1)
            levels += 1

        # insert new value
        cdef INDEX_T i = ((1 << levels) - 1) + count  # LevelStart + n
        self._values[i] = value
        self._references[count] = reference

        # update
        self.count += 1
        self._update_one(i)

        # return
        return count

    cdef VALUE_T pop_fast(self) nogil:
        """The c-method for fast popping.

        Returns the minimum value. The reference is put in self._popped_ref.

        """
        # shorter name for values
        cdef VALUE_T *values = self._values

        # init index. start at 1 because we start in level 1
        cdef LEVELS_T level
        cdef INDEX_T i = 1
        cdef LEVELS_T levels = self.levels
        # search tree (using absolute indices)
        for level in range(1, levels):
            if values[i] <= values[i+1]:
                i = i * 2 + 1  # CalcNextAbs
            else:
                i = (i+1) * 2 + 1  # CalcNextAbs

        # select best one in last level
        if values[i] <= values[i+1]:
            i = i
        else:
            i += 1

        # get values
        cdef INDEX_T ir = i - ((1 << levels) - 1) # (2**self.levels-1)
                                                  # LevelStart
        cdef VALUE_T value = values[i]
        self._popped_ref = self._references[ir]

        # remove it
        if self.count:
            self._remove(i)

        # return
        return value

    ## Python Public methods (that do not need to be VERY fast)

    def push(self, VALUE_T value, REFERENCE_T reference=-1):
        """push(value, reference=-1)

        Append a value to the heap, with optional reference.

        Parameters
        ----------
        value : float
            Value to push onto the heap
        reference : int, optional
            Reference to associate with the given value.

        """
        self.push_fast(value, reference)

    def min_val(self):
        """min_val()

        Get the minimum value on the heap.

        Returns only the value, and does not remove it from the heap.

        """
        # shorter name for values
        cdef VALUE_T *values = self._values

        # select best one in last level
        if values[1] < values[2]:
            return values[1]
        else:
            return values[2]

    def values(self):
        """values()

        Get the values in the heap as a list.

        """
        cdef INDEX_T i0 = 2**self.levels - 1  # LevelStart
        return [self._values[i] for i in range(i0, i0+self.count)]

    def references(self):
        """references()

        Get the references in the heap as a list.

        """
        return [self._references[i] for i in range(self.count)]

    def pop(self):
        """pop()

        Get the minimum value and remove it from the list.

        Returns
        -------
        value : float
        reference : int
            If no reference was provided, -1 is returned here.

        Raises
        ------
        IndexError
            On attempt to pop from an empty heap

        """
        if self.count == 0:
            raise IndexError('pop from an empty heap')
        value = self.pop_fast()
        ref = self._popped_ref
        return value, ref


cdef class FastUpdateBinaryHeap(BinaryHeap):
    """FastUpdateBinaryHeap(initial_capacity=128, max_reference=None)

    Binary heap that allows the value of a reference to be updated quickly.

    This heap class keeps cross-references so that the value associated with a
    given reference can be quickly queried (O(1) time) or updated (O(log2(N))
    time). This is ideal for pathfinding algorithms that implement some
    variant of Dijkstra's algorithm.

    Parameters
    ----------
    initial_capacity : int
        Estimate of the size of the heap, if known in advance. (In any case,
        the heap will dynamically grow and shrink as required, though never
        below the `initial_capacity`.)

    max_reference : int, optional
        Largest reference value that might be pushed to the heap. (Pushing a
        larger value will result in an error.) If no value is provided,
        `1-initial_capacity` will be used. For the cross-reference index to
        work, all references must be in the range [0, max_reference];
        references pushed outside of that range will not be added to the heap.
        The cross-references are kept as a 1-d array of length
        `max_reference+1', so memory use of this heap is effectively
        O(max_reference)

    Attributes
    ----------
    count : int
        The number of values in the heap
    levels : int
        The number of levels in the binary heap (see Notes below). The values
        are stored in the last level, so 2**levels is the capacity of the
        heap before another resize is required.
    min_levels : int
        The minimum number of levels in the heap (relates to the
        `initial_capacity` parameter.)
    max_reference : int
        The provided or calculated maximum allowed reference value.

    Notes
    -----
    The cross-references map data[reference]->internalindex, such that the
    value corresponding to a given reference can be found efficiently. This
    can be queried with the value_of() method.

    A special method, push_if_lower() is provided that will update the heap if
    the given reference is not in the heap, or if it is and the provided value
    is lower than the current value in the heap. This is again useful for
    pathfinding algorithms.

    """

    cdef readonly REFERENCE_T max_reference
    cdef INDEX_T *_crossref
    cdef BOOL_T _invalid_ref
    cdef BOOL_T _pushed

    def __cinit__(self, INDEX_T initial_capacity=128, max_reference=None):
        if max_reference is None:
            max_reference = initial_capacity - 1
        self.max_reference = max_reference
        self._crossref = <INDEX_T *>malloc((self.max_reference + 1) *
                                           sizeof(INDEX_T))

    def __init__(self, INDEX_T initial_capacity=128, max_reference=None):
        """__init__(initial_capacity=128, max_reference=None)

        Class constructor.

        """
        if self._crossref is NULL:
            raise MemoryError()
        # below will call self.reset
        BinaryHeap.__init__(self, initial_capacity)

    def __dealloc__(self):
        free(self._crossref)

    def reset(self):
        """reset()

        Reset the heap to default, empty state.

        """
        BinaryHeap.reset(self)
        # set default values of crossrefs
        cdef INDEX_T i
        for i in range(self.max_reference + 1):
            self._crossref[i] = -1

    cdef void _remove(self, INDEX_T i1) nogil:
        """Remove a value from the heap. By index."""
        cdef LEVELS_T levels = self.levels
        cdef INDEX_T count = self.count

        # get indices
        cdef INDEX_T i0 = (1 << levels) - 1  # 2**self.levels - 1 # LevelStart
        cdef INDEX_T i2 = i0 + count - 1

        # get relative indices
        cdef INDEX_T r1 = i1 - i0
        cdef INDEX_T r2 = count - 1

        cdef VALUE_T *values = self._values
        cdef REFERENCE_T *references = self._references
        cdef INDEX_T *crossref = self._crossref

        # update cross reference
        crossref[references[r2]] = r1
        crossref[references[r1]] = -1  # disable removed item

        # swap with last
        values[i1] = values[i2]
        references[r1] = references[r2]

        # make last Null
        values[i2] = inf

        # update
        self.count -= 1
        count -= 1
        if (levels > self.min_levels) & (count < (1 << (levels-2))):
            self._add_or_remove_level(-1)
        else:
            self._update_one(i1)
            self._update_one(i2)

    cdef INDEX_T push_fast(self, VALUE_T value, REFERENCE_T reference) nogil:
        """The c method for fast pushing.

        If the reference is already present, will update its value, otherwise
        will append it.

        If -1 is returned, the provided reference was out-of-bounds and no
        value was pushed to the heap.

        """
        if not (0 <= reference <= self.max_reference):
            return -1

        # init variable to store the index-in-the-heap
        cdef INDEX_T i

        # Reference is the index in the array where MCP is applied to.
        # Find the index-in-the-heap using the crossref array.
        cdef INDEX_T ir = self._crossref[reference]

        if ir != -1:
            # update
            i = (1 << self.levels) - 1 + ir
            self._values[i] = value
            self._update_one(i)
            return ir

        # if not updated: append normally and store reference
        ir = BinaryHeap.push_fast(self, value, reference)
        self._crossref[reference] = ir
        return ir

    cdef INDEX_T push_if_lower_fast(self, VALUE_T value,
                                    REFERENCE_T reference) nogil:
        """If the reference is already present, will update its value ONLY if
        the new value is lower than the old one. If the reference is not
        present, this append it. If a value was appended, self._pushed is
        set to 1.

        If -1 is returned, the provided reference was out-of-bounds and no
        value was pushed to the heap.

        """
        if not (0 <= reference <= self.max_reference):
            return -1

        # init variable to store the index-in-the-heap
        cdef INDEX_T i

        # Reference is the index in the array where MCP is applied to.
        # Find the index-in-the-heap using the crossref array.
        cdef INDEX_T ir = self._crossref[reference]
        cdef VALUE_T *values = self._values
        self._pushed = 1
        if ir != -1:
            # update
            i = (1 << self.levels) - 1 + ir
            if values[i] > value:
                values[i] = value
                self._update_one(i)
            else:
                self._pushed = 0
            return ir

        # if not updated: append normally and store reference
        ir = BinaryHeap.push_fast(self, value, reference)
        self._crossref[reference] = ir
        return ir

    cdef VALUE_T value_of_fast(self, REFERENCE_T reference):
        """Return the value corresponding to the given reference.

        If inf is returned, the reference may be invalid: check the
        _invaild_ref field in this case.

        """
        if not (0 <= reference <= self.max_reference):
            self._invalid_ref = 1
            return inf

        # init variable to store the index-in-the-heap
        cdef INDEX_T i

        # Reference is the index in the array where MCP is applied to.
        # Find the index-in-the-heap using the crossref array.
        cdef INDEX_T ir = self._crossref[reference]
        self._invalid_ref = 0
        if ir == -1:
            self._invalid_ref = 1
            return inf
        i = (1 << self.levels) - 1 + ir
        return self._values[i]

    def push(self, double value, int reference):
        """push(value, reference)

        Append/update a value in the heap.

        Parameters
        ----------
        value : float
        reference : int
            If the reference is already present in the array, the value for
            that reference will be updated, otherwise the (value, reference)
            pair will be added to the heap.

        Raises
        ------
        ValueError
            On pushing a reference outside the range [0, max_reference].

        """
        if self.push_fast(value, reference) == -1:
            raise ValueError('reference outside of range [0, max_reference]')

    def push_if_lower(self, double value, int reference):
        """push_if_lower(value, reference)

        Append/update a value in the heap if the extant value is lower.

        If the reference is already in the heap, update only of the new value
        is lower than the current one. If the reference is not present, the
        value will always be pushed to the heap.

        Parameters
        ----------
        value : float
        reference : int
            If the reference is already present in the array, the value for
            that reference will be updated, otherwise the (value, reference)
            pair will be added to the heap.

        Returns
        -------
        pushed : bool
            True if an append/update occurred, False if otherwise.

        Raises
        ------
        ValueError
            On pushing a reference outside the range [0, max_reference].

        """
        if self.push_if_lower_fast(value, reference) == -1:
            raise ValueError('reference outside of range [0, max_reference]')
        return self._pushed == 1

    def value_of(self, int reference):
        """value_of(reference)

        Get the value corresponding to a given reference.

        Parameters
        ----------
        reference : int
            A reference already pushed to the heap.

        Returns
        -------
        value : float

        Raises
        ------
        ValueError
            On querying a reference outside the range [0, max_reference], or
            not already pushed to the heap.

        """
        value = self.value_of_fast(reference)
        if self._invalid_ref:
            raise ValueError('invalid reference')
        return value

    def cross_references(self):
        """Get the cross references in the heap as a list."""
        return [self._crossref[i] for i in range(self.max_reference + 1)]


# ***** Centerpath construction *****

# The reason that heap implementation is in the same file is because I cannt make cimport
# work with bazel. I keep getting the following error:
#
#   centerpath_construction.pyx:5:8: 'tenxnet/vision/post_analysis/heap.pxd' not found
#
# even though heap.pxd is included as the pxd_hdrs of centerpath_construction

"""The following functions construct direction map from label map"""

import numpy as np
import cython
# TODO: 
# cimport tenxnet.vision.post_analysis.heap as heap
cimport numpy as cnp
cnp.import_array()

DTYPE = np.intp
ctypedef Py_ssize_t DTYPE_t

cdef struct s_shpinfo

ctypedef s_shpinfo shape_info


cdef struct s_shpinfo:
    DTYPE_t row
    DTYPE_t col
    DTYPE_t num_elems # number of total elements
    DTYPE_t eight_neighbor[9]


cdef void get_shape_info(array_shape, shape_info *shapeinfo) except *:
    """Calculate shape information and store."""
    shapeinfo.row = array_shape[0]
    shapeinfo.col = array_shape[1]
    shapeinfo.num_elems = shapeinfo.row * shapeinfo.col
    shapeinfo.eight_neighbor[0] = - shapeinfo.col - 1
    shapeinfo.eight_neighbor[1] = - shapeinfo.col
    shapeinfo.eight_neighbor[2] = - shapeinfo.col + 1
    shapeinfo.eight_neighbor[3] = -1
    shapeinfo.eight_neighbor[4] = 0
    shapeinfo.eight_neighbor[5] = 1
    shapeinfo.eight_neighbor[6] = shapeinfo.col - 1
    shapeinfo.eight_neighbor[7] = shapeinfo.col
    shapeinfo.eight_neighbor[8] = shapeinfo.col + 1


@cython.cdivision(True)
cdef bint within_bound(
    DTYPE_t center,
    DTYPE_t delta,
    DTYPE_t n_row,
    DTYPE_t n_col,
    DTYPE_t radius
):
    cdef DTYPE_t col, row
    cdef DTYPE_t diameter = radius * 2 + 1
    cdef bint inbound = 1
    col = center % n_col - radius + delta % diameter
    row = center // n_col - radius + delta // diameter
    if col < 0 or col >= n_col or row < 0 or row >= n_row:
        inbound = 0
    return inbound


cdef void _multi_target_direction(
    DTYPE_t *label_arr_p,
    DTYPE_t *direction_arr_p,
    DTYPE_t *temp_direct_arr_p,
    VALUE_T *neighbor_cost_p,
    DTYPE_t *node_reached_arr_p,
    DTYPE_t *target_arr_p,
    shape_info *shapeinfo,
) except *:
    """Direction from every labeled element to the closest target.

    There are possibly multiple targets. Each labeled element will
    try to find the closest one.
    """

    cdef DTYPE_t i, label

    cdef BinaryHeap pqueue = FastUpdateBinaryHeap(
        initial_capacity=128,
        max_reference=shapeinfo.num_elems
    )
    for i in range(shapeinfo.num_elems):
        label = label_arr_p[i]
        # labeled pixel but the direction is not yet found
        if label != 0 and target_arr_p[i] != 1:
            pqueue.reset()
            _shortest_path(
                i,
                target_arr_p,
                label_arr_p,
                direction_arr_p,
                temp_direct_arr_p,
                node_reached_arr_p,
                i,
                neighbor_cost_p,
                pqueue,
                shapeinfo
            )


cdef void _shortest_path(
    DTYPE_t start,
    DTYPE_t *target_arr_p,
    DTYPE_t *label_arr_p,
    DTYPE_t *direction_arr_p,
    DTYPE_t *temp_direct_arr_p,
    DTYPE_t *reached_map_p,
    DTYPE_t mark_reached,
    VALUE_T *neighbor_cost_p,
    FastUpdateBinaryHeap pqueue,
    shape_info *shapeinfo,
) except *:
    # print("run node: ", start, direction_arr_p[start])
    # this node is a target
    """Shortest path from a starting point to multiple target.

    start: starting index
    target_arr_p: pointer to the target array. in the target array,
        1 means it's a target, 0 means it's not a target.
    label_arr_p:
    direction_arr_p: array to store direction that leads to the target.
    temp_direct_arr_p: save the direction when looking for targets. used for tracing
        back.
    reached_map_p: 
    """
    # if this point is a target, stay put and return
    if target_arr_p[start] == 1:
        direction_arr_p[start] = 4
        return

    cdef DTYPE_t i, n, label, index, neighbor, previous_index
    cdef VALUE_T cost, new_cost

    label = label_arr_p[start]
    pqueue.push_fast(0, start)

    # for loop to serve as while loop to prevent potential inf loop
    for i in range(shapeinfo.num_elems):
        if pqueue.count == 0:
            break
        cost = pqueue.pop_fast()
        index = pqueue._popped_ref
        reached_map_p[index] = mark_reached
        # one of the target is found
        if target_arr_p[index] == 1:
            # a target is found, trace back to the start and assign correct direction along
            # the shortest path.
            previous_index = index + shapeinfo.eight_neighbor[temp_direct_arr_p[index]]
            while (previous_index != start):
                index = previous_index
                previous_index = index + shapeinfo.eight_neighbor[temp_direct_arr_p[index]]
            
            direction_arr_p[previous_index] = 8 - temp_direct_arr_p[index]
            break
        for n in range(9):
            if n != 4 and within_bound(index, n, shapeinfo.row, shapeinfo.col, 1):
                neighbor = index + shapeinfo.eight_neighbor[n]
                if label == label_arr_p[neighbor] and reached_map_p[neighbor] != mark_reached:
                    new_cost = cost + neighbor_cost_p[n]
                    # there is a numerical accuracy problem here. this is to break the 
                    # path with equal cost
                    if pqueue.value_of_fast(neighbor) - new_cost > 0.001:
                        # this is a trick: revert the direction just use 8
                        # minus the current direction
                        # print("push neighbor ", neighbor)
                        pqueue.push_if_lower_fast(new_cost, neighbor)
                        # use temp_direct_arr to remember the way back
                        temp_direct_arr_p[neighbor] = 8 - n


cdef VALUE_T _shortest_path_tree(
    DTYPE_t start,
    DTYPE_t *label_arr_p,
    DTYPE_t *direction_arr_p,
    DTYPE_t *reached_map_p,
    DTYPE_t mark_reached,
    VALUE_T *neighbor_cost_p,
    FastUpdateBinaryHeap pqueue,
    shape_info *shapeinfo,
):
    cdef DTYPE_t i, n, label, index, neighbor
    cdef VALUE_T cost, new_cost, total_cost

    label = label_arr_p[start]
    total_cost = 0
    pqueue.push_fast(0, start)
    direction_arr_p[start] = 4
    reached_map_p[start] = mark_reached

    for i in range(shapeinfo.num_elems):
        if pqueue.count == 0:
            break
        cost = pqueue.pop_fast()
        index = pqueue._popped_ref
        # print("current queue: ", pqueue.values(), pqueue.references())
        # print("process node: ", index)
        # when a node is poped, the shortest path to that node has been found.
        reached_map_p[index] = mark_reached
        total_cost += cost
        # n represent different neighbor directions
        for n in range(9):
            if n != 4 and within_bound(index, n, shapeinfo.row, shapeinfo.col, 1):
                neighbor = index + shapeinfo.eight_neighbor[n]
                # print("process neighbor: ", neighbor)
                if label == label_arr_p[neighbor] and reached_map_p[neighbor] != mark_reached:
                    new_cost = cost + neighbor_cost_p[n]
                    # there is a numerical accuracy problem here to break the 
                    # path with equal cost
                    if pqueue.value_of_fast(neighbor) - new_cost > 0.1:
                        # this is a trick: revert the direction just use 8
                        # minus the current direction
                        pqueue.push_if_lower_fast(new_cost, neighbor)
                        direction_arr_p[neighbor] = 8 - n
    return total_cost


cdef void _single_target_direction(
    DTYPE_t *label_arr_p,
    DTYPE_t *direction_arr_p,
    DTYPE_t *center_idx_p,
    VALUE_T *total_cost_arr_p,
    DTYPE_t max_label,
    VALUE_T *neighbor_cost_p,
    DTYPE_t *node_reached_arr_p,
    DTYPE_t *target_arr_p,
    shape_info *shapeinfo,
) except *:
    cdef DTYPE_t i, label, center
    cdef VALUE_T total_cost

    cdef BinaryHeap pqueue = FastUpdateBinaryHeap(
        initial_capacity=2048,
        max_reference=shapeinfo.num_elems
    )
    for i in range(shapeinfo.num_elems):
        label = label_arr_p[i]
        if label > max_label:
            raise ValueError("found label bigger than the maximum label")
        if label != 0 and target_arr_p[i] > 0:
            pqueue.reset()
            total_cost = _shortest_path_tree(
                i,
                label_arr_p,
                direction_arr_p,
                node_reached_arr_p,
                i,
                neighbor_cost_p,
                pqueue,
                shapeinfo
            )
            if total_cost < total_cost_arr_p[label]:
                total_cost_arr_p[label] = total_cost
                center_idx_p[label] = i
    
    for i in range(1, max_label+1):
        center = center_idx_p[i]
        if center == -1:
            # warn(f"No region labeled as {i}")
            continue
        _shortest_path_tree(
            center,
            label_arr_p,
            direction_arr_p,
            node_reached_arr_p,
            -99,
            neighbor_cost_p,
            pqueue,
            shapeinfo
        )


def label_to_direction(label_map, max_label, candidate_target_map=None):
    """Generate direction arr for each labeled pixel.

    The target point of each label region is defined as the pixel within the labeled region,
    that has the smallest sum of the shortest paths from that pixel to every other pixels
    with that label.

    Background label is assumed to be 0. The notation of direction follows the same protocols as
    in this scrip.
    """

    shape = label_map.shape
    cdef shape_info shapeinfo

    label_arr = label_map.flatten(order="C").astype(DTYPE)
    center_idx = np.ones(max_label + 1, dtype=DTYPE) * -1
    direction_arr = np.ones(label_map.size, dtype=DTYPE) * -1
    total_cost_arr = np.ones(max_label + 1, dtype=np.double) * np.finfo(np.double).max
    neighbor_cost = np.array([1.4, 1.0, 1.4, 1.0, 0, 1.0, 1.4, 1.0, 1.4], dtype=np.double)
    node_reached_arr = np.ones(label_map.size, dtype=DTYPE) * -1

    cdef DTYPE_t *label_arr_p = <DTYPE_t*>cnp.PyArray_DATA(label_arr)
    cdef DTYPE_t *center_idx_p = <DTYPE_t*>cnp.PyArray_DATA(center_idx)
    cdef DTYPE_t *direction_arr_p = <DTYPE_t*>cnp.PyArray_DATA(direction_arr)
    cdef VALUE_T *total_cost_arr_p = <VALUE_T*>cnp.PyArray_DATA(total_cost_arr)
    cdef VALUE_T *neighbor_cost_p = <VALUE_T*>cnp.PyArray_DATA(neighbor_cost)
    cdef DTYPE_t *node_reached_arr_p = <DTYPE_t*>cnp.PyArray_DATA(node_reached_arr)
    cdef DTYPE_t *target_arr_p

    if candidate_target_map is None:
        target_arr_p = label_arr_p
    else:
        target_arr = candidate_target_map.flatten(order="C").astype(DTYPE)
        target_arr_p = <DTYPE_t*>cnp.PyArray_DATA(target_arr)

    get_shape_info(shape, &shapeinfo)
    
    _single_target_direction(
        label_arr_p,
        direction_arr_p,
        center_idx_p,
        total_cost_arr_p,
        max_label,
        neighbor_cost_p,
        node_reached_arr_p,
        target_arr_p,
        &shapeinfo
    )

    return direction_arr.reshape(shape)


def label_to_ridge_direction(label_map, target_map):
    shape = label_map.shape
    cdef shape_info shapeinfo

    label_arr = label_map.flatten(order="C").astype(DTYPE)
    target_arr = target_map.flatten(order="C").astype(DTYPE)
    direction_arr = np.ones(label_map.size, dtype=DTYPE) * -1
    temp_direction_arr = np.ones(label_map.size, dtype=DTYPE) * -1
    neighbor_cost = np.array([1.4, 1.0, 1.4, 1.0, 0, 1.0, 1.4, 1.0, 1.4], dtype=np.double)
    node_reached_arr = np.ones(label_map.size, dtype=DTYPE) * -1

    cdef DTYPE_t *label_arr_p = <DTYPE_t*>cnp.PyArray_DATA(label_arr)
    cdef DTYPE_t *target_arr_p = <DTYPE_t*>cnp.PyArray_DATA(target_arr)
    cdef DTYPE_t *direction_arr_p = <DTYPE_t*>cnp.PyArray_DATA(direction_arr)
    cdef DTYPE_t *temp_direction_arr_p = <DTYPE_t*>cnp.PyArray_DATA(temp_direction_arr)
    cdef VALUE_T *neighbor_cost_p = <VALUE_T*>cnp.PyArray_DATA(neighbor_cost)
    cdef DTYPE_t *node_reached_arr_p = <DTYPE_t*>cnp.PyArray_DATA(node_reached_arr)

    get_shape_info(shape, &shapeinfo)
    _multi_target_direction(
        label_arr_p,
        direction_arr_p,
        temp_direction_arr_p,
        neighbor_cost_p,
        node_reached_arr_p,
        target_arr_p,
        &shapeinfo,
    )
    return direction_arr.reshape(shape)


def test_shortest_path_tree(label_map, start):
    shape = label_map.shape
    cdef shape_info shapeinfo

    label_arr = label_map.flatten(order="C").astype(DTYPE)
    direction_arr = np.ones(label_map.size, dtype=DTYPE) * -1
    neighbor_cost = np.array([1.4, 1.0, 1.4, 1.0, 0, 1.0, 1.4, 1.0, 1.4], dtype=np.double)
    node_reached_arr = np.ones(label_map.size, dtype=DTYPE) * -1

    cdef DTYPE_t *label_arr_p = <DTYPE_t*>cnp.PyArray_DATA(label_arr)
    cdef DTYPE_t *direction_arr_p = <DTYPE_t*>cnp.PyArray_DATA(direction_arr)
    cdef VALUE_T *neighbor_cost_p = <VALUE_T*>cnp.PyArray_DATA(neighbor_cost)
    cdef DTYPE_t *node_reached_arr_p = <DTYPE_t*>cnp.PyArray_DATA(node_reached_arr)

    cdef FastUpdateBinaryHeap pqueue = FastUpdateBinaryHeap(initial_capacity=128, max_reference=label_map.size)
    

    get_shape_info(shape, &shapeinfo)
    _shortest_path_tree(start, label_arr_p, direction_arr_p, node_reached_arr_p, 1, neighbor_cost_p, pqueue, &shapeinfo)

    return direction_arr, node_reached_arr


def test_shortest_path(label_map, start, target_map):
    shape = label_map.shape
    cdef shape_info shapeinfo

    label_arr = label_map.flatten(order="C").astype(DTYPE)
    target_arr = target_map.flatten(order="C").astype(DTYPE)
    direction_arr = np.ones(label_map.size, dtype=DTYPE) * -1
    temp_direction_arr = np.ones(label_map.size, dtype=DTYPE) * -1
    neighbor_cost = np.array([1.4, 1.0, 1.4, 1.0, 0, 1.0, 1.4, 1.0, 1.4], dtype=np.double)
    node_reached_arr = np.ones(label_map.size, dtype=DTYPE) * -1

    cdef DTYPE_t *label_arr_p = <DTYPE_t*>cnp.PyArray_DATA(label_arr)
    cdef DTYPE_t *target_arr_p = <DTYPE_t*>cnp.PyArray_DATA(target_arr)
    cdef DTYPE_t *direction_arr_p = <DTYPE_t*>cnp.PyArray_DATA(direction_arr)
    cdef DTYPE_t *temp_direction_arr_p = <DTYPE_t*>cnp.PyArray_DATA(temp_direction_arr)
    cdef VALUE_T *neighbor_cost_p = <VALUE_T*>cnp.PyArray_DATA(neighbor_cost)
    cdef DTYPE_t *node_reached_arr_p = <DTYPE_t*>cnp.PyArray_DATA(node_reached_arr)

    cdef FastUpdateBinaryHeap pqueue = FastUpdateBinaryHeap(initial_capacity=128, max_reference=label_map.size)

    get_shape_info(shape, &shapeinfo)
    _shortest_path(
        start,
        target_arr_p,
        label_arr_p,
        direction_arr_p,
        temp_direction_arr_p,
        node_reached_arr_p,
        1,
        neighbor_cost_p,
        pqueue, 
        &shapeinfo
    )

    return direction_arr, node_reached_arr
