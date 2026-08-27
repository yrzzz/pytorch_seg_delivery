# Functions for path representation of instance segmentation
import numpy as np
from warnings import warn

import cython
cimport numpy as cnp
cnp.import_array()

DTYPE = np.int32
ctypedef cnp.int32_t DTYPE_t

cdef DTYPE_t BG_NODE_NULL = -999
cdef DTYPE_t NO_COLOR = -99

cdef struct s_shpinfo

ctypedef s_shpinfo shape_info


cdef struct s_shpinfo:
    DTYPE_t row
    DTYPE_t col
    DTYPE_t num_elems # number of total elements
    DTYPE_t eight_neighbor[9]


cdef void get_shape_info(DTYPE_t height, DTYPE_t width, shape_info *shapeinfo) nogil:
    """Calculate shape information and store."""
    shapeinfo.row = height
    shapeinfo.col = width
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
) noexcept nogil:
    cdef DTYPE_t col, row
    cdef DTYPE_t diameter = radius * 2 + 1
    cdef bint inbound = 1
    col = center % n_col - radius + delta % diameter
    row = center // n_col - radius + delta // diameter
    if col < 0 or col >= n_col or row < 0 or row >= n_row:
        inbound = 0
    return inbound


@cython.cdivision(True)
cdef void get_window_search_idx(
    DTYPE_t[:] plateau_search_idx_p,
    DTYPE_t radius,
    DTYPE_t num_search,
    shape_info *shapeinfo,
) nogil:
    """Calculate index to search within a window

    radius: radius of the search window. The overall size of the
        window is (radius * 2 + 1) x (radius * 2 + 1)
    """
    cdef DTYPE_t i
    cdef DTYPE_t col = shapeinfo.col
    cdef DTYPE_t diameter = radius * 2 + 1
    for i in range(num_search):
        plateau_search_idx_p[i] = - radius * col - radius + (i // diameter) * col + i % diameter


@cython.boundscheck(False)
cdef DTYPE_t find_root(DTYPE_t[:] forest_p, DTYPE_t n) noexcept nogil:
    # NOTE: this lead to infinite loop if cycle could exist in the forest. Need to be very careful
    # to make sure NO Cycle in the forest !! 
    cdef DTYPE_t root = n
    while (forest_p[root] != root):
        root = forest_p[root]
        if root == BG_NODE_NULL:
            break
    return root


@cython.boundscheck(False)
cdef inline void set_root(DTYPE_t[:] forest_p, DTYPE_t n, DTYPE_t root) noexcept nogil:
    """Set the root of n to root.

    At the same time, all the nodes along the same path are
    set to the same root to achieve path compression.
    """
    cdef DTYPE_t j
    while (forest_p[n] != n):
        j = forest_p[n]
        forest_p[n] = root
        n = j
    forest_p[n] = root


@cython.boundscheck(False)
cdef inline void join_trees(DTYPE_t[:] forest_p, DTYPE_t n, DTYPE_t m, DTYPE_t[:] support) noexcept nogil:
    """Join two trees containing nodes n and m.

    A delibrate decision is made that always choose to use the smaller
    index as the shared root. A comparable decision is in union-find, use
    rank to choose which tree to merge to the other.

    By making this decision: always use the smaller index, in find_root
    we can also use `forest_p[root] > root` as the condition in while loop.
    """
    cdef DTYPE_t root
    cdef DTYPE_t root_m

    if n == m:
        return
    
    root = find_root(forest_p, n)
    root_m = find_root(forest_p, m)

    if root == root_m:
        return

    if (root > root_m):
        # merge root to root_m
        support[root_m] = support[root_m] + support[root]
        support[root] = 0 
        root = root_m
    else:
        # merge root_m to root
        support[root] = support[root] + support[root_m]
        support[root_m] = 0

    set_root(forest_p, n, root)
    set_root(forest_p, m, root)


@cython.boundscheck(False)
cdef void find_parent(
    DTYPE_t[:] direction_p,
    DTYPE_t[:] forest_p,
    shape_info *shapeinfo,
) noexcept nogil:
    """Locate parent of each node.

    This function serves as the first raster scan of the image,
    similar to the watershed algorithm. 
    """
    cdef DTYPE_t i, direct, gap, n

    for i in range(shapeinfo.num_elems):
        direct = direction_p[i]
        if direct == BG_NODE_NULL:
            forest_p[i] = BG_NODE_NULL
        elif direct != 4:
            if within_bound(i, direct, shapeinfo.row, shapeinfo.col, 1):
                forest_p[i] = i + shapeinfo.eight_neighbor[direct]
            else:
                # the direction leads to outside of the image, set as background.
                forest_p[i] = BG_NODE_NULL


@cython.boundscheck(False)
cdef DTYPE_t follow_to_root(
    DTYPE_t[:] forest_p,
    DTYPE_t[:] node_color_p,
    DTYPE_t[:] support_p,
    DTYPE_t[:] direction_p,
    DTYPE_t[:] stationary_p,
    DTYPE_t current_idx,
) noexcept nogil:
    """Find root of each node given a direction map.

    Similar to the "find" operation in union-find with path compression
    using recursion. However cannot use the simple find_root function defined
    here to do this because we need to deal with the following senarios:

    * the direction leads to a background node.
        - solution: mark all nodes leading to BG the BG nodes.
    * cycle exists in the path.
        - solution: use color to detect cycle and break it by setting
                    a new root.
    
    Also in this process a support count is maintained for each root.

    A biproduct of setting a new root when cycle is detected is that after this,
    we need to merge roots that are close enough to each other.
    """
    cdef DTYPE_t root
    cdef DTYPE_t parent = forest_p[current_idx]
    if parent == current_idx:
        # first call of the function starts with a root. Accumulate support only for
        # strong stationary px.
        # This is alwyas the first call of the functino rather than part of the recursion
        # because this function exam whether parent is a root in the last else. So it never
        # step in this if block during recursion.
        if stationary_p[parent] == 2:
            support_p[parent] += 1
        return parent
    if parent == BG_NODE_NULL:
        # background node. also make any nodes lead to this background node background.
        return parent
    if node_color_p[current_idx] == NO_COLOR:
        # start the coloring process
        node_color_p[current_idx] = current_idx
    if node_color_p[parent] == node_color_p[current_idx]:
        # cycle detected, set the touching node as a new root, set the touching node as a 
        # strong stationary point as well
        forest_p[parent] = parent
        stationary_p[parent] = 2
        if direction_p[current_idx] != 4:
            # current_idx is not a stay pixel and contribute to the support of its root
            # which is the parent
            support_p[parent] += 1
        return parent
    elif forest_p[parent] != parent:
        # parent is not a root and no cycle detected
        if node_color_p[parent] == NO_COLOR:
            # if the parent is not colored due to a previous search, color it.
            node_color_p[parent] = node_color_p[current_idx]
        # recursive call to achieve path compression
        root = follow_to_root(forest_p, node_color_p, support_p, direction_p, stationary_p, parent)
        forest_p[current_idx] = root
        if direction_p[current_idx] != 4 and root != BG_NODE_NULL:
            # current_idx is not a stay pixel and contribute to the support of the root
            support_p[root] += 1
        return root
    else:
        # BUG: cannot tell whether should accumulate count or not with this logic
        # this cause the support to be undercounted by up to 8

        # parent is a root, i.e., forest_p[parent] == parent
        if stationary_p[parent] != 2:
            stationary_p[parent] = 2
        return parent


@cython.boundscheck(False)
cdef void find_center(
    DTYPE_t[:] forest_p,
    DTYPE_t[:] node_color_p,
    DTYPE_t[:] support_p,
    DTYPE_t[:] direction_p,
    DTYPE_t[:] stationary_p,
    shape_info *shapeinfo,
    DTYPE_t[:] plateau_search_idx_p,
    DTYPE_t plateau_search_radius,
    DTYPE_t merge_threshold,
) noexcept nogil:
    """Find center of each node."""
    cdef DTYPE_t i, j, neighbor, neigh_root, root
    cdef DTYPE_t edge = 2 * plateau_search_radius + 1
    cdef DTYPE_t num_search = edge * edge // 2  # only top half

    # First raster scan after which each node points directly to its
    # tentative root after following the directions.
    for i in range(shapeinfo.num_elems):
        if forest_p[i] != BG_NODE_NULL:
            follow_to_root(forest_p, node_color_p, support_p, direction_p, stationary_p, i)

    # After the second and third raster scan, each node is NOT guaranteed to point directly to the root.

    # Second raster scan to merge nearby strong stationary px with either strong stationary px or
    # soft stationary px with stay direction
    for i in range(shapeinfo.num_elems):
        if stationary_p[i] == 0:
            continue
        root = find_root(forest_p, i)
        if root == BG_NODE_NULL:
            continue
        # do not merge soft stationary px with direction hint
        if stationary_p[i] == 1 and direction_p[i] != 4:
            continue
        for j in range(num_search):
            if not within_bound(i, j, shapeinfo.row, shapeinfo.col, plateau_search_radius):
                continue
            neighbor = i + plateau_search_idx_p[j]
            if stationary_p[neighbor] == 0:
                continue
            neigh_root = find_root(forest_p, neighbor)
            if neigh_root == BG_NODE_NULL:
                continue
            # do not merge two soft stationary px
            if stationary_p[i] == 1 and stationary_p[neighbor] == 1:
                continue
            # do not merge soft stationary px with direction hint
            if stationary_p[neighbor] == 1 and direction_p[neighbor] != 4:
                continue

            join_trees(forest_p, i, neighbor, support_p)

    # Third raster scan to merge nearby strong stationary px and soft stationary px with non-stay direction
    # if any of them have support less than threshold
    for i in range(shapeinfo.num_elems):
        if stationary_p[i] == 0:
            continue
        # do not merge soft stationary px with no direction hint
        if stationary_p[i] == 1 and direction_p[i] == 4:
            continue
        root = find_root(forest_p, i)
        if root == BG_NODE_NULL:
            continue
        for j in range(num_search):
            if not within_bound(i, j, shapeinfo.row, shapeinfo.col, plateau_search_radius):
                continue
            neighbor = i + plateau_search_idx_p[j]
            if stationary_p[neighbor] == 0:
                continue
            neigh_root = find_root(forest_p, neighbor)
            if neigh_root == BG_NODE_NULL:
                continue
            # do not merge two strong stationary px or two soft stationary px
            if stationary_p[i] == stationary_p[neighbor]:
                continue
            # do not merge soft stationary px with no direction hint
            if stationary_p[neighbor] == 1 and direction_p[neighbor] == 4:
                continue
            # do not merge if both supports are larger than threshold (two big objects)
            if support_p[root] > merge_threshold and support_p[neigh_root] > merge_threshold:
                continue

            join_trees(forest_p, i, neighbor, support_p)


cdef DTYPE_t resolve_labels(
    DTYPE_t[:] forest_p,
    DTYPE_t[:] label_p,
    DTYPE_t[:] support_p,
    DTYPE_t[:] support_count_p,
    DTYPE_t min_support,
    DTYPE_t[:] area_p,
    shape_info *shapeinfo,
) noexcept nogil:
    """Assign final labels.
    
    If support of a label is below the min_support, such label and
    corresponding pixels are not included in the final label, i.e.,
    changed to background.
    """
    cdef DTYPE_t i
    cdef DTYPE_t counter = 1
    cdef DTYPE_t root
    for i in range(shapeinfo.num_elems):
        if forest_p[i] != BG_NODE_NULL:
            root = find_root(forest_p, i)
            # root should never be BG_NODE_NULL but just be safe
            if root != BG_NODE_NULL and support_p[root] > min_support:
                if label_p[root] == 0:
                    label_p[root] = counter
                    support_count_p[counter] = support_p[root]
                    counter += 1
                label_p[i] = label_p[root]
                area_p[label_p[root]] += 1
    return counter


# TODO (dongyao): make background_val a real arg of the function
def direction_to_label(
    cnp.ndarray[DTYPE_t, ndim=1] direction,
    cnp.ndarray[DTYPE_t, ndim=1] stationary_px, 
    DTYPE_t height,
    DTYPE_t width,
    DTYPE_t background_val,
    DTYPE_t min_support = 30,
    DTYPE_t merge_threshold = 30,
    DTYPE_t merge_center_radius = 2
):
    """Generate label map with directions.

    Integer to represent 
              x
       -------->
      | 
      |  0 1 2 
      |  3 4 5 
      |  6 7 8
      |
    y V

    Args:
        direction_map: flattened 1D array of the direction of each pixel. Refer to the map above
            for the meaning of each integer.
        stationary_px: mark whether each pixel is 0: non-stationary pixel; 1: soft-stationary pixel;
            2: strong stationary pixel. Nearby strong stationary pixels are always merged. soft-stationary
            pixels are merged if it meets certain condition. Refer to second and third raster scan of find_center
        height: original image height
        width: original image width
        background_val: value represent background in the direction map.
        min_support: instance with support less than this threshold is not included.
        merge_threshold: support threshold to determine whether two objects can be merged based on soft-stationary pixel
            with direction hint
        merge_center_radius: radius to search for valid pixel to merge.
    """
    if direction.size != (height * width):
        raise ValueError(f"The size of direction array {direction.size} do not match height={height} and width={width}")
    if direction.size != stationary_px.size:
        raise ValueError("Size of direction and stationary_px do not match")
    if background_val != -999:
        raise ValueError("Background value has to be -999")

    cdef shape_info shapeinfo
    cdef DTYPE_t num_search = (2 * merge_center_radius + 1)**2 // 2

    cdef cnp.ndarray[DTYPE_t, ndim=1] forest = np.arange(direction.size, dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] support = np.zeros(direction.size, dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] support_count = np.zeros(direction.size, dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] area = np.zeros(direction.size, dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] plateau_search_idx = np.zeros(num_search, dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] node_color = np.ones(direction.size, dtype=DTYPE) * NO_COLOR
    cdef cnp.ndarray[DTYPE_t, ndim=1] label = np.zeros(direction.size, dtype=DTYPE)

    cdef DTYPE_t [:] plateau_search_idx_view = plateau_search_idx
    cdef DTYPE_t [:] direction_view = direction
    cdef DTYPE_t [:] stationary_px_view = stationary_px
    cdef DTYPE_t [:] forest_view = forest
    cdef DTYPE_t [:] node_color_view = node_color
    cdef DTYPE_t [:] support_view = support
    cdef DTYPE_t [:] support_count_view = support_count
    cdef DTYPE_t [:] label_view = label
    cdef DTYPE_t [:] area_view = area

    with nogil:
        get_shape_info(height, width, &shapeinfo)
        get_window_search_idx(plateau_search_idx_view, merge_center_radius, num_search, &shapeinfo)
        find_parent(direction_view, forest_view, &shapeinfo)
        find_center(forest_view, node_color_view, support_view, direction_view, stationary_px_view, &shapeinfo, plateau_search_idx_view, merge_center_radius, merge_threshold)
        counter = resolve_labels(forest_view, label_view, support_view, support_count_view, min_support, area_view, &shapeinfo)
    return label, support_count[:counter], area[:counter]


@cython.boundscheck(False)
cdef (DTYPE_t, DTYPE_t) follow_to_distance(
    DTYPE_t[:] forest_p,
    DTYPE_t[:] node_color_p,
    DTYPE_t[:] distance_p,
    DTYPE_t current_idx,
    DTYPE_t current_color,
) nogil:
    cdef DTYPE_t root, parent_to_root_dist, dist_to_root
    cdef DTYPE_t parent = forest_p[current_idx]
    if parent == current_idx:
        # first call of the function starts with a root. Distance should be 0
        # No need to change distance_p since distance_p is initialized with 0
        return parent, 0

    if node_color_p[current_idx] == NO_COLOR:
        # node not visited before; mark the node as visited by this current path
        node_color_p[current_idx] = current_color
    elif node_color_p[current_idx] == current_color:
        # cycle detected, the current path revisit the current node
        # set the current node as root
        forest_p[current_idx] = current_idx
        return current_idx, 0
    else:
        # node visited before by a different path; take the existing result
        dist_to_root = distance_p[current_idx]
        return parent, dist_to_root

    if parent == BG_NODE_NULL:
        # background node. also make any nodes lead to this background node background.
        distance_p[current_idx] = 1
        return parent, 1
    elif forest_p[parent] != parent:
        # parent is not a root
        root, parent_to_root_dist = follow_to_distance(forest_p, node_color_p, distance_p, parent, current_color)
        if root != current_idx:
            dist_to_root = parent_to_root_dist + 1
        else:
            # when cycle exist, the root could be visited again
            dist_to_root = 0
        forest_p[current_idx] = root
        distance_p[current_idx] = dist_to_root
        return root, dist_to_root
    else:
        # parent is a root, i.e., forest_p[parent] == parent
        distance_p[current_idx] = 1
        return parent, 1


cdef void find_distance(
    DTYPE_t[:] forest_p,
    DTYPE_t[:] node_color_p,
    DTYPE_t[:] distance_p,
    shape_info *shapeinfo,
) nogil:
    """Find distance of each node to root."""
    cdef DTYPE_t i
    for i in range(shapeinfo.num_elems):
        if forest_p[i] != BG_NODE_NULL:
            follow_to_distance(forest_p, node_color_p, distance_p, i, i)

def direction_to_distance(
    cnp.ndarray[DTYPE_t, ndim=1] direction,
    DTYPE_t height,
    DTYPE_t width,
    DTYPE_t background_val,
):
    if direction.size != (height * width):
        raise ValueError(f"The size of direction array {direction.size} do not match height={height} and width={width}")
    if background_val != -999:
        raise ValueError("Background value has to be -999")

    cdef shape_info shapeinfo

    cdef cnp.ndarray[DTYPE_t, ndim=1] forest = np.arange(direction.size, dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] distance = np.zeros(direction.size, dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] node_color = np.ones(direction.size, dtype=DTYPE) * NO_COLOR

    cdef DTYPE_t [:] direction_view = direction
    cdef DTYPE_t [:] forest_view = forest
    cdef DTYPE_t [:] node_color_view = node_color
    cdef DTYPE_t [:] distance_view = distance

    with nogil:
        get_shape_info(height, width, &shapeinfo)
        find_parent(direction_view, forest_view, &shapeinfo)
        find_distance(forest_view, node_color_view, distance_view, &shapeinfo)
    return distance
