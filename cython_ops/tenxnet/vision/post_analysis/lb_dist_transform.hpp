/* Multi-Label Euclidean Distance Transform 2D

The code is modified from: https://github.com/seung-lab/euclidean-distance-transform-3d

License of the original code:

License: GNU 3.0

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: July 2018 - April 2021

*/

#ifndef EDT_H
#define EDT_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "threadpool.h"

// The pyedt namespace contains the primary implementation,
// but users will probably want to use the edt namespace (bottom)
// as the function sigs are a bit cleaner.
// pyedt names are underscored to prevent namespace collisions
// in the Cython wrapper.

namespace label_edt {

#define sq(x) (static_cast<float>(x) * static_cast<float>(x))

/* 1D Euclidean Distance Transform for Multiple Segids
 *
 * Map a row of segids to a euclidean distance transform.
 * Zero is considered a universal boundary as are differing
 * segids. Segments touching the boundary are mapped to 1.
 *
 * T* segids: 1d array of (un)signed integers
 * *d: write destination, equal sized array as *segids
 * n: size of segids, d
 * stride: typically 1, but can be used on a 
 *    multi dimensional array, in which case it is nx, nx*ny, etc
 * anisotropy: physical distance of each voxel
 *
 * Writes output to *d
 */
template <typename T>
void squared_edt_1d_multi_seg(T* segids,
                              float* dist,
                              const int n,
                              const long int stride) {
  long int i;

  T working_segid = segids[0];

  dist[0] = static_cast<float>(working_segid != 0);

  for (i = stride; i < n * stride; i += stride) {
    if (segids[i] == 0) {
      dist[i] = 0.0;
    } else if (segids[i] == working_segid) {
      dist[i] = dist[i - stride] + 1;
    } else {
      dist[i] = 1;
      dist[i - stride] = static_cast<float>(segids[i - stride] != 0);
      working_segid = segids[i];
    }
  }

  // reverse scan
  dist[n - stride] = static_cast<float>(segids[n - stride] != 0);
  long int min_bound = stride;

  for (i = (n - 2) * stride; i >= min_bound; i -= stride) {
    dist[i] = std::fminf(dist[i], dist[i + stride] + 1);
  }

  for (i = 0; i < n * stride; i += stride) {
    dist[i] *= dist[i];
  }
}

/* 1D Euclidean Distance Transform based on:
 * 
 * http://cs.brown.edu/people/pfelzens/dt/
 * 
 * Felzenszwalb and Huttenlocher. 
 * Distance Transforms of Sampled Functions.
 * Theory of Computing, Volume 8. p415-428. 
 * (Sept. 2012) doi: 10.4086/toc.2012.v008a019
 *
 */
void squared_edt_1d_parabolic(float* f,
                              float* d,
                              const int n,
                              const long int stride,
                              const bool black_border_left,
                              const bool black_border_right) {
  if (n == 0) {
    return;
  }

  int k = 0;
  int* v = new int[n]();
  float* ff = new float[n]();
  for (long int i = 0; i < n; i++) {
    ff[i] = f[i * stride];
  }

  float* ranges = new float[n + 1]();

  ranges[0] = -INFINITY;
  ranges[1] = +INFINITY;

  /* Unclear if this adds much but I certainly find it easier to get the parens right.
   *
   * Eqn: s = ( f(r) + r^2 ) - ( f(p) + p^2 ) / ( 2r - 2p )
   * 1: s = (f(r) - f(p) + (r^2 - p^2)) / 2(r-p)
   * 2: s = (f(r) - r(p) + (r+p)(r-p)) / 2(r-p) <-- can reuse r-p, replace mult w/ add
   */
  float s;
  float factor1, factor2;
  for (long int i = 1; i < n; i++) {
    factor1 = i - v[k];
    factor2 = i + v[k];
    s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);

    while (k > 0 && s <= ranges[k]) {
      k--;
      factor1 = i - v[k];
      factor2 = i + v[k];
      s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);
    }

    k++;
    v[k] = i;
    ranges[k] = s;
    ranges[k + 1] = +INFINITY;
  }

  k = 0;
  float envelope;
  for (long int i = 0; i < n; i++) {
    while (ranges[k + 1] < i) {
      k++;
    }

    d[i * stride] = sq(i - v[k]) + ff[v[k]];
    if (black_border_left && black_border_right) {
      envelope = std::fminf(sq(i + 1), sq(n - i));
      d[i * stride] = std::fminf(envelope, d[i * stride]);
    } else if (black_border_left) {
      d[i * stride] = std::fminf(sq(i + 1), d[i * stride]);
    } else if (black_border_right) {
      d[i * stride] = std::fminf(sq(n - i), d[i * stride]);
    }
  }

  delete[] v;
  delete[] ff;
  delete[] ranges;
}

template <typename T>
void squared_edt_1d_parabolic_multi_seg(
    T* segids, float* field, float* dist, const int n, const long int stride) {
  T working_segid = segids[0];
  T segid;
  long int last = 0;

  for (int i = 1; i < n; i++) {
    segid = segids[i * stride];
    if (segid == 0) {
      continue;
    } else if (segid != working_segid) {
      if (working_segid != 0) {
        squared_edt_1d_parabolic(field + last * stride,
                                 dist + last * stride,
                                 i - last,
                                 stride,
                                 (last > 0),
                                 (i < n - 1));
      }
      working_segid = segid;
      last = i;
    }
  }

  if (working_segid != 0 && last < n) {
    squared_edt_1d_parabolic(field + last * stride,
                             dist + last * stride,
                             n - last,
                             stride,
                             (last > 0),
                             true);
  }
}

template <typename T>
float* _edt2dsq(T* input,
                const size_t sx,
                const size_t sy,
                const int parallel = 1,
                float* workspace = NULL) {
  const size_t voxels = sx * sy;

  if (workspace == NULL) {
    workspace = new float[voxels]();
  }

  for (size_t y = 0; y < sy; y++) {
    squared_edt_1d_multi_seg<T>((input + sx * y), (workspace + sx * y), sx, 1);
  }

  ThreadPool pool(parallel);

  for (size_t x = 0; x < sx; x++) {
    pool.enqueue([input, x, workspace, sy, sx]() {
      squared_edt_1d_parabolic_multi_seg<T>(
          (input + x), (workspace + x), (workspace + x), sy, sx);
    });
  }

  pool.join();

  return workspace;
}

}  // namespace label_edt

#undef sq

#endif