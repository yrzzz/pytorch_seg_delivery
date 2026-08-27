/**************************************************************************
* Microsoft COCO Toolbox.      version 2.0
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
* Licensed under the Simplified BSD License [see coco/license.txt]
*
* Modified by 10x Genomics
**************************************************************************/
#include "maskApi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

uint umin(uint a, uint b) {
  return (a < b) ? a : b;
}
uint umax(uint a, uint b) {
  return (a > b) ? a : b;
}

void rleInit(RLE *R, siz h, siz w, siz m, uint *cnts) {
  R->h = h;
  R->w = w;
  R->m = m;
  R->cnts = (m == 0) ? 0 : malloc(sizeof(uint) * m);
  siz j;
  if (cnts)
    for (j = 0; j < m; j++)
      R->cnts[j] = cnts[j];
}

void rleFree(RLE *R) {
  free(R->cnts);
  R->cnts = 0;
}

void rlesInit(RLE **R, siz n) {
  siz i;
  *R = (RLE *)malloc(sizeof(RLE) * n);
  for (i = 0; i < n; i++)
    rleInit((*R) + i, 0, 0, 0, 0);
}

void rlesFree(RLE **R, siz n) {
  siz i;
  for (i = 0; i < n; i++)
    rleFree((*R) + i);
  free(*R);
  *R = 0;
}

void rleEncode(RLE *R, const byte *M, siz h, siz w, siz n) {
  siz i, j, k, a = w * h;
  uint c, *cnts;
  byte p;
  cnts = malloc(sizeof(uint) * (a + 1));
  for (i = 0; i < n; i++) {
    const byte *T = M + a * i;
    k = 0;
    p = 0;
    c = 0;
    for (j = 0; j < a; j++) {
      if (T[j] != p) {
        cnts[k++] = c;
        c = 0;
        p = T[j];
      }
      c++;
    }
    cnts[k++] = c;
    rleInit(R + i, h, w, k, cnts);
  }
  free(cnts);
}

/*
 * R - length n array of RLE structures 
 * Rlabels - length n array of uint32 label values (should be non-zero)
 * M - length w * h array of uint32 labels (the labelmap image)
 * 
 */
//void rleEncodeLabels(RLE *R, const uint32 *Rlabels, const uint32 *M, siz h, siz w, siz n ) {
void rleEncodeLabels(
    RLE *R, const uint32 *Rlabels, const uint32 *M, siz h, siz w, siz n) {
  siz a = w * h;

  struct count {
    uint cnt;
    uint32 label;
  };

  struct count *counts = (struct count *)malloc(sizeof(struct count) * (a + 1));
  siz k = 0;
  uint c = 0;
  uint32 p = 0;

  /* much like regular rleEncode, but we do one scan and create a "run" for 
     each different value, zero or otherwise */
  for (siz j = 0; j < a; j++) {
    if (M[j] != p) {
      counts[k].cnt = c;
      counts[k++].label = p;
      c = 0;
      p = M[j];
    }
    c++;
  }
  counts[k].cnt = c;
  counts[k++].label = p;

  uint *tmp_cnts = (uint *)malloc(sizeof(uint) * k);

  /* for each target label, scan the master run list and make a sub list where
     each label that isn't a target during a given iteration is considered a zero. */
  for (siz i = 0; i < n; i++) {
    uint c = 0;
    siz tmp_k = 0;

    for (siz j = 0; j < k; j++) {
      if (counts[j].label == Rlabels[i]) {
        tmp_cnts[tmp_k++] = c;             /* previous zeroes*/
        tmp_cnts[tmp_k++] = counts[j].cnt; /* count of current label */
        c = 0;
      } else
        c += counts[j].cnt; /* everything else treated as a zero */
    }
    if (c > 0)
      tmp_cnts[tmp_k++] = c;

    rleInit(R + i, h, w, tmp_k, tmp_cnts);
  }

  free(counts);
  free(tmp_cnts);
}

void rleDecode(const RLE *R, byte *M, siz n) {
  siz i, j, k;
  for (i = 0; i < n; i++) {
    byte v = 0;
    for (j = 0; j < R[i].m; j++) {
      for (k = 0; k < R[i].cnts[j]; k++)
        *(M++) = v;
      v = !v;
    }
  }
}

void rleMerge(const RLE *R, RLE *M, siz n, int intersect) {
  uint *cnts, c, ca, cb, cc, ct;
  int v, va, vb, vp;
  siz i, a, b, h = R[0].h, w = R[0].w, m = R[0].m;
  RLE A, B;
  if (n == 0) {
    rleInit(M, 0, 0, 0, 0);
    return;
  }
  if (n == 1) {
    rleInit(M, h, w, m, R[0].cnts);
    return;
  }
  cnts = malloc(sizeof(uint) * (h * w + 1));
  for (a = 0; a < m; a++)
    cnts[a] = R[0].cnts[a];
  for (i = 1; i < n; i++) {
    B = R[i];
    if (B.h != h || B.w != w) {
      h = w = m = 0;
      break;
    }
    rleInit(&A, h, w, m, cnts);
    ca = A.cnts[0];
    cb = B.cnts[0];
    v = va = vb = 0;
    m = 0;
    a = b = 1;
    cc = 0;
    ct = 1;
    while (ct > 0) {
      c = umin(ca, cb);
      cc += c;
      ct = 0;
      ca -= c;
      if (!ca && a < A.m) {
        ca = A.cnts[a++];
        va = !va;
      }
      ct += ca;
      cb -= c;
      if (!cb && b < B.m) {
        cb = B.cnts[b++];
        vb = !vb;
      }
      ct += cb;
      vp = v;
      if (intersect)
        v = va && vb;
      else
        v = va || vb;
      if (v != vp || ct == 0) {
        cnts[m++] = cc;
        cc = 0;
      }
    }
    rleFree(&A);
  }
  rleInit(M, h, w, m, cnts);
  free(cnts);
}

void rleArea(const RLE *R, siz n, uint *a) {
  siz i, j;
  for (i = 0; i < n; i++) {
    a[i] = 0;
    for (j = 1; j < R[i].m; j += 2)
      a[i] += R[i].cnts[j];
  }
}

void rleIou(RLE *dt, RLE *gt, siz m, siz n, byte *iscrowd, double *o) {
  siz g, d;
  BB db, gb;
  int crowd;
  db = malloc(sizeof(double) * m * 4);
  rleToBbox(dt, db, m);
  gb = malloc(sizeof(double) * n * 4);
  rleToBbox(gt, gb, n);
  bbIou(db, gb, m, n, iscrowd, o);
  free(db);
  free(gb);
  for (g = 0; g < n; g++)
    for (d = 0; d < m; d++)
      if (o[g * m + d] > 0) {
        crowd = iscrowd != NULL && iscrowd[g];
        if (dt[d].h != gt[g].h || dt[d].w != gt[g].w) {
          o[g * m + d] = -1;
          continue;
        }
        siz ka, kb, a, b;
        uint c, ca, cb, ct, i, u;
        int va, vb;
        ca = dt[d].cnts[0];
        ka = dt[d].m;
        va = vb = 0;
        cb = gt[g].cnts[0];
        kb = gt[g].m;
        a = b = 1;
        i = u = 0;
        ct = 1;
        while (ct > 0) {
          c = umin(ca, cb);
          if (va || vb) {
            u += c;
            if (va && vb)
              i += c;
          }
          ct = 0;
          ca -= c;
          if (!ca && a < ka) {
            ca = dt[d].cnts[a++];
            va = !va;
          }
          ct += ca;
          cb -= c;
          if (!cb && b < kb) {
            cb = gt[g].cnts[b++];
            vb = !vb;
          }
          ct += cb;
        }
        if (i == 0)
          u = 1;
        else if (crowd)
          rleArea(dt + d, 1, &u);
        o[g * m + d] = (double)i / (double)u;
      }
}

void rleNms(RLE *dt, siz n, uint *keep, double thr) {
  siz i, j;
  double u;
  for (i = 0; i < n; i++)
    keep[i] = 1;
  for (i = 0; i < n; i++)
    if (keep[i]) {
      for (j = i + 1; j < n; j++)
        if (keep[j]) {
          rleIou(dt + i, dt + j, 1, 1, 0, &u);
          if (u > thr)
            keep[j] = 0;
        }
    }
}

void bbIou(BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o) {
  double h, w, i, u, ga, da;
  siz g, d;
  int crowd;
  for (g = 0; g < n; g++) {
    BB G = gt + g * 4;
    ga = G[2] * G[3];
    crowd = iscrowd != NULL && iscrowd[g];
    for (d = 0; d < m; d++) {
      BB D = dt + d * 4;
      da = D[2] * D[3];
      o[g * m + d] = 0;
      w = fmin(D[2] + D[0], G[2] + G[0]) - fmax(D[0], G[0]);
      if (w <= 0)
        continue;
      h = fmin(D[3] + D[1], G[3] + G[1]) - fmax(D[1], G[1]);
      if (h <= 0)
        continue;
      i = w * h;
      u = crowd ? da : da + ga - i;
      o[g * m + d] = i / u;
    }
  }
}

void bbNms(BB dt, siz n, uint *keep, double thr) {
  siz i, j;
  double u;
  for (i = 0; i < n; i++)
    keep[i] = 1;
  for (i = 0; i < n; i++)
    if (keep[i]) {
      for (j = i + 1; j < n; j++)
        if (keep[j]) {
          bbIou(dt + i * 4, dt + j * 4, 1, 1, 0, &u);
          if (u > thr)
            keep[j] = 0;
        }
    }
}

void rleToBbox(const RLE *R, BB bb, siz n) {
  siz i;
  for (i = 0; i < n; i++) {
    uint h, w, x, y, xs, ys, xe, ye, xp, cc, t;
    siz j, m;
    h = (uint)R[i].h;
    w = (uint)R[i].w;
    m = R[i].m;
    m = ((siz)(m / 2)) * 2;
    xs = w;
    ys = h;
    xe = ye = 0;
    cc = 0;
    if (m == 0) {
      bb[4 * i + 0] = bb[4 * i + 1] = bb[4 * i + 2] = bb[4 * i + 3] = 0;
      continue;
    }
    for (j = 0; j < m; j++) {
      cc += R[i].cnts[j];
      t = cc - j % 2;
      y = t % h;
      x = (t - y) / h;
      if (j % 2 == 0)
        xp = x;
      else if (xp < x) {
        ys = 0;
        ye = h - 1;
      }
      xs = umin(xs, x);
      xe = umax(xe, x);
      ys = umin(ys, y);
      ye = umax(ye, y);
    }
    bb[4 * i + 0] = xs;
    bb[4 * i + 2] = xe - xs + 1;
    bb[4 * i + 1] = ys;
    bb[4 * i + 3] = ye - ys + 1;
  }
}

void rleFrBbox(RLE *R, const BB bb, siz h, siz w, siz n) {
  siz i;
  for (i = 0; i < n; i++) {
    double xs = bb[4 * i + 0], xe = xs + bb[4 * i + 2];
    double ys = bb[4 * i + 1], ye = ys + bb[4 * i + 3];
    double xy[8] = {xs, ys, xs, ye, xe, ye, xe, ys};
    rleFrPoly(R + i, xy, 4, h, w);
  }
}

int uintCompare(const void *a, const void *b) {
  uint c = *((uint *)a), d = *((uint *)b);
  return c > d ? 1 : c < d ? -1 : 0;
}

void rleFrPoly(RLE *R, const double *xy, siz k, siz h, siz w) {
  /* upsample and get discrete points densely along entire boundary */
  siz j, m = 0;
  double scale = 5;
  int *x, *y, *u, *v;
  uint *a, *b;
  x = malloc(sizeof(int) * (k + 1));
  y = malloc(sizeof(int) * (k + 1));
  for (j = 0; j < k; j++)
    x[j] = (int)(scale * xy[j * 2 + 0] + .5);
  x[k] = x[0];
  for (j = 0; j < k; j++)
    y[j] = (int)(scale * xy[j * 2 + 1] + .5);
  y[k] = y[0];
  for (j = 0; j < k; j++)
    m += umax(abs(x[j] - x[j + 1]), abs(y[j] - y[j + 1])) + 1;
  u = malloc(sizeof(int) * m);
  v = malloc(sizeof(int) * m);
  m = 0;
  for (j = 0; j < k; j++) {
    int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, d;
    int flip;
    double s;
    dx = abs(xe - xs);
    dy = abs(ys - ye);
    flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
    if (flip) {
      t = xs;
      xs = xe;
      xe = t;
      t = ys;
      ys = ye;
      ye = t;
    }
    s = dx >= dy ? (double)(ye - ys) / dx : (double)(xe - xs) / dy;
    if (dx >= dy)
      for (d = 0; d <= dx; d++) {
        t = flip ? dx - d : d;
        u[m] = t + xs;
        v[m] = (int)(ys + s * t + .5);
        m++;
      }
    else
      for (d = 0; d <= dy; d++) {
        t = flip ? dy - d : d;
        v[m] = t + ys;
        u[m] = (int)(xs + s * t + .5);
        m++;
      }
  }
  /* get points along y-boundary and downsample */
  free(x);
  free(y);
  k = m;
  m = 0;
  double xd, yd;
  x = malloc(sizeof(int) * k);
  y = malloc(sizeof(int) * k);
  for (j = 1; j < k; j++)
    if (u[j] != u[j - 1]) {
      xd = (double)(u[j] < u[j - 1] ? u[j] : u[j] - 1);
      xd = (xd + .5) / scale - .5;
      if (floor(xd) != xd || xd < 0 || xd > w - 1)
        continue;
      yd = (double)(v[j] < v[j - 1] ? v[j] : v[j - 1]);
      yd = (yd + .5) / scale - .5;
      if (yd < 0)
        yd = 0;
      else if (yd > h)
        yd = h;
      yd = ceil(yd);
      x[m] = (int)xd;
      y[m] = (int)yd;
      m++;
    }
  /* compute rle encoding given y-boundary points */
  k = m;
  a = malloc(sizeof(uint) * (k + 1));
  for (j = 0; j < k; j++)
    a[j] = (uint)(x[j] * (int)(h) + y[j]);
  a[k++] = (uint)(h * w);
  free(u);
  free(v);
  free(x);
  free(y);
  qsort(a, k, sizeof(uint), uintCompare);
  uint p = 0;
  for (j = 0; j < k; j++) {
    uint t = a[j];
    a[j] -= p;
    p = t;
  }
  b = malloc(sizeof(uint) * k);
  j = m = 0;
  b[m++] = a[j++];
  while (j < k)
    if (a[j] > 0)
      b[m++] = a[j++];
    else {
      j++;
      if (j < k)
        b[m - 1] += a[j++];
    }
  rleInit(R, h, w, m, b);
  free(a);
  free(b);
}

char *rleToString(const RLE *R) {
  /* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
  siz i, m = R->m, p = 0;
  long x;
  int more;
  char *s = malloc(sizeof(char) * m * 6);
  for (i = 0; i < m; i++) {
    x = (long)R->cnts[i];
    if (i > 2)
      x -= (long)R->cnts[i - 2];
    more = 1;
    while (more) {
      char c = x & 0x1f;
      x >>= 5;
      more = (c & 0x10) ? x != -1 : x != 0;
      if (more)
        c |= 0x20;
      c += 48;
      s[p++] = c;
    }
  }
  s[p] = 0;
  return s;
}

void rleFrString(RLE *R, char *s, siz h, siz w) {
  siz m = 0, p = 0, k;
  long x;
  int more;
  uint *cnts;
  while (s[m])
    m++;
  cnts = malloc(sizeof(uint) * m);
  m = 0;
  while (s[p]) {
    x = 0;
    k = 0;
    more = 1;
    while (more) {
      char c = s[p] - 48;
      x |= (c & 0x1f) << 5 * k;
      more = c & 0x20;
      p++;
      k++;
      if (!more && (c & 0x10))
        x |= -1 << 5 * k;
    }
    if (m > 2)
      x += (long)cnts[m - 2];
    cnts[m++] = (uint)x;
  }
  rleInit(R, h, w, m, cnts);
  free(cnts);
}

void rlesOffset(const RLE *orig_rles,
                RLE *new_rles,
                siz n,
                siz new_h,
                siz new_w,
                siz offset_h,
                siz offset_w) {
  siz i;
  for (i = 0; i < n; i++) {
    rleOffset(orig_rles + i, new_rles + i, new_h, new_w, offset_h, offset_w);
  }
}

void rleOffset(const RLE *orig_rle,
               RLE *new_rle,
               siz new_h,
               siz new_w,
               siz offset_h,
               siz offset_w) {
  uint *cnts, orig_cnt;
  siz i, j, k = 0, orig_h = orig_rle[0].h, orig_w = orig_rle[0].w,
            orig_m = orig_rle[0].m;
  siz a, b, extra, overhead, remain_h, remain_w, sandwich_zero, tail,
      remain = orig_h;
  byte label = 0;
  cnts = malloc(sizeof(uint) * (orig_rle[0].m + new_w * 5 + 1));
  remain_h = new_h - orig_h - offset_h;
  remain_w = new_w - orig_w - offset_w;
  // initial overhead
  overhead = new_h * offset_w + offset_h;
  sandwich_zero = offset_h + remain_h;
  for (i = 0; i < orig_m; i++) {
    orig_cnt = orig_rle[0].cnts[i];
    if (orig_cnt > remain) {
      // spill over to more columns
      extra = orig_cnt - remain;
      a = extra / orig_h;
      b = extra % orig_h;
      if (label == 0 || sandwich_zero == 0) {
        // if label is 0, it can include the any sandwich_zero, reach to the start of the possible next 1
        // if label is 1 and sandwich_zero is 0, then can add continuously.
        cnts[k++] = remain + remain_h + a * new_h + offset_h + b + overhead;
        remain = orig_h - b;
        overhead = 0;
      } else {
        // spill over 1s, and extra 0s when go to a new column
        cnts[k++] = remain;
        for (j = 0; j < a; j++) {
          cnts[k++] = sandwich_zero;
          cnts[k++] = orig_h;
        }
        if (b > 0) {
          cnts[k++] = sandwich_zero;
          cnts[k++] = b;
          // 1 ends at the middle of the column. No overhead for next 0.
          overhead = 0;
        } else {
          // 1 ends at the end of the column. Next 0 has overhead > 0.
          overhead = sandwich_zero;
        }
        remain = orig_h - b;
      }
    } else {
      // stay at the current column, possibly ends the column.
      cnts[k++] = orig_cnt + overhead * (label == 0);
      remain = remain - orig_cnt;
      // default overhead to be 0 for remain > 0.
      overhead = 0;
      if (remain == 0) {
        remain = orig_h;
        if (label == 0) {
          // partial 0, column ends with 0. add the extra 0 and go to the start
          // of the next 1
          cnts[k - 1] += sandwich_zero;
        } else {
          // column ends with 1; Next 0 has overhead > 0.
          overhead = sandwich_zero;
        }
      }
    }
    label = !label;
  }

  if (label == 1) {
    // ends with 0. 0 always reach to the next start so remove the extra offset_h.
    cnts[k - 1] += new_h * remain_w - offset_h;
  } else {
    // ends with 1. Add overhead except the extra offset_h, if it's none zero
    tail = new_h * remain_w + overhead - offset_h;
    if (tail > 0) {
      cnts[k++] = tail;
    }
  }
  rleInit(new_rle, new_h, new_w, k, cnts);
  free(cnts);
}

void rleIntersectPercent(RLE *dt, RLE *gt, siz m, siz n, double *o) {
  siz g, d;
  BB db, gb;
  db = malloc(sizeof(double) * m * 4);
  rleToBbox(dt, db, m);
  gb = malloc(sizeof(double) * n * 4);
  rleToBbox(gt, gb, n);
  bbIou(db, gb, m, n, 0, o);
  free(db);
  free(gb);
  for (g = 0; g < n; g++)
    for (d = 0; d < m; d++)
      if (o[g * m + d] > 0) {
        if (dt[d].h != gt[g].h || dt[d].w != gt[g].w) {
          o[g * m + d] = -1;
          continue;
        }
        siz ka, kb, a, b;
        uint c, ca, cb, ct, i, u, area_d, area_g;
        int va, vb;
        ca = dt[d].cnts[0];
        ka = dt[d].m;
        va = vb = 0;
        cb = gt[g].cnts[0];
        kb = gt[g].m;
        a = b = 1;
        i = u = 0;
        ct = 1;
        while (ct > 0) {
          c = umin(ca, cb);
          if (va || vb) {
            u += c;
            if (va && vb)
              i += c;
          }
          ct = 0;
          ca -= c;
          if (!ca && a < ka) {
            ca = dt[d].cnts[a++];
            va = !va;
          }
          ct += ca;
          cb -= c;
          if (!cb && b < kb) {
            cb = gt[g].cnts[b++];
            vb = !vb;
          }
          ct += cb;
        }
        if (i == 0)
          o[g * m + d] = 0;
        else {
          rleArea(dt + d, 1, &area_d);
          rleArea(gt + g, 1, &area_g);
          if (area_d > area_g) {
            o[g * m + d] = (double)i / (double)area_g;
          } else {
            o[g * m + d] = (double)i / (double)area_d;
          }
        }
      }
}

/*
 * ref_rle - reference rle
 * compare_rle - rle that compares with the reference rle
 * intersect_rle - intersection rle between reference and compare rle
 * exclude_rle - excluded rle of the compare rle
 */
void rleIntersectExclude(const RLE *ref_rle,
                         const RLE *compare_rle,
                         RLE *intersect_rle,
                         RLE *exclude_rle) {
  uint *inter_cnts, *exclude_cnts;
  uint ref_len, comp_len, next_step_len, curr_len_inter, curr_len_exclude,
      remain_len;
  int ref_val, comp_val, curr_inter_val, curr_exclude_val, next_inter_val,
      next_exclude_val;
  siz height = ref_rle->h, width = ref_rle->w;
  siz ref_idx, comp_idx, inter_idx, exclude_idx;
  inter_cnts = malloc(sizeof(uint) * (height * width + 1));
  exclude_cnts = malloc(sizeof(uint) * (height * width + 1));
  ref_len = ref_rle->cnts[0];
  comp_len = compare_rle->cnts[0];
  ref_val = comp_val = 0;
  curr_inter_val = curr_exclude_val = next_inter_val = next_exclude_val = 0;
  ref_idx = comp_idx = 1;                 // the next idx
  curr_len_inter = curr_len_exclude = 0;  // initialize
  inter_idx = exclude_idx = 0;            // initialize
  remain_len = 1;                         // hack to enter the loop
  while (remain_len > 0) {
    // step forward in the loop
    remain_len = 0;
    next_step_len = umin(ref_len, comp_len);
    curr_len_exclude += next_step_len;
    curr_len_inter += next_step_len;
    // impact of current step on ref_rle
    ref_len -= next_step_len;
    if (!ref_len && ref_idx < ref_rle->m) {
      // next step exhausts current reference length and ref is not ended yet
      ref_len = ref_rle->cnts[ref_idx++];
      ref_val = !ref_val;
    }
    remain_len += ref_len;  // update remain length after stepping forward
    // impact of current step on comp_rle
    comp_len -= next_step_len;
    if (!comp_len && comp_idx < compare_rle->m) {
      // next step exhausts current comp_rle length and comp_rle is not ended yet
      comp_len = compare_rle->cnts[comp_idx++];
      comp_val = !comp_val;
    }
    remain_len += comp_len;  // update remain length after stepping forward
    // figure out whether to record length or not
    curr_inter_val = next_inter_val;
    curr_exclude_val = next_exclude_val;
    next_inter_val = ref_val && comp_val;
    next_exclude_val = (comp_val == 1 && ref_val == 0) ? 1 : 0;
    if (curr_inter_val != next_inter_val || remain_len == 0) {
      inter_cnts[inter_idx++] = curr_len_inter;
      curr_len_inter = 0;
    }
    if (curr_exclude_val != next_exclude_val || remain_len == 0) {
      exclude_cnts[exclude_idx++] = curr_len_exclude;
      curr_len_exclude = 0;
    }
  }
  rleInit(intersect_rle, height, width, inter_idx, inter_cnts);
  rleInit(exclude_rle, height, width, exclude_idx, exclude_cnts);
  free(inter_cnts);
  free(exclude_cnts);
}
