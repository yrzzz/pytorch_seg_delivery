/**************************************************************************
* Microsoft COCO Toolbox.      version 2.0
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
* Licensed under the Simplified BSD License [see coco/license.txt]
*
* Modified by 10x Genomics
**************************************************************************/
#pragma once
#include <stdint.h>

typedef unsigned int uint;
typedef unsigned long siz;
typedef unsigned char byte;
typedef uint32_t uint32;

/* Bounding box is represented as: start_x, start_y, width, height */
typedef double *BB;
typedef struct {
  siz h, w, m;
  uint *cnts;
} RLE;

/* Initialize/destroy RLE. */
void rleInit(RLE *R, siz h, siz w, siz m, uint *cnts);
void rleFree(RLE *R);

/* Initialize/destroy RLE array. */
void rlesInit(RLE **R, siz n);
void rlesFree(RLE **R, siz n);

/* Encode binary masks using RLE. */
void rleEncode(RLE *R, const byte *mask, siz h, siz w, siz n);

/* Encode non-binary masks using RLE and return a list of RLEs */
void rleEncodeLabels(
    RLE *R, const uint32 *Rlabels, const uint32 *M, siz h, siz w, siz n);

/* Decode binary masks encoded via RLE. */
void rleDecode(const RLE *R, byte *mask, siz n);

/* Compute union or intersection of encoded masks. */
void rleMerge(const RLE *R, RLE *M, siz n, int intersect);

/* Compute area of encoded masks. */
void rleArea(const RLE *R, siz n, uint *a);

/* Compute intersection over union between masks. */
void rleIou(RLE *dt, RLE *gt, siz m, siz n, byte *iscrowd, double *o);

/* Compute non-maximum suppression between bounding masks */
void rleNms(RLE *dt, siz n, uint *keep, double thr);

/* Compute intersection over union between bounding boxes. */
void bbIou(BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o);

/* Compute non-maximum suppression between bounding boxes */
void bbNms(BB dt, siz n, uint *keep, double thr);

/* Get bounding boxes surrounding encoded masks. */
void rleToBbox(const RLE *R, BB bb, siz n);

/* Convert bounding boxes to encoded masks. */
void rleFrBbox(RLE *R, const BB bb, siz h, siz w, siz n);

/* Convert polygon to encoded mask. */
void rleFrPoly(RLE *R, const double *xy, siz k, siz h, siz w);

/* Get compressed string representation of encoded mask. */
char *rleToString(const RLE *R);

/* Convert from compressed string representation of encoded mask. */
void rleFrString(RLE *R, char *s, siz h, siz w);

/* Offset a list of rles. */
void rlesOffset(const RLE *orig_rles,
                RLE *new_rles,
                siz n,
                siz new_h,
                siz new_w,
                siz offset_h,
                siz offset_w);

/* offset the rle to a new window with the offset */
void rleOffset(const RLE *orig_rle,
               RLE *new_rle,
               siz new_h,
               siz new_w,
               siz offset_h,
               siz offset_w);

/* calculate intersection and exclusion of a rle relative to the reference rle*/
void rleIntersectExclude(const RLE *ref_rle,
                         const RLE *compare_rle,
                         RLE *intersect_rle,
                         RLE *exclude_rle);

void rleIntersectPercent(RLE *dt, RLE *gt, siz m, siz n, double *o);
