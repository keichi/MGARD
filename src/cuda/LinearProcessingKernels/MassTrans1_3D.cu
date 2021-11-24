/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include "cuda/CommonInternal.h"
 
#include "cuda/DataRefactoring/Correction/LinearProcessingKernel3D.h"
#include "cuda/DataRefactoring/Correction/LinearProcessingKernel3D.hpp"

namespace mgard_x {

#define KERNELS(D, T)                                                          \
  template void lpk_reo_1_3d<D, T>(                                            \
      Handle<D, T> & handle, SIZE nr, SIZE nc, SIZE nf, SIZE nf_c, SIZE zero_r,     \
      SIZE zero_c, SIZE zero_f, T *ddist_f, T *dratio_f, T *dv1, SIZE lddv11,     \
      SIZE lddv12, T *dv2, SIZE lddv21, SIZE lddv22, T *dw, SIZE lddw1, SIZE lddw2, \
      int queue_idx, int config);\
  template class Lpk1Reo3D<D, T, CUDA>;

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)
KERNELS(4, double)
KERNELS(4, float)
KERNELS(5, double)
KERNELS(5, float)

#undef KERNELS

} // namespace mgard_x