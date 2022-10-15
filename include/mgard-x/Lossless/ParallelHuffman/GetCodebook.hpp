/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "FillArraySequence.hpp"
#include "GenerateCL.hpp"
#include "GenerateCW.hpp"
#include "GetFirstNonzeroIndex.hpp"
#include "ReorderByIndex.hpp"
#include "ReverseArray.hpp"

#ifndef MGARD_X_GET_CODEBOOK_TEMPLATE_HPP
#define MGARD_X_GET_CODEBOOK_TEMPLATE_HPP

namespace mgard_x {
// Parallel codebook generation wrapper
template <typename Q, typename H, typename DeviceType>
void GetCodebook(int dict_size,
                 SubArray<1, unsigned int, DeviceType> _d_freq_subarray,
                 SubArray<1, H, DeviceType> _d_codebook_subarray,
                 SubArray<1, uint8_t, DeviceType> _d_decode_meta_subarray,
                 HuffmanWorkspace<Q, H, DeviceType> &workspace) {
  // Metadata
  auto type_bw = sizeof(H) * 8;

  SubArray<1, H, DeviceType> _d_first_subarray(
      {(SIZE)type_bw}, (H *)_d_decode_meta_subarray((IDX)0));
  SubArray<1, H, DeviceType> _d_entry_subarray(
      {(SIZE)type_bw}, (H *)_d_decode_meta_subarray(sizeof(H) * type_bw));
  SubArray<1, Q, DeviceType> _d_qcode_subarray(
      {(SIZE)dict_size}, (Q *)_d_decode_meta_subarray(sizeof(H) * 2 * type_bw));

  // Sort Qcodes by frequency
  DeviceLauncher<DeviceType>::Execute(
      FillArraySequenceKernel(_d_qcode_subarray), 0);

  MemoryManager<DeviceType>::Copy1D(workspace._d_freq_copy_subarray.data(),
                                    _d_freq_subarray.data(), dict_size, 0);
  MemoryManager<DeviceType>::Copy1D(workspace._d_qcode_copy_subarray.data(),
                                    _d_qcode_subarray.data(), dict_size, 0);
  DeviceCollective<DeviceType>::SortByKey(
      (SIZE)dict_size, workspace._d_freq_copy_subarray,
      workspace._d_qcode_copy_subarray, _d_freq_subarray, _d_qcode_subarray,
      workspace.sort_by_key_workspace, 0);

  DeviceLauncher<DeviceType>::Execute(
      GetFirstNonzeroIndexKernel<unsigned int, DeviceType>(
          _d_freq_subarray, workspace.first_nonzero_index_subarray),
      0);

  unsigned int first_nonzero_index;
  MemoryManager<DeviceType>().Copy1D(
      &first_nonzero_index, workspace.first_nonzero_index_subarray(IDX(0)), 1,
      0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  if (debug_print_huffman) {
    PrintSubarray("SortByKey::_d_freq_subarray", _d_freq_subarray);
    PrintSubarray("SortByKey::_d_qcode_subarray", _d_qcode_subarray);
    // std::cout << "first_nonzero_index: " << first_nonzero_index << std::endl;
  }

  int nz_dict_size = dict_size - first_nonzero_index;

  SubArray<1, unsigned int, DeviceType> _nz_d_freq_subarray(
      {(SIZE)nz_dict_size}, _d_freq_subarray(first_nonzero_index));
  SubArray<1, H, DeviceType> _nz_d_codebook_subarray(
      {(SIZE)nz_dict_size}, _d_codebook_subarray(first_nonzero_index));

  DeviceLauncher<DeviceType>::Execute(
      GenerateCLKernel<unsigned int, DeviceType>(
          _nz_d_freq_subarray, workspace.CL_subarray, nz_dict_size,
          _nz_d_freq_subarray, workspace.lNodesLeader_subarray,
          workspace.iNodesFreq_subarray, workspace.iNodesLeader_subarray,
          workspace.tempFreq_subarray, workspace.tempIsLeaf_subarray,
          workspace.tempIndex_subarray, workspace.copyFreq_subarray,
          workspace.copyIsLeaf_subarray, workspace.copyIndex_subarray,
          workspace.diagonal_path_intersections_subarray,
          workspace.status_subarray),
      0);

  unsigned int max_CL;
  MemoryManager<DeviceType>().Copy1D(&max_CL, workspace.CL_subarray(IDX(0)), 1,
                                     0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  if (debug_print_huffman) {
    PrintSubarray("GenerateCL::CL_subarray", workspace.CL_subarray);
    std::cout << "GenerateCL: max_CL" << max_CL << std::endl;
  }

  int max_CW_bits = (sizeof(H) * 8) - 8;
  if (max_CL > max_CW_bits) {
    std::cout << log::log_err << "Cannot store all Huffman codewords in "
              << max_CW_bits + 8 << "-bit representation" << std::endl;
    std::cout << log::log_err
              << "Huffman codeword representation requires at least "
              << max_CL + 8 << " bits (longest codeword: " << max_CL << " bits)"
              << std::endl;
    exit(1);
  }

  DeviceLauncher<DeviceType>::Execute(
      GenerateCWKernel<unsigned int, H, DeviceType>(
          workspace.CL_subarray, _nz_d_codebook_subarray, _d_first_subarray,
          _d_entry_subarray, nz_dict_size, workspace.status_subarray),
      0);

  DeviceLauncher<DeviceType>::Execute(
      ReverseArrayKernel<H, DeviceType>(_d_codebook_subarray), 0);
  DeviceLauncher<DeviceType>::Execute(
      ReverseArrayKernel<Q, DeviceType>(_d_qcode_subarray), 0);

  MemoryManager<DeviceType>().Copy1D(workspace._d_codebook_subarray_org.data(),
                                     _d_codebook_subarray.data(),
                                     _d_codebook_subarray.shape(0), 0);

  DeviceLauncher<DeviceType>::Execute(
      ReorderByIndexKernel<H, Q, DeviceType>(workspace._d_codebook_subarray_org,
                                             _d_codebook_subarray,
                                             _d_qcode_subarray),
      0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
}

} // namespace mgard_x

#endif