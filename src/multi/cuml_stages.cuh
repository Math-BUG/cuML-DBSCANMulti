/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Universidade Federal de Viçosa (UFV)
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fronteira mínima para os estágios do DBSCAN vendorizado do cuML v26.02.00.
 *
 * Não há uma segunda implementação dos estágios aqui. AdjGraph, CorePoints,
 * weak_cc_batched e MergeLabels continuam sendo chamados diretamente pelos runners. Os
 * únicos helpers abaixo adaptam assinaturas que diferem do caminho multiparamétrico:
 *
 *   - o relabel final combina as mesmas primitivas RAFT/Thrust usadas pelo cuML;
 *   - o vertex degree fixa o caminho single-GPU, brute-force, L2 e sem pesos/RBC.
 *
 * CorePoints::compute fica disponível por este header para uma comparação futura com
 * Multi::CorePoints::compute_grid. A implementação fusionada não é substituída nesta fase.
 */

#pragma once

#include <adjgraph/runner.cuh>
#include <corepoints/compute.cuh>
#include <mergelabels/runner.cuh>

#ifdef DBSCANMULTI_USE_CUVS
#include <vertexdeg/runner.cuh>

#include <cuml/common/distance_type.hpp>
#endif

#include <raft/core/handle.hpp>
#include <raft/core/error.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/sparse/csr.hpp>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include <cuda_runtime_api.h>

#include <limits>

namespace ML {
namespace Dbscan {
namespace Multi {
namespace CumlStages {

/**
 * Aplica o mesmo contrato final de labels do runner do cuML: labels monotônicos, ruído -1
 * e clusters numerados a partir de zero.
 */
template <typename Index_ = int>
void relabel_for_sklearn(const raft::handle_t& handle,
                         Index_* labels,
                         Index_ N,
                         cudaStream_t stream)
{
  const Index_ max_label = std::numeric_limits<Index_>::max();
  raft::label::make_monotonic(
    labels, labels, N, stream, [max_label] __device__(Index_ value) {
      return value == max_label;
    });

  auto first = thrust::device_pointer_cast(labels);
  thrust::transform(handle.get_thrust_policy(),
                    first,
                    first + N,
                    first,
                    [max_label] __device__(Index_ value) {
                      return value == max_label ? static_cast<Index_>(-1)
                                                : static_cast<Index_>(value - 1);
                    });
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

#ifdef DBSCANMULTI_USE_CUVS

/** Chama diretamente o estágio VertexDeg do cuML no caminho brute-force L2. */
template <typename Type_f, typename Index_ = int>
void vertex_degree_l2(const raft::handle_t& handle,
                      const Type_f* x,
                      Index_ N,
                      Index_ D,
                      Index_ start_vertex_id,
                      Index_ batch_size,
                      Type_f eps,
                      bool* adj,
                      Index_* vd,
                      cudaStream_t stream)
{
  ML::Dbscan::VertexDeg::run<Type_f, Index_>(handle,
                                             nullptr,  // sem índice RBC
                                             nullptr,  // ia não usado no brute-force
                                             nullptr,  // ja não usado no brute-force
                                             static_cast<Index_>(0),
                                             adj,
                                             vd,
                                             nullptr,  // sem graus ponderados
                                             x,
                                             nullptr,  // sem sample_weight
                                             eps,
                                             N,
                                             D,
                                             1,  // VertexDeg::Algo
                                             start_vertex_id,
                                             batch_size,
                                             stream,
                                             ML::distance::DistanceType::L2SqrtUnexpanded);
}

#endif

}  // namespace CumlStages
}  // namespace Multi
}  // namespace Dbscan
}  // namespace ML
