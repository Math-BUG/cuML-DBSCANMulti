/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modifications Copyright (c) 2026, Universidade Federal de Viçosa (UFV).
 *
 * Derivado de third_party/cuml/cpp/src/dbscan/vertexdeg/algo.cuh (cuML v26.02.00).
 *
 * Modificações em relação ao original:
 *   1. Mantida apenas a chamada do caminho de força bruta com métrica L2 — exatamente a
 *      mesma chamada ao cuVS que o DBSCAN do cuML faz;
 *   2. removidos os caminhos fora de escopo: RBC/ball_cover, sample_weight (e o
 *      accumulateWeights que depende dele), métrica Cosine e o backend Precomputed.
 *
 * Nada aqui é reimplementação: a busca de vizinhança continua sendo a do cuVS. A
 * funcionalidade multi entra depois, sobre o CSR que o cuML já constrói — ver
 * csr_multi_eps.cuh.
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/epsilon_neighborhood.hpp>

namespace ML {
namespace Dbscan {
namespace Multi {
namespace VertexDeg {

/**
 * Matriz de adjacência densa e graus de um lote, para um único raio.
 *
 * Assinatura reduzida da Algo::launcher do cuML. `eps2` é o raio AO QUADRADO, como o cuVS
 * espera. `vd` tem n_points + 1 posições: a última recebe a soma, que é o nnz do lote — o
 * mesmo contrato do cuML, do qual o AdjGraph depende.
 */
template <typename value_t, typename index_t>
void run_cuvs(const raft::handle_t& handle,
              const value_t* x,
              index_t N,
              index_t D,
              index_t start_row,
              index_t n_points,
              value_t eps2,
              bool* adj,
              index_t* vd,
              cudaStream_t /* stream */)
{
  cuvs::neighbors::epsilon_neighborhood::compute<value_t, index_t, int64_t>(
    handle,
    raft::make_device_matrix_view<const value_t, int64_t, raft::row_major>(
      x + static_cast<std::int64_t>(start_row) * D, n_points, D),
    raft::make_device_matrix_view<const value_t, int64_t, raft::row_major>(x, N, D),
    raft::make_device_matrix_view<bool, int64_t, raft::row_major>(adj, n_points, N),
    raft::make_device_vector_view<index_t, int64_t>(vd, n_points + 1),
    eps2,
    cuvs::distance::DistanceType::L2Unexpanded);
}

}  // namespace VertexDeg
}  // namespace Multi
}  // namespace Dbscan
}  // namespace ML
