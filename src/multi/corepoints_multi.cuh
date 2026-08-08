/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modifications Copyright (c) 2026, Universidade Federal de Viçosa (UFV).
 *
 * Derivado de third_party/cuml/cpp/src/dbscan/corepoints/compute.cuh (cuML v26.02.00).
 *
 * Modificação: a versão original avalia um único critério `vd[i] >= min_pts` e escreve uma
 * máscara de pontos centrais. Aqui os graus de k raios são comparados contra l valores de
 * minPts, produzindo as k*l máscaras da grade em uma única varredura.
 *
 * É onde as duas monotonicidades se encontram, e por isso a contagem de vizinhos não
 * precisa saber nada sobre minPts:
 *   - em ε (Algoritmo 2): o grau já vem calculado para cada raio pelo kernel de vizinhança;
 *   - em minPts (Algoritmo 3): com os valores em ordem crescente, basta achar o maior
 *     índice em que o ponto ainda é central; a partir do primeiro que falha, nenhum vale.
 *
 * Ordem das configurações: config = e * l + m (eps-major), a mesma usada nos rótulos.
 */

#pragma once

#include <raft/core/handle.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstddef>

namespace ML {
namespace Dbscan {
namespace Multi {
namespace CorePoints {

/**
 * Marca os pontos centrais do lote para cada combinação (ε, minPts).
 *
 * @param[in]  handle          raft handle
 * @param[in]  vd              graus por ε: vd[e * vd_stride + i]
 * @param[in]  vd_stride       passo entre as fatias de vd
 * @param[in]  n_eps           k, número de valores de ε
 * @param[in]  min_pts         l valores de minPts no device, em ordem CRESCENTE
 * @param[in]  n_min_pts       l
 * @param[out] masks           k*l máscaras contíguas de N posições: masks[config * N + i]
 * @param[in]  N               número total de pontos
 * @param[in]  start_vertex_id primeiro ponto do lote
 * @param[in]  batch_size      número de pontos do lote
 * @param[in]  stream          CUDA stream
 */
template <typename Values_ = int, typename Index_ = int>
void compute_grid(const raft::handle_t& handle,
                  const Values_* vd,
                  Index_ vd_stride,
                  int n_eps,
                  const Index_* min_pts,
                  int n_min_pts,
                  bool* masks,
                  Index_ N,
                  Index_ start_vertex_id,
                  Index_ batch_size,
                  cudaStream_t stream)
{
  auto counting = thrust::make_counting_iterator<Index_>(0);
  thrust::for_each(
    handle.get_thrust_policy(), counting, counting + batch_size, [=] __device__(Index_ idx) {
      const std::size_t point = static_cast<std::size_t>(idx + start_vertex_id);

      for (int e = 0; e < n_eps; ++e) {
        const Index_ degree = static_cast<Index_>(vd[e * vd_stride + idx]);

        // Algoritmo 3: maior índice m em que o ponto ainda é central sob ε_e.
        int limit = -1;
        for (int m = 0; m < n_min_pts; ++m) {
          if (degree >= min_pts[m]) {
            limit = m;
          } else {
            break;  // monotonicidade: uma vez que falha, não volta a ser ponto central
          }
        }

        for (int m = 0; m < n_min_pts; ++m) {
          const std::size_t config = static_cast<std::size_t>(e) * n_min_pts + m;
          masks[config * static_cast<std::size_t>(N) + point] = (m <= limit);
        }
      }
    });
}

}  // namespace CorePoints
}  // namespace Multi
}  // namespace Dbscan
}  // namespace ML
