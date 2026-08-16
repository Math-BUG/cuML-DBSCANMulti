/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Universidade Federal de Viçosa (UFV)
 * SPDX-License-Identifier: Apache-2.0
 *
 * Busca de vizinhança para múltiplos ε, com tiling em memória compartilhada.
 *
 * Substitui cuvs::neighbors::epsilon_neighborhood::compute, usada por
 * ML::Dbscan::VertexDeg::Algo::launcher (third_party/cuml/cpp/src/dbscan/vertexdeg/algo.cuh),
 * que só aceita ε escalar.
 *
 * Ideia central (Algoritmo 2 do artigo). A distância de cada par é calculada UMA vez, com
 * corte pelo maior ε. Explorando a monotonicidade do raio — se dois pontos são vizinhos
 * sob um raio menor, também são sob todos os maiores — basta guardar, por par, o índice do
 * MENOR ε que o contém:
 *
 *     codes[i][j] = min { e : dist²(i,j) <= eps²_e },  ou 255 se nenhum
 *
 * São 8 bits por par, em vez de uma matriz booleana por ε. A adjacência de qualquer ε_e
 * sai daí por `codes <= e`, sem recalcular distância, e as contagens de vizinhos de todos
 * os raios saem da mesma varredura.
 *
 * Saídas:
 *   - codes: n_points x N, row-major, uint8;
 *   - vd:    k x vd_stride, com vd[e * vd_stride + i] = grau do ponto i sob ε_e e
 *            vd[e * vd_stride + n_points] = soma dos graus do lote sob ε_e (lida pelo
 *            runner como nnz do lote, no mesmo contrato do cuML).
 *
 * Métrica: L2, comparada em distância ao quadrado contra eps², como no cuML.
 */

#pragma once

#include "atomics.cuh"

#include <raft/core/error.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cub/cub.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ML {
namespace Dbscan {
namespace Multi {
namespace VertexDeg {

/** Código de "não é vizinho sob nenhum ε". */
static constexpr std::uint8_t kNoNeighbor = 255;

/** Número máximo de valores de ε suportados pelo kernel. */
static constexpr int kMaxEps = 16;

/** Orçamento de memória compartilhada por bloco para o tile de pontos, em bytes.
 *  32 KiB fica abaixo do limite padrão de 48 KiB em todas as arquiteturas suportadas. */
static constexpr int kTileSharedBytes = 32 * 1024;

/** Número máximo de colunas por tile. */
static constexpr int kMaxTileCols = 256;

/**
 * Acumula os graus do bloco em vd[e * vd_stride + n_points], para cada ε.
 * O TempStorage do BlockReduce é reutilizado entre os ε, por isso o __syncthreads().
 */
template <typename index_t, int TPB, int MAX_K>
__device__ inline void reduceAndAccumulate(const index_t (&cnt)[MAX_K],
                                           bool active,
                                           int k,
                                           index_t* vd,
                                           index_t vd_stride,
                                           index_t n_points)
{
  typedef cub::BlockReduce<index_t, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

#pragma unroll
  for (int c = 0; c < MAX_K; ++c) {
    if (c < k) {  // uniforme no bloco: k é o mesmo para todas as threads
      const index_t total = BlockReduce(temp_storage).Sum(active ? cnt[c] : index_t(0));
      if (threadIdx.x == 0) atomicAddIndex<index_t>(vd + c * vd_stride + n_points, total);
      __syncthreads();
    }
  }
}

/**
 * Caminho rápido: o ponto de consulta cabe em registradores (D <= MAX_D).
 *
 * Os laços sobre dimensões e sobre ε são desenrolados com limites de compilação, e os
 * índices excedentes são descartados por operador ternário — que não avalia o ramo não
 * tomado, portanto não há leitura fora dos limites do tile. É esse desenrolamento que
 * mantém q[] e cnt[] em registradores (MAX_D_REG e MAX_PARAM_REG no artigo).
 */
template <typename value_t, typename index_t, int TPB, int MAX_D, int MAX_K>
__global__ void epsNeighborhoodRegKernel(const value_t* __restrict__ x,
                                         index_t N,
                                         index_t D,
                                         index_t start_row,
                                         index_t n_points,
                                         const value_t* __restrict__ eps2,
                                         int k,
                                         index_t tile_cols,
                                         std::uint8_t* __restrict__ codes,
                                         index_t* __restrict__ vd,
                                         index_t vd_stride)
{
  extern __shared__ char smem_raw[];
  value_t* tile = reinterpret_cast<value_t*>(smem_raw);

  const index_t row = static_cast<index_t>(blockIdx.x) * TPB + static_cast<index_t>(threadIdx.x);
  const bool active = row < n_points;

  value_t q[MAX_D];
  if (active) {
    const value_t* src = x + static_cast<std::size_t>(start_row + row) * D;
#pragma unroll
    for (int d = 0; d < MAX_D; ++d) {
      q[d] = (d < D) ? src[d] : value_t(0);
    }
  }

  value_t e2[MAX_K];
#pragma unroll
  for (int c = 0; c < MAX_K; ++c) {
    e2[c] = (c < k) ? eps2[c] : value_t(0);
  }
  const value_t cutoff = eps2[k - 1];  // maior raio: critério de corte da varredura

  index_t cnt[MAX_K];
#pragma unroll
  for (int c = 0; c < MAX_K; ++c) {
    cnt[c] = 0;
  }

  std::uint8_t* codes_row = active ? codes + static_cast<std::size_t>(row) * N : nullptr;

  for (index_t j0 = 0; j0 < N; j0 += tile_cols) {
    const index_t tj = (tile_cols < N - j0) ? tile_cols : (N - j0);

    for (index_t idx = threadIdx.x; idx < tj * D; idx += TPB) {
      const index_t r = idx / D;
      const index_t c = idx - r * D;
      tile[idx]       = x[static_cast<std::size_t>(j0 + r) * D + c];
    }
    __syncthreads();

    if (active) {
      for (index_t t = 0; t < tj; ++t) {
        const value_t* p = tile + static_cast<std::size_t>(t) * D;
        value_t acc      = value_t(0);
#pragma unroll
        for (int d = 0; d < MAX_D; ++d) {
          const value_t diff = (d < D) ? (q[d] - p[d]) : value_t(0);
          acc += diff * diff;
        }

        // Menor ε que contém o par. A varredura desce, então o último acerto é o menor.
        std::uint8_t code = kNoNeighbor;
        if (acc <= cutoff) {
#pragma unroll
          for (int c = MAX_K - 1; c >= 0; --c) {
            if (c < k && acc <= e2[c]) code = static_cast<std::uint8_t>(c);
          }
        }
        codes_row[j0 + t] = code;

        // Um vizinho sob ε_e também conta para todos os raios maiores.
#pragma unroll
        for (int c = 0; c < MAX_K; ++c) {
          cnt[c] += (c < k && code <= static_cast<std::uint8_t>(c)) ? index_t(1) : index_t(0);
        }
      }
    }
    __syncthreads();
  }

  if (active) {
#pragma unroll
    for (int c = 0; c < MAX_K; ++c) {
      if (c < k) vd[c * vd_stride + row] = cnt[c];
    }
  }

  reduceAndAccumulate<index_t, TPB, MAX_K>(cnt, active, k, vd, vd_stride, n_points);
}

/**
 * Caminho genérico para D alto: o ponto de consulta é relido da memória global e o acúmulo
 * da distância tem corte antecipado ao ultrapassar o maior ε — o que compensa justamente
 * quando D é grande.
 */
template <typename value_t, typename index_t, int TPB, int MAX_K>
__global__ void epsNeighborhoodGenericKernel(const value_t* __restrict__ x,
                                             index_t N,
                                             index_t D,
                                             index_t start_row,
                                             index_t n_points,
                                             const value_t* __restrict__ eps2,
                                             int k,
                                             index_t tile_cols,
                                             std::uint8_t* __restrict__ codes,
                                             index_t* __restrict__ vd,
                                             index_t vd_stride)
{
  extern __shared__ char smem_raw[];
  value_t* tile = reinterpret_cast<value_t*>(smem_raw);

  const index_t row = static_cast<index_t>(blockIdx.x) * TPB + static_cast<index_t>(threadIdx.x);
  const bool active = row < n_points;

  value_t e2[MAX_K];
#pragma unroll
  for (int c = 0; c < MAX_K; ++c) {
    e2[c] = (c < k) ? eps2[c] : value_t(0);
  }
  const value_t cutoff = eps2[k - 1];

  index_t cnt[MAX_K];
#pragma unroll
  for (int c = 0; c < MAX_K; ++c) {
    cnt[c] = 0;
  }

  const value_t* q       = active ? x + static_cast<std::size_t>(start_row + row) * D : nullptr;
  std::uint8_t* codes_row = active ? codes + static_cast<std::size_t>(row) * N : nullptr;

  for (index_t j0 = 0; j0 < N; j0 += tile_cols) {
    const index_t tj = (tile_cols < N - j0) ? tile_cols : (N - j0);

    for (index_t idx = threadIdx.x; idx < tj * D; idx += TPB) {
      const index_t r = idx / D;
      const index_t c = idx - r * D;
      tile[idx]       = x[static_cast<std::size_t>(j0 + r) * D + c];
    }
    __syncthreads();

    if (active) {
      for (index_t t = 0; t < tj; ++t) {
        const value_t* p = tile + static_cast<std::size_t>(t) * D;
        value_t acc      = value_t(0);
        for (index_t d = 0; d < D; ++d) {
          const value_t diff = q[d] - p[d];
          acc += diff * diff;
          if (acc > cutoff) break;
        }

        std::uint8_t code = kNoNeighbor;
        if (acc <= cutoff) {
#pragma unroll
          for (int c = MAX_K - 1; c >= 0; --c) {
            if (c < k && acc <= e2[c]) code = static_cast<std::uint8_t>(c);
          }
        }
        codes_row[j0 + t] = code;

#pragma unroll
        for (int c = 0; c < MAX_K; ++c) {
          cnt[c] += (c < k && code <= static_cast<std::uint8_t>(c)) ? index_t(1) : index_t(0);
        }
      }
    }
    __syncthreads();
  }

  if (active) {
#pragma unroll
    for (int c = 0; c < MAX_K; ++c) {
      if (c < k) vd[c * vd_stride + row] = cnt[c];
    }
  }

  reduceAndAccumulate<index_t, TPB, MAX_K>(cnt, active, k, vd, vd_stride, n_points);
}

/**
 * Fallback para D tão alto que nem uma coluna cabe no orçamento de shared memory.
 *
 * O caminho tiled não pode lançar `D*sizeof(value_t)` bytes quando isso ultrapassa o
 * limite do bloco. Este kernel relê os dois pontos da memória global, preserva o mesmo
 * corte antecipado e troca apenas desempenho por correção/portabilidade.
 */
template <typename value_t, typename index_t, int TPB, int MAX_K>
__global__ void epsNeighborhoodGlobalKernel(const value_t* __restrict__ x,
                                             index_t N,
                                             index_t D,
                                             index_t start_row,
                                             index_t n_points,
                                             const value_t* __restrict__ eps2,
                                             int k,
                                             std::uint8_t* __restrict__ codes,
                                             index_t* __restrict__ vd,
                                             index_t vd_stride)
{
  const index_t row = static_cast<index_t>(blockIdx.x) * TPB +
                      static_cast<index_t>(threadIdx.x);
  const bool active = row < n_points;

  value_t e2[MAX_K];
#pragma unroll
  for (int c = 0; c < MAX_K; ++c) {
    e2[c] = (c < k) ? eps2[c] : value_t(0);
  }
  const value_t cutoff = eps2[k - 1];

  index_t cnt[MAX_K];
#pragma unroll
  for (int c = 0; c < MAX_K; ++c) {
    cnt[c] = 0;
  }

  if (active) {
    const value_t* q = x + static_cast<std::size_t>(start_row + row) * D;
    std::uint8_t* codes_row = codes + static_cast<std::size_t>(row) * N;
    for (index_t j = 0; j < N; ++j) {
      const value_t* p = x + static_cast<std::size_t>(j) * D;
      value_t acc      = value_t(0);
      for (index_t dim = 0; dim < D; ++dim) {
        const value_t diff = q[dim] - p[dim];
        acc += diff * diff;
        if (acc > cutoff) break;
      }

      std::uint8_t code = kNoNeighbor;
      if (acc <= cutoff) {
#pragma unroll
        for (int c = MAX_K - 1; c >= 0; --c) {
          if (c < k && acc <= e2[c]) code = static_cast<std::uint8_t>(c);
        }
      }
      codes_row[j] = code;
#pragma unroll
      for (int c = 0; c < MAX_K; ++c) {
        cnt[c] += (c < k && code <= static_cast<std::uint8_t>(c)) ? index_t(1) : index_t(0);
      }
    }

#pragma unroll
    for (int c = 0; c < MAX_K; ++c) {
      if (c < k) vd[c * vd_stride + row] = cnt[c];
    }
  }

  reduceAndAccumulate<index_t, TPB, MAX_K>(cnt, active, k, vd, vd_stride, n_points);
}

/**
 * Materializa a matriz booleana de adjacência de um ε a partir dos códigos.
 *
 * É o que permite reaproveitar AdjGraph::run (adj_to_csr) do cuML sem alteração: uma
 * passagem sobre uma matriz de bytes, sem recalcular nenhuma distância.
 *
 * Template só para dar linkagem de template ao kernel: `__global__` ignora `inline`, e
 * sem isso o cabeçalho daria símbolo duplicado se um dia for incluído em mais de um .cu.
 */
template <typename code_t>
__global__ void codesToAdjKernel(const code_t* __restrict__ codes,
                                 bool* __restrict__ adj,
                                 std::size_t count,
                                 code_t eps_index)
{
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) adj[idx] = codes[idx] <= eps_index;
}

/** Colunas por tile que cabem no orçamento de memória compartilhada. */
template <typename value_t, typename index_t>
inline index_t tile_cols_for(index_t D, index_t N)
{
  const std::size_t per_col = static_cast<std::size_t>(D) * sizeof(value_t);
  std::size_t cols          = per_col > 0 ? (kTileSharedBytes / per_col) : kMaxTileCols;
  if (cols < 1) cols = 1;
  if (cols > static_cast<std::size_t>(kMaxTileCols)) cols = kMaxTileCols;
  if (cols > static_cast<std::size_t>(N)) cols = static_cast<std::size_t>(N);
  return static_cast<index_t>(cols);
}

namespace detail {

template <typename value_t, typename index_t, int TPB, int MAX_K>
void launch_by_dim(const value_t* x,
                   index_t N,
                   index_t D,
                   index_t start_row,
                   index_t n_points,
                   const value_t* eps2,
                   int k,
                   index_t tile_cols,
                   std::uint8_t* codes,
                   index_t* vd,
                   index_t vd_stride,
                   int blocks,
                   std::size_t shmem,
                   cudaStream_t stream)
{
#define DBSCANMULTI_LAUNCH_REG(MAX_D)                                       \
  epsNeighborhoodRegKernel<value_t, index_t, TPB, MAX_D, MAX_K>             \
    <<<blocks, TPB, shmem, stream>>>(                                       \
      x, N, D, start_row, n_points, eps2, k, tile_cols, codes, vd, vd_stride)

  if (D <= 4) {
    DBSCANMULTI_LAUNCH_REG(4);
  } else if (D <= 8) {
    DBSCANMULTI_LAUNCH_REG(8);
  } else if (D <= 16) {
    DBSCANMULTI_LAUNCH_REG(16);
  } else if (D <= 32) {
    DBSCANMULTI_LAUNCH_REG(32);
  } else if (static_cast<std::size_t>(D) * sizeof(value_t) > kTileSharedBytes) {
    epsNeighborhoodGlobalKernel<value_t, index_t, TPB, MAX_K>
      <<<blocks, TPB, 0, stream>>>(
        x, N, D, start_row, n_points, eps2, k, codes, vd, vd_stride);
  } else {
    epsNeighborhoodGenericKernel<value_t, index_t, TPB, MAX_K>
      <<<blocks, TPB, shmem, stream>>>(
        x, N, D, start_row, n_points, eps2, k, tile_cols, codes, vd, vd_stride);
  }

#undef DBSCANMULTI_LAUNCH_REG
}

}  // namespace detail

/**
 * Calcula os códigos de ε e os graus de vizinhança de um lote de linhas.
 *
 * @param[in]  x          pontos, N x D row-major, no device
 * @param[in]  N          número de pontos
 * @param[in]  D          dimensionalidade
 * @param[in]  start_row  primeira linha do lote
 * @param[in]  n_points   número de linhas do lote
 * @param[in]  eps2       k raios AO QUADRADO, no device, em ordem CRESCENTE
 * @param[in]  k          número de valores de ε (1 <= k <= 16)
 * @param[out] codes      n_points x N, índice do menor ε por par (255 = nenhum)
 * @param[out] vd         k x vd_stride, graus por ε; a posição n_points recebe a soma
 * @param[in]  vd_stride  passo entre as fatias de vd (>= n_points + 1)
 */
template <typename value_t, typename index_t>
void run_multi(const value_t* x,
               index_t N,
               index_t D,
               index_t start_row,
               index_t n_points,
               const value_t* eps2,
               int k,
               std::uint8_t* codes,
               index_t* vd,
               index_t vd_stride,
               cudaStream_t stream)
{
  constexpr int TPB = 128;

  ASSERT(k >= 1 && k <= kMaxEps, "Número de valores de eps deve estar entre 1 e %d", kMaxEps);

  const index_t tile_cols = tile_cols_for<value_t, index_t>(D, N);
  const std::size_t shmem =
    static_cast<std::size_t>(tile_cols) * static_cast<std::size_t>(D) * sizeof(value_t);
  const int blocks = static_cast<int>((n_points + TPB - 1) / TPB);

  // As posições de soma acumulam por atomicAdd e precisam começar zeradas.
  for (int c = 0; c < k; ++c) {
    RAFT_CUDA_TRY(cudaMemsetAsync(vd + c * vd_stride + n_points, 0, sizeof(index_t), stream));
  }

  if (k <= 1) {
    detail::launch_by_dim<value_t, index_t, TPB, 1>(
      x, N, D, start_row, n_points, eps2, k, tile_cols, codes, vd, vd_stride, blocks, shmem, stream);
  } else if (k <= 4) {
    detail::launch_by_dim<value_t, index_t, TPB, 4>(
      x, N, D, start_row, n_points, eps2, k, tile_cols, codes, vd, vd_stride, blocks, shmem, stream);
  } else {
    detail::launch_by_dim<value_t, index_t, TPB, 16>(
      x, N, D, start_row, n_points, eps2, k, tile_cols, codes, vd, vd_stride, blocks, shmem, stream);
  }

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/** Constrói a adjacência booleana do ε de índice eps_index a partir dos códigos. */
inline void codes_to_adj(const std::uint8_t* codes,
                         bool* adj,
                         std::size_t count,
                         int eps_index,
                         cudaStream_t stream)
{
  constexpr int TPB = 256;
  const std::size_t blocks = (count + TPB - 1) / TPB;
  codesToAdjKernel<<<static_cast<int>(blocks), TPB, 0, stream>>>(
    codes, adj, count, static_cast<std::uint8_t>(eps_index));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace VertexDeg
}  // namespace Multi
}  // namespace Dbscan
}  // namespace ML
