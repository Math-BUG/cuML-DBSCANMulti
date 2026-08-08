/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Universidade Federal de Viçosa (UFV)
 * SPDX-License-Identifier: Apache-2.0
 *
 * Código novo: é aqui que a funcionalidade multi-ε entra, SOBRE o pipeline do cuML, sem
 * substituir nenhuma peça dele.
 *
 * Ideia. O DBSCAN do cuML, no caminho de força bruta, faz por lote:
 *
 *     cuVS epsilon_neighborhood::compute  ->  adj densa + graus
 *     AdjGraph (exclusive_scan + adj_to_csr)  ->  CSR do lote
 *     weak_cc_batched + MergeLabels  ->  rótulos
 *
 * Para k raios, chamar isso k vezes repetiria a etapa cara — a distância par-a-par, que é
 * O(N² · D) — k vezes. Explorando a monotonicidade do raio (vizinho sob um raio menor é
 * vizinho sob todos os maiores), o CSR do MAIOR raio já contém todos os pares de que
 * qualquer raio menor precisa. Então:
 *
 *   1. roda-se o cuVS UMA vez, no maior ε, e o AdjGraph do cuML sobre o resultado;
 *   2. anota-se cada entrada do CSR com o índice do menor ε que a contém — custo
 *      O(nnz · D), não O(N² · D), porque só os pares que já são vizinhos são visitados;
 *   3. o CSR de cada ε menor sai por compactação das entradas anotadas — custo O(nnz),
 *      sem nenhuma distância recalculada.
 *
 * O grau de cada ponto em cada raio cai de graça do passo 2, como prefixo do histograma
 * dos códigos da linha, e é o que alimenta as máscaras de pontos centrais por minPts.
 *
 * Vale quando o maior ε ainda é seletivo, que é o regime de interesse do DBSCAN: o ganho é
 * a razão entre N² · D e nnz · D. Se o maior ε aproximar nnz de N², a anotação custa o
 * mesmo que uma passagem completa e o ganho some — mas nesse regime todos os pontos são
 * vizinhos de todos e o agrupamento perdeu o sentido.
 */

#pragma once

#include "atomics.cuh"

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cstddef>
#include <cstdint>

namespace ML {
namespace Dbscan {
namespace Multi {
namespace CsrEps {

/** Limite de raios por execução: o código de ε é um byte e o histograma vive em shared. */
static constexpr int kMaxEps = 16;

/** Orçamento de shared memory para cachear a linha de consulta. */
static constexpr std::size_t kMaxQuerySharedBytes = 32 * 1024;

/**
 * Anota o CSR do maior ε e produz os graus de todos os raios.
 *
 * Um bloco por linha do lote. Cada thread percorre uma fatia das entradas da linha,
 * calcula a distância ao quadrado do par e determina o menor ε que o contém. O histograma
 * dos códigos fica em shared; seu prefixo é exatamente o grau da linha em cada raio.
 *
 * @param[in]  row_len   graus no maior ε (o `vd` que o cuVS produziu)
 * @param[out] codes     um byte por entrada do CSR
 * @param[out] vd_multi  k x vd_stride; a posição n_points de cada fatia recebe a soma
 */
template <typename value_t, typename index_t, int TPB>
__global__ void annotateKernel(const value_t* __restrict__ x,
                               index_t D,
                               index_t start_row,
                               index_t n_points,
                               const index_t* __restrict__ ex_scan,
                               const index_t* __restrict__ row_len,
                               const index_t* __restrict__ adj_graph,
                               const value_t* __restrict__ eps2,
                               int k,
                               std::uint8_t* __restrict__ codes,
                               index_t* __restrict__ vd_multi,
                               index_t vd_stride,
                               bool query_in_shared)
{
  const index_t row = static_cast<index_t>(blockIdx.x);
  if (row >= n_points) return;

  // Layout da shared dinâmica: [eps2 (k)][linha de consulta (D), se couber]
  extern __shared__ char smem_raw[];
  value_t* s_eps2 = reinterpret_cast<value_t*>(smem_raw);
  value_t* s_q    = s_eps2 + k;

  __shared__ index_t hist[kMaxEps];

  for (int e = threadIdx.x; e < k; e += TPB) {
    s_eps2[e] = eps2[e];
    hist[e]   = 0;
  }

  const value_t* xq = x + static_cast<std::size_t>(start_row + row) * D;
  if (query_in_shared) {
    for (index_t t = threadIdx.x; t < D; t += TPB) s_q[t] = xq[t];
  }
  __syncthreads();

  const value_t* q    = query_in_shared ? s_q : xq;
  const index_t begin = ex_scan[row];
  const index_t len   = row_len[row];

  for (index_t p = threadIdx.x; p < len; p += TPB) {
    const index_t col = adj_graph[begin + p];
    const value_t* xc = x + static_cast<std::size_t>(col) * D;

    value_t dist = 0;
    for (index_t t = 0; t < D; ++t) {
      const value_t diff = q[t] - xc[t];
      dist += diff * diff;
    }

    // A entrada veio do maior raio, então k-1 sempre satisfaz. Varrendo em ordem
    // decrescente, o último raio que satisfaz é o menor — a monotonicidade garante que
    // não há buracos.
    int code = k - 1;
    for (int e = k - 2; e >= 0; --e) {
      if (dist <= s_eps2[e]) code = e;
    }

    codes[begin + p] = static_cast<std::uint8_t>(code);
    atomicAddIndex<index_t>(&hist[code], static_cast<index_t>(1));
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    index_t acc = 0;
    for (int e = 0; e < k; ++e) {
      acc += hist[e];
      vd_multi[static_cast<std::size_t>(e) * vd_stride + row] = acc;
      atomicAddIndex<index_t>(&vd_multi[static_cast<std::size_t>(e) * vd_stride + n_points], acc);
    }
  }
}

/**
 * Compacta o CSR do maior ε para o CSR de um ε menor.
 *
 * Um bloco por linha, com prefix sum de bloco em vez de atômico: preserva a ordem das
 * colunas dentro da linha e torna a saída determinística entre execuções, o que é o que
 * permite comparar rótulos de duas execuções sem tolerância.
 */
template <typename index_t, int TPB>
__global__ void filterKernel(const index_t* __restrict__ ex_scan_src,
                             const index_t* __restrict__ row_len_src,
                             const index_t* __restrict__ adj_graph_src,
                             const std::uint8_t* __restrict__ codes,
                             index_t n_points,
                             std::uint8_t eps_index,
                             const index_t* __restrict__ ex_scan_dst,
                             index_t* __restrict__ adj_graph_dst)
{
  const index_t row = static_cast<index_t>(blockIdx.x);
  if (row >= n_points) return;

  using BlockScan = cub::BlockScan<index_t, TPB>;
  __shared__ typename BlockScan::TempStorage temp;
  __shared__ index_t base;

  if (threadIdx.x == 0) base = 0;
  __syncthreads();

  const index_t begin = ex_scan_src[row];
  const index_t len   = row_len_src[row];
  const index_t out   = ex_scan_dst[row];

  for (index_t p0 = 0; p0 < len; p0 += TPB) {
    const index_t p = p0 + static_cast<index_t>(threadIdx.x);
    const bool keep = (p < len) && (codes[begin + p] <= eps_index);

    index_t offset = 0;
    index_t total  = 0;
    BlockScan(temp).ExclusiveSum(keep ? static_cast<index_t>(1) : static_cast<index_t>(0),
                                 offset,
                                 total);

    if (keep) adj_graph_dst[out + base + offset] = adj_graph_src[begin + p];
    __syncthreads();

    if (threadIdx.x == 0) base += total;
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Invólucros de host
// ---------------------------------------------------------------------------

/** Ver annotateKernel. Zera as somas de vd_multi antes, pois são acumuladas por atômico. */
template <typename value_t, typename index_t>
void annotate(const value_t* x,
              index_t D,
              index_t start_row,
              index_t n_points,
              const index_t* ex_scan,
              const index_t* row_len,
              const index_t* adj_graph,
              const value_t* eps2,
              int k,
              std::uint8_t* codes,
              index_t* vd_multi,
              index_t vd_stride,
              cudaStream_t stream)
{
  constexpr int TPB = 128;

  const std::size_t eps_bytes   = static_cast<std::size_t>(k) * sizeof(value_t);
  const std::size_t query_bytes = static_cast<std::size_t>(D) * sizeof(value_t);
  const bool query_in_shared    = (eps_bytes + query_bytes) <= kMaxQuerySharedBytes;
  const std::size_t shmem       = eps_bytes + (query_in_shared ? query_bytes : 0);

  for (int e = 0; e < k; ++e) {
    RAFT_CUDA_TRY(cudaMemsetAsync(
      vd_multi + static_cast<std::size_t>(e) * vd_stride + n_points, 0, sizeof(index_t), stream));
  }

  annotateKernel<value_t, index_t, TPB>
    <<<static_cast<int>(n_points), TPB, shmem, stream>>>(x,
                                                         D,
                                                         start_row,
                                                         n_points,
                                                         ex_scan,
                                                         row_len,
                                                         adj_graph,
                                                         eps2,
                                                         k,
                                                         codes,
                                                         vd_multi,
                                                         vd_stride,
                                                         query_in_shared);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * CSR de um ε menor: exclusive_scan dos graus daquele raio (mesma etapa do AdjGraph do
 * cuML) seguido da compactação das entradas anotadas.
 */
template <typename index_t>
void filter(const raft::handle_t& handle,
            const index_t* ex_scan_src,
            const index_t* row_len_src,
            const index_t* adj_graph_src,
            const std::uint8_t* codes,
            index_t n_points,
            int eps_index,
            const index_t* row_len_dst,
            index_t* ex_scan_dst,
            index_t* adj_graph_dst,
            cudaStream_t stream)
{
  constexpr int TPB = 128;

  thrust::exclusive_scan(handle.get_thrust_policy(),
                         thrust::device_pointer_cast(row_len_dst),
                         thrust::device_pointer_cast(row_len_dst) + n_points,
                         thrust::device_pointer_cast(ex_scan_dst));

  filterKernel<index_t, TPB><<<static_cast<int>(n_points), TPB, 0, stream>>>(
    ex_scan_src,
    row_len_src,
    adj_graph_src,
    codes,
    n_points,
    static_cast<std::uint8_t>(eps_index),
    ex_scan_dst,
    adj_graph_dst);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace CsrEps
}  // namespace Multi
}  // namespace Dbscan
}  // namespace ML
