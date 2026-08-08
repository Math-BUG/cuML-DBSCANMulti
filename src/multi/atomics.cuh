/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Universidade Federal de Viçosa (UFV)
 * SPDX-License-Identifier: Apache-2.0
 *
 * atomicAdd para o tipo de índice do DBSCAN.
 *
 * O CUDA só oferece atomicAdd para int, unsigned int, unsigned long long, float e double —
 * não há sobrecarga para inteiro de 64 bits COM SINAL, que é o `Index_ = int64_t` usado
 * quando N ultrapassa ~46341 e o int32 passa a limitar o tamanho do lote.
 *
 * A reinterpretação para unsigned é segura: em complemento de dois a soma tem exatamente a
 * mesma representação nos dois casos, e aqui só se acumulam contagens não negativas.
 */

#pragma once

#include <cstddef>

namespace ML {
namespace Dbscan {
namespace Multi {

template <typename index_t>
__device__ inline void atomicAddIndex(index_t* addr, index_t val)
{
  if constexpr (sizeof(index_t) == 8) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr), static_cast<unsigned long long>(val));
  } else {
    atomicAdd(reinterpret_cast<int*>(addr), static_cast<int>(val));
  }
}

}  // namespace Multi
}  // namespace Dbscan
}  // namespace ML
