/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modifications Copyright (c) 2026, Universidade Federal de Viçosa (UFV).
 *
 * Este caminho histórico fazia a adaptação local de
 * third_party/cuml/cpp/src/dbscan/vertexdeg/algo.cuh (cuML v26.02.00).
 *
 * A adaptação foi eliminada: o runner agora chama ML::Dbscan::VertexDeg::run
 * diretamente por cuml_stages.cuh. O arquivo permanece apenas como encaminhamento de
 * compatibilidade e para preservar o contrato de proveniência/NOTICE desta fase.
 */

#pragma once

#include "cuml_stages.cuh"
