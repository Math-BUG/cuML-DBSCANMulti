/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Universidade Federal de Viçosa (UFV)
 * SPDX-License-Identifier: Apache-2.0
 *
 * Substituto ("shim") de <cuml/common/logger.hpp>.
 *
 * Por que existe: o logger.hpp original do cuML inclui <cuml/common/logger_macros.hpp>,
 * que NÃO existe no repositório do cuML — é gerado em tempo de build pelo CMake deles
 * (`create_logger_macros(CUML "ML::default_logger()" include/cuml/common)`,
 * cpp/CMakeLists.txt:259) — e depende da biblioteca externa rapids_logger. Como aqui o
 * build é um nvcc direto, sem o CMake do cuML, esse cabeçalho nunca seria gerado.
 *
 * Como funciona: este arquivo é encontrado antes do original porque `src/compat` vem
 * primeiro na ordem dos `-I` (ver Makefile). Os `#include <cuml/common/logger.hpp>`
 * dos arquivos vendorizados resolvem para cá, e nem logger_macros.hpp nem rapids_logger
 * entram no build.
 *
 * As macros são no-ops por padrão. Compile com -DDBSCANMULTI_VERBOSE_LOG para ligar as
 * mensagens no stderr — útil para depurar tamanho de lote, nnz por lote e número de
 * configurações, sem poluir o stdout, cuja última linha é o JSON de resultado.
 */

#pragma once

#include <cstdio>

#ifdef DBSCANMULTI_VERBOSE_LOG

#define DBSCANMULTI_LOG_IMPL(level, fmt, ...) \
  std::fprintf(stderr, "[" level "] " fmt "\n", ##__VA_ARGS__)

#define CUML_LOG_TRACE(fmt, ...)    DBSCANMULTI_LOG_IMPL("trace", fmt, ##__VA_ARGS__)
#define CUML_LOG_DEBUG(fmt, ...)    DBSCANMULTI_LOG_IMPL("debug", fmt, ##__VA_ARGS__)
#define CUML_LOG_INFO(fmt, ...)     DBSCANMULTI_LOG_IMPL("info", fmt, ##__VA_ARGS__)
#define CUML_LOG_WARN(fmt, ...)     DBSCANMULTI_LOG_IMPL("warn", fmt, ##__VA_ARGS__)
#define CUML_LOG_ERROR(fmt, ...)    DBSCANMULTI_LOG_IMPL("error", fmt, ##__VA_ARGS__)
#define CUML_LOG_CRITICAL(fmt, ...) DBSCANMULTI_LOG_IMPL("critical", fmt, ##__VA_ARGS__)

#else

#define CUML_LOG_TRACE(...)    ((void)0)
#define CUML_LOG_DEBUG(...)    ((void)0)
#define CUML_LOG_INFO(...)     ((void)0)
#define CUML_LOG_WARN(...)     ((void)0)
#define CUML_LOG_ERROR(...)    ((void)0)
#define CUML_LOG_CRITICAL(...) ((void)0)

#endif
