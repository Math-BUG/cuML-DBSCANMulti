/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Universidade Federal de Viçosa (UFV)
 * SPDX-License-Identifier: Apache-2.0
 *
 * Executável CUDA do cuML-DBSCANMulti.
 *
 * Implementa o contrato de linha de comando documentado por este projeto, usado pelas
 * ferramentas locais de benchmark e validação:
 *
 *   ./dbscan_multi --input points.f32 --output labels.i32 --n N --d D \
 *                  --eps E[,E2,...] --min-samples M[,M2,...] --json
 *   ./dbscan_multi ... --eps-min 0.1 --eps-max 0.5 --eps-step 0.1 ...
 *
 * - points.f32: matriz row-major float32, sem cabeçalho;
 * - labels.i32: um int32 por ponto e por configuração, em ordem config-major;
 * - a última linha do stdout é o JSON de resultado, com fit_ms e configuration_count.
 *
 * Extensão em relação ao contrato original: --min-samples também aceita lista. Com k
 * valores de ε e l de minPts saem k*l configurações, em ordem eps-major
 * (config = e * l + m), como reportado no campo "config_order" do JSON.
 *
 * fit_ms cobre somente a execução GPU com dados já residentes. O workspace cresce na
 * primeira chamada e é reutilizado: com --warmup >= 1 sua alocação fica fora da medição;
 * com --warmup 0 ela entra na primeira amostra. Leitura e transferências ficam fora. Com
 * --repeat R é a mediana de R execuções.
 */

#include "multi/runner_multi.cuh"

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef DBSCANMULTI_GIT_SHA
#define DBSCANMULTI_GIT_SHA "unknown"
#endif
#ifndef DBSCANMULTI_GIT_DIRTY
#define DBSCANMULTI_GIT_DIRTY -1
#endif
#ifndef DBSCANMULTI_REVISION_KIND
#define DBSCANMULTI_REVISION_KIND "unknown"
#endif
#ifndef DBSCANMULTI_BUILD_ID
#define DBSCANMULTI_BUILD_ID "unknown"
#endif
#ifndef DBSCANMULTI_CUDA_ARCH
#define DBSCANMULTI_CUDA_ARCH "unknown"
#endif
#ifndef DBSCANMULTI_BUILD_BACKEND
#define DBSCANMULTI_BUILD_BACKEND "unknown"
#endif
#ifndef DBSCANMULTI_BUILD_FLAGS
#define DBSCANMULTI_BUILD_FLAGS "unknown"
#endif

namespace {

struct Options {
  std::string input;
  std::string output;
  int n       = 0;
  int d       = 0;
  std::vector<float> eps;
  float eps_min                   = 0.0f;
  float eps_max                   = 0.0f;
  float eps_step                  = 0.0f;
  bool has_eps_range              = false;
  std::vector<int> min_samples;
  std::size_t max_bytes_per_batch = 0;
  int warmup                      = 0;
  int repeat                      = 1;
  std::string backend             = "cuvs";
  std::string index               = "auto";
  std::string route               = "auto";
  int neigh_per_row               = 0;
  bool solo                       = false;
  bool solo_isolado               = false;  // recria tudo por configuração (reprodutor)
  bool json                       = false;
  bool selftest                   = false;
  bool perfil                     = false;  // tempo por fase dentro do runner
  bool build_info                 = false;
};

std::string json_escape(const std::string& value)
{
  std::ostringstream out;
  for (const unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          const char hex[] = "0123456789abcdef";
          out << "\\u00" << hex[(c >> 4) & 0xf] << hex[c & 0xf];
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

void write_build_json(std::ostream& out)
{
  out << "{\"git_sha\":\"" << json_escape(DBSCANMULTI_GIT_SHA)
      << "\",\"revision_kind\":\"" << json_escape(DBSCANMULTI_REVISION_KIND)
      << "\",\"git_dirty\":";
  if (DBSCANMULTI_GIT_DIRTY == 0) {
    out << "false";
  } else if (DBSCANMULTI_GIT_DIRTY == 1) {
    out << "true";
  } else {
    out << "null";
  }
  out << ",\"build_id\":\"" << json_escape(DBSCANMULTI_BUILD_ID)
      << "\",\"cuda_arch\":\"" << json_escape(DBSCANMULTI_CUDA_ARCH)
      << "\",\"configured_backend\":\"" << json_escape(DBSCANMULTI_BUILD_BACKEND)
      << "\",\"compiled_backends\":[";
#ifdef DBSCANMULTI_USE_CUVS
  out << "\"cuvs\",\"codes\"";
#else
  out << "\"codes\"";
#endif
  out << "],\"flags\":\"" << json_escape(DBSCANMULTI_BUILD_FLAGS) << "\"}";
}

void write_cuda_runtime_json(std::ostream& out)
{
  int runtime_version = 0;
  int driver_version  = 0;
  int device          = -1;
  cudaDeviceProp prop{};
  const bool runtime_ok = cudaRuntimeGetVersion(&runtime_version) == cudaSuccess;
  const bool driver_ok  = cudaDriverGetVersion(&driver_version) == cudaSuccess;
  const bool device_ok  = cudaGetDevice(&device) == cudaSuccess &&
                         cudaGetDeviceProperties(&prop, device) == cudaSuccess;

  out << "{\"runtime_version\":" << (runtime_ok ? runtime_version : 0)
      << ",\"driver_version\":" << (driver_ok ? driver_version : 0)
      << ",\"device\":" << (device_ok ? device : -1);
  if (device_ok) {
    out << ",\"gpu_name\":\"" << json_escape(prop.name) << "\",\"compute_capability\":\""
        << prop.major << "." << prop.minor << "\",\"total_global_memory\":"
        << static_cast<unsigned long long>(prop.totalGlobalMem);
  }
  out << "}";
}

void write_argv_json(std::ostream& out, int argc, char** argv)
{
  out << "[";
  for (int i = 0; i < argc; ++i) {
    if (i != 0) out << ",";
    out << "\"" << json_escape(argv[i] ? argv[i] : "") << "\"";
  }
  out << "]";
}

long long parse_integer(const std::string& raw, const char* flag)
{
  try {
    std::size_t consumed = 0;
    const long long value = std::stoll(raw, &consumed, 10);
    if (consumed != raw.size()) throw std::invalid_argument("sufixo");
    return value;
  } catch (const std::exception&) {
    std::cerr << "erro: " << flag << " exige um inteiro válido, recebi '" << raw << "'\n";
    std::exit(2);
  }
}

int parse_int_value(const std::string& raw, const char* flag)
{
  const long long value = parse_integer(raw, flag);
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
    std::cerr << "erro: " << flag << " está fora do intervalo de int: '" << raw << "'\n";
    std::exit(2);
  }
  return static_cast<int>(value);
}

float parse_float_value(const std::string& raw, const char* flag)
{
  try {
    std::size_t consumed = 0;
    const float value = std::stof(raw, &consumed);
    if (consumed != raw.size() || !std::isfinite(value)) throw std::invalid_argument("não finito");
    return value;
  } catch (const std::exception&) {
    std::cerr << "erro: " << flag << " exige um número finito válido, recebi '" << raw << "'\n";
    std::exit(2);
  }
}

/** Traduz o nome do backend, validando cedo para não falhar só na GPU. */
ML::Dbscan::Multi::Backend parse_backend(const std::string& nome)
{
  if (nome == "cuvs") {
#ifdef DBSCANMULTI_USE_CUVS
    return ML::Dbscan::Multi::Backend::Cuvs;
#else
    std::cerr << "erro: este binário foi compilado sem cuVS; use --backend codes\n";
    std::exit(2);
#endif
  }
  if (nome == "codes") return ML::Dbscan::Multi::Backend::Codes;
  std::cerr << "erro: --backend deve ser 'cuvs' ou 'codes', recebi '" << nome << "'\n";
  std::exit(2);
}

void print_usage()
{
  std::cerr
    << "uso: dbscan_multi --input points.f32 --n N --d D\n"
       "                  (--eps E[,E2,...] | --eps-min A --eps-max B --eps-step S)\n"
       "                  --min-samples M[,M2,...]\n"
       "                  [--output labels.i32] [--max-mbytes-per-batch MB]\n"
       "                  [--backend cuvs|codes] [--index auto|int32|int64]\n"
       "                  [--route auto|annotated|dense]\n"
       "                  [--warmup W] [--repeat R] [--json]\n"
       "     dbscan_multi --selftest [--backend cuvs|codes] [--json]\n"
       "     dbscan_multi --build-info\n"
       "\n"
       "  --backend cuvs (padrão) usa a mesma busca de vizinhança do DBSCAN do cuML e faz\n"
       "  o multi-eps sobre o CSR dele; codes usa o kernel próprio, sem libcuvs.\n"
       "  --index auto (padrão) usa int64 quando N*N estoura int32 (N >= 46341), porque\n"
       "  int32 trava o tamanho do lote em MAX_INT/N e cada lote a mais custa uma rodada\n"
       "  de fusão de rótulos POR CONFIGURAÇÃO.\n"
       "  --route força, apenas no backend cuvs, a rota multi-eps que anota/compacta o\n"
       "  maior CSR ou a rota densa que reconstrói um CSR por raio. Use auto para medir.\n"
       "  --neigh-per-row V dimensiona o lote supondo V vizinhos por linha em vez do pior\n"
       "  caso N. É o parâmetro neigh_per_row do cuML, que o cuML não expõe. Reduz o número\n"
       "  de lotes; se V ficar muito abaixo do grau real, a alocação do CSR falha.\n"
       "  --eps e --min-samples aceitam listas separadas por vírgula. Com k valores de eps\n"
       "  e l de min-samples são avaliadas k*l configurações, em ordem eps-major\n"
       "  (config = e * l + m). Os valores efetivos e sua ordem vão no JSON.\n"
       "  A faixa --eps-min/--eps-max/--eps-step é mutuamente exclusiva com --eps.\n"
       "  Limite: no máximo 16 valores de eps.\n"
       "  --warmup W descarta W execuções antes de medir (a primeira paga o carregamento\n"
       "  do módulo CUDA); --repeat R mede R execuções e reporta a mediana em fit_ms, com\n"
       "  todos os tempos em fit_ms_all.\n"
       "  --solo mede também cada configuração ISOLADA (k=1, l=1), no mesmo processo,\n"
       "  compartilhando handle e workspace entre elas — que é como se deve chamar em laço.\n"
       "  Sai em solo_ms, e é o denominador do ganho do multi.\n"
       "  --solo-isolado recria tudo por configuração, inclusive o workspace. Existe para\n"
       "  reproduzir a degradação de 2,3x dos jobs 4862/4863; não use para medir.\n";
}

std::vector<int> parse_int_list(const std::string& raw)
{
  if (raw.empty() || raw.front() == ',' || raw.back() == ',' || raw.find(",,") != std::string::npos) {
    std::cerr << "erro: --min-samples contém um item vazio\n";
    std::exit(2);
  }
  std::vector<int> values;
  std::stringstream ss(raw);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) values.push_back(parse_int_value(item, "--min-samples"));
  }
  return values;
}

std::vector<float> parse_float_list(const std::string& raw)
{
  if (raw.empty() || raw.front() == ',' || raw.back() == ',' || raw.find(",,") != std::string::npos) {
    std::cerr << "erro: --eps contém um item vazio\n";
    std::exit(2);
  }
  std::vector<float> values;
  std::stringstream ss(raw);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) values.push_back(parse_float_value(item, "--eps"));
  }
  return values;
}

/**
 * Expande a faixa de ε. O limite máximo entra quando pertence à progressão: uma faixa não
 * divisível como 0,1..0,55 com passo 0,2 produz 0,1 / 0,3 / 0,5. Essa regra faz parte do
 * contrato de CLI documentado pelo projeto.
 */
std::vector<float> expand_eps_range(float eps_min, float eps_max, float eps_step)
{
  std::vector<float> values;
  if (eps_step <= 0.0f) {
    std::cerr << "erro: --eps-step deve ser positivo\n";
    std::exit(2);
  }
  if (eps_max < eps_min) {
    std::cerr << "erro: --eps-max deve ser maior ou igual a --eps-min\n";
    std::exit(2);
  }
  const double tolerance = 1e-9 * std::max(1.0, static_cast<double>(std::fabs(eps_max)));
  for (int i = 0;; ++i) {
    const double value = static_cast<double>(eps_min) + i * static_cast<double>(eps_step);
    if (value > static_cast<double>(eps_max) + tolerance) break;
    values.push_back(static_cast<float>(value));
    if (values.size() > static_cast<std::size_t>(ML::Dbscan::Multi::VertexDeg::kMaxEps)) break;
  }
  return values;
}

bool parse_args(int argc, char** argv, Options& opt)
{
  bool has_eps_list = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next             = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "erro: " << name << " exige um valor\n";
        std::exit(2);
      }
      return argv[++i];
    };

    if (arg == "--input") {
      opt.input = next("--input");
    } else if (arg == "--output") {
      opt.output = next("--output");
    } else if (arg == "--n") {
      opt.n = parse_int_value(next("--n"), "--n");
    } else if (arg == "--d") {
      opt.d = parse_int_value(next("--d"), "--d");
    } else if (arg == "--eps") {
      opt.eps      = parse_float_list(next("--eps"));
      has_eps_list = true;
    } else if (arg == "--eps-min") {
      opt.eps_min        = parse_float_value(next("--eps-min"), "--eps-min");
      opt.has_eps_range  = true;
    } else if (arg == "--eps-max") {
      opt.eps_max       = parse_float_value(next("--eps-max"), "--eps-max");
      opt.has_eps_range = true;
    } else if (arg == "--eps-step") {
      opt.eps_step      = parse_float_value(next("--eps-step"), "--eps-step");
      opt.has_eps_range = true;
    } else if (arg == "--min-samples") {
      opt.min_samples = parse_int_list(next("--min-samples"));
    } else if (arg == "--max-mbytes-per-batch") {
      const long long mb = parse_integer(next("--max-mbytes-per-batch"),
                                         "--max-mbytes-per-batch");
      if (mb < 0 || static_cast<unsigned long long>(mb) >
                      std::numeric_limits<std::size_t>::max() / 1000000ull) {
        std::cerr << "erro: --max-mbytes-per-batch deve ser >= 0 e caber em size_t\n";
        return false;
      }
      opt.max_bytes_per_batch = static_cast<std::size_t>(mb) * 1000000ull;
    } else if (arg == "--backend") {
      opt.backend = next("--backend");
    } else if (arg == "--index") {
      opt.index = next("--index");
    } else if (arg == "--route") {
      opt.route = next("--route");
    } else if (arg == "--neigh-per-row") {
      opt.neigh_per_row = parse_int_value(next("--neigh-per-row"), "--neigh-per-row");
    } else if (arg == "--solo") {
      opt.solo = true;
    } else if (arg == "--perfil") {
      opt.perfil = true;
    } else if (arg == "--solo-isolado") {
      opt.solo         = true;
      opt.solo_isolado = true;
    } else if (arg == "--warmup") {
      opt.warmup = parse_int_value(next("--warmup"), "--warmup");
    } else if (arg == "--repeat") {
      opt.repeat = parse_int_value(next("--repeat"), "--repeat");
    } else if (arg == "--json") {
      opt.json = true;
    } else if (arg == "--selftest") {
      opt.selftest = true;
    } else if (arg == "--build-info") {
      opt.build_info = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      std::exit(0);
    } else {
      std::cerr << "erro: argumento desconhecido '" << arg << "'\n";
      print_usage();
      return false;
    }
  }

  if (has_eps_list && opt.has_eps_range) {
    std::cerr << "erro: --eps e a faixa --eps-min/--eps-max/--eps-step são mutuamente "
                 "exclusivos\n";
    return false;
  }
  if (opt.repeat < 1) {
    std::cerr << "erro: --repeat deve ser >= 1\n";
    return false;
  }
  if (opt.warmup < 0) {
    std::cerr << "erro: --warmup deve ser >= 0\n";
    return false;
  }
  if (opt.neigh_per_row < 0) {
    std::cerr << "erro: --neigh-per-row deve ser >= 0\n";
    return false;
  }
  if (opt.has_eps_range) opt.eps = expand_eps_range(opt.eps_min, opt.eps_max, opt.eps_step);
  return true;
}

/** Ordena crescente e remove duplicatas — pré-condição dos Algoritmos 2 e 3. */
template <typename T>
void normalize_list(std::vector<T>& values, const char* flag)
{
  const std::vector<T> original = values;
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  if (values != original) {
    std::cerr << "aviso: " << flag
              << " foi ordenado/deduplicado; a ordem das configurações na saída segue os "
                 "campos \"eps\" e \"min_samples\" do JSON\n";
  }
}

std::vector<float> read_points(const std::string& path, int n, int d)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "erro: não foi possível abrir '" << path << "'\n";
    std::exit(1);
  }
  const std::streamsize bytes = f.tellg();
  const std::size_t expected  = static_cast<std::size_t>(n) * d * sizeof(float);
  if (static_cast<std::size_t>(bytes) != expected) {
    std::cerr << "erro: '" << path << "' tem " << bytes << " bytes, esperado " << expected
              << " para n=" << n << " d=" << d << " em float32\n";
    std::exit(1);
  }
  f.seekg(0);
  std::vector<float> data(static_cast<std::size_t>(n) * d);
  f.read(reinterpret_cast<char*>(data.data()), bytes);
  if (!f) {
    std::cerr << "erro: falha ao ler '" << path << "'\n";
    std::exit(1);
  }
  return data;
}

void write_labels(const std::string& path, const std::vector<int>& labels)
{
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "erro: não foi possível escrever '" << path << "'\n";
    std::exit(1);
  }
  f.write(reinterpret_cast<const char*>(labels.data()),
          static_cast<std::streamsize>(labels.size() * sizeof(int)));
  if (!f) {
    std::cerr << "erro: falha ao escrever '" << path << "'\n";
    std::exit(1);
  }
}

/** Mediana de uma amostra pequena (copia o vetor de propósito: a ordem original é reportada). */
float median_of(std::vector<float> values)
{
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  return (values.size() % 2) ? values[mid] : 0.5f * (values[mid - 1] + values[mid]);
}

/**
 * Executa o ajuste e devolve os rótulos no host, em ordem config-major.
 * O tempo medido cobre apenas a execução em GPU com os dados já residentes.
 *
 * `warmup` execuções são descartadas antes de medir. Isso importa porque a primeira
 * chamada do processo paga o carregamento do módulo CUDA e a materialização dos kernels,
 * custo que não é do algoritmo e que o baseline em Python não pagaria dentro da medição.
 * Em seguida `repeat` execuções são cronometradas individualmente; `timings` recebe todas,
 * na ordem em que ocorreram, e o chamador resume como preferir.
 */
template <typename Index_>
std::vector<int> fit_com_handle(const raft::handle_t& handle,
                                const std::vector<float>& host_points,
                                int n,
                                int d,
                                const std::vector<float>& eps,
                                const std::vector<int>& min_samples,
                                std::size_t max_bytes_per_batch,
                                ML::Dbscan::Multi::Backend backend,
                                int neigh_per_row,
                                int warmup,
                                int repeat,
                                std::vector<float>& timings,
                                rmm::device_uvector<char>* workspace = nullptr,
                                ML::Dbscan::Multi::PerfilMulti* perfil = nullptr,
                                ML::Dbscan::Multi::BuffersCsr<Index_>* buffers = nullptr)
{
  cudaStream_t stream = handle.get_stream();

  const int n_eps          = static_cast<int>(eps.size());
  const int n_min_pts      = static_cast<int>(min_samples.size());
  const std::size_t n_cfg  = static_cast<std::size_t>(n_eps) * n_min_pts;

  // O kernel compara distâncias ao quadrado; os raios são elevados aqui, uma única vez.
  std::vector<float> eps2(eps.size());
  std::transform(eps.begin(), eps.end(), eps2.begin(), [](float e) { return e * e; });

  const std::vector<Index_> min_pts_idx(min_samples.begin(), min_samples.end());

  rmm::device_uvector<float> d_points(host_points.size(), stream);
  raft::update_device(d_points.data(), host_points.data(), host_points.size(), stream);

  rmm::device_uvector<float> d_eps2(eps2.size(), stream);
  raft::update_device(d_eps2.data(), eps2.data(), eps2.size(), stream);

  rmm::device_uvector<Index_> d_min_pts(min_pts_idx.size(), stream);
  raft::update_device(d_min_pts.data(), min_pts_idx.data(), min_pts_idx.size(), stream);

  rmm::device_uvector<Index_> d_labels(static_cast<std::size_t>(n) * n_cfg, stream);

  handle.sync_stream(stream);

  auto run_once = [&]() {
    ML::Dbscan::Multi::fit_multi<float, Index_>(handle,
                                                d_points.data(),
                                                static_cast<Index_>(n),
                                                static_cast<Index_>(d),
                                                eps.data(),
                                                d_eps2.data(),
                                                n_eps,
                                                d_min_pts.data(),
                                                n_min_pts,
                                                d_labels.data(),
                                                max_bytes_per_batch,
                                                stream,
                                                backend,
                                                static_cast<Index_>(neigh_per_row),
                                                workspace,
                                                perfil,
                                                buffers);
  };

  for (int i = 0; i < warmup; ++i) {
    run_once();
    handle.sync_stream(stream);
  }

  cudaEvent_t start, stop;
  RAFT_CUDA_TRY(cudaEventCreate(&start));
  RAFT_CUDA_TRY(cudaEventCreate(&stop));

  timings.clear();
  timings.reserve(static_cast<std::size_t>(repeat));
  for (int i = 0; i < repeat; ++i) {
    float ms = 0.0f;
    RAFT_CUDA_TRY(cudaEventRecord(start, stream));
    run_once();
    RAFT_CUDA_TRY(cudaEventRecord(stop, stream));
    RAFT_CUDA_TRY(cudaEventSynchronize(stop));
    RAFT_CUDA_TRY(cudaEventElapsedTime(&ms, start, stop));
    timings.push_back(ms);
  }

  RAFT_CUDA_TRY(cudaEventDestroy(start));
  RAFT_CUDA_TRY(cudaEventDestroy(stop));

  // Os rótulos são estreitados para int32 na saída: o contrato de arquivo é .i32 e todo
  // rótulo é menor que N. A cópia extra fica fora da região cronometrada.
  std::vector<Index_> raw(static_cast<std::size_t>(n) * n_cfg);
  raft::update_host(raw.data(), d_labels.data(), raw.size(), stream);
  handle.sync_stream(stream);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    if (raw[i] < static_cast<Index_>(-1) || raw[i] >= static_cast<Index_>(n)) {
      std::ostringstream message;
      message << "rótulo CUDA inválido na posição " << i << ": " << raw[i]
              << " (esperado -1 <= label < " << n << ")";
      throw std::runtime_error(message.str());
    }
  }
  return std::vector<int>(raw.begin(), raw.end());
}

/**
 * Ver fit_com_handle; cria um raft::handle_t próprio, como faria um chamador avulso.
 *
 * O workspace é criado aqui e reaproveitado entre o warmup e os repeats. Sem isso,
 * `fit_multi` aloca e libera dezenas de GB DENTRO da região cronometrada a cada repetição —
 * a adjacência densa sozinha é N x batch_size bytes, 15 GB em N=256000 — e alocar 10 GB
 * custou 14 ms nos jobs 4864/4865.
 *
 * O ponto não é o tempo em si, é a assimetria: o laço de configurações isoladas
 * (medir_solo_impl) já compartilhava um workspace, então só a GRADE pagava. E `ganho_puro`
 * é grade dividido por isoladas — a métrica saía subestimada por construção, justamente
 * onde se quer medir o ganho.
 */
template <typename Index_>
std::vector<int> fit_impl(const std::vector<float>& host_points,
                          int n,
                          int d,
                          const std::vector<float>& eps,
                          const std::vector<int>& min_samples,
                          std::size_t max_bytes_per_batch,
                          ML::Dbscan::Multi::Backend backend,
                          int neigh_per_row,
                          int warmup,
                          int repeat,
                          std::vector<float>& timings,
                          bool reusar_workspace = true)
{
  raft::handle_t handle;
  rmm::device_uvector<char> workspace(0, handle.get_stream());
  ML::Dbscan::Multi::BuffersCsr<Index_> buffers(handle.get_stream());
  return fit_com_handle<Index_>(handle,
                                host_points,
                                n,
                                d,
                                eps,
                                min_samples,
                                max_bytes_per_batch,
                                backend,
                                neigh_per_row,
                                warmup,
                                repeat,
                                timings,
                                reusar_workspace ? &workspace : nullptr,
                                /* perfil */ nullptr,
                                reusar_workspace ? &buffers : nullptr);
}

/**
 * Uma execução instrumentada, para o --perfil. Separada de `fit` de propósito: warmup e
 * repeat não fazem sentido aqui (cada fase é cronometrada com sincronização, então repetir
 * só multiplica o custo) e o resultado nunca deve entrar em fit_ms.
 */
template <typename Index_>
void fit_perfilado_impl(const std::vector<float>& host_points,
                        int n,
                        int d,
                        const std::vector<float>& eps,
                        const std::vector<int>& min_samples,
                        std::size_t max_bytes_per_batch,
                        ML::Dbscan::Multi::Backend backend,
                        int neigh_per_row,
                        std::vector<float>& timings,
                        ML::Dbscan::Multi::PerfilMulti& perfil)
{
  raft::handle_t handle;
  fit_com_handle<Index_>(handle, host_points, n, d, eps, min_samples, max_bytes_per_batch,
                         backend, neigh_per_row, /* warmup */ 1, /* repeat */ 1, timings,
                         /* workspace */ nullptr, &perfil);
}

/** Tipo de índice do DBSCAN. `Auto` decide a partir de N — ver `int32_limita_lote`. */
enum class IndexType { Auto, Int32, Int64 };

/**
 * O tamanho de lote do cuML é limitado por `(MAX_LABEL - 1) / N`, porque o CSR de um lote indexa
 * `N * batch_size` elementos. Com int32 isso trava o lote assim que `N² >= INT_MAX`, ou
 * seja, a partir de N ≈ 46341 — e cada lote a mais significa recalcular a vizinhança na
 * segunda passagem e uma rodada extra de fusão de rótulos POR CONFIGURAÇÃO. É o mesmo
 * aviso que o cuML imprime ("Using the larger integer type might result in better
 * performance").
 */
bool int32_limita_lote(int n)
{
  return static_cast<long long>(n) * n >= static_cast<long long>(std::numeric_limits<int>::max());
}

bool usa_int64(IndexType tipo, int n)
{
  return tipo == IndexType::Int64 || (tipo == IndexType::Auto && int32_limita_lote(n));
}

IndexType parse_index_type(const std::string& nome)
{
  if (nome == "auto") return IndexType::Auto;
  if (nome == "int32") return IndexType::Int32;
  if (nome == "int64") return IndexType::Int64;
  std::cerr << "erro: --index deve ser 'auto', 'int32' ou 'int64', recebi '" << nome << "'\n";
  std::exit(2);
}

int parse_route_type(const std::string& nome)
{
  if (nome == "auto") return 0;
  if (nome == "annotated") return 1;
  if (nome == "dense") return 2;
  std::cerr << "erro: --route deve ser 'auto', 'annotated' ou 'dense', recebi '" << nome
            << "'\n";
  std::exit(2);
}

/** Ver fit_perfilado_impl; despacha entre as instanciações de índice. */
void fit_perfilado(const std::vector<float>& host_points,
                   int n,
                   int d,
                   const std::vector<float>& eps,
                   const std::vector<int>& min_samples,
                   std::size_t max_bytes_per_batch,
                   ML::Dbscan::Multi::Backend backend,
                   IndexType index_type,
                   int neigh_per_row,
                   std::vector<float>& timings,
                   ML::Dbscan::Multi::PerfilMulti& perfil)
{
  if (usa_int64(index_type, n)) {
    fit_perfilado_impl<std::int64_t>(host_points, n, d, eps, min_samples, max_bytes_per_batch,
                                     backend, neigh_per_row, timings, perfil);
  } else {
    fit_perfilado_impl<int>(host_points, n, d, eps, min_samples, max_bytes_per_batch,
                            backend, neigh_per_row, timings, perfil);
  }
}

/** Ver fit_impl; despacha entre as instanciações de índice. */
std::vector<int> fit(const std::vector<float>& host_points,
                     int n,
                     int d,
                     const std::vector<float>& eps,
                     const std::vector<int>& min_samples,
                     std::size_t max_bytes_per_batch,
                     ML::Dbscan::Multi::Backend backend,
                     IndexType index_type,
                     int neigh_per_row,
                     int warmup,
                     int repeat,
                     std::vector<float>& timings)
{
  if (usa_int64(index_type, n)) {
    return fit_impl<std::int64_t>(host_points,
                                  n,
                                  d,
                                  eps,
                                  min_samples,
                                  max_bytes_per_batch,
                                  backend,
                                  neigh_per_row,
                                  warmup,
                                  repeat,
                                  timings);
  }
  return fit_impl<int>(host_points,
                       n,
                       d,
                       eps,
                       min_samples,
                       max_bytes_per_batch,
                       backend,
                       neigh_per_row,
                       warmup,
                       repeat,
                       timings);
}

/**
 * Mede cada configuração isolada (k=1, l=1), no mesmo processo.
 *
 * `handle_unico` decide o que se compartilha entre as configurações. É um experimento, não
 * uma preferência: o job 4862 mostrou que, com um raft::handle_t novo por chamada, a
 * segunda configuração em diante custa 2,3x a primeira — mesmo quando o trabalho é
 * idêntico (variar só minPts não muda a busca de vizinhança). Com handle compartilhado, se
 * a degradação sumir, a causa é a construção do handle; se persistir, é a alocação.
 */
template <typename Index_>
void medir_solo_impl(const std::vector<float>& points,
                     int n,
                     int d,
                     const std::vector<float>& eps,
                     const std::vector<int>& min_samples,
                     std::size_t max_bytes_per_batch,
                     ML::Dbscan::Multi::Backend backend,
                     int neigh_per_row,
                     int warmup,
                     int repeat,
                     bool handle_unico,
                     std::vector<std::vector<float>>& tempos_por_config)
{
  // O workspace é o que domina: a adjacência densa é N x batch_size bytes e chega a
  // dezenas de GB. Alocá-lo e liberá-lo por configuração foi a causa da degradação de
  // 2,3x medida no job 4863 — reaproveitá-lo é a correção, e é também o que qualquer
  // chamador em laço deveria fazer.
  raft::handle_t handle;
  rmm::device_uvector<char> workspace(0, handle.get_stream());
  // Os buffers do CSR (adj_graph, adj_graph_f, codes) somam mais que o workspace em dados
  // densos — 20 GB contra 3,8 GB — e também precisam sobreviver ao laço.
  ML::Dbscan::Multi::BuffersCsr<Index_> buffers(handle.get_stream());

  for (std::size_t e = 0; e < eps.size(); ++e) {
    for (std::size_t m = 0; m < min_samples.size(); ++m) {
      const std::vector<float> eps1{eps[e]};
      const std::vector<int> mp1{min_samples[m]};
      std::vector<float> t;

      if (handle_unico) {
        // Handle e workspace compartilhados: é assim que se deve chamar em laço.
        fit_com_handle<Index_>(handle, points, n, d, eps1, mp1, max_bytes_per_batch,
                               backend, neigh_per_row, warmup, repeat, t, &workspace,
                               /* perfil */ nullptr, &buffers);
      } else {
        // Tudo novo a cada configuração, para reproduzir o comportamento antigo.
        fit_impl<Index_>(points, n, d, eps1, mp1, max_bytes_per_batch, backend,
                         neigh_per_row, warmup, repeat, t,
                         /* reusar_workspace */ false);
      }
      tempos_por_config.push_back(t);
    }
  }
}

void medir_solo(const std::vector<float>& points,
                int n,
                int d,
                const std::vector<float>& eps,
                const std::vector<int>& min_samples,
                std::size_t max_bytes_per_batch,
                ML::Dbscan::Multi::Backend backend,
                IndexType index_type,
                int neigh_per_row,
                int warmup,
                int repeat,
                bool handle_unico,
                std::vector<std::vector<float>>& tempos_por_config)
{
  if (usa_int64(index_type, n)) {
    medir_solo_impl<std::int64_t>(points, n, d, eps, min_samples, max_bytes_per_batch, backend,
                                  neigh_per_row, warmup, repeat, handle_unico,
                                  tempos_por_config);
  } else {
    medir_solo_impl<int>(points, n, d, eps, min_samples, max_bytes_per_batch, backend,
                         neigh_per_row, warmup, repeat, handle_unico, tempos_por_config);
  }
}

template <typename T>
std::string join(const std::vector<T>& values)
{
  std::ostringstream os;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) os << ",";
    os << values[i];
  }
  return os.str();
}

/** Restaura um knob de teste mesmo se uma chamada CUDA lançar uma exceção. */
template <typename T>
class ScopedValue {
 public:
  ScopedValue(T& target, T value) : target_(target), previous_(target) { target_ = value; }
  ~ScopedValue() { target_ = previous_; }
  ScopedValue(const ScopedValue&)            = delete;
  ScopedValue& operator=(const ScopedValue&) = delete;

 private:
  T& target_;
  T previous_;
};

/** Número de clusters e de pontos de ruído de uma configuração. */
void summarize(const std::vector<int>& labels,
               int n,
               std::size_t config,
               int& n_clusters,
               int& n_noise)
{
  n_clusters   = 0;
  n_noise      = 0;
  const int* p = labels.data() + config * static_cast<std::size_t>(n);
  for (int i = 0; i < n; ++i) {
    if (p[i] < 0) {
      ++n_noise;
    } else if (p[i] + 1 > n_clusters) {
      n_clusters = p[i] + 1;
    }
  }
}

/**
 * Três blobs gaussianos bem separados nas duas primeiras dimensões; as demais ficam em
 * zero.
 *
 * O preenchimento com zeros é proposital: dimensões constantes não contribuem para a
 * distância ao quadrado, então a geometria — e portanto a resposta esperada — é idêntica
 * para qualquer D. Isso permite variar D só para exercitar os ramos especializados do
 * kernel (MAX_D = 4, 8, 16, 32 e o genérico) mantendo uma asserção exata: três clusters.
 */
std::vector<float> make_blobs(int n_per_cluster, int d, unsigned seed = 42)
{
  const int n               = 3 * n_per_cluster;
  const float centers[3][2] = {{0.0f, 0.0f}, {10.0f, 0.0f}, {0.0f, 10.0f}};

  std::mt19937 rng(seed);
  std::normal_distribution<float> noise(0.0f, 0.5f);
  std::vector<float> points(static_cast<std::size_t>(n) * d, 0.0f);
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < n_per_cluster; ++i) {
      const std::size_t row = static_cast<std::size_t>(c) * n_per_cluster + i;
      points[row * d + 0]   = centers[c][0] + noise(rng);
      points[row * d + 1]   = centers[c][1] + noise(rng);
    }
  }
  return points;
}

/**
 * Renumera os clusters de uma configuração pela ordem de primeira aparição, preservando
 * -1. Rótulos de DBSCAN são invariantes a permutação: sem canonizar, duas partições iguais
 * poderiam ser declaradas diferentes só por terem numerado os grupos em outra ordem.
 */
std::vector<int> canonicalize(const std::vector<int>& labels, int n, std::size_t config)
{
  const int* p = labels.data() + config * static_cast<std::size_t>(n);
  std::vector<int> out(static_cast<std::size_t>(n), -1);
  std::vector<int> mapa(static_cast<std::size_t>(n), -1);
  int proximo = 0;
  for (int i = 0; i < n; ++i) {
    if (p[i] < 0) continue;
    if (p[i] >= n) {
      // `fit_com_handle` rejeita isto antes de chegar aqui. A guarda mantém o helper
      // seguro caso ele venha a ser reutilizado diretamente em outro teste.
      throw std::out_of_range("canonicalize recebeu rótulo fora de [0, N)");
    }
    if (mapa[static_cast<std::size_t>(p[i])] < 0) mapa[static_cast<std::size_t>(p[i])] = proximo++;
    out[static_cast<std::size_t>(i)] = mapa[static_cast<std::size_t>(p[i])];
  }
  return out;
}

/**
 * Verificação rápida sem dependência de Python, em três partes:
 *
 *   1. grade 3x3 sobre três blobs bem separados, checando as duas monotonicidades que a
 *      implementação multiparamétrica assume — com ε fixo o ruído não diminui quando
 *      minPts cresce; com minPts fixo o ruído não aumenta quando ε cresce;
 *   2. a mesma grade com o lote forçado a ser pequeno, exigindo rótulos idênticos aos do
 *      caso de lote único. É o que exercita a fusão de rótulos entre lotes, que em dados
 *      pequenos nunca roda e é a parte mais delicada herdada do cuML;
 *   3. varredura de D e de k, para percorrer os ramos especializados do kernel de
 *      vizinhança em vez de deixá-los quebrar na primeira execução com dados reais.
 */
int run_selftest(bool json, ML::Dbscan::Multi::Backend backend)
{
  const bool use_cuvs     = ML::Dbscan::Multi::backend_uses_cuvs(backend);
  const int n_per_cluster = 1000;
  const int n             = 3 * n_per_cluster;
  const int d             = 2;
  const std::vector<float> eps{0.3f, 0.5f, 0.8f};
  const std::vector<int> min_samples{4, 8, 16};

  const std::vector<float> points = make_blobs(n_per_cluster, d);

  std::vector<float> timings;
  const std::vector<int> labels =
    fit(points, n, d, eps, min_samples, 0, backend, IndexType::Int32, 0, 0, 1, timings);
  const float fit_ms = timings.front();

  const int k = static_cast<int>(eps.size());
  const int l = static_cast<int>(min_samples.size());

  std::vector<int> clusters(static_cast<std::size_t>(k) * l);
  std::vector<int> noise_counts(static_cast<std::size_t>(k) * l);
  std::ostringstream configs;

  for (int e = 0; e < k; ++e) {
    for (int m = 0; m < l; ++m) {
      const std::size_t config = static_cast<std::size_t>(e) * l + m;
      summarize(labels, n, config, clusters[config], noise_counts[config]);
      std::cerr << "[selftest] eps=" << eps[e] << " minPts=" << min_samples[m]
                << " clusters=" << clusters[config] << " ruido=" << noise_counts[config] << "\n";
      if (config) configs << ",";
      configs << "{\"eps\":" << eps[e] << ",\"min_samples\":" << min_samples[m]
              << ",\"n_clusters\":" << clusters[config] << ",\"n_noise\":" << noise_counts[config]
              << "}";
    }
  }

  bool ok = true;

  // No build cuVS os dois caminhos estão disponíveis no mesmo executável. Compará-los
  // aqui evita o falso conforto de dois selftests independentes que nunca confrontam os
  // vetores produzidos. Três execuções também detectam instabilidade do backend codes.
  bool backend_comparison_performed = false;
  int backend_mismatches            = 0;
  int backend_repeat_mismatches     = 0;
#ifdef DBSCANMULTI_USE_CUVS
  if (use_cuvs) {
    backend_comparison_performed = true;
    const std::size_t n_cfg       = static_cast<std::size_t>(k) * l;
    std::vector<bool> mismatch(n_cfg, false);
    std::vector<bool> repeat_mismatch(n_cfg, false);
    std::vector<int> codes_reference;
    for (int repetition = 0; repetition < 3; ++repetition) {
      std::vector<float> codes_timings;
      const std::vector<int> codes_labels = fit(points,
                                                n,
                                                d,
                                                eps,
                                                min_samples,
                                                0,
                                                ML::Dbscan::Multi::Backend::Codes,
                                                IndexType::Int32,
                                                0,
                                                0,
                                                1,
                                                codes_timings);
      for (std::size_t config = 0; config < n_cfg; ++config) {
        if (canonicalize(labels, n, config) != canonicalize(codes_labels, n, config)) {
          mismatch[config] = true;
        }
        if (repetition > 0 &&
            canonicalize(codes_reference, n, config) != canonicalize(codes_labels, n, config)) {
          repeat_mismatch[config] = true;
        }
      }
      if (repetition == 0) codes_reference = codes_labels;
    }
    backend_mismatches = static_cast<int>(std::count(mismatch.begin(), mismatch.end(), true));
    backend_repeat_mismatches =
      static_cast<int>(std::count(repeat_mismatch.begin(), repeat_mismatch.end(), true));
    if (backend_mismatches != 0 || backend_repeat_mismatches != 0) ok = false;
    std::cerr << "[selftest] backends: " << (n_cfg - backend_mismatches) << "/" << n_cfg
              << " configurações idênticas entre cuVS e codes; "
              << (n_cfg - backend_repeat_mismatches) << "/" << n_cfg
              << " determinísticas em 3 execuções codes\n";
  }
#endif

  // O maior ε com o menor minPts deve enxergar os três blobs.
  const std::size_t easiest = static_cast<std::size_t>(k - 1) * l + 0;
  if (clusters[easiest] != 3) {
    std::cerr << "[selftest] FALHA: esperava 3 clusters em eps=" << eps[k - 1]
              << " minPts=" << min_samples[0] << ", obtive " << clusters[easiest] << "\n";
    ok = false;
  }

  for (int e = 0; e < k; ++e) {
    for (int m = 1; m < l; ++m) {
      const std::size_t cur  = static_cast<std::size_t>(e) * l + m;
      const std::size_t prev = static_cast<std::size_t>(e) * l + (m - 1);
      if (noise_counts[cur] < noise_counts[prev]) {
        std::cerr << "[selftest] FALHA: com eps=" << eps[e] << " o ruído caiu de "
                  << noise_counts[prev] << " para " << noise_counts[cur]
                  << " ao aumentar minPts\n";
        ok = false;
      }
    }
  }

  for (int m = 0; m < l; ++m) {
    for (int e = 1; e < k; ++e) {
      const std::size_t cur  = static_cast<std::size_t>(e) * l + m;
      const std::size_t prev = static_cast<std::size_t>(e - 1) * l + m;
      if (noise_counts[cur] > noise_counts[prev]) {
        std::cerr << "[selftest] FALHA: com minPts=" << min_samples[m] << " o ruído subiu de "
                  << noise_counts[prev] << " para " << noise_counts[cur] << " ao aumentar eps\n";
        ok = false;
      }
    }
  }

  // --- parte 2: múltiplos lotes ---------------------------------------------
  // Em dados pequenos o lote cobre N inteiro e a fusão de rótulos entre lotes nunca roda,
  // apesar de ser a parte mais delicada herdada do cuML. Forçando um orçamento apertado, o
  // resultado tem de ser o MESMO: o particionamento em lotes é detalhe de execução, não do
  // algoritmo. Uma divergência aqui aponta para a fusão, não para a vizinhança.
  const std::size_t alvo_lote = static_cast<std::size_t>(n) / 4;
  const std::size_t orcamento =
    ML::Dbscan::Multi::max_bytes_for_batch_size<int>(n, k, l, alvo_lote, use_cuvs);

  std::size_t memoria_estimada = 0;
  const std::size_t lote_efetivo = ML::Dbscan::Multi::compute_batch_size_multi<int>(
    memoria_estimada, n, k, l, orcamento, use_cuvs);
  const int n_lotes = static_cast<int>((n + lote_efetivo - 1) / lote_efetivo);

  std::vector<float> timings_lotes;
  const std::vector<int> labels_lotes =
    fit(points, n, d, eps, min_samples, orcamento, backend, IndexType::Int32, 0, 0, 1,
        timings_lotes);

  int configs_divergentes = 0;
  for (std::size_t config = 0; config < static_cast<std::size_t>(k) * l; ++config) {
    if (canonicalize(labels, n, config) != canonicalize(labels_lotes, n, config)) {
      ++configs_divergentes;
      int c_ref = 0, r_ref = 0, c_lot = 0, r_lot = 0;
      summarize(labels, n, config, c_ref, r_ref);
      summarize(labels_lotes, n, config, c_lot, r_lot);
      std::cerr << "[selftest] FALHA: config " << config << " diverge entre 1 lote e "
                << n_lotes << " lotes (clusters " << c_ref << " vs " << c_lot << ", ruído "
                << r_ref << " vs " << r_lot << ")\n";
      ok = false;
    }
  }
  std::cerr << "[selftest] lotes: " << n_lotes << " x " << lote_efetivo << " pontos, "
            << (static_cast<std::size_t>(k) * l - configs_divergentes) << "/"
            << (static_cast<std::size_t>(k) * l) << " configurações idênticas ao lote único\n";
  if (n_lotes < 2) {
    std::cerr << "[selftest] FALHA: o orçamento apertado não produziu mais de um lote; a "
                 "fusão de rótulos não foi exercitada\n";
    ok = false;
  }

  // --- parte 3: ramos especializados do kernel -------------------------------
  // D escolhe entre os kernels de registrador (MAX_D = 4, 8, 16, 32) e o genérico; k
  // escolhe entre MAX_K = 1, 4 e 16. Sem esta varredura, um ramo com erro só apareceria na
  // primeira execução com dados reais daquela dimensionalidade.
  const int dims_teste[]       = {2, 8, 16, 32, 33};
  const std::vector<std::vector<float>> eps_teste{
    {0.8f}, {0.3f, 0.5f, 0.8f}, {0.2f, 0.3f, 0.5f, 0.65f, 0.8f}};
  const std::vector<int> min_pts_teste{4};
  const int n_pc_teste = 500;
  const int n_teste    = 3 * n_pc_teste;

  for (int dt : dims_teste) {
    const std::vector<float> pts = make_blobs(n_pc_teste, dt);
    bool dim_ok                  = true;
    for (const auto& eps_t : eps_teste) {
      std::vector<float> t;
      const std::vector<int> lab =
        fit(pts, n_teste, dt, eps_t, min_pts_teste, 0, backend, IndexType::Int32, 0, 0, 1, t);

      // O maior ε é o último; com minPts=4 tem de enxergar exatamente os três blobs.
      const std::size_t maior = eps_t.size() - 1;
      int c = 0, r = 0;
      summarize(lab, n_teste, maior, c, r);
      if (c != 3) {
        std::cerr << "[selftest] FALHA: D=" << dt << " k=" << eps_t.size()
                  << " esperava 3 clusters em eps=" << eps_t.back() << ", obtive " << c << "\n";
        ok = dim_ok = false;
      }

      // Monotonicidade em ε dentro deste k, que é o que valida os códigos daquele ramo.
      int r_anterior = -1;
      for (std::size_t e = 0; e < eps_t.size(); ++e) {
        int ce = 0, re = 0;
        summarize(lab, n_teste, e, ce, re);
        if (r_anterior >= 0 && re > r_anterior) {
          std::cerr << "[selftest] FALHA: D=" << dt << " k=" << eps_t.size()
                    << " ruído subiu de " << r_anterior << " para " << re << " ao aumentar eps\n";
          ok = dim_ok = false;
        }
        r_anterior = re;
      }
    }
    std::cerr << "[selftest] D=" << dt << (dim_ok ? " ok" : " FALHOU") << " para k = 1, 3 e 5\n";
  }

  // --- parte 3b: fallback sem shared memory em dimensão muito alta ----------
  // float32 em D=8193 já exige mais de 32 KiB até para uma única coluna do tile. A
  // geometria usa só as duas primeiras coordenadas; portanto D=33 e D=8193 devem gerar
  // exatamente a mesma partição no backend codes.
  const int n_pc_high = 16;
  const int n_high    = 3 * n_pc_high;
  // Cinco raios exercitam a especialização MAX_K=16 também no fallback; testar apenas
  // k=1 deixaria justamente os vetores de códigos multiparamétricos fora desta cobertura.
  const std::vector<float> eps_high{0.5f, 1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<int> min_pts_high{2};
  std::vector<float> high_timing_ref;
  std::vector<float> high_timing_fallback;
  const std::vector<int> high_ref = fit(make_blobs(n_pc_high, 33),
                                        n_high,
                                        33,
                                        eps_high,
                                        min_pts_high,
                                        0,
                                        ML::Dbscan::Multi::Backend::Codes,
                                        IndexType::Int32,
                                        0,
                                        0,
                                        1,
                                        high_timing_ref);
  const int high_d = ML::Dbscan::Multi::VertexDeg::kTileSharedBytes /
                       static_cast<int>(sizeof(float)) +
                     1;
  const std::vector<int> high_fallback = fit(make_blobs(n_pc_high, high_d),
                                             n_high,
                                             high_d,
                                             eps_high,
                                             min_pts_high,
                                             0,
                                             ML::Dbscan::Multi::Backend::Codes,
                                             IndexType::Int32,
                                             0,
                                             0,
                                             1,
                                             high_timing_fallback);
  int high_dim_mismatches = 0;
  for (std::size_t config = 0; config < eps_high.size() * min_pts_high.size(); ++config) {
    if (canonicalize(high_ref, n_high, config) !=
        canonicalize(high_fallback, n_high, config)) {
      ++high_dim_mismatches;
    }
  }
  if (high_dim_mismatches != 0) ok = false;
  std::cerr << "[selftest] alta dimensão codes: "
            << (high_dim_mismatches == 0 ? "idêntico" : "DIVERGENTE") << " em "
            << (eps_high.size() - static_cast<std::size_t>(high_dim_mismatches)) << "/"
            << eps_high.size() << " configurações entre D=33 e D=" << high_d << "\n";

  // --- parte 4: tipo de índice ----------------------------------------------
  // int64 existe para destravar o tamanho do lote (int32 o limita a MAX_INT/N), e com
  // N grande é o caminho padrão. O tipo do índice é detalhe de representação: os rótulos
  // têm de ser os mesmos. Aqui N é pequeno, então esta é a única oportunidade de exercitar
  // a instanciação int64 com uma resposta conhecida.
  std::vector<float> timings_i64;
  const std::vector<int> labels_i64 =
    fit(points, n, d, eps, min_samples, 0, backend, IndexType::Int64, 0, 0, 1, timings_i64);

  int divergentes_i64 = 0;
  for (std::size_t config = 0; config < static_cast<std::size_t>(k) * l; ++config) {
    if (canonicalize(labels, n, config) != canonicalize(labels_i64, n, config)) {
      ++divergentes_i64;
      std::cerr << "[selftest] FALHA: config " << config << " diverge entre int32 e int64\n";
      ok = false;
    }
  }
  std::cerr << "[selftest] índice: " << (static_cast<std::size_t>(k) * l - divergentes_i64) << "/"
            << (static_cast<std::size_t>(k) * l) << " configurações idênticas entre int32 e int64"
            << " (auto para N=" << n << " escolheria "
            << (usa_int64(IndexType::Auto, n) ? "int64" : "int32") << ")\n";

  // --- parte 5: correção do lote quando neigh_per_row mente -------------------
  // `--neigh-per-row` dimensiona o lote ANTES da busca, então é uma promessa sobre os
  // dados. Os jobs 4866-4871 morreram em `std::bad_alloc` porque a promessa (512) estava
  // 110x abaixo do grau real de heterogeneous_blobs: o lote saiu do tamanho do dataset e o
  // CSR não coube. Agora o runner mede o grau, aborta antes de alocar e refaz o lote.
  //
  // Aqui o teto artificial faz o papel da GPU cheia: um palpite de 1 vizinho por linha
  // produz o lote máximo, e o teto obriga a correção a acontecer. O que se exige é que os
  // rótulos saiam idênticos aos da execução sem aperto — a correção muda o lote, não a
  // resposta.
  //
  // 1 MB: apertado o bastante para o lote de 3000 linhas estourar em qualquer densidade
  // plausível, e folgado o bastante para o lote corrigido ficar na casa das dezenas de
  // linhas. Um teto muito menor levaria o pior caso a lotes de 1 linha e o teste passaria
  // a gastar minutos em 3000 fusões de rótulos.
  std::vector<float> timings_ajuste;
  std::vector<int> labels_ajuste;
  int correcoes = 0;
  {
    ScopedValue<std::size_t> teto_guard(ML::Dbscan::Multi::csr_teto_de_teste(), 1024u * 1024u);
    ScopedValue<int> correcoes_guard(ML::Dbscan::Multi::csr_correcoes_de_lote(), 0);
    labels_ajuste = fit(points, n, d, eps, min_samples, 0, backend, IndexType::Int32,
                        /* neigh_per_row */ 1, 0, 1, timings_ajuste);
    correcoes = ML::Dbscan::Multi::csr_correcoes_de_lote();
  }
  if (correcoes == 0) {
    std::cerr << "[selftest] FALHA: o teto apertado não obrigou nenhuma correção de lote; "
                 "o teste não exercitou nada\n";
    ok = false;
  }

  int divergentes_ajuste = 0;
  for (std::size_t config = 0; config < static_cast<std::size_t>(k) * l; ++config) {
    if (canonicalize(labels, n, config) != canonicalize(labels_ajuste, n, config)) {
      ++divergentes_ajuste;
      std::cerr << "[selftest] FALHA: config " << config
                << " diverge depois da correção do lote\n";
      ok = false;
    }
  }
  std::cerr << "[selftest] lote corrigido (" << correcoes << "x): "
            << (static_cast<std::size_t>(k) * l - divergentes_ajuste) << "/"
            << (static_cast<std::size_t>(k) * l)
            << " configurações idênticas com neigh-per-row mentindo (1) e o CSR apertado\n";

  // --- parte 6: as duas rotas do multi-eps ----------------------------------
  // A partir do job 4895, a rota é escolhida pelo nnz medido: grafo esparso anota o CSR do
  // maior raio e compacta; grafo denso reconstrói um CSR por raio chamando o cuVS de novo.
  // A escolha é de DESEMPENHO — anotar custava 68% do tempo em dados densos, e era o que
  // fazia o multi perder para o cuML. Os rótulos têm de ser os mesmos nas duas.
  //
  // Sem forçar, o dataset do teste escolheria uma só e a outra nunca rodaria.
  int divergentes_rota = 0;
  const bool route_tested = use_cuvs;
  if (route_tested) {
    std::vector<int> labels_rota[2];
    {
      ScopedValue<int> rota_guard(ML::Dbscan::Multi::rota_forcada(), 0);
      for (int rota = 1; rota <= 2; ++rota) {
        ML::Dbscan::Multi::rota_forcada() = rota;
        std::vector<float> t_rota;
        labels_rota[rota - 1] = fit(points,
                                    n,
                                    d,
                                    eps,
                                    min_samples,
                                    0,
                                    backend,
                                    IndexType::Int32,
                                    0,
                                    0,
                                    1,
                                    t_rota);
      }
    }

    for (std::size_t config = 0; config < static_cast<std::size_t>(k) * l; ++config) {
      if (canonicalize(labels_rota[0], n, config) != canonicalize(labels_rota[1], n, config)) {
        ++divergentes_rota;
        std::cerr << "[selftest] FALHA: config " << config
                  << " diverge entre a rota que anota e a rota densa\n";
        ok = false;
      }
    }
    std::cerr << "[selftest] rotas: " << (static_cast<std::size_t>(k) * l - divergentes_rota)
              << "/" << (static_cast<std::size_t>(k) * l)
              << " configurações idênticas entre anotar-e-compactar e um CSR por raio\n";
  } else {
    std::cerr << "[selftest] rotas: não aplicável ao backend codes\n";
  }

  std::cerr << "[selftest] " << (ok ? "PASSOU" : "FALHOU") << "\n";

  if (json) {
    std::cout << "{\"selftest\":" << (ok ? "true" : "false")
              << ",\"backend\":\"" << (use_cuvs ? "cuvs" : "codes") << "\""
              << ",\"fit_ms\":" << fit_ms
              << ",\"n\":" << n << ",\"d\":" << d << ",\"eps\":[" << join(eps)
              << "],\"min_samples\":[" << join(min_samples)
              << "],\"configuration_count\":" << (k * l)
              << ",\"config_order\":\"eps_major\",\"batches\":" << n_lotes
              << ",\"batch_size\":" << lote_efetivo
              << ",\"batched_mismatches\":" << configs_divergentes
              << ",\"index_mismatches\":" << divergentes_i64
              << ",\"batch_fixup_mismatches\":" << divergentes_ajuste
              << ",\"route_mismatches\":" << divergentes_rota
              << ",\"route_tested\":" << (route_tested ? "true" : "false")
              << ",\"backend_comparison_performed\":"
              << (backend_comparison_performed ? "true" : "false")
              << ",\"backend_mismatches\":" << backend_mismatches
              << ",\"backend_repeat_mismatches\":" << backend_repeat_mismatches
              << ",\"high_dim_mismatches\":" << high_dim_mismatches
              << ",\"configurations\":[" << configs.str() << "],\"build\":";
    write_build_json(std::cout);
    std::cout << ",\"cuda\":";
    write_cuda_runtime_json(std::cout);
    std::cout << "}" << std::endl;
  }
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
  Options opt;
  if (!parse_args(argc, argv, opt)) return 2;

  if (opt.build_info) {
    std::cout << "{\"build\":";
    write_build_json(std::cout);
    std::cout << "}" << std::endl;
    return 0;
  }

  const ML::Dbscan::Multi::Backend backend = parse_backend(opt.backend);
  const IndexType index_type               = parse_index_type(opt.index);
  const int route_type                     = parse_route_type(opt.route);

  if (route_type != 0 && !ML::Dbscan::Multi::backend_uses_cuvs(backend)) {
    std::cerr << "erro: --route só se aplica ao backend cuvs\n";
    return 2;
  }
  if (opt.selftest && route_type != 0) {
    std::cerr << "erro: --route não deve ser combinado com --selftest; o teste força as duas rotas\n";
    return 2;
  }

  if (opt.selftest) return run_selftest(opt.json, backend);

  ML::Dbscan::Multi::rota_forcada() = route_type;

  if (opt.input.empty() || opt.n <= 0 || opt.d <= 0 || opt.eps.empty() ||
      opt.min_samples.empty()) {
    std::cerr << "erro: --input, --n, --d, --eps (ou a faixa) e --min-samples são "
                 "obrigatórios\n";
    print_usage();
    return 2;
  }
  normalize_list(opt.eps, "--eps");
  normalize_list(opt.min_samples, "--min-samples");

  const float largest_safe_eps = std::sqrt(std::numeric_limits<float>::max());
  if (std::any_of(opt.eps.begin(), opt.eps.end(), [largest_safe_eps](float value) {
        return !std::isfinite(value) || value <= 0.0f || value > largest_safe_eps;
      })) {
    std::cerr << "erro: valores de eps devem ser finitos, positivos e não estourar eps²\n";
    return 2;
  }
  if (std::any_of(opt.min_samples.begin(), opt.min_samples.end(), [](int value) {
        return value <= 0;
      })) {
    std::cerr << "erro: valores de min-samples devem ser positivos\n";
    return 2;
  }
  if (opt.eps.size() > static_cast<std::size_t>(ML::Dbscan::Multi::VertexDeg::kMaxEps)) {
    std::cerr << "erro: no máximo " << ML::Dbscan::Multi::VertexDeg::kMaxEps
              << " valores de eps por execução\n";
    return 2;
  }

  const std::vector<float> points = read_points(opt.input, opt.n, opt.d);
  if (std::any_of(points.begin(), points.end(), [](float value) { return !std::isfinite(value); })) {
    std::cerr << "erro: o arquivo de entrada contém NaN ou infinito\n";
    return 2;
  }

  std::vector<float> timings;
  const std::vector<int> labels = fit(points,
                                      opt.n,
                                      opt.d,
                                      opt.eps,
                                      opt.min_samples,
                                      opt.max_bytes_per_batch,
                                      backend,
                                      index_type,
                                      opt.neigh_per_row,
                                      opt.warmup,
                                      opt.repeat,
                                      timings);
  const ML::Dbscan::Multi::ExecutionStats execution_stats =
    ML::Dbscan::Multi::last_execution_stats();
  const float fit_ms            = median_of(timings);

  if (!opt.output.empty()) write_labels(opt.output, labels);

  // O perfil roda UMA passagem extra, depois da medição e fora dela: os eventos por fase
  // sincronizam o stream e contaminariam fit_ms. Serve para localizar o gargalo, não para
  // reportar tempo.
  if (opt.perfil) {
    ML::Dbscan::Multi::PerfilMulti perf;
    std::vector<float> descartado;
    fit_perfilado(points, opt.n, opt.d, opt.eps, opt.min_samples, opt.max_bytes_per_batch,
                  backend, index_type, opt.neigh_per_row, descartado, perf);

    const std::size_t n_cfg = opt.eps.size() * opt.min_samples.size();
    const double tot = perf.total() > 0 ? perf.total() : 1.0;
    // `por_config` é o que decide o teto do ganho, então é campo explícito e não texto:
    // distinguir por substring dependia de acento e de caixa, e "relabel final" caía no
    // lado errado justamente por isso.
    struct Fase {
      const char* nome;
      double ms;
      const char* quando;
      bool por_config;
    };
    const Fase fases[] = {
      {"busca (passagem 1)", perf.busca_p1, "por lote", false},
      {"CSR (passagem 1)", perf.csr_p1, "por lote", false},
      {"anotacao (passagem 1)", perf.anotacao_p1, "por lote", false},
      {"pontos centrais", perf.nucleo, "por lote", false},
      {"busca (passagem 2)", perf.busca_p2, "por lote > 0", false},
      {"CSR (passagem 2)", perf.csr_p2, "por lote", false},
      {"anotacao (passagem 2)", perf.anotacao_p2, "por lote", false},
      {"filtro por eps", perf.filtro, "por lote e eps", false},
      {"rotulagem (weak_cc)", perf.rotulagem, "POR CONFIGURACAO", true},
      {"fusao de rotulos", perf.fusao, "POR CONFIGURACAO", true},
      {"relabel final", perf.relabel, "POR CONFIGURACAO", true},
    };

    std::fprintf(stderr,
                 "\n[perfil] %d lote(s), nnz maximo %lld, %zu configuracoes\n",
                 perf.n_lotes,
                 perf.nnz_max,
                 n_cfg);
    std::fprintf(stderr, "[perfil] %-22s %8s %6s   %s\n", "fase", "ms", "%", "quando");

    double compartilhado = 0, por_config = 0;
    for (const Fase& f : fases) {
      std::fprintf(
        stderr, "[perfil] %-22s %8.1f %5.1f%%   %s\n", f.nome, f.ms, 100.0 * f.ms / tot, f.quando);
      if (f.por_config) {
        por_config += f.ms;
      } else {
        compartilhado += f.ms;
      }
    }

    std::fprintf(stderr, "[perfil] %-22s %8.1f\n", "total", perf.total());

    // Os tempos acima somam TODAS as passagens de fit_multi que couberam no perfil, warmup
    // inclusive. Ler um deles como o custo de uma operação única foi exatamente o erro que
    // me fez enxergar uma anomalia no weak_cc que não existia: `busca` cobre 2 execuções e
    // k raios, não uma busca. A abertura abaixo já sai normalizada.
    const int execs = perf.execucoes > 0 ? perf.execucoes : 1;
    std::fprintf(stderr,
                 "\n[perfil] rotulagem por raio, POR EXECUCAO (%d execucao(oes) no total)\n",
                 execs);
    std::fprintf(stderr,
                 "[perfil] %5s %12s %14s %10s %12s\n",
                 "eps#",
                 "chamadas",
                 "arestas",
                 "ms",
                 "ms/G-aresta");
    for (int e = 0; e < static_cast<int>(opt.eps.size())
                    && e < ML::Dbscan::Multi::PerfilMulti::kMaxEpsPerfil;
         ++e) {
      const double ms       = perf.rotulagem_eps[e] / execs;
      const double arestas  = static_cast<double>(perf.nnz_rotulado[e]) / execs;
      const double por_giga = arestas > 0 ? ms / (arestas / 1e9) : 0.0;
      std::fprintf(stderr,
                   "[perfil] %5d %12.1f %14.3e %10.1f %12.1f\n",
                   e,
                   static_cast<double>(perf.chamadas_eps[e]) / execs,
                   arestas,
                   ms,
                   por_giga);
    }
    // Na rota densa a passagem 1 chama o cuVS uma vez por raio; na que anota, uma só. Sem
    // saber a rota, dividir busca_p1 pelo número de raios daria o valor errado por 3x.
    const int buscas_p1_por_exec = perf.lotes_densos > 0 ? static_cast<int>(opt.eps.size()) : 1;
    std::fprintf(stderr,
                 "[perfil] rota: %d lote(s) densos, %d anotado(s) (por execucao)\n",
                 perf.lotes_densos / execs,
                 perf.lotes_anotados / execs);
    std::fprintf(stderr,
                 "[perfil] uma varredura N^2*D custa ~%.1f ms\n",
                 perf.busca_p1 / execs / buscas_p1_por_exec);
    std::fprintf(stderr,
                 "[perfil] compartilhado entre configuracoes: %8.1f ms (%.1f%%)\n"
                 "[perfil] pago por configuracao:             %8.1f ms (%.1f ms cada)\n"
                 "[perfil] teto do ganho = total/por_config = %.1fx\n",
                 compartilhado,
                 100.0 * compartilhado / tot,
                 por_config,
                 por_config / static_cast<double>(n_cfg),
                 por_config > 0 ? perf.total() / por_config : 0.0);
  }

  // Cada configuração isolada, no MESMO processo. Compartilhar o processo é o que torna
  // esta medição comparável com a do baseline, que percorre as configurações num processo
  // Python só e portanto já roda com contexto CUDA quente e clocks em boost. Tudo o mais
  // continua isolado: handle, upload, workspace, warmup e repeat são refeitos por
  // configuração.
  std::vector<float> solo_ms;
  std::ostringstream solo_all;
  if (opt.solo) {
    std::vector<std::vector<float>> tempos;
    medir_solo(points,
               opt.n,
               opt.d,
               opt.eps,
               opt.min_samples,
               opt.max_bytes_per_batch,
               backend,
               index_type,
               opt.neigh_per_row,
               opt.warmup,
               opt.repeat,
               !opt.solo_isolado,
               tempos);
    for (const auto& t : tempos) {
      if (!solo_ms.empty()) solo_all << ",";
      solo_all << "[" << join(t) << "]";
      solo_ms.push_back(median_of(t));
    }
  }

  if (opt.json) {
    const std::size_t n_cfg = opt.eps.size() * opt.min_samples.size();
    const char* route_observed = "unknown";
    if (!ML::Dbscan::Multi::backend_uses_cuvs(backend) || opt.eps.size() <= 1) {
      route_observed = "not-applicable";
    } else if (execution_stats.annotated_batches > 0 && execution_stats.dense_batches > 0) {
      route_observed = "mixed";
    } else if (execution_stats.annotated_batches > 0) {
      route_observed = "annotated";
    } else if (execution_stats.dense_batches > 0) {
      route_observed = "dense";
    }
    std::cout << "{\"fit_ms\":" << fit_ms << ",\"fit_ms_all\":[" << join(timings)
              << "],\"backend\":\"" << (ML::Dbscan::Multi::backend_uses_cuvs(backend) ? "cuvs"
                                                                                      : "codes")
              << "\",\"index\":\"" << (usa_int64(index_type, opt.n) ? "int64" : "int32")
              << "\",\"neigh_per_row\":" << opt.neigh_per_row
              << ",\"warmup\":" << opt.warmup << ",\"repeat\":" << opt.repeat
              << ",\"configuration_count\":" << n_cfg << ",\"eps_count\":" << opt.eps.size()
              << ",\"min_samples_count\":" << opt.min_samples.size() << ",\"eps\":["
              << join(opt.eps) << "],\"min_samples\":[" << join(opt.min_samples)
              << "],\"config_order\":\"eps_major\",\"n\":" << opt.n << ",\"d\":" << opt.d;
    if (opt.solo) {
      std::cout << ",\"solo_ms\":[" << join(solo_ms) << "],\"solo_ms_all\":[" << solo_all.str()
                << "]";
    }
    std::cout << ",\"input\":\"" << json_escape(opt.input)
              << "\",\"input_bytes\":"
              << (static_cast<unsigned long long>(opt.n) * static_cast<unsigned long long>(opt.d) *
                  sizeof(float))
              << ",\"max_bytes_per_batch\":"
              << static_cast<unsigned long long>(opt.max_bytes_per_batch)
              << ",\"requested_index\":\"" << json_escape(opt.index)
              << "\",\"requested_route\":\"" << json_escape(opt.route)
              << "\",\"execution\":{\"effective_max_bytes_per_batch\":"
              << static_cast<unsigned long long>(execution_stats.max_bytes_per_batch)
              << ",\"batch_size\":" << execution_stats.batch_size
              << ",\"batches\":" << execution_stats.batches
              << ",\"attempts\":" << execution_stats.attempts
              << ",\"batch_corrections\":" << execution_stats.batch_corrections
              << ",\"dense_batches\":" << execution_stats.dense_batches
              << ",\"annotated_batches\":" << execution_stats.annotated_batches
              << ",\"batch_routes\":[";
    for (std::size_t i = 0; i < execution_stats.batch_routes.size(); ++i) {
      if (i != 0) std::cout << ",";
      std::cout << "\""
                << ML::Dbscan::Multi::batch_route_name(execution_stats.batch_routes[i])
                << "\"";
    }
    std::cout << "]"
              << ",\"route_observed\":\"" << route_observed << "\""
              << ",\"stats_scope\":\"last_measured_repeat\""
              << ",\"max_nnz\":" << execution_stats.max_nnz
              << ",\"total_nnz_max_eps\":" << execution_stats.total_nnz_max_eps
              << ",\"mean_degree_max_eps\":"
              << (opt.n > 0 ? static_cast<double>(execution_stats.total_nnz_max_eps) / opt.n
                            : 0.0)
              << "},\"build\":";
    write_build_json(std::cout);
    std::cout << ",\"cuda\":";
    write_cuda_runtime_json(std::cout);
    std::cout << ",\"argv\":";
    write_argv_json(std::cout, argc, argv);
    std::cout << "}" << std::endl;
  }
  return 0;
}
